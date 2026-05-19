#!/usr/bin/env python3
"""Evaluate topic-partitioned CRAG on NQ-Open."""

from __future__ import annotations

import argparse
import csv
import json
import re
import string
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from crag_topic_rag.crag import combine_ambiguous_knowledge, evaluate_retrieval, load_t5_retrieval_evaluator, prepare_knowledge
from crag_topic_rag.topic_router import SemanticTopicRouter, extract_knowledge_points, load_topic_profiles
from naive_rag.index_io import l2_normalize_vector, load_json, read_jsonl
from naive_rag.llm_client import call_chat_completion, load_default_env_files, resolve_llm_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CRAG + Kiraffe-style topic RAG on NQ-Open.")
    parser.add_argument("--index-dir", type=Path, default=Path("datasets/nq_open/indexes/fullwiki/topic_index"))
    parser.add_argument("--data-file", type=Path, default=Path("datasets/nq_open/raw/nq_open_test.jsonl"))
    parser.add_argument("--topic-info", type=Path, default=Path("datasets/nq_open/indexes/fullwiki/topic_partition/topic_info_40.csv"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/nq_open/topic_crag/reference_crag_topic_fullwiki"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fallback-top-k", type=int, default=5)
    parser.add_argument("--router-top-topics", type=int, default=5)
    parser.add_argument("--router-max-knowledge-points", type=int, default=6)
    parser.add_argument("--router-max-keywords", type=int, default=12)
    parser.add_argument("--include-outlier-topic", action="store_true")
    parser.add_argument("--evaluator-path", help="Optional CRAG T5 retrieval evaluator path. If omitted, embedding scores use the same CRAG flag process.")
    parser.add_argument("--evaluator-device")
    parser.add_argument("--upper-threshold", type=float, default=0.60, help="CRAG score threshold for a relevant/correct passage flag.")
    parser.add_argument("--lower-threshold", type=float, default=0.45, help="CRAG score threshold for an ambiguous passage flag.")
    parser.add_argument("--decompose-mode", choices=["selection", "excerption", "fixed_num"], default="selection")
    parser.add_argument("--internal-top-n", type=int, default=3)
    parser.add_argument("--external-top-n", type=int, default=5)
    parser.add_argument("--device")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--router-temperature", type=float, default=0.0)
    parser.add_argument("--answer-temperature", type=float, default=0.0)
    parser.add_argument("--answer-max-tokens", type=int, default=48)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-save-prompts", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def normalize_answer(text: Any) -> str:
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_tokens(text: Any) -> list[str]:
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def relaxed_accuracy(prediction: str, gold: str) -> float:
    pred = normalize_answer(prediction)
    truth = normalize_answer(gold)
    if not pred or not truth:
        return float(pred == truth)
    if truth in {"yes", "no"} or pred in {"yes", "no"}:
        return float(pred == truth)
    return float(pred == truth or truth in pred or pred in truth)


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = answer_tokens(prediction)
    gold_tokens = answer_tokens(gold)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max(metric_fn: Any, prediction: str, golds: list[str]) -> float:
    return max((metric_fn(prediction, gold) for gold in golds), default=0.0)


def semantic_similarity(model: Any, prediction: str, gold: str) -> float:
    import numpy as np

    pred = normalize_answer(prediction)
    truth = normalize_answer(gold)
    if not pred or not truth:
        return float(pred == truth)
    embeddings = model.encode([pred, truth], convert_to_numpy=True, show_progress_bar=False)
    left = l2_normalize_vector(np.asarray(embeddings[0], dtype=np.float32))
    right = l2_normalize_vector(np.asarray(embeddings[1], dtype=np.float32))
    return float(left @ right)


def load_examples(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_index(index_dir: Path):
    import numpy as np

    manifest = load_json(index_dir / "manifest.json")
    chunks_path = Path(manifest["chunks_path"])
    embeddings_path = Path(manifest["embeddings_path"])
    chunks = read_jsonl(chunks_path)
    embeddings = np.load(embeddings_path, mmap_mode="r")
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(f"Index mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings")
    return manifest, chunks, embeddings


def build_topic_index(chunks: list[dict[str, Any]]) -> dict[int, list[int]]:
    topic_index: dict[int, list[int]] = {}
    for index, chunk in enumerate(chunks):
        topic_id = chunk.get("topic_id")
        if topic_id is None:
            continue
        topic_index.setdefault(int(topic_id), []).append(index)
    return topic_index


def retrieve_from_indices(question_embedding: Any, chunks: list[dict[str, Any]], embeddings: Any, candidate_indices: Any, top_k: int) -> list[dict[str, Any]]:
    import numpy as np

    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if candidate_indices.size == 0:
        return []
    candidate_scores = embeddings[candidate_indices] @ question_embedding
    top_k = min(top_k, len(candidate_indices))
    if top_k == len(candidate_indices):
        local_top_indices = np.argsort(-candidate_scores)
    else:
        local_top_indices = np.argpartition(-candidate_scores, top_k - 1)[:top_k]
        local_top_indices = local_top_indices[np.argsort(-candidate_scores[local_top_indices])]
    results = []
    for rank, local_index in enumerate(local_top_indices, 1):
        index = int(candidate_indices[int(local_index)])
        chunk = dict(chunks[index])
        chunk["rank"] = rank
        chunk["score"] = float(candidate_scores[int(local_index)])
        results.append(chunk)
    return results


def retrieve_from_topics(question_embedding: Any, chunks: list[dict[str, Any]], embeddings: Any, topic_index: dict[int, list[int]], topic_ids: list[int], top_k: int) -> list[dict[str, Any]]:
    candidate_indices = [index for topic_id in topic_ids for index in topic_index.get(int(topic_id), [])]
    return retrieve_from_indices(question_embedding, chunks, embeddings, candidate_indices, top_k)


def retrieve_full(question_embedding: Any, chunks: list[dict[str, Any]], embeddings: Any, top_k: int) -> list[dict[str, Any]]:
    import numpy as np

    return retrieve_from_indices(question_embedding, chunks, embeddings, np.arange(len(chunks)), top_k)


def build_prompt(question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    context_blocks = []
    for item in retrieved_chunks:
        context_blocks.append(
            "\n".join(
                [
                    f"[{item['rank']}] Title: {item.get('title', 'Unknown')}",
                    f"Article ID: {item.get('article_id')}",
                    f"Topic ID: {item.get('topic_id')}",
                    f"Context: {item.get('text', '')}",
                ]
            )
        )
    return f"""Use only the context below to answer the NQ-Open question.
Return the shortest possible answer only.
Return only the entity, name, date, number, yes, or no when possible.
Do not write a sentence.
Do not explain.
Do not include citations.
If the answer is not in the context, make the best short answer guess from the context.

Context:
{chr(10).join(context_blocks)}

Question:
{question}

Short answer:
"""


def evidence_titles(example: dict[str, Any]) -> list[str]:
    titles = []
    for item in example.get("evidence_list") or []:
        title = item.get("title")
        if title and title not in titles:
            titles.append(title)
    return titles


def compact_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": item.get("rank"),
            "score": item.get("score"),
            "topic_id": item.get("topic_id"),
            "article_id": item.get("article_id"),
            "title": item.get("title"),
            "chunk_id": item.get("chunk_id"),
            "source_rank": item.get("source_rank"),
            "strip_index": item.get("strip_index"),
            "crag_sentence_score": item.get("crag_sentence_score"),
            "knowledge_source": item.get("knowledge_source"),
        }
        for item in chunks
    ]


def existing_ids(records_path: Path) -> set[str]:
    ids = set()
    if not records_path.exists():
        return ids
    with records_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in record:
                ids.add(str(record["id"]))
    return ids


def read_existing_rows(records_path: Path) -> list[dict[str, Any]]:
    if not records_path.exists():
        return []
    rows = []
    with records_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "id", "question", "gold_answers", "prediction", "em", "acc", "f1",
        "crag_state", "crag_confidence", "selected_topic_ids", "retrieved_titles", "latency_sec",
    ]
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in fieldnames})


def write_summary(path: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    count = len(rows)
    summary = {
        "count": count,
        "config": config,
        "metrics": {
            "EM": sum(row["em"] for row in rows) / count if count else 0.0,
            "Acc": sum(row["acc"] for row in rows) / count if count else 0.0,
            "F1": sum(row["f1"] for row in rows) / count if count else 0.0,
        },
        "metrics_by_question_type": {},
        "crag": {
            "correct": sum(1 for row in rows if row.get("crag_state") == "correct"),
            "ambiguous": sum(1 for row in rows if row.get("crag_state") == "ambiguous"),
            "incorrect": sum(1 for row in rows if row.get("crag_state") == "incorrect"),
        },
    }
    for question_type in sorted({str(row.get("question_type", "unknown")) for row in rows}):
        subset = [row for row in rows if str(row.get("question_type", "unknown")) == question_type]
        n = len(subset)
        summary["metrics_by_question_type"][question_type] = {
            "count": n,
            "EM": sum(row["em"] for row in subset) / n if n else 0.0,
            "Acc": sum(row["acc"] for row in subset) / n if n else 0.0,
            "F1": sum(row["f1"] for row in subset) / n if n else 0.0,
        }
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    import numpy as np
    from sentence_transformers import SentenceTransformer

    load_default_env_files(args.env_file)
    base_url, model_name, api_key = resolve_llm_settings(args.base_url, args.model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    predictions_csv_path = args.output_dir / "predictions.csv"
    summary_path = args.output_dir / "summary.json"

    manifest, chunks, embeddings = load_index(args.index_dir)
    topic_profiles = load_topic_profiles(args.topic_info, args.router_max_keywords, args.include_outlier_topic)
    topic_index = build_topic_index(chunks)
    if not topic_index:
        raise ValueError("The CRAG topic index has no topic_id assignments. Run build_crag_topic_index.py first.")

    embedding_model = SentenceTransformer(manifest["embedding_model"], device=args.device)
    retrieval_evaluator = None
    if args.evaluator_path:
        retrieval_evaluator = load_t5_retrieval_evaluator(args.evaluator_path, args.evaluator_device or args.device)
    router = SemanticTopicRouter(topic_profiles, embedding_model, args.router_top_topics)
    examples = load_examples(args.data_file)
    end = len(examples) if args.limit is None else min(len(examples), args.start + args.limit)
    selected_indices = list(range(args.start, end))
    done_ids = existing_ids(records_path) if args.resume else set()
    rows_for_summary = read_existing_rows(records_path) if args.resume else []

    config = {
        "dataset": "nq_open",
        "source": "ehsk/OpenQA-eval via NQ_FiD gold rows",
        "split": "test",
        "start": args.start,
        "limit": args.limit,
        "index_dir": str(args.index_dir.resolve()),
        "topic_info_path": str(args.topic_info.resolve()),
        "top_k": args.top_k,
        "fallback_top_k": args.fallback_top_k,
        "router_top_topics": args.router_top_topics,
        "router_max_knowledge_points": args.router_max_knowledge_points,
        "router_max_keywords": args.router_max_keywords,
        "topic_partitioning": True,
        "crag": True,
        "crag_pipeline": "official_adapted_topic_partition_fullwiki_fallback",
        "evaluator_path": args.evaluator_path,
        "evaluator_type": "t5" if args.evaluator_path else "embedding_fallback",
        "upper_threshold": args.upper_threshold,
        "lower_threshold": args.lower_threshold,
        "decompose_mode": args.decompose_mode,
        "internal_top_n": args.internal_top_n,
        "external_top_n": args.external_top_n,
        "model_name": model_name,
        "embedding_model": manifest["embedding_model"],
    }

    with records_path.open("a", encoding="utf-8") as records_file:
        for ordinal, index in enumerate(selected_indices, 1):
            example = examples[index]
            example_id = str(example.get("_id") or example.get("id") or index)
            if example_id in done_ids:
                continue
            started_at = time.perf_counter()
            question = str(example["question"])
            gold_answers = [str(answer) for answer in example["answer"]]

            knowledge_points = extract_knowledge_points(
                question=question,
                base_url=base_url,
                model_name=model_name,
                api_key=api_key,
                max_knowledge_points=args.router_max_knowledge_points,
                temperature=args.router_temperature,
            )
            ranked_topics = router.rank(knowledge_points)
            selected_topic_ids = [topic["topic_id"] for topic in ranked_topics]
            query_embedding = embedding_model.encode([question], convert_to_numpy=True, show_progress_bar=False)[0]
            query_embedding = l2_normalize_vector(np.asarray(query_embedding, dtype=np.float32))

            topic_chunks = retrieve_from_topics(query_embedding, chunks, embeddings, topic_index, selected_topic_ids, args.top_k)
            crag_eval = evaluate_retrieval(
                question,
                query_embedding,
                topic_chunks,
                embedding_model,
                upper_threshold=args.upper_threshold,
                lower_threshold=args.lower_threshold,
                evaluator=retrieval_evaluator,
            )
            crag_state = crag_eval["state"]

            internal_chunks, internal_evidence = prepare_knowledge(
                question,
                query_embedding,
                topic_chunks,
                embedding_model,
                decompose_mode=args.decompose_mode,
                top_n=args.internal_top_n,
                evaluator=retrieval_evaluator,
                knowledge_source="internal",
            )
            external_chunks = []
            external_knowledge_chunks = []
            external_evidence = []
            if crag_state in {"ambiguous", "incorrect"}:
                # CRAG uses web search for this branch. For HotpotQA we adapt it to
                # query the larger fullwiki index instead, preserving the branch logic.
                external_chunks = retrieve_full(query_embedding, chunks, embeddings, args.fallback_top_k)
                external_knowledge_chunks, external_evidence = prepare_knowledge(
                    question,
                    query_embedding,
                    external_chunks,
                    embedding_model,
                    decompose_mode=args.decompose_mode,
                    top_n=args.external_top_n,
                    evaluator=retrieval_evaluator,
                    knowledge_source="external_fullwiki",
                )

            if crag_state == "correct":
                selected_knowledge_chunks = internal_chunks
            elif crag_state == "ambiguous":
                selected_knowledge_chunks = combine_ambiguous_knowledge(internal_chunks, external_knowledge_chunks)
            else:
                selected_knowledge_chunks = external_knowledge_chunks
            selected_evidence = internal_evidence + external_evidence
            prompt = build_prompt(question, selected_knowledge_chunks)
            prediction = call_chat_completion(
                messages=[
                    {"role": "system", "content": "You answer NQ-Open questions with the shortest possible answer only."},
                    {"role": "user", "content": prompt},
                ],
                base_url=base_url,
                model_name=model_name,
                api_key=api_key,
                temperature=args.answer_temperature,
                max_tokens=args.answer_max_tokens,
            ).strip()

            row = {
                "id": example_id,
                "question": question,
                "gold_answers": gold_answers,
                "prediction": prediction,
                "em": metric_max(exact_match, prediction, gold_answers),
                "acc": metric_max(relaxed_accuracy, prediction, gold_answers),
                "f1": metric_max(token_f1, prediction, gold_answers),
                "knowledge_points": knowledge_points,
                "router_topics": ranked_topics,
                "selected_topic_ids": selected_topic_ids,
                "crag_state": crag_state,
                "crag_confidence": crag_eval["confidence"],
                "crag_flags": crag_eval["flags"],
                "crag_chunk_scores": crag_eval["chunk_scores"],
                "topic_retrieved_chunks": compact_chunks(topic_chunks),
                "external_retrieved_chunks": compact_chunks(external_chunks),
                "internal_knowledge_chunks": compact_chunks(internal_chunks),
                "external_knowledge_chunks": compact_chunks(external_knowledge_chunks),
                "selected_knowledge_chunks": compact_chunks(selected_knowledge_chunks),
                "selected_evidence": selected_evidence,
                "retrieved_titles": [item.get("title") for item in selected_knowledge_chunks],
                "latency_sec": round(time.perf_counter() - started_at, 3),
            }
            if not args.no_save_prompts:
                row["prompt"] = prompt
            records_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            records_file.flush()
            append_csv_row(
                predictions_csv_path,
                {
                    **row,
                    "selected_topic_ids": json.dumps(selected_topic_ids),
                    "retrieved_titles": json.dumps(row["retrieved_titles"], ensure_ascii=False),
                },
            )
            rows_for_summary.append(row)
            write_summary(summary_path, rows_for_summary, config)
            if args.progress_every and len(rows_for_summary) % args.progress_every == 0:
                metrics = load_json(summary_path)["metrics"]
                print(
                    f"{len(rows_for_summary)}/{end - args.start} examples | "
                    f"EM {metrics['EM']:.4f} Acc {metrics['Acc']:.4f} F1 {metrics['F1']:.4f} "
                    ""
                )
            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

    write_summary(summary_path, rows_for_summary, config)
    print(f"Records: {records_path.resolve()}")
    print(f"Predictions CSV: {predictions_csv_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
