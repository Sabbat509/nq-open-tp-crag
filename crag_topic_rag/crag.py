from __future__ import annotations

import re
from typing import Any


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def extract_strips_from_psg(text: str, mode: str = "selection") -> list[str]:
    """Mirror the CRAG repo decomposition modes: selection, excerption, fixed_num."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return []
    if mode == "selection":
        return [text]
    if mode == "fixed_num":
        final_strips = []
        window_length = 50
        words = text.split(" ")
        buf = []
        for word in words:
            buf.append(word)
            if len(buf) == window_length:
                final_strips.append(" ".join(buf))
                buf = []
        if buf:
            if final_strips and len(buf) < 10:
                final_strips[-1] += " " + " ".join(buf)
            else:
                final_strips.append(" ".join(buf))
        return final_strips
    if mode == "excerption":
        num_concatenate_strips = 3
        origin_strips = []
        for question_strip in text.split("?"):
            origin_strips.extend(split_sentences(question_strip))
        strips = []
        for strip in origin_strips:
            if strip in strips:
                continue
            if not strips or len(strip.split()) > 5:
                strips.append(strip)
            else:
                strips[-1] += " " + strip
        final_strips = []
        buf = []
        for strip in strips:
            buf.append(strip)
            if len(buf) == num_concatenate_strips:
                final_strips.append(" ".join(buf))
                buf = []
        if buf:
            final_strips.append(" ".join(buf))
        return final_strips
    raise ValueError(f"Unsupported decompose mode: {mode}")


def load_t5_retrieval_evaluator(model_path: str, device: str | None = None) -> dict[str, Any]:
    """Load the official CRAG-style T5 sequence-classification evaluator."""
    import torch
    from transformers import T5ForSequenceClassification, T5Tokenizer

    actual_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForSequenceClassification.from_pretrained(model_path, num_labels=1)
    model.to(actual_device)
    model.eval()
    return {"tokenizer": tokenizer, "model": model, "device": actual_device}


def _score_with_t5(question: str, texts: list[str], evaluator: dict[str, Any], max_length: int = 512) -> list[float]:
    import torch

    tokenizer = evaluator["tokenizer"]
    model = evaluator["model"]
    device = evaluator["device"]
    scores = []
    for text in texts:
        if len(str(text).split()) < 4:
            scores.append(-1.0)
            continue
        input_content = f"{question} [SEP] {text}"
        inputs = tokenizer(
            input_content,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        with torch.no_grad():
            outputs = model(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device))
        scores.append(float(outputs["logits"].detach().cpu().reshape(-1)[0]))
    return scores


def _score_with_embeddings(question_embedding: Any, texts: list[str], embedding_model: Any) -> list[float]:
    import numpy as np

    if not texts:
        return []
    embeddings = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    return [float(score) for score in embeddings @ question_embedding]


def score_texts(
    question: str,
    texts: list[str],
    *,
    question_embedding: Any,
    embedding_model: Any,
    evaluator: dict[str, Any] | None = None,
) -> list[float]:
    if evaluator is not None:
        return _score_with_t5(question, texts, evaluator)
    return _score_with_embeddings(question_embedding, texts, embedding_model)


def crag_action_from_scores(scores: list[float], upper_threshold: float, lower_threshold: float) -> tuple[str, list[str]]:
    """Official CRAG flag reduction: any 2 => correct, else any 1 => ambiguous, else incorrect."""
    flags = []
    for score in scores:
        if score >= upper_threshold:
            flags.append("2")
        elif score >= lower_threshold:
            flags.append("1")
        else:
            flags.append("0")
    if "2" in flags:
        return "correct", flags
    if "1" in flags:
        return "ambiguous", flags
    return "incorrect", flags


def evaluate_retrieval(
    question: str,
    question_embedding: Any,
    retrieved_chunks: list[dict[str, Any]],
    embedding_model: Any,
    upper_threshold: float,
    lower_threshold: float,
    evaluator: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate retrieved passages using the CRAG repo's score->flag->action process."""
    if not retrieved_chunks:
        return {"state": "incorrect", "confidence": 0.0, "flags": [], "chunk_scores": []}

    texts = [str(chunk.get("text", "")) for chunk in retrieved_chunks]
    scores = score_texts(
        question,
        texts,
        question_embedding=question_embedding,
        embedding_model=embedding_model,
        evaluator=evaluator,
    )
    state, flags = crag_action_from_scores(scores, upper_threshold, lower_threshold)
    chunk_scores = []
    for chunk, score, flag in zip(retrieved_chunks, scores, flags):
        chunk_scores.append(
            {
                "rank": chunk.get("rank"),
                "score": chunk.get("score"),
                "crag_score": float(score),
                "crag_flag": flag,
                "topic_id": chunk.get("topic_id"),
                "article_id": chunk.get("article_id"),
                "title": chunk.get("title"),
                "chunk_id": chunk.get("chunk_id"),
            }
        )
    return {"state": state, "confidence": max(scores) if scores else 0.0, "flags": flags, "chunk_scores": chunk_scores}


def prepare_knowledge(
    question: str,
    question_embedding: Any,
    retrieved_chunks: list[dict[str, Any]],
    embedding_model: Any,
    *,
    decompose_mode: str,
    top_n: int,
    evaluator: dict[str, Any] | None = None,
    knowledge_source: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Decompose retrieved passages and select the top relevant strips, as in CRAG knowledge prep."""
    strip_rows = []
    for chunk in retrieved_chunks:
        for strip_index, strip in enumerate(extract_strips_from_psg(str(chunk.get("text", "")), decompose_mode)):
            strip_rows.append({"chunk": chunk, "strip_index": strip_index, "strip": strip})
    if not strip_rows:
        return [], []

    scores = score_texts(
        question,
        [row["strip"] for row in strip_rows],
        question_embedding=question_embedding,
        embedding_model=embedding_model,
        evaluator=evaluator,
    )
    for row, score in zip(strip_rows, scores):
        row["score"] = float(score)
    strip_rows.sort(key=lambda row: row["score"], reverse=True)
    selected = strip_rows[: max(1, min(top_n, len(strip_rows)))]

    recomposed = []
    evidence = []
    for rank, row in enumerate(selected, 1):
        chunk = row["chunk"]
        item = {
            **chunk,
            "rank": rank,
            "text": row["strip"],
            "source_rank": chunk.get("rank"),
            "strip_index": row["strip_index"],
            "crag_sentence_score": row["score"],
            "knowledge_source": knowledge_source,
        }
        recomposed.append(item)
        evidence.append(
            {
                "rank": chunk.get("rank"),
                "title": chunk.get("title"),
                "topic_id": chunk.get("topic_id"),
                "strip_index": row["strip_index"],
                "score": row["score"],
                "source": knowledge_source,
                "text": row["strip"],
            }
        )
    return recomposed, evidence


def combine_ambiguous_knowledge(
    internal_chunks: list[dict[str, Any]], external_chunks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Mirror CRAG's ambiguous branch: Knowledge1 internal + Knowledge2 external."""
    combined = []
    for rank, chunk in enumerate(internal_chunks + external_chunks, 1):
        label = "Knowledge1" if chunk.get("knowledge_source") == "internal" else "Knowledge2"
        item = dict(chunk)
        item["rank"] = rank
        item["text"] = f"{label}: {chunk.get('text', '')}"
        item["knowledge_source"] = "combined"
        combined.append(item)
    return combined
