#!/usr/bin/env python3
"""Attach BERTopic topic IDs to the existing fullwiki vector index for CRAG."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from naive_rag.index_io import load_json, read_jsonl, save_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a topic-aware index for CRAG from an existing vector index.")
    parser.add_argument("--base-index-dir", type=Path, default=Path("result/naive_rag"))
    parser.add_argument("--topic-map", type=Path, default=Path("result/crag_topic_rag/topic_partition/topic_article_map.jsonl"))
    parser.add_argument("--topic-info", type=Path, default=Path("result/crag_topic_rag/topic_partition/topic_info_40.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("result/crag_topic_rag/topic_index"))
    return parser.parse_args()


def load_topic_map(path: Path) -> dict[str, int]:
    rows = read_jsonl(path)
    mapping = {}
    for row in rows:
        article_id = str(row.get("article_id"))
        if article_id and article_id != "None":
            mapping[article_id] = int(row["topic_id"])
    if not mapping:
        raise ValueError(f"No article topics loaded from {path}")
    return mapping


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(args.base_index_dir / "manifest.json")
    chunks_path = Path(manifest["chunks_path"])
    embeddings_path = Path(manifest["embeddings_path"])
    if not chunks_path.is_absolute():
        chunks_path = args.base_index_dir / chunks_path.name
    if not embeddings_path.is_absolute():
        embeddings_path = args.base_index_dir / embeddings_path.name

    topic_map = load_topic_map(args.topic_map)
    chunks = read_jsonl(chunks_path)
    topic_counts: Counter[int] = Counter()
    missing = 0
    for chunk in chunks:
        article_id = str(chunk.get("article_id"))
        topic_id = topic_map.get(article_id)
        if topic_id is None:
            missing += 1
            chunk["topic_id"] = None
        else:
            chunk["topic_id"] = int(topic_id)
            topic_counts[int(topic_id)] += 1

    out_chunks = args.output_dir / "chunks.jsonl"
    out_manifest = args.output_dir / "manifest.json"
    write_jsonl(out_chunks, chunks)
    save_json(
        out_manifest,
        {
            "created_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "base_index_dir": str(args.base_index_dir.resolve()),
            "topic_map_path": str(args.topic_map.resolve()),
            "topic_info_path": str(args.topic_info.resolve()) if args.topic_info else None,
            "embedding_model": manifest["embedding_model"],
            "chunk_count": len(chunks),
            "topic_assigned_chunk_count": len(chunks) - missing,
            "topic_missing_chunk_count": missing,
            "topic_count_including_outlier": len(topic_counts),
            "chunks_path": str(out_chunks.resolve()),
            "embeddings_path": str(embeddings_path.resolve()),
            "base_manifest": manifest,
        },
    )
    print(f"Saved topic-aware chunks: {out_chunks}")
    print(f"Saved manifest: {out_manifest}")
    print(f"Assigned topics to {len(chunks) - missing:,}/{len(chunks):,} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
