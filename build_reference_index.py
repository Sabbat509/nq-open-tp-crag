#!/usr/bin/env python3
"""Build the same normalized NumPy vector index used by the reference naive RAG."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from naive_rag.chunking import chunk_documents
from naive_rag.index_io import l2_normalize, load_pickle, save_json, write_jsonl


def encode_texts(texts: list[str], embedding_model_name: str, batch_size: int, device: str | None):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    if not texts:
        raise ValueError("No chunks were created; cannot build an index.")
    model_kwargs = {"device": device} if device else {}
    model = SentenceTransformer(embedding_model_name, **model_kwargs)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return l2_normalize(np.asarray(embeddings, dtype=np.float32))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-data", type=Path, default=Path("result/intermediate/doc_data.pkl"))
    parser.add_argument("--output-dir", type=Path, default=Path("result/naive_rag"))
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size-words", type=int, default=400)
    parser.add_argument("--chunk-overlap-words", type=int, default=80)
    parser.add_argument("--min-chunk-words", type=int, default=40)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()

    started = time.perf_counter()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_data = load_pickle(args.doc_data)
    print(f"Loaded doc_data: {args.doc_data}")
    print(f"Documents available: {len(doc_data):,}")
    chunks = chunk_documents(
        doc_data=doc_data,
        chunk_size_words=args.chunk_size_words,
        chunk_overlap_words=args.chunk_overlap_words,
        min_chunk_words=args.min_chunk_words,
        max_documents=args.max_documents,
    )
    print(f"Chunks created: {len(chunks):,}")
    embeddings = encode_texts(
        [chunk["text"] for chunk in chunks],
        args.embedding_model,
        args.batch_size,
        args.device,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = args.output_dir / "chunks.jsonl"
    embeddings_path = args.output_dir / "embeddings.npy"
    manifest_path = args.output_dir / "manifest.json"
    write_jsonl(chunks_path, chunks)
    import numpy as np
    np.save(embeddings_path, embeddings)
    manifest: dict[str, Any] = {
        "created_at": timestamp,
        "doc_data_path": str(args.doc_data),
        "embedding_model": args.embedding_model,
        "embedding_shape": list(embeddings.shape),
        "chunk_count": len(chunks),
        "document_count_available": len(doc_data),
        "document_count_indexed": args.max_documents or len(doc_data),
        "chunk_size_words": args.chunk_size_words,
        "chunk_overlap_words": args.chunk_overlap_words,
        "min_chunk_words": args.min_chunk_words,
        "chunks_path": str(chunks_path),
        "embeddings_path": str(embeddings_path),
        "topic_map_path": None,
        "topic_aware": False,
        "topic_assigned_chunk_count": 0,
        "topic_missing_chunk_count": len(chunks),
        "index_type": "normalized_numpy_dot_product",
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    save_json(manifest_path, manifest)
    print(f"Saved chunks: {chunks_path}")
    print(f"Saved embeddings: {embeddings_path}")
    print(f"Saved manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
