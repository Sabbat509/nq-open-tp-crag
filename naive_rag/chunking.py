from __future__ import annotations

import re
from typing import Any, Iterator


def normalize_whitespace(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def word_chunks(
    text: str,
    chunk_size_words: int,
    chunk_overlap_words: int,
    min_chunk_words: int,
) -> Iterator[tuple[int, int, str]]:
    """Yield word-window chunks as (start_word, end_word, chunk_text)."""
    text = normalize_whitespace(text)
    if not text:
        return

    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than 0.")
    if chunk_overlap_words < 0:
        raise ValueError("chunk_overlap_words cannot be negative.")
    if chunk_overlap_words >= chunk_size_words:
        raise ValueError("chunk_overlap_words must be smaller than chunk_size_words.")
    if min_chunk_words <= 0:
        raise ValueError("min_chunk_words must be greater than 0.")

    words = text.split()
    step = chunk_size_words - chunk_overlap_words

    for start in range(0, len(words), step):
        end = min(start + chunk_size_words, len(words))
        if end - start < min_chunk_words:
            break
        yield start, end, " ".join(words[start:end])
        if end == len(words):
            break


def chunk_documents(
    doc_data: list[dict[str, Any]],
    chunk_size_words: int,
    chunk_overlap_words: int,
    min_chunk_words: int,
    max_documents: int | None = None,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    documents = doc_data[:max_documents] if max_documents else doc_data

    for doc_index, item in enumerate(documents):
        title = normalize_whitespace(item.get("title", "Unknown")) or "Unknown"
        article_id = item.get("id", doc_index)
        source_file = item.get("source_file")
        text = normalize_whitespace(item.get("text", ""))

        for chunk_index, (start_word, end_word, chunk_text) in enumerate(
            word_chunks(
                text=text,
                chunk_size_words=chunk_size_words,
                chunk_overlap_words=chunk_overlap_words,
                min_chunk_words=min_chunk_words,
            )
        ):
            chunks.append(
                {
                    "chunk_id": f"{article_id}:{chunk_index}",
                    "doc_index": doc_index,
                    "article_id": article_id,
                    "title": title,
                    "source_file": source_file,
                    "chunk_index": chunk_index,
                    "start_word": start_word,
                    "end_word": end_word,
                    "text": chunk_text,
                }
            )

    return chunks
