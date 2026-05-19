from __future__ import annotations

import ast
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

from naive_rag.llm_client import call_chat_completion


def parse_listish(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    return [part.strip() for part in re.split(r"[,;|]", text) if part.strip()]


def load_topic_profiles(
    topic_info_path: Path,
    max_keywords: int,
    include_outlier_topic: bool = False,
) -> list[dict[str, Any]]:
    if not topic_info_path.exists():
        raise FileNotFoundError(f"topic_info file does not exist: {topic_info_path}")
    csv.field_size_limit(sys.maxsize)
    profiles = []
    with topic_info_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            topic_id = int(row["Topic"])
            if topic_id == -1 and not include_outlier_topic:
                continue
            representation = parse_listish(row.get("Representation"))[:max_keywords]
            name = row.get("Name") or f"topic_{topic_id}"
            profile_text = (
                f"Topic {topic_id}. "
                f"Name: {name}. "
                f"Keywords: {', '.join(representation)}."
            )
            profiles.append(
                {
                    "topic_id": topic_id,
                    "count": int(row.get("Count") or 0),
                    "name": name,
                    "keywords": representation,
                    "profile_text": profile_text,
                }
            )
    if not profiles:
        raise ValueError(f"No usable topic profiles were loaded from {topic_info_path}.")
    return profiles


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
    stripped = re.sub(r"```$", "", stripped).strip()
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(stripped[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def extract_knowledge_points(
    question: str,
    base_url: str,
    model_name: str,
    api_key: str,
    max_knowledge_points: int,
    temperature: float,
) -> list[str]:
    prompt = f"""Extract concise knowledge points needed to answer the question.
Return strict JSON only, with this schema:
{{"knowledge_points": ["short phrase", "short phrase"]}}

Rules:
- Return at most {max_knowledge_points} knowledge points.
- Prefer English noun phrases because the topic keywords are English.
- Do not answer the question.
- Do not include explanations.

Question:
{question}
"""
    response = call_chat_completion(
        messages=[
            {"role": "system", "content": "You extract retrieval-oriented knowledge points as strict JSON."},
            {"role": "user", "content": prompt},
        ],
        base_url=base_url,
        model_name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=256,
    )
    parsed = extract_json_object(response)
    if parsed:
        raw_points = parsed.get("knowledge_points", [])
        if isinstance(raw_points, list):
            points = [str(point).strip() for point in raw_points if str(point).strip()]
            if points:
                return points[:max_knowledge_points]
    fallback_points = [line.strip("-* \t") for line in response.splitlines() if line.strip("-* \t")]
    return (fallback_points or [question])[:max_knowledge_points]


class SemanticTopicRouter:
    """Kiraffe-style topic router with cached profile embeddings."""

    def __init__(self, topic_profiles: list[dict[str, Any]], embedding_model: Any, top_topics: int) -> None:
        import numpy as np

        self.topic_profiles = topic_profiles
        self.embedding_model = embedding_model
        self.top_topics = top_topics
        profile_embeddings = embedding_model.encode(
            [profile["profile_text"] for profile in topic_profiles],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        norms = np.linalg.norm(profile_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.profile_embeddings = profile_embeddings / norms

    def rank(self, knowledge_points: list[str]) -> list[dict[str, Any]]:
        import numpy as np

        point_embeddings = self.embedding_model.encode(
            knowledge_points,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        norms = np.linalg.norm(point_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        point_embeddings = point_embeddings / norms

        similarities = self.profile_embeddings @ point_embeddings.T
        topic_scores = similarities.max(axis=1)
        best_point_indices = similarities.argmax(axis=1)
        top_count = min(self.top_topics, len(self.topic_profiles))
        top_indices = np.argsort(-topic_scores)[:top_count]

        ranked = []
        for index in top_indices:
            profile = dict(self.topic_profiles[int(index)])
            best_point_index = int(best_point_indices[int(index)])
            profile["score"] = float(topic_scores[int(index)])
            profile["best_knowledge_point"] = knowledge_points[best_point_index]
            ranked.append(profile)
        return ranked
