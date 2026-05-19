from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL_NAME = "gemma2:2b"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_default_env_files(env_file: Path | None) -> None:
    if env_file is not None:
        load_dotenv(env_file)
        return
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / "naive_rag" / ".env")


def env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def resolve_llm_settings(base_url: str | None = None, model_name: str | None = None) -> tuple[str, str, str]:
    resolved_base_url = base_url or env_value("BASE_URL", "NAIVE_RAG_LLM_BASE_URL", default=DEFAULT_BASE_URL)
    resolved_model_name = model_name or env_value("MODEL_NAME", "NAIVE_RAG_LLM_MODEL", default=DEFAULT_MODEL_NAME)
    api_key = env_value("API_KEY", "NAIVE_RAG_LLM_API_KEY", default="ollama")
    return resolved_base_url, resolved_model_name, api_key


def chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def call_chat_completion(
    messages: list[dict[str, str]],
    base_url: str,
    model_name: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
) -> str:
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        chat_completions_url(base_url),
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "BERTopic-for-RAG/0.1 Python",
        },
    )
    try:
        with urlopen(request, timeout=120) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API request failed with HTTP {exc.code}:\n{error_body}") from exc
    except URLError as exc:
        raise RuntimeError(f"LLM API request failed: {exc.reason}") from exc
    try:
        return response_payload["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        formatted = json.dumps(response_payload, ensure_ascii=False, indent=2)
        raise RuntimeError(f"Unexpected LLM API response:\n{formatted}") from exc
