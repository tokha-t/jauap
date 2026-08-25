"""Single-provider boundary for Anthropic calls with a mandatory disk cache."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = DATA_DIR / ".llm_cache"
DEFAULT_MODEL = "claude-sonnet-4-5"


def _cache_path(system: str, user: str, model: str) -> Path:
    digest = hashlib.sha256((system + user + model).encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def _extract_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].lstrip()
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Structured completion must be a JSON object")
    return parsed


def complete(system: str, user: str, schema: dict | None = None) -> dict | str:
    """Complete one prompt, with all provider access and caching in this function.

    ``JAUAP_OFFLINE=1`` is a hard network kill switch. The app itself loads the
    frozen demo results in that mode and therefore never calls this function.
    """
    if os.environ.get("JAUAP_OFFLINE") == "1":
        raise RuntimeError("LLM calls are disabled while JAUAP_OFFLINE=1")

    model = os.environ.get("JAUAP_MODEL", DEFAULT_MODEL)
    cache_path = _cache_path(system, user, model)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return cached["result"]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    schema_instruction = ""
    if schema is not None:
        schema_instruction = (
            "\n\nReturn only one valid JSON object matching this JSON Schema:\n"
            + json.dumps(schema, ensure_ascii=False)
        )
    message = client.messages.create(
        model=model,
        max_tokens=1800,
        temperature=0,
        system=system + schema_instruction,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
    result: dict | str = _extract_json(text) if schema is not None else text

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"model": model, "result": result}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
