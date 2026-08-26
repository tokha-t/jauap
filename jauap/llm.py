"""Single provider boundary with provider-aware, auditable disk caching."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CACHE_DIR = DATA_DIR / ".llm_cache"
DEFAULT_PROVIDER = "gemini"
PROVIDERS = {
    "gemini": {
        "key_env": "GOOGLE_API_KEY",
        "default_model": "gemini-3.6-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "groq": {
        "key_env": "GROQ_API_KEY",
        "default_model": "openai/gpt-oss-120b",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "anthropic": {
        "key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4-5",
        "base_url": None,
    },
}


def provider_metadata() -> dict[str, str]:
    """Return the selected provider/model without exposing credentials."""
    provider = os.environ.get("JAUAP_PROVIDER", DEFAULT_PROVIDER).strip().casefold()
    if provider not in PROVIDERS:
        allowed = ", ".join(PROVIDERS)
        raise ValueError(f"Unsupported JAUAP_PROVIDER={provider!r}; choose one of: {allowed}")
    model = os.environ.get("JAUAP_MODEL", "").strip() or PROVIDERS[provider]["default_model"]
    return {"provider": provider, "model": model}


def provider_key_env(provider: str | None = None) -> str:
    """Return the environment variable used by a provider."""
    selected = provider or provider_metadata()["provider"]
    if selected not in PROVIDERS:
        raise ValueError(f"Unsupported provider: {selected!r}")
    return str(PROVIDERS[selected]["key_env"])


def _cache_path(system: str, user: str, provider: str, model: str) -> Path:
    material = provider + model + system + user
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
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


def _schema_instruction(schema: dict, *, retry: bool = False) -> str:
    prefix = ""
    if retry:
        prefix = (
            "Your previous response could not be parsed. Do not use Markdown fences or commentary. "
            "Return exactly one JSON object and nothing else.\n"
        )
    return (
        "\n\n"
        + prefix
        + "Return only one valid JSON object matching this JSON Schema:\n"
        + json.dumps(schema, ensure_ascii=False)
    )


def _openai_compatible_text(
    *, provider: str, api_key: str, model: str, system: str, user: str
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=PROVIDERS[provider]["base_url"])
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = response.choices[0].message.content
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{provider} returned an empty completion")
    return text


def _anthropic_text(*, api_key: str, model: str, system: str, user: str) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1800,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
    if not text.strip():
        raise ValueError("anthropic returned an empty completion")
    return text


def _provider_text(*, provider: str, api_key: str, model: str, system: str, user: str) -> str:
    if provider == "anthropic":
        return _anthropic_text(api_key=api_key, model=model, system=system, user=user)
    return _openai_compatible_text(
        provider=provider,
        api_key=api_key,
        model=model,
        system=system,
        user=user,
    )


def complete(system: str, user: str, schema: dict | None = None) -> dict | str:
    """Complete one prompt; provider SDK access and caching stay behind this boundary."""
    metadata = provider_metadata()
    provider = metadata["provider"]
    model = metadata["model"]
    cache_path = _cache_path(system, user, provider, model)
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("provider") != provider or cached.get("model") != model:
            raise ValueError(f"Cache metadata mismatch: {cache_path}")
        return cached["result"]

    key_env = provider_key_env(provider)
    api_key = os.environ.get(key_env)
    if not api_key:
        raise RuntimeError(f"{key_env} is not set for JAUAP_PROVIDER={provider}")

    prompted_system = system + (_schema_instruction(schema) if schema is not None else "")
    text = _provider_text(
        provider=provider,
        api_key=api_key,
        model=model,
        system=prompted_system,
        user=user,
    )
    if schema is None:
        result: dict | str = text
    else:
        try:
            result = _extract_json(text)
        except (json.JSONDecodeError, ValueError):
            retry_system = system + _schema_instruction(schema, retry=True)
            retry_text = _provider_text(
                provider=provider,
                api_key=api_key,
                model=model,
                system=retry_system,
                user=user,
            )
            result = _extract_json(retry_text)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(
            {"provider": provider, "model": model, "result": result},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(cache_path)
    return result
