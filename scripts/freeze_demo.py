#!/usr/bin/env python3
"""Freeze honest, resumable model classifications for the offline demo corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jauap.classify import (  # noqa: E402
    CLASSIFICATION_CONTRACT_SHA256,
    SYSTEM_PROMPT,
    classify_text,
)
from jauap.llm import is_cached, provider_key_env, provider_metadata  # noqa: E402
from jauap.pipeline import process_records  # noqa: E402


CORPUS_PATH = ROOT / "data" / "demo_corpus.json"
RESULTS_PATH = ROOT / "data" / "demo_results.json"


def _timestamp() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _corpus_digest() -> str:
    return hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _load_checkpoint(metadata: dict[str, str], corpus_sha256: str) -> dict[str, dict[str, Any]]:
    if not RESULTS_PATH.exists():
        return {}
    payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    expected = (
        metadata["provider"],
        metadata["model"],
        corpus_sha256,
        CLASSIFICATION_CONTRACT_SHA256,
    )
    actual = (
        payload.get("provider"),
        payload.get("model"),
        payload.get("corpus_sha256"),
        payload.get("classification_contract_sha256"),
    )
    if actual != expected:
        raise SystemExit(
            "Existing freeze checkpoint belongs to a different provider, model, corpus, "
            "or classification contract; "
            "move it aside before starting another run."
        )
    if payload.get("status") == "complete":
        raise SystemExit(
            f"{RESULTS_PATH} is already complete for "
            f"{payload.get('provider', 'unknown')}/{payload.get('model', 'unknown')}."
        )
    classifications = payload.get("classifications", {})
    if not isinstance(classifications, dict):
        raise SystemExit("Existing freeze checkpoint is malformed.")
    return classifications


def _checkpoint_payload(
    *,
    metadata: dict[str, str],
    corpus_sha256: str,
    classifications: dict[str, dict[str, Any]],
    total: int,
) -> dict[str, Any]:
    return {
        "status": "in_progress",
        "checkpointed_at": _timestamp(),
        "provider": metadata["provider"],
        "model": metadata["model"],
        "classification_source": f"{metadata['provider']} API via classify_text",
        "classification_contract_sha256": CLASSIFICATION_CONTRACT_SHA256,
        "corpus_sha256": corpus_sha256,
        "processed_cases": len(classifications),
        "total_cases": total,
        "classifications": classifications,
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze provider-produced classifications with per-record checkpoints."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=float(os.environ.get("JAUAP_FREEZE_DELAY", "4")),
        help="Seconds to wait after each uncached provider call (default: 4).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=int(os.environ.get("JAUAP_FREEZE_RETRIES", "3")),
        help="Retries for a record that transiently degrades to fallback (default: 3).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=float(os.environ.get("JAUAP_FREEZE_RETRY_DELAY", "30")),
        help="Base seconds between retries; backoff is linear (default: 30).",
    )
    arguments = parser.parse_args()
    if arguments.delay < 0:
        parser.error("--delay must be zero or greater")
    if arguments.retries < 0:
        parser.error("--retries must be zero or greater")
    if arguments.retry_delay < 0:
        parser.error("--retry-delay must be zero or greater")
    return arguments


def _classify_with_retry(
    record: dict[str, Any],
    *,
    retries: int,
    retry_delay: float,
) -> dict[str, Any]:
    """Retry transient provider degradation without ever checkpointing fallback output."""
    for attempt in range(retries + 1):
        result = classify_text(record["raw_text"], record.get("settlement", "Кокшетау"))
        if not result.get("warning"):
            return result
        if attempt < retries:
            wait_seconds = retry_delay * (attempt + 1)
            print(
                f"{record['id']} degraded to fallback; retry "
                f"{attempt + 1}/{retries} in {wait_seconds:g}s.",
                flush=True,
            )
            if wait_seconds:
                time.sleep(wait_seconds)
    raise RuntimeError(f"{record['id']} still degraded after {retries} retries")


def main() -> None:
    arguments = _arguments()
    metadata = provider_metadata()
    key_env = provider_key_env(metadata["provider"])
    if not os.environ.get(key_env):
        raise SystemExit(
            f"{key_env} is required for JAUAP_PROVIDER={metadata['provider']}; "
            "no provider call was attempted."
        )

    records = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    corpus_sha256 = _corpus_digest()
    classifications = _load_checkpoint(metadata, corpus_sha256)
    print(
        f"Freezing {len(records)} appeals through {metadata['provider']}/{metadata['model']}; "
        f"resuming after {len(classifications)} completed records.",
        flush=True,
    )

    for index, record in enumerate(records, start=1):
        appeal_id = record["id"]
        if appeal_id in classifications:
            continue
        cached_before_call = is_cached(SYSTEM_PROMPT, record["raw_text"])
        try:
            result = _classify_with_retry(
                record,
                retries=arguments.retries,
                retry_delay=arguments.retry_delay,
            )
        except RuntimeError:
            raise SystemExit(
                f"{appeal_id} degraded to fallback; checkpoint retained at "
                f"{len(classifications)}/{len(records)}. Retry later."
            ) from None
        classifications[appeal_id] = result
        checkpoint = _checkpoint_payload(
            metadata=metadata,
            corpus_sha256=corpus_sha256,
            classifications=classifications,
            total=len(records),
        )
        _write_json_atomic(RESULTS_PATH, checkpoint)
        source = "cache" if cached_before_call else "provider"
        print(f"[{index}/{len(records)}] {appeal_id} · {source} · checkpoint saved", flush=True)
        if not cached_before_call and arguments.delay:
            time.sleep(arguments.delay)

    cases, clusters, classifier_warnings = process_records(records, deterministic_support=True)
    if classifier_warnings:
        sample = "\n".join(classifier_warnings[:5])
        raise SystemExit(
            f"Cached classification replay degraded for {len(classifier_warnings)} records; "
            f"checkpoint retained.\n{sample}"
        )

    payload = {
        "status": "complete",
        "generated_at": _timestamp(),
        "as_of_date": date.today().isoformat(),
        "provider": metadata["provider"],
        "model": metadata["model"],
        "classification_source": f"{metadata['provider']} API via classify_text",
        "classification_contract_sha256": CLASSIFICATION_CONTRACT_SHA256,
        "corpus_sha256": corpus_sha256,
        "case_count": len(cases),
        "cases": cases,
        "clusters": clusters,
    }
    _write_json_atomic(RESULTS_PATH, payload)
    print(f"Wrote {len(cases)} model-classified cases to {RESULTS_PATH}")
    print("Run scripts/score_demo.py and commit data/.llm_cache/ with the results.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Freeze interrupted; the last completed checkpoint is intact.") from None
