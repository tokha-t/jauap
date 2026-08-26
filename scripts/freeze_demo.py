#!/usr/bin/env python3
"""Freeze honest Anthropic classifications for the offline demo corpus."""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jauap.pipeline import process_records  # noqa: E402


CORPUS_PATH = ROOT / "data" / "demo_corpus.json"
RESULTS_PATH = ROOT / "data" / "demo_results.json"


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is required; no demo results were written.")

    records = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    print(
        f"Classifying {len(records)} appeals through Anthropic; cached records will be reused.",
        flush=True,
    )
    cases, clusters, classifier_warnings = process_records(records, deterministic_support=True)
    if classifier_warnings:
        sample = "\n".join(classifier_warnings[:5])
        raise SystemExit(
            f"Model classification degraded for {len(classifier_warnings)} records; "
            f"no demo results were written.\n{sample}"
        )

    payload = {
        "generated_at": datetime.now().replace(microsecond=0).isoformat(),
        "as_of_date": date.today().isoformat(),
        "classification_source": "Anthropic API via classify_text",
        "model": os.environ.get("JAUAP_MODEL", "claude-sonnet-4-5"),
        "cases": cases,
        "clusters": clusters,
    }
    temporary_path = RESULTS_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(RESULTS_PATH)
    print(f"Wrote {len(cases)} model-classified cases to {RESULTS_PATH}")
    print("Run scripts/score_demo.py and commit data/.llm_cache/ with the results.")


if __name__ == "__main__":
    main()
