#!/usr/bin/env python3
"""Score frozen demo outputs against isolated human-authored ground truth."""

from __future__ import annotations

import json
import hashlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "data" / "demo_results.json"
GROUND_TRUTH_PATH = ROOT / "data" / "demo_ground_truth.json"
SCORE_PATH = ROOT / "data" / "demo_score.json"
FIELDS = ("appeal_type", "topic", "settlement")


def _predicted(case: dict[str, Any], field: str) -> str:
    if field == "settlement":
        location = case.get("location") or {}
        return str(case.get("settlement") or location.get("settlement") or "")
    return str(case.get(field, ""))


def _print_confusion(field: str, pairs: list[tuple[str, str]]) -> None:
    labels = sorted({value for pair in pairs for value in pair})
    counts = Counter(pairs)
    width = max(8, min(24, max(map(len, labels), default=8)))
    print(f"\nConfusion matrix: {field} (rows=true, columns=predicted)")
    print(" " * (width + 2) + " ".join(f"{label[:width]:>{width}}" for label in labels))
    for actual in labels:
        cells = " ".join(f"{counts[(actual, predicted)]:>{width}}" for predicted in labels)
        print(f"{actual[:width]:>{width}}  {cells}")


def main() -> None:
    if not RESULTS_PATH.exists():
        raise SystemExit("data/demo_results.json is missing; run scripts/freeze_demo.py first.")
    if not GROUND_TRUTH_PATH.exists():
        raise SystemExit("data/demo_ground_truth.json is missing; cannot score the demo.")

    result_bytes = RESULTS_PATH.read_bytes()
    results = json.loads(result_bytes)
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))["ground_truth"]
    cases = results["cases"]
    by_id = {case["id"]: case for case in cases}
    if set(by_id) != set(ground_truth):
        missing = sorted(set(ground_truth) - set(by_id))
        extra = sorted(set(by_id) - set(ground_truth))
        raise SystemExit(f"Result/ground-truth ID mismatch. Missing={missing[:5]} Extra={extra[:5]}")

    accuracies: dict[str, float] = {}
    for field in FIELDS:
        pairs = [
            (str(ground_truth[appeal_id][field]), _predicted(by_id[appeal_id], field))
            for appeal_id in sorted(ground_truth)
        ]
        _print_confusion(field, pairs)
        accuracies[field] = sum(actual == predicted for actual, predicted in pairs) / len(pairs)
        print(f"Accuracy {field}: {accuracies[field]:.2%} ({sum(a == p for a, p in pairs)}/{len(pairs)})")

    if accuracies["appeal_type"] >= 1.0:
        SCORE_PATH.unlink(missing_ok=True)
        raise SystemExit("FAIL: appeal_type accuracy is 100%; investigate a ground-truth leak.")

    score_payload = {
        "generated_at": datetime.now().replace(microsecond=0).isoformat(),
        "sample_size": len(cases),
        "results_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "accuracy": accuracies,
    }
    SCORE_PATH.write_text(
        json.dumps(score_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote measured scores to {SCORE_PATH}")


if __name__ == "__main__":
    main()
