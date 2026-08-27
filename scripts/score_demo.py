#!/usr/bin/env python3
"""Score frozen demo outputs against isolated human-authored ground truth."""

from __future__ import annotations

import json
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "data" / "demo_results.json"
GROUND_TRUTH_PATH = ROOT / "data" / "demo_ground_truth.json"
SCORE_PATH = ROOT / "data" / "demo_score.json"
CORPUS_PATH = ROOT / "data" / "demo_corpus.json"
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


def review_calibration(
    cases: list[dict[str, Any]], ground_truth: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Measure whether structural review flags separate harder appeal-type cases."""
    flagged = [case for case in cases if case.get("needs_human_review")]
    unflagged = [case for case in cases if not case.get("needs_human_review")]
    if any(not case.get("review_reasons") for case in flagged):
        raise ValueError("Every flagged case must carry at least one review reason")

    def accuracy(partition: list[dict[str, Any]]) -> float | None:
        if not partition:
            return None
        return sum(
            case["appeal_type"] == ground_truth[case["id"]]["appeal_type"]
            for case in partition
        ) / len(partition)

    flagged_accuracy = accuracy(flagged)
    unflagged_accuracy = accuracy(unflagged)
    gap = (
        unflagged_accuracy - flagged_accuracy
        if flagged_accuracy is not None and unflagged_accuracy is not None
        else None
    )
    return {
        "flagged_count": len(flagged),
        "flagged_share": len(flagged) / len(cases) if cases else 0.0,
        "flagged_accuracy": flagged_accuracy,
        "unflagged_count": len(unflagged),
        "unflagged_accuracy": unflagged_accuracy,
        "accuracy_gap": gap,
    }


def main() -> None:
    if not RESULTS_PATH.exists():
        raise SystemExit("data/demo_results.json is missing; run scripts/freeze_demo.py first.")
    if not GROUND_TRUTH_PATH.exists():
        raise SystemExit("data/demo_ground_truth.json is missing; cannot score the demo.")

    result_bytes = RESULTS_PATH.read_bytes()
    results = json.loads(result_bytes)
    ground_truth_payload = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    corpus_sha256 = hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()
    if results.get("corpus_sha256") != corpus_sha256:
        raise SystemExit(
            "Frozen results belong to a different corpus; run scripts/freeze_demo.py first."
        )
    if ground_truth_payload.get("corpus_sha256") != corpus_sha256:
        raise SystemExit(
            "Ground truth belongs to a different corpus; run python -m jauap.corpus first."
        )
    ground_truth = ground_truth_payload["ground_truth"]
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

    appeal_type_distribution = Counter(
        str(item["appeal_type"])
        for item in ground_truth.values()
    )
    majority_label = "сообщение"
    majority_baseline = appeal_type_distribution[majority_label] / len(cases)
    accuracy_gap = accuracies["appeal_type"] - majority_baseline
    print(
        f"\nModel appeal-type accuracy: {accuracies['appeal_type']:.2%}\n"
        f"Baseline (always {majority_label}): {majority_baseline:.2%}\n"
        f"Measured gap: {accuracy_gap:+.2%}"
    )

    calibration = review_calibration(cases, ground_truth)
    print(
        f"\nFlagged for review: {calibration['flagged_share']:.2%} "
        f"({calibration['flagged_count']}/{len(cases)})\n"
        f"Appeal-type accuracy, flagged: {calibration['flagged_accuracy']:.2%}\n"
        f"Appeal-type accuracy, unflagged: {calibration['unflagged_accuracy']:.2%}\n"
        f"Unflagged minus flagged: {calibration['accuracy_gap']:+.2%}"
    )

    failures: list[str] = []
    if accuracies["appeal_type"] < 0.876:
        failures.append("appeal_type accuracy is below 87.60%")
    if accuracies["topic"] < 0.96:
        failures.append("topic accuracy is below 96%")
    if not 0.08 <= calibration["flagged_share"] <= 0.15:
        failures.append("review share is outside 8–15%")
    if calibration["accuracy_gap"] is None or calibration["accuracy_gap"] < 0.20:
        failures.append("review accuracy gap is below 20 percentage points")
    if failures:
        SCORE_PATH.unlink(missing_ok=True)
        raise SystemExit("FAIL: " + "; ".join(failures))

    if accuracies["appeal_type"] >= 1.0:
        SCORE_PATH.unlink(missing_ok=True)
        raise SystemExit("FAIL: appeal_type accuracy is 100%; investigate a ground-truth leak.")

    score_payload = {
        "generated_at": results.get("generated_at"),
        "sample_size": len(cases),
        "results_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "accuracy": accuracies,
        "appeal_type_distribution": dict(appeal_type_distribution),
        "majority_class_baseline": {
            "label": majority_label,
            "accuracy": majority_baseline,
        },
        "appeal_type_accuracy_gap": accuracy_gap,
        "review_calibration": calibration,
    }
    SCORE_PATH.write_text(
        json.dumps(score_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote measured scores to {SCORE_PATH}")


if __name__ == "__main__":
    main()
