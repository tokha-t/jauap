from __future__ import annotations

import unittest
import json
from pathlib import Path

from scripts.score_demo import review_calibration


class ReviewCalibrationTests(unittest.TestCase):
    def test_committed_demo_meets_calibration_gates(self) -> None:
        score = json.loads(
            (Path(__file__).resolve().parents[1] / "data" / "demo_score.json").read_text(
                encoding="utf-8"
            )
        )
        calibration = score["review_calibration"]
        self.assertGreaterEqual(score["accuracy"]["appeal_type"], 0.876)
        self.assertGreaterEqual(score["accuracy"]["topic"], 0.96)
        self.assertGreaterEqual(calibration["flagged_share"], 0.08)
        self.assertLessEqual(calibration["flagged_share"], 0.15)
        self.assertGreaterEqual(calibration["accuracy_gap"], 0.20)

    def test_partitions_appeal_type_accuracy_by_review_flag(self) -> None:
        cases = [
            {"id": "A", "appeal_type": "запрос", "needs_human_review": True,
             "review_reasons": ["Неоднозначно"]},
            {"id": "B", "appeal_type": "сообщение", "needs_human_review": False,
             "review_reasons": []},
        ]
        truth = {
            "A": {"appeal_type": "сообщение"},
            "B": {"appeal_type": "сообщение"},
        }
        measured = review_calibration(cases, truth)
        self.assertEqual(measured["flagged_share"], 0.5)
        self.assertEqual(measured["flagged_accuracy"], 0.0)
        self.assertEqual(measured["unflagged_accuracy"], 1.0)
        self.assertEqual(measured["accuracy_gap"], 1.0)

    def test_flag_without_reason_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "review reason"):
            review_calibration(
                [{"id": "A", "appeal_type": "запрос", "needs_human_review": True,
                  "review_reasons": []}],
                {"A": {"appeal_type": "запрос"}},
            )


if __name__ == "__main__":
    unittest.main()
