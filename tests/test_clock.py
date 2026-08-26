from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from jauap import clock
from jauap.deadline_engine import working_days_between


class FutureDate(date):
    @classmethod
    def today(cls) -> FutureDate:
        return cls(2026, 9, 25)


class DemoClockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.results_path = Path(self.temporary_directory.name) / "demo_results.json"
        self.results_path.write_text(
            json.dumps({"as_of_date": "2026-08-26"}),
            encoding="utf-8",
        )
        self.path_patch = patch.object(clock, "RESULTS_PATH", self.results_path)
        self.date_patch = patch.object(clock, "date", FutureDate)
        self.path_patch.start()
        self.date_patch.start()

    def tearDown(self) -> None:
        self.date_patch.stop()
        self.path_patch.stop()
        self.temporary_directory.cleanup()

    def test_frozen_mode_ignores_system_clock_advance(self) -> None:
        with clock.clock_mode(frozen=True):
            self.assertEqual(clock.demo_now(), date(2026, 8, 26))

    def test_live_mode_uses_system_date(self) -> None:
        with clock.clock_mode(frozen=False):
            self.assertEqual(clock.demo_now(), date(2026, 9, 25))

    def test_committed_queue_and_map_deadlines_share_reference_date(self) -> None:
        committed = json.loads(
            (Path(__file__).resolve().parents[1] / "data" / "demo_results.json").read_text(
                encoding="utf-8"
            )
        )
        self.results_path.write_text(
            json.dumps({"as_of_date": committed["as_of_date"]}),
            encoding="utf-8",
        )
        with clock.clock_mode(frozen=True):
            reference = clock.demo_now()
            for case in committed["cases"]:
                expected = working_days_between(reference, date.fromisoformat(case["deadline"]))
                self.assertEqual(case["working_days_remaining"], expected, case["id"])
            for cluster in committed["clusters"]:
                expected = working_days_between(
                    reference, date.fromisoformat(cluster["earliest_deadline"])
                )
                member_values = [
                    case["working_days_remaining"]
                    for case in committed["cases"]
                    if case["id"] in cluster["member_ids"]
                ]
                self.assertEqual(expected, min(member_values), cluster["cluster_id"])


if __name__ == "__main__":
    unittest.main()
