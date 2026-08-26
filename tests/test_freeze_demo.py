from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import freeze_demo


def classification(appeal_type: str) -> dict:
    return {
        "appeal_type": appeal_type,
        "topic": "water_supply",
        "routing_targets": ["owner"],
        "language_detected": "ru",
        "urgency": "routine",
        "confidence": 0.8,
        "reasoning": "model output",
        "needs_human_review": False,
    }


class FreezeDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.corpus_path = self.root / "demo_corpus.json"
        self.results_path = self.root / "demo_results.json"
        self.records = [
            {
                "id": "AP-0001",
                "raw_text": "first",
                "received_at": "2026-01-01T09:00:00",
                "settlement": "Кокшетау",
            },
            {
                "id": "AP-0002",
                "raw_text": "second",
                "received_at": "2026-01-01T09:01:00",
                "settlement": "Кокшетау",
            },
        ]
        self.corpus_path.write_text(json.dumps(self.records), encoding="utf-8")
        self.path_patches = [
            patch.object(freeze_demo, "CORPUS_PATH", self.corpus_path),
            patch.object(freeze_demo, "RESULTS_PATH", self.results_path),
        ]
        for path_patch in self.path_patches:
            path_patch.start()

    def tearDown(self) -> None:
        for path_patch in reversed(self.path_patches):
            path_patch.stop()
        self.temporary_directory.cleanup()

    def test_checkpoints_every_new_record_then_writes_complete_payload(self) -> None:
        snapshots: list[dict] = []
        real_write = freeze_demo._write_json_atomic

        def capture(path: Path, payload: dict) -> None:
            snapshots.append(json.loads(json.dumps(payload)))
            real_write(path, payload)

        with (
            patch.dict(
                os.environ,
                {"GOOGLE_API_KEY": "test-only", "JAUAP_PROVIDER": "gemini"},
                clear=True,
            ),
            patch.object(sys, "argv", ["freeze_demo.py", "--delay", "0"]),
            patch.object(
                freeze_demo,
                "provider_metadata",
                return_value={"provider": "gemini", "model": "test-model"},
            ),
            patch.object(freeze_demo, "provider_key_env", return_value="GOOGLE_API_KEY"),
            patch.object(freeze_demo, "is_cached", side_effect=[False, False]),
            patch.object(
                freeze_demo,
                "classify_text",
                side_effect=[classification("заявление"), classification("жалоба")],
            ) as classify_mock,
            patch.object(
                freeze_demo,
                "process_records",
                return_value=([{"id": "AP-0001"}, {"id": "AP-0002"}], [], []),
            ),
            patch.object(freeze_demo, "_write_json_atomic", side_effect=capture),
        ):
            freeze_demo.main()

        self.assertEqual(classify_mock.call_count, 2)
        self.assertEqual(
            [(item["status"], item.get("processed_cases")) for item in snapshots],
            [("in_progress", 1), ("in_progress", 2), ("complete", None)],
        )
        final = json.loads(self.results_path.read_text(encoding="utf-8"))
        self.assertEqual(final["provider"], "gemini")
        self.assertEqual(final["model"], "test-model")
        self.assertEqual(
            final["classification_contract_sha256"],
            freeze_demo.CLASSIFICATION_CONTRACT_SHA256,
        )
        self.assertEqual(final["case_count"], 2)

    def test_resume_skips_classifications_already_in_checkpoint(self) -> None:
        corpus_sha256 = hashlib.sha256(self.corpus_path.read_bytes()).hexdigest()
        self.results_path.write_text(
            json.dumps(
                {
                    "status": "in_progress",
                    "provider": "gemini",
                    "model": "test-model",
                    "corpus_sha256": corpus_sha256,
                    "classification_contract_sha256": freeze_demo.CLASSIFICATION_CONTRACT_SHA256,
                    "classifications": {"AP-0001": classification("заявление")},
                }
            ),
            encoding="utf-8",
        )

        with (
            patch.dict(
                os.environ,
                {"GOOGLE_API_KEY": "test-only", "JAUAP_PROVIDER": "gemini"},
                clear=True,
            ),
            patch.object(sys, "argv", ["freeze_demo.py", "--delay", "0"]),
            patch.object(
                freeze_demo,
                "provider_metadata",
                return_value={"provider": "gemini", "model": "test-model"},
            ),
            patch.object(freeze_demo, "provider_key_env", return_value="GOOGLE_API_KEY"),
            patch.object(freeze_demo, "is_cached", return_value=True),
            patch.object(
                freeze_demo,
                "classify_text",
                return_value=classification("жалоба"),
            ) as classify_mock,
            patch.object(
                freeze_demo,
                "process_records",
                return_value=([{"id": "AP-0001"}, {"id": "AP-0002"}], [], []),
            ),
        ):
            freeze_demo.main()

        classify_mock.assert_called_once_with("second", "Кокшетау")

    def test_mismatched_checkpoint_is_rejected(self) -> None:
        self.results_path.write_text(
            json.dumps(
                {
                    "status": "in_progress",
                    "provider": "groq",
                    "model": "other-model",
                    "corpus_sha256": "different",
                    "classifications": {},
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(SystemExit, "different provider, model, corpus"):
            freeze_demo._load_checkpoint(
                {"provider": "gemini", "model": "test-model"},
                "expected",
            )

    def test_stale_classification_contract_is_rejected(self) -> None:
        corpus_sha256 = hashlib.sha256(self.corpus_path.read_bytes()).hexdigest()
        self.results_path.write_text(
            json.dumps(
                {
                    "status": "in_progress",
                    "provider": "gemini",
                    "model": "test-model",
                    "corpus_sha256": corpus_sha256,
                    "classification_contract_sha256": "stale",
                    "classifications": {},
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(SystemExit, "classification contract"):
            freeze_demo._load_checkpoint(
                {"provider": "gemini", "model": "test-model"},
                corpus_sha256,
            )

    def test_transient_fallback_is_retried_without_being_returned(self) -> None:
        fallback = {**classification("сообщение"), "warning": "temporary fallback"}
        provider_result = classification("заявление")
        with (
            patch.object(
                freeze_demo,
                "classify_text",
                side_effect=[fallback, provider_result],
            ) as classify_mock,
            patch.object(freeze_demo.time, "sleep") as sleep_mock,
        ):
            result = freeze_demo._classify_with_retry(
                self.records[0],
                retries=3,
                retry_delay=2,
            )

        self.assertEqual(result, provider_result)
        self.assertEqual(classify_mock.call_count, 2)
        sleep_mock.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
