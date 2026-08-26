from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jauap import llm


SCHEMA = {"type": "object", "required": ["answer"]}


class FakeOpenAI:
    responses: list[str | None] = []
    calls: list[dict] = []

    def __init__(self, *, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append({"base_url": self.base_url, **kwargs})
        text = self.responses.pop(0)
        if text is None:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=None, finish_reason="content_filter")]
            )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=text),
                    finish_reason="stop",
                )
            ]
        )


class LlmBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeOpenAI.responses = []
        FakeOpenAI.calls = []
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.cache_patch = patch.object(llm, "CACHE_DIR", Path(self.temporary_directory.name))
        self.cache_patch.start()
        self.module_patch = patch.dict(
            sys.modules,
            {"openai": SimpleNamespace(OpenAI=FakeOpenAI)},
        )
        self.module_patch.start()
        self.environment_patch = patch.dict(os.environ, {}, clear=True)
        self.environment_patch.start()

    def tearDown(self) -> None:
        self.environment_patch.stop()
        self.module_patch.stop()
        self.cache_patch.stop()
        self.temporary_directory.cleanup()

    def test_gemini_is_the_safe_default(self) -> None:
        self.assertEqual(
            llm.provider_metadata(),
            {"provider": "gemini", "model": "gemini-3.5-flash-lite"},
        )
        self.assertEqual(llm.provider_key_env(), "GOOGLE_API_KEY")

    def test_runtime_api_key_exists_only_inside_live_request(self) -> None:
        with llm.runtime_api_key("  ui-gemini-key  "):
            self.assertEqual(os.environ["GOOGLE_API_KEY"], "ui-gemini-key")
        self.assertNotIn("GOOGLE_API_KEY", os.environ)

    def test_runtime_api_key_restores_existing_environment_value(self) -> None:
        os.environ["GOOGLE_API_KEY"] = "environment-key"
        with llm.runtime_api_key("ui-gemini-key"):
            self.assertEqual(os.environ["GOOGLE_API_KEY"], "ui-gemini-key")
        self.assertEqual(os.environ["GOOGLE_API_KEY"], "environment-key")

    def test_unknown_provider_fails_before_any_call(self) -> None:
        os.environ["JAUAP_PROVIDER"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unsupported JAUAP_PROVIDER"):
            llm.complete("system", "user")
        self.assertEqual(FakeOpenAI.calls, [])

    def test_provider_and_model_create_distinct_cache_entries(self) -> None:
        os.environ.update(
            {
                "GOOGLE_API_KEY": "google-test-value",
                "GROQ_API_KEY": "groq-test-value",
                "JAUAP_MODEL": "shared-model-for-test",
            }
        )
        FakeOpenAI.responses = ['{"answer": "gemini"}', '{"answer": "groq"}']

        gemini_result = llm.complete("system", "user", SCHEMA)
        os.environ["JAUAP_PROVIDER"] = "groq"
        groq_result = llm.complete("system", "user", SCHEMA)

        self.assertEqual(gemini_result, {"answer": "gemini"})
        self.assertEqual(groq_result, {"answer": "groq"})
        cache_files = sorted(Path(self.temporary_directory.name).glob("*.json"))
        self.assertEqual(len(cache_files), 2)
        metadata = [json.loads(path.read_text(encoding="utf-8")) for path in cache_files]
        self.assertEqual({item["provider"] for item in metadata}, {"gemini", "groq"})
        self.assertEqual(len(FakeOpenAI.calls), 2)

    def test_structured_output_retries_once_with_stricter_instruction(self) -> None:
        os.environ["GOOGLE_API_KEY"] = "google-test-value"
        FakeOpenAI.responses = ["not json", '```json\n{"answer": "ok"}\n```']

        result = llm.complete("system", "user", SCHEMA)

        self.assertEqual(result, {"answer": "ok"})
        self.assertEqual(len(FakeOpenAI.calls), 2)
        retry_system = FakeOpenAI.calls[1]["messages"][0]["content"]
        self.assertIn("previous response could not be parsed", retry_system)

    def test_filtered_empty_completion_retries_as_synthetic_triage(self) -> None:
        os.environ["GOOGLE_API_KEY"] = "google-test-value"
        FakeOpenAI.responses = [None, '{"answer": "classified"}']

        result = llm.complete("system", "UPPERCASE USER", SCHEMA)

        self.assertEqual(result, {"answer": "classified"})
        self.assertEqual(len(FakeOpenAI.calls), 2)
        retry_system = FakeOpenAI.calls[1]["messages"][0]["content"]
        self.assertIn("synthetic administrative evidence", retry_system)
        retry_user = FakeOpenAI.calls[1]["messages"][1]["content"]
        self.assertEqual(retry_user, "uppercase user")

    def test_cached_result_avoids_a_second_provider_call(self) -> None:
        os.environ["GOOGLE_API_KEY"] = "google-test-value"
        FakeOpenAI.responses = ['{"answer": "cached"}']

        first = llm.complete("system", "user", SCHEMA)
        del os.environ["GOOGLE_API_KEY"]
        second = llm.complete("system", "user", SCHEMA)

        self.assertEqual(first, second)
        self.assertEqual(len(FakeOpenAI.calls), 1)


if __name__ == "__main__":
    unittest.main()
