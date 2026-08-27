from __future__ import annotations

import unittest
import hashlib
import json
from collections import Counter, defaultdict

from jauap.classify import _language
from jauap.corpus import (
    DATA_PATH,
    GROUND_TRUTH_PATH,
    LANGUAGE_QUOTAS,
    TOPIC_QUOTAS,
    TYPE_QUOTAS,
    TYPE_TEMPLATES,
    UNVERIFIED_TOPICS,
    generate,
    validate,
)


class CorpusCredibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = generate()

    def test_exact_type_distribution_and_baseline(self) -> None:
        distribution = Counter(
            record["_ground_truth"]["appeal_type"]
            for record in self.records
        )
        self.assertEqual(distribution, TYPE_QUOTAS)
        self.assertLess(distribution["сообщение"] / len(self.records), 0.65)

    def test_type_templates_cover_every_topic_language_and_type(self) -> None:
        for language in LANGUAGE_QUOTAS:
            for topic in TOPIC_QUOTAS:
                self.assertEqual(set(TYPE_TEMPLATES[language][topic]), set(TYPE_QUOTAS))

    def test_ten_adversarial_triples_share_topic_address_and_language(self) -> None:
        groups = defaultdict(list)
        for record in self.records:
            marker = record["hard_case"] or ""
            if marker.startswith("adversarial_type_triple_"):
                groups[marker].append(record)
        self.assertEqual(len(groups), 10)
        for group in groups.values():
            self.assertEqual(len(group), 3)
            self.assertEqual(len({item["_ground_truth"]["topic"] for item in group}), 1)
            self.assertEqual(len({item["language_detected"] for item in group}), 1)
            self.assertEqual(
                {item["_ground_truth"]["appeal_type"] for item in group},
                {"сообщение", "заявление", "жалоба"},
            )

    def test_full_validation_passes(self) -> None:
        validate(self.records)

    def test_unverified_competence_topics_have_unknown_ground_truth(self) -> None:
        for record in self.records:
            if record["_source_topic"] in UNVERIFIED_TOPICS:
                self.assertEqual(
                    record["_ground_truth"]["topic"],
                    "НЕ ОПРЕДЕЛЕНО",
                    record["id"],
                )

    def test_construction_ground_truth_remains_construction(self) -> None:
        construction = [
            record for record in self.records
            if record["_source_topic"] == "construction"
        ]
        self.assertTrue(construction)
        self.assertTrue(all(
            record["_ground_truth"]["topic"] == "construction"
            for record in construction
        ))

    def test_every_language_label_is_derived_from_its_text(self) -> None:
        for record in self.records:
            self.assertEqual(_language(record["raw_text"]), record["language_detected"], record["id"])

    def test_committed_ground_truth_is_bound_to_public_corpus(self) -> None:
        payload = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["corpus_sha256"],
            hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
