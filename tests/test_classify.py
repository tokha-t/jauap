from __future__ import annotations

import unittest
from unittest.mock import patch

from jauap.classify import classify_text, fallback_classification


def model_result(*, topic: str, routing_targets: list[str] | None = None) -> dict:
    return {
        "appeal_type": "сообщение",
        "topic": topic,
        "routing_targets": routing_targets or ["model-suggested owner"],
        "language_detected": "ru",
        "urgency": "routine",
        "confidence": 0.9,
        "reasoning": "model output",
    }


class DeterministicRoutingTests(unittest.TestCase):
    def test_fallback_topic_rules_cover_existing_kazakh_and_latin_templates(self) -> None:
        examples = {
            "Отдел месяц не отвечает на моё заявление о воде": "water_supply",
            "Күрделі жөндеу қашан басталады?": "construction",
            "Әлеуметтік төлем түспеді": "social",
            "Ayaldamasyna avtobus uaqytynda kelmeidi": "public_transport",
        }
        for text, expected in examples.items():
            with self.subTest(text=text):
                self.assertEqual(fallback_classification(text)["topic"], expected)

    def test_localised_model_topic_is_normalised_to_routing_key(self) -> None:
        with patch(
            "jauap.classify.complete",
            return_value=model_result(topic="Водоснабжение"),
        ):
            result = classify_text("Во дворе течёт вода")

        self.assertEqual(result["topic"], "water_supply")
        self.assertEqual(
            result["routing_targets"],
            ["ГКП на ПХВ «Көкшетау Су Арнасы»", "Отдел ЖКХ, ПТ и АД"],
        )

    def test_verified_unknown_competence_overrides_model_topic(self) -> None:
        with patch(
            "jauap.classify.complete",
            return_value=model_result(topic="street_lighting"),
        ):
            result = classify_text("Со вчерашнего вечера нет света")

        self.assertEqual(result["topic"], "НЕ ОПРЕДЕЛЕНО")
        self.assertEqual(
            result["routing_targets"],
            ["НЕ ОПРЕДЕЛЕНО — требует уточнения"],
        )

    def test_spurious_unknown_uses_recognised_rule_topic(self) -> None:
        with patch(
            "jauap.classify.complete",
            return_value=model_result(topic="НЕ ОПРЕДЕЛЕНО"),
        ):
            result = classify_text("В школе протекает крыша")

        self.assertEqual(result["topic"], "education")

    def test_model_topic_uses_two_hop_water_route(self) -> None:
        with patch(
            "jauap.classify.complete",
            return_value=model_result(topic="water_supply"),
        ):
            result = classify_text("Во дворе течёт вода")

        self.assertEqual(
            result["routing_targets"],
            ["ГКП на ПХВ «Көкшетау Су Арнасы»", "Отдел ЖКХ, ПТ и АД"],
        )

    def test_model_topic_preserves_article_65_2_split(self) -> None:
        with patch(
            "jauap.classify.complete",
            return_value=model_result(topic="road_condition"),
        ):
            result = classify_text("На дороге яма и незаконный павильон")

        self.assertEqual(
            result["routing_targets"],
            [
                "Отдел ЖКХ, ПТ и АД",
                "Отдел земельных отношений, архитектуры и градостроительства",
            ],
        )

    def test_model_topic_uses_rural_settlement_route(self) -> None:
        with patch(
            "jauap.classify.complete",
            return_value=model_result(topic="road_condition"),
        ):
            result = classify_text("Разбита дорога", settlement="Красный Яр")

        self.assertEqual(
            result["routing_targets"],
            ["Аппарат акима Красноярского сельского округа"],
        )


if __name__ == "__main__":
    unittest.main()
