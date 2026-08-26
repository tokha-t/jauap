from __future__ import annotations

import unittest
from unittest.mock import patch

from jauap.classify import classify_text


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
