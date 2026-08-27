from __future__ import annotations

import unittest

from jauap.review import review_flags


def case(**overrides: object) -> dict:
    value = {
        "appeal_type": "сообщение",
        "topic": "water_supply",
        "location": {"confidence": 0.99},
    }
    value.update(overrides)
    return value


class StructuralReviewTests(unittest.TestCase):
    def test_request_and_inaction_boundary_is_flagged(self) -> None:
        reasons = review_flags(
            case(appeal_type="жалоба"),
            "Ответа нет, прошу восстановить воду.",
        )
        self.assertTrue(any("заявлением и жалобой" in reason for reason in reasons))

    def test_defect_and_information_boundary_is_flagged(self) -> None:
        reasons = review_flags(case(), "Во дворе нет воды. Когда будет график ремонта?")
        self.assertTrue(any("дефекте и запросом" in reason for reason in reasons))

    def test_minority_label_without_marker_is_flagged(self) -> None:
        reasons = review_flags(case(appeal_type="предложение"), "Во дворе нет воды.")
        self.assertTrue(any("нет лексического маркера" in reason for reason in reasons))

    def test_unknown_topic_and_unresolved_address_are_separate_reasons(self) -> None:
        reasons = review_flags(
            case(topic="НЕ ОПРЕДЕЛЕНО", location=None),
            "Во дворе стая собак.",
        )
        self.assertTrue(any("Компетенция не определена" in reason for reason in reasons))
        self.assertTrue(any("Адрес не распознан" in reason for reason in reasons))

    def test_multiple_topics_are_flagged(self) -> None:
        reasons = review_flags(case(), "На дороге яма, рядом незаконный павильон.")
        self.assertTrue(any("ст. 65(2)" in reason for reason in reasons))

    def test_model_confidence_is_not_a_review_signal(self) -> None:
        reasons = review_flags(case(confidence=0.01), "Во дворе нет воды.")
        self.assertEqual(reasons, [])

    def test_model_label_disagreeing_with_explicit_type_marker_is_flagged(self) -> None:
        reasons = review_flags(
            case(appeal_type="сообщение"),
            "Для моего дома нужно устранить проблему; прошу включить работу в исполнение.",
        )
        self.assertTrue(any("упростила «заявление»" in reason for reason in reasons))

    def test_matching_explicit_type_marker_does_not_add_disagreement(self) -> None:
        reasons = review_flags(
            case(appeal_type="заявление"),
            "Для моего дома нужно устранить проблему; прошу включить работу в исполнение.",
        )
        self.assertFalse(any("упростила «заявление»" in reason for reason in reasons))


if __name__ == "__main__":
    unittest.main()
