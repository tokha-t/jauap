"""Structural review signals independent of model-reported confidence."""

from __future__ import annotations

from typing import Any


REQUEST_MARKERS = ("прошу", "сұраймын", "өтінемін", "требую")
INACTION_MARKERS = ("ответа нет", "не ответили", "отказал", "бездейств", "жауап жоқ")
INFORMATION_MARKERS = ("когда", "қашан", "qashan", "график", "сроки", "мерзім", "кесте")
DEFECT_MARKERS = (
    "нет вод", "су жоқ", "su joq", "прорвал", "порыв", "труба жарыл", "труба сломан",
    "батареи холод", "жылу жоқ", "jylu joq", "канализация ағып", "течет канализ", "кәріз суы ағып",
    "нет света", "жарық жоқ", "jaryq joq", "не вывозят мусор", "мусор", "қоқыс", "qoqys",
    "не чистят", "тазаланбаған", "tazalanbagan", "яма", "шұңқыр", "shungqyr",
    "не горят фонар", "фонари жанбай", "шамдар жанбай", "пропускает рейс", "автобус уақытында келмей",
    "сухое дерево", "қураған ағаш", "протекает крыш", "шатыры ағып", "выплата", "соцвыплат", "төлем түспеді",
)

PROPOSAL_MARKERS = ("предлаг", "ұсын", "usyn")
RESPONSE_MARKERS = ("поддержив", "қолдай", "qoldai", "пікір", "pikir")
TYPE_LEXICAL_MARKERS = {
    "сообщение": (
        "сообщаю этот факт", "бұл жағдайды тіркеу", "фактіні тіркеуді", "bul jagdaidy tirkeuge",
    ),
    "заявление": (
        "для моего дома нужно устранить", "өз құқығымды іске асыру", "oz quqygymdy iske asyru",
    ),
    "жалоба": ("обжалую", "шағымданамын", "shagymdanamyn", "требую признать бездействие"),
    "запрос": ("нужна информация", "ақпарат беруді", "бекітілген мерзім мен график", "bekitilgen merzim men jumys kestesi"),
    "предложение": PROPOSAL_MARKERS,
    "отклик": RESPONSE_MARKERS,
}
LOCATION_CONFIDENCE_THRESHOLD = 0.8

TOPIC_MARKERS = {
    "water_supply": ("вода", "воды", "су жоқ", "су құбыр", "su joq", "труб", "порыв"),
    "sewerage": ("канализ", "кәріз", "kariz", "кнс"),
    "heating": ("отоплен", "батаре", "жылу", "jylu"),
    "unverified": ("собак", "иттер", "кск", "оси", "электр", "жарық жоқ", "света нет"),
    "waste_removal": ("мусор", "қоқыс", "qoqys"),
    "snow_cleaning": ("снег", "қар ", "мұз", "лёд", "лед"),
    "road_condition": ("яма", "разбита дорог", "шұңқыр", "shungqyr"),
    "street_lighting": ("фонар", "шамдар", "освещен"),
    "public_transport": ("автобус", "маршрут", "аялдама", "ayaldama"),
    "landscaping": ("дерев", "ағаш", "agash"),
    "illegal_construction": ("незакон", "самовол", "павильон"),
    "construction": ("капитал", "капремонт", "құрылыс", "күрделі жөндеу", "kapitaldy jondeu"),
    "land": ("земел", "участок", "жер телім", "jer shekar"),
    "education": ("школ", "мектеп", "детсад"),
    "social": ("соцвыплат", "социальная выплат", "әлеуметтік төлем", "aleumettik tolem"),
}


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _detected_topics(text: str) -> set[str]:
    return {
        topic
        for topic, markers in TOPIC_MARKERS.items()
        if _contains_any(text, markers)
    }


def _lexical_types(text: str) -> set[str]:
    return {
        appeal_type
        for appeal_type, markers in TYPE_LEXICAL_MARKERS.items()
        if _contains_any(text, markers)
    }


def review_flags(case: dict[str, Any], text: str) -> list[str]:
    """Return operator-facing reasons why a classification needs review."""
    lowered = text.casefold()
    reasons: list[str] = []

    if _contains_any(lowered, REQUEST_MARKERS) and _contains_any(lowered, INACTION_MARKERS):
        reasons.append("Неоднозначная граница между заявлением и жалобой: есть и просьба, и признак бездействия.")
    if _contains_any(lowered, DEFECT_MARKERS) and _contains_any(lowered, INFORMATION_MARKERS):
        reasons.append("Неоднозначная граница между сообщением о дефекте и запросом информации.")

    appeal_type = str(case.get("appeal_type", ""))
    if appeal_type == "предложение" and not _contains_any(lowered, PROPOSAL_MARKERS):
        reasons.append("Модель выбрала «предложение», но в тексте нет лексического маркера предложения.")
    if appeal_type == "отклик" and not _contains_any(lowered, RESPONSE_MARKERS):
        reasons.append("Модель выбрала «отклик», но в тексте нет лексического маркера отклика.")

    lexical_types = _lexical_types(lowered)
    if appeal_type == "сообщение" and "заявление" in lexical_types:
        reasons.append(
            "Модель упростила «заявление» до «сообщения», хотя в тексте есть явный маркер осуществления права."
        )

    if case.get("topic") == "НЕ ОПРЕДЕЛЕНО":
        reasons.append("Компетенция не определена: нужно подтвердить ответственный орган.")

    location = case.get("location")
    if not location:
        reasons.append("Адрес не распознан: географическую маршрутизацию нужно проверить вручную.")
    elif float(location.get("confidence", 0.0)) < LOCATION_CONFIDENCE_THRESHOLD:
        reasons.append("Низкая уверенность распознавания адреса: географическую маршрутизацию нужно проверить.")

    if len(_detected_topics(lowered)) >= 2:
        reasons.append("В тексте обнаружено несколько тем: нужно проверить разделение по АППК ст. 65(2).")

    return reasons
