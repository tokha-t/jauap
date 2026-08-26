"""LLM-first appeal classification with deterministic, honest degradation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

from .llm import complete


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ROUTING = json.loads((DATA_DIR / "routing.json").read_text(encoding="utf-8"))
TOPIC_ALIASES = {
    alias.casefold(): topic
    for topic, route in ROUTING.items()
    for alias in (topic, route["display_name"])
}
FALLBACK_NOTICE = "Резервный режим: классификация по правилам, без модели. Уверенность занижена намеренно."

APPEAL_DEFINITIONS = """АППК статья 4 — типы обращений:
- заявление — Request for assistance in exercising rights/freedoms/lawful interests
- жалоба — Demand to restore or protect rights violated by an administrative act, action, or inaction
- сообщение — Notification of a violation of law, or of defects in the work of a state body
- предложение — Recommendation to improve legislation or the work of state bodies
- отклик — Expression of attitude toward state policy or public events
- запрос — Request for information on matters of personal or public interest

Apply these distinctions in this order:
1. Use жалоба only when the citizen explicitly challenges an identifiable administrative
   refusal, decision, act, action, or inaction and asks to cancel, recognise, or remedy that
   violation. A dissatisfied tone or a request to fix a service is not by itself a жалоба.
2. Use запрос when the primary purpose is to obtain information, a date, a schedule, or an
   explanation (for example, "Когда планируется ремонт?").
3. Use предложение for an idea intended to improve public services (for example,
   "Предлагаю установить дополнительные урны").
4. Use заявление for a first-instance request for assistance in exercising a right or
   receiving/restoring a service (for example, "Прошу восстановить подачу воды").
5. Use сообщение for a factual notification of a current defect or service lapse (for
   example, "Во дворе не вывозят мусор"), even if the wording is emotional or asks staff
   to fix it, unless rule 1 applies."""

RESULT_SCHEMA = {
    "type": "object",
    "required": ["appeal_type", "topic", "routing_targets", "language_detected", "urgency", "confidence", "reasoning"],
    "properties": {
        "appeal_type": {"enum": ["заявление", "жалоба", "сообщение", "предложение", "отклик", "запрос"]},
        "topic": {"type": "string"},
        "routing_targets": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "language_detected": {"enum": ["kk", "ru", "mixed", "latin"]},
        "urgency": {"enum": ["routine", "elevated", "emergency"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reasoning": {"type": "string"},
    },
}

SYSTEM_PROMPT = f"""You classify raw citizen appeals for an internal Kokshetau akimat triage console.
Classify Kazakh, Russian, and intra-sententially code-switched input as-is. Do not translate first and do not run a separate language-detection step.

{APPEAL_DEFINITIONS}

Topic taxonomy and routing data:
{json.dumps(ROUTING, ensure_ascii=False)}

Return appeal_type, topic, routing_targets (a list which may contain more than one target under АППК ст. 65(2)), language_detected, urgency (routine/elevated/emergency), confidence from 0 to 1, and one-sentence reasoning.
For housing inspection/КСК–ОСИ, electricity outages, stray animals, or any other unverified competence, return topic exactly "НЕ ОПРЕДЕЛЕНО".
If confidence is below 0.7 the caller will require human review.
Appeals about prepared/committed criminal offences or threats to state/public safety are emergency under АППК ст. 64(7-2)."""

CLASSIFICATION_CONTRACT_SHA256 = hashlib.sha256(
    (
        SYSTEM_PROMPT
        + "\0"
        + json.dumps(RESULT_SCHEMA, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\0validator:topic-normalisation-v2"
    ).encode("utf-8")
).hexdigest()


TOPIC_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("НЕ ОПРЕДЕЛЕНО", ("собак", "иттер", "ксж", "кск", "оси", "электр", "жарық жоқ", "света нет")),
    ("water_supply", ("вода", "воды", "о воде", "су жоқ", "су құбыр", "su joq", "подачу воды", "труба", "порыв")),
    ("sewerage", ("канализац", "кәріз", "kariz", "кнс")),
    ("heating", ("отоплен", "батаре", "жылу", "jylu")),
    ("waste_removal", ("мусор", "қоқыс", "qoqys")),
    ("snow_cleaning", ("снег", "қар ", "мұз", "лед", "лёд")),
    ("street_lighting", ("фонар", "шамдар", "освещен")),
    ("road_condition", ("яма", "дорог", "шұңқыр", "shungqyr", "ремонт улиц")),
    ("public_transport", ("автобус", "маршрут", "аялдама", "ayaldama")),
    ("landscaping", ("дерев", "ағаш", "agash", "двор", "аулас")),
    ("illegal_construction", ("незакон", "самоволь", "павильон")),
    ("construction", ("капитал", "строитель", "құрылыс", "күрделі жөндеу", "kapitaldy jondeu")),
    ("land", ("земел", "участ", "жер ", "jer ")),
    ("education", ("школ", "мектеп", "детсад")),
    ("social", ("соц", "выплат", "жәрдем", "әлеуметтік", "aleumettik")),
]


def _language(text: str) -> str:
    lowered = text.casefold()
    kazakh_letters = len(re.findall(r"[әғқңөұүһі]", lowered))
    russian_markers = len(re.findall(r"\b(нет|уже|прошу|когда|дом|двор|опять|пожалуйста|требую)\b", lowered))
    latin_markers = len(re.findall(r"\b(joq|qashan|uide|jatyr|tuspedi|janbaidy)\b", lowered))
    if latin_markers:
        return "latin"
    if kazakh_letters and russian_markers:
        return "mixed"
    return "kk" if kazakh_letters else "ru"


def _topic(text: str) -> str:
    lowered = text.casefold()
    for topic, keywords in TOPIC_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return topic
    return "НЕ ОПРЕДЕЛЕНО"


def _appeal_type(text: str) -> str:
    lowered = text.casefold()
    if any(word in lowered for word in ("требую признать", "отказал", "бездейств", "шағым", "ответа нет")):
        return "жалоба"
    if any(word in lowered for word in ("когда планируется", "предоставить", "разъяснить", "объяснить", "ақпарат")):
        return "запрос"
    if any(word in lowered for word in ("предлагаю", "ұсынамын")):
        return "предложение"
    if any(word in lowered for word in ("поддерживаю", "не поддерживаю", "пікір")):
        return "отклик"
    if any(word in lowered for word in ("прошу восстановить", "қалпына келтір")):
        return "заявление"
    return "сообщение"


def _routing_targets(topic: str, text: str, settlement: str = "Кокшетау") -> list[str]:
    if settlement in {"Красный Яр", "Кызыл-Жулдыз"}:
        return ["Аппарат акима Красноярского сельского округа"]
    if settlement == "Станционный":
        return ["Аппарат акима Станционной поселковой администрации"]
    lowered = text.casefold()
    if "павильон" in lowered and any(word in lowered for word in ("яма", "дорог")):
        return ["Отдел ЖКХ, ПТ и АД", "Отдел земельных отношений, архитектуры и градостроительства"]
    route = ROUTING.get(topic, ROUTING["НЕ ОПРЕДЕЛЕНО"])
    operational = route["operational_owner"]
    statutory = route["statutory_clock_holder"]
    return [operational] if operational == statutory else [operational, statutory]


def _urgency(text: str) -> str:
    lowered = text.casefold()
    if any(term in lowered for term in ("бомба", "теракт", "убийство", "взрыв", "угроза безопасности")):
        return "emergency"
    if any(term in lowered for term in ("опасно", "қауіпті", "авария", "срочно", "тезірек")):
        return "elevated"
    return "routine"


def fallback_classification(text: str, settlement: str = "Кокшетау") -> dict[str, Any]:
    """Conservative fallback required by §10.4; confidence remains 0.3."""
    topic = _topic(text)
    return {
        "appeal_type": _appeal_type(text),
        "topic": topic,
        "routing_targets": _routing_targets(topic, text, settlement),
        "language_detected": _language(text),
        "urgency": _urgency(text),
        "confidence": 0.3,
        "reasoning": "Резервная классификация по ключевым признакам; требуется проверка специалистом.",
        "needs_human_review": True,
        "warning": FALLBACK_NOTICE,
    }


def _normalise_model_topic(topic: Any, text: str) -> str:
    """Constrain provider labels to routing keys and preserve verified unknown competence."""
    rule_topic = _topic(text)
    if rule_topic == "НЕ ОПРЕДЕЛЕНО":
        return rule_topic
    normalised = TOPIC_ALIASES.get(str(topic).strip().casefold())
    if normalised and normalised != "НЕ ОПРЕДЕЛЕНО":
        return normalised
    return rule_topic


def _validate_result(result: dict[str, Any], text: str, settlement: str) -> dict[str, Any]:
    missing = set(RESULT_SCHEMA["required"]) - result.keys()
    if missing:
        raise ValueError(f"Missing classification fields: {sorted(missing)}")
    result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))
    result["topic"] = _normalise_model_topic(result["topic"], text)
    result["routing_targets"] = _routing_targets(result["topic"], text, settlement)
    result["needs_human_review"] = result["confidence"] < 0.7
    return result


def classify_text(text: str, settlement: str = "Кокшетау") -> dict[str, Any]:
    try:
        result = complete(SYSTEM_PROMPT, text, RESULT_SCHEMA)
        if not isinstance(result, dict):
            raise ValueError("Classifier did not return an object")
        return _validate_result(result, text, settlement)
    except Exception:
        return fallback_classification(text, settlement)


def classify_batch(
    records: list[dict[str, Any]],
    *,
    on_warning: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Classify every record independently so one failure never kills a batch."""
    outputs: list[dict[str, Any]] = []
    for record in records:
        settlement = record.get("settlement", "Кокшетау")
        result = classify_text(record["raw_text"], settlement)
        if result.get("warning") and on_warning:
            on_warning(f"{record.get('id', 'без ID')}: {result['warning']}")
        outputs.append({"id": record.get("id"), **result})
    return outputs
