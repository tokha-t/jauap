"""Local-only address extraction, normalisation, and approximate resolution."""

from __future__ import annotations

import json
import math
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .llm import complete
from .schema import Location


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
STREET_DATA = json.loads((DATA_DIR / "streets.json").read_text(encoding="utf-8"))
STREETS = STREET_DATA["streets"]
SETTLEMENTS = STREET_DATA["settlements"]
DISTRICTS = STREET_DATA["districts"]

EXTRACTION_SCHEMA = {
    "type": "object",
    "required": ["raw_mention", "street", "building", "district", "settlement"],
    "properties": {
        "raw_mention": {"type": "string"},
        "street": {"type": ["string", "null"]},
        "building": {"type": ["string", "null"]},
        "district": {"type": ["string", "null"]},
        "settlement": {"enum": ["Кокшетау", "Красный Яр", "Станционный", "Кызыл-Жулдыз"]},
    },
}
EXTRACTION_SYSTEM = """Extract one address mention from a raw Kokshetau-area citizen appeal.
Return the exact source substring as raw_mention and best guesses for street, building, district, and settlement.
Do not geocode and do not invent missing components."""

PREFIXES = re.compile(r"\b(?:ул\.?|улица|көшесі|көше|проспект|пр\.?|дом|д\.?|үй)\b", re.IGNORECASE)
PUNCTUATION = re.compile(r"[^0-9a-zа-яәғқңөұүһі\s-]", re.IGNORECASE)
SPACES = re.compile(r"\s+")


def _latin_to_cyrillic(value: str) -> str:
    """Small, intentionally conservative normaliser for demo transliteration."""
    replacements = [
        ("sh", "ш"), ("ch", "ч"), ("zh", "ж"), ("ng", "ң"),
        ("q", "қ"), ("gh", "ғ"), ("j", "ж"), ("y", "ы"),
    ]
    result = value
    for source, target in replacements:
        result = result.replace(source, target)
    return result


def normalize_address(value: str) -> str:
    normalized = value.casefold().replace("-үй", " ")
    normalized = PREFIXES.sub(" ", normalized)
    normalized = PUNCTUATION.sub(" ", normalized)
    if re.search(r"\b(?:joq|qashan|uide|koshesi)\b", normalized):
        normalized = _latin_to_cyrillic(normalized)
    return SPACES.sub(" ", normalized).strip()


def _settlement(text: str) -> str:
    lowered = text.casefold()
    if "қызыл-жұлдыз" in lowered or "кызыл-жулдыз" in lowered:
        return "Кызыл-Жулдыз"
    if "красный яр" in lowered:
        return "Красный Яр"
    if "станционн" in lowered:
        return "Станционный"
    return "Кокшетау"


def _district(text: str) -> str | None:
    lowered = text.casefold()
    return next((district for district in DISTRICTS if district.casefold() in lowered), None)


def _similarity(left: str, right: str) -> float:
    try:
        from rapidfuzz.fuzz import ratio

        return float(ratio(left, right))
    except ImportError:
        return SequenceMatcher(None, left, right).ratio() * 100


def _candidate_phrases(text: str, max_words: int = 5) -> list[str]:
    tokens = normalize_address(text).split()
    phrases: list[str] = []
    for length in range(1, min(max_words, len(tokens)) + 1):
        phrases.extend(" ".join(tokens[index:index + length]) for index in range(len(tokens) - length + 1))
    return phrases


def _match_street(text: str, settlement: str, extracted_street: str | None = None) -> tuple[dict[str, Any] | None, float]:
    available = [street for street in STREETS if street["settlement"] == settlement]
    if settlement == "Кызыл-Жулдыз":
        return None, 0.0
    normalized_text = normalize_address(text)

    exact = [street for street in available if normalize_address(street["name"]) in normalized_text]
    if exact:
        match = max(exact, key=lambda street: len(street["name"]))
        return match, 100.0

    phrases = [normalize_address(extracted_street)] if extracted_street else []
    phrases.extend(_candidate_phrases(text))
    best: dict[str, Any] | None = None
    best_score = 0.0
    for street in available:
        street_name = normalize_address(street["name"])
        score = max((_similarity(street_name, phrase) for phrase in phrases if phrase), default=0.0)
        if score > best_score:
            best, best_score = street, score
    return (best, best_score) if best_score >= 80 else (None, best_score)


def _local_extract(text: str) -> dict[str, Any]:
    settlement = _settlement(text)
    street, _ = _match_street(text, settlement)
    number = re.search(r"(?<![A-Za-zА-Яа-яӘәҒғҚқҢңӨөҰұҮүҺһІі])([1-9]\d{0,2})(?:-[үйа-я])?", text)
    street_name = street["name"] if street else None
    raw_mention = ""
    if street_name:
        street_tokens = street_name.split()
        token = street_tokens[-1]
        found = re.search(re.escape(token), text, re.IGNORECASE)
        if found:
            end = number.end() if number and number.start() >= found.start() else found.end()
            raw_mention = text[found.start():end]
    return {
        "raw_mention": raw_mention,
        "street": street_name,
        "building": number.group(1) if number else None,
        "district": _district(text),
        "settlement": settlement,
    }


def extract_address(text: str, *, use_llm: bool = True) -> dict[str, Any]:
    if use_llm:
        try:
            result = complete(EXTRACTION_SYSTEM, text, EXTRACTION_SCHEMA)
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    return _local_extract(text)


def resolve_location(text: str, *, use_llm: bool = True) -> Location | None:
    """Resolve to a deterministic placeholder coordinate without network access."""
    extracted = extract_address(text, use_llm=use_llm)
    settlement = extracted.get("settlement") or _settlement(text)
    street, score = _match_street(text, settlement, extracted.get("street"))
    if street is None:
        return None

    building = extracted.get("building")
    number = int(building) if building and str(building).isdigit() else 0
    lat_jitter = ((number % 7) - 3) * 0.00004
    lon_jitter = ((number % 11) - 5) * 0.00004 / max(math.cos(math.radians(street["lat"])), 0.1)
    return Location(
        raw_mention=extracted.get("raw_mention") or street["name"],
        street=street["name"],
        building=str(building) if building else None,
        district=extracted.get("district"),
        settlement=settlement,
        lat=float(street["lat"]) + lat_jitter,
        lon=float(street["lon"]) + lon_jitter,
        confidence=round(min(0.99, score / 100), 2),
    )
