"""Transparent single-linkage duplicate detection for triaged appeals."""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date, datetime
from typing import Any


EARTH_RADIUS_M = 6_371_000
TOKEN_RE = re.compile(r"[0-9a-zа-яәғқңөұүһі]+", re.IGNORECASE)


def haversine(left: dict[str, Any], right: dict[str, Any]) -> float:
    lat1, lon1 = math.radians(float(left["lat"])), math.radians(float(left["lon"]))
    lat2, lon2 = math.radians(float(right["lat"])), math.radians(float(right["lon"]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))


def _received(case: dict[str, Any]) -> datetime:
    value = case.get("received_at") or case.get("appeal", {}).get("received_at")
    return value if isinstance(value, datetime) else datetime.fromisoformat(value)


def _deadline(case: dict[str, Any]) -> date:
    value = case["deadline"]
    return value if isinstance(value, date) else date.fromisoformat(value)


def _text(case: dict[str, Any]) -> str:
    return case.get("raw_text") or case.get("appeal", {}).get("raw_text", "")


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def _tfidf_vectors(texts: list[str]) -> list[dict[str, float]]:
    documents = [_tokens(text) for text in texts]
    document_frequency = Counter(token for tokens in documents for token in set(tokens))
    count = len(documents)
    vectors: list[dict[str, float]] = []
    for tokens in documents:
        term_frequency = Counter(tokens)
        vector = {
            token: frequency * (math.log((count + 1) / (document_frequency[token] + 1)) + 1)
            for token, frequency in term_frequency.items()
        }
        vectors.append(vector)
    return vectors


def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
    numerator = sum(value * right.get(token, 0.0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _find(parent: list[int], value: int) -> int:
    while parent[value] != value:
        parent[value] = parent[parent[value]]
        value = parent[value]
    return value


def _union(parent: list[int], left: int, right: int) -> None:
    root_left, root_right = _find(parent, left), _find(parent, right)
    if root_left != root_right:
        parent[root_right] = root_left


def cluster_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cluster on topic × time × location, with TF-IDF fallback when unlocated."""
    parent = list(range(len(cases)))
    vectors = _tfidf_vectors([_text(case) for case in cases])
    low_confidence_edges: set[tuple[int, int]] = set()

    for left_index, left in enumerate(cases):
        for right_index in range(left_index + 1, len(cases)):
            right = cases[right_index]
            if left["topic"] != right["topic"]:
                continue
            if abs((_received(left) - _received(right)).total_seconds()) >= 21 * 86400:
                continue
            left_location, right_location = left.get("location"), right.get("location")
            if left_location is not None and right_location is not None:
                if haversine(left_location, right_location) < 150:
                    _union(parent, left_index, right_index)
            elif left_location is None and right_location is None and _cosine(vectors[left_index], vectors[right_index]) > 0.75:
                _union(parent, left_index, right_index)
                low_confidence_edges.add((left_index, right_index))

    groups: dict[int, list[int]] = {}
    for index in range(len(cases)):
        groups.setdefault(_find(parent, index), []).append(index)

    ordered_groups = sorted(groups.values(), key=lambda indices: min(cases[index]["id"] for index in indices))
    summaries: list[dict[str, Any]] = []
    for sequence, indices in enumerate(ordered_groups, start=1):
        members = [cases[index] for index in indices]
        cluster_id = f"CL-{sequence:03d}"
        for member in members:
            member["cluster_id"] = cluster_id
        clearest = max(
            members,
            key=lambda case: (float(case.get("confidence", 0)), -abs(len(_text(case)) - 110)),
        )
        group_low_confidence = any(
            left in indices and right in indices for left, right in low_confidence_edges
        )
        summaries.append({
            "cluster_id": cluster_id,
            "member_ids": [case["id"] for case in members],
            "member_count": len(members),
            "oldest_received_at": min(_received(case) for case in members).isoformat(),
            "earliest_deadline": min(_deadline(case) for case in members).isoformat(),
            "representative_text": _text(clearest),
            "distinct_applicants": len({case.get("applicant_name") or case.get("appeal", {}).get("applicant_name") for case in members}),
            "low_confidence": group_low_confidence,
            "resolved": False,
            "notification_messages": [],
        })
    return summaries
