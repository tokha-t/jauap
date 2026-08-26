"""Pure appeal-processing pipeline shared by the app and demo freezer."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from .classify import classify_batch
from .cluster import cluster_cases
from .deadline_engine import DEADLINES, deadline_for, register_date, working_days_between
from .draft import draft_response
from .geo import resolve_location
from .risk import apply_risk


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ROUTING = json.loads((DATA_DIR / "routing.json").read_text(encoding="utf-8"))


def _route_metadata(classification: dict[str, Any]) -> dict[str, Any]:
    targets = classification["routing_targets"]
    rural = targets and targets[0].startswith("Аппарат акима")
    if rural:
        return {
            "operational_owner": targets[0],
            "statutory_clock_holder": targets[0],
            "entity_type": "ГУ",
            "oblast_escalation": None,
        }
    route = ROUTING.get(classification["topic"], ROUTING["НЕ ОПРЕДЕЛЕНО"])
    return {
        key: route.get(key)
        for key in ("operational_owner", "statutory_clock_holder", "entity_type", "oblast_escalation")
    }


def process_records(
    records: list[dict[str, Any]],
    *,
    deterministic_support: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Classify and enrich records without importing the Streamlit application.

    ``deterministic_support`` keeps geocoding and response drafting local while
    classification still runs through the real ``classify_text`` path. The
    demo freezer uses it so its model cache records classification evidence,
    not unrelated drafting or address-extraction calls.
    """
    classifier_warnings: list[str] = []
    classified = classify_batch(records, on_warning=classifier_warnings.append)
    by_id = {result["id"]: result for result in classified}
    today = date.today()
    cases: list[dict[str, Any]] = []

    for record in records:
        result = by_id[record["id"]]
        received = datetime.fromisoformat(record["received_at"])
        registered = register_date(received)
        appeal_type = result["appeal_type"] if result["appeal_type"] in DEADLINES else "сообщение"
        deadline = deadline_for(appeal_type, registered)
        location = resolve_location(record["raw_text"], use_llm=not deterministic_support)
        route = _route_metadata(result)
        settlement = record.get("settlement", "Кокшетау")
        case = {
            "id": record["id"],
            "raw_text": record["raw_text"],
            "received_at": record["received_at"],
            "channel": record.get("channel", "ввод оператора"),
            "applicant_name": record.get("applicant_name", "Синтетический заявитель"),
            "settlement": settlement,
            "language_detected": result["language_detected"],
            "appeal_type": appeal_type,
            "topic": result["topic"],
            "routing_targets": result["routing_targets"],
            **route,
            "registered_date": registered.isoformat(),
            "deadline": deadline.isoformat(),
            "deadline_basis": DEADLINES[appeal_type]["basis"],
            "working_days_remaining": working_days_between(today, deadline),
            "deemed_refusal_date": deadline.isoformat(),
            "deemed_refusal": working_days_between(today, deadline) < 0,
            "location": asdict(location) if location else None,
            "cluster_id": None,
            "escalation_risk": 0.0,
            "risk_factors": [],
            "misroute_cost_avoided": 0 if result["topic"] == "НЕ ОПРЕДЕЛЕНО" else 3,
            "draft_response": None,
            "confidence": result["confidence"],
            "needs_human_review": bool(result["needs_human_review"] or location is None),
            "classification_reasoning": result["reasoning"],
            "urgency": result["urgency"],
            "emotional_escalation": False,
        }
        cases.append(case)

    clusters = cluster_cases(cases)
    apply_risk(cases, clusters)
    for case in cases:
        case["draft_response"] = draft_response(case, frozen_demo=deterministic_support)
    return cases, clusters, classifier_warnings
