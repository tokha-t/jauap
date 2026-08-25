"""Explainable escalation-risk scoring; no model or hidden features."""

from __future__ import annotations

from datetime import date
from typing import Any


def risk_band(score: float) -> str:
    if score < 0.35:
        return "green"
    if score <= 0.65:
        return "amber"
    return "red"


def score_case(case: dict[str, Any], cluster: dict[str, Any]) -> tuple[float, list[str]]:
    """Return the exact weighted sum and human-readable reasons from §8.6."""
    score = 0.0
    factors: list[str] = []
    member_count = int(cluster["member_count"])
    remaining = int(case["working_days_remaining"])

    if member_count >= 2:
        score += 0.30
        factors.append(f"Повторное обращение по тому же объекту ({member_count} обращений в текущем кластере)")
    if remaining <= 3:
        score += 0.20
        factors.append(f"До истечения срока осталось {remaining} раб. дн.; приближается состояние отказа по ст. 91(2)")
    if remaining < 0:
        score += 0.25
        factors.append("Срок уже истёк: считается отказом по АППК ст. 91(2)")
    if member_count >= 5:
        score += 0.15
        factors.append(f"В кластере {member_count} обращений от {cluster['distinct_applicants']} разных заявителей")
    if case["appeal_type"] == "жалоба":
        score += 0.10
        factors.append("Тип — жалоба; срок по ст. 99 не подлежит продлению")
    if case.get("emotional_escalation"):
        score += 0.10
        factors.append("В тексте упомянуты прокуратура, суд, СМИ или вышестоящий орган")
    if case.get("urgency") == "emergency":
        score += 0.30
        factors.append("Требуется немедленная передача по АППК ст. 64(7-2)")
    if case["topic"] == "НЕ ОПРЕДЕЛЕНО" or case.get("entity_type") == "НЕ ОПРЕДЕЛЕНО":
        score += 0.10
        factors.append("Компетентный орган не определён; требуется ручное уточнение маршрута")

    return round(min(1.0, score), 2), factors


def detect_emotional_escalation(text: str) -> bool:
    lowered = text.casefold()
    return any(term in lowered for term in (
        "прокуратур", "президент", "суд", "сми", "журналист", "аким области",
        "президент әкімшілігі", "сотқа", "бұқаралық ақпарат",
    ))


def apply_risk(cases: list[dict[str, Any]], clusters: list[dict[str, Any]]) -> None:
    by_cluster = {cluster["cluster_id"]: cluster for cluster in clusters}
    for case in cases:
        cluster = by_cluster[case["cluster_id"]]
        case["emotional_escalation"] = detect_emotional_escalation(case["raw_text"])
        score, factors = score_case(case, cluster)
        case["escalation_risk"] = score
        case["risk_factors"] = factors
        case["risk_band"] = risk_band(score)
