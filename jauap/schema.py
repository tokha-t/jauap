"""Serializable domain records used across the JAUAP pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class Appeal:
    id: str
    raw_text: str
    received_at: datetime
    channel: str
    applicant_name: str
    language_detected: str


@dataclass
class Location:
    raw_mention: str
    street: str | None
    building: str | None
    district: str | None
    settlement: str
    lat: float
    lon: float
    confidence: float


@dataclass
class TriagedCase:
    appeal: Appeal
    appeal_type: str
    topic: str
    routing_targets: list[str]
    statutory_clock_holder: str
    operational_owner: str
    entity_type: str
    oblast_escalation: str | None
    registered_date: date
    deadline: date
    deadline_basis: str
    working_days_remaining: int
    deemed_refusal_date: date
    location: Location | None
    cluster_id: str | None
    escalation_risk: float
    risk_factors: list[str]
    misroute_cost_avoided: int
    draft_response: str | None
    confidence: float
    needs_human_review: bool
    classification_reasoning: str = ""
    urgency: str = "routine"
    emotional_escalation: bool = False


@dataclass
class ClusterSummary:
    cluster_id: str
    member_ids: list[str]
    member_count: int
    oldest_received_at: datetime
    earliest_deadline: date
    representative_text: str
    distinct_applicants: int
    low_confidence: bool = False
    resolved: bool = False
    notification_messages: list[dict[str, str]] = field(default_factory=list)
