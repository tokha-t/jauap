"""Kazakhstan administrative deadline calculations for the demo.

The legal counting rule used here is deliberately explicit: a period stated as
N working days starts on the day *after* registration (АППК ст. 76(2)).

Hand-checked examples with no intervening public holiday:

* registered Monday 2026-02-02 + 1 working day = Tuesday 2026-02-03
* registered Friday 2026-02-06 + 1 working day = Monday 2026-02-09
* registered Monday 2026-02-02 + 15 working days = Monday 2026-02-23
* registered Monday 2026-02-02 + 20 working days = Monday 2026-03-02

The holiday file intentionally contains a TODO marker for Kurban Ait. It moves
each year and is never inferred by this module.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
HOLIDAYS_PATH = DATA_DIR / "holidays.json"
WORKDAY_END = time(18, 0)

DEADLINES = {
    "заявление": {"days": 15, "unit": "working", "basis": "АППК ст. 76(1)", "extendable": True},
    "жалоба": {"days": 20, "unit": "working", "basis": "АППК ст. 99", "extendable": False},
    "сообщение": {"days": 15, "unit": "working", "basis": "АППК ст. 87 + 76(1)", "extendable": True},
    "предложение": {"days": 15, "unit": "working", "basis": "АППК ст. 87 + 76(1)", "extendable": True},
    "отклик": {"days": 15, "unit": "working", "basis": "АППК ст. 87 + 76(1)", "extendable": True},
    "запрос": {"days": 15, "unit": "working", "basis": "АППК ст. 87 + 76(1)", "extendable": True},
    "запрос_401V": {"days": 15, "unit": "calendar", "basis": "Закон № 401-V ст. 11(10)", "extendable": True},
    "петиция_мест": {"days": 20, "unit": "working", "basis": "АППК ст. 90-5(1)(2)", "extendable": False},
}

ARTICLE_76 = (
    "ст. 76(1): «Срок административной процедуры, возбужденной на основании "
    "обращения, составляет пятнадцать рабочих дней со дня регистрации обращения, "
    "если иное не предусмотрено законами Республики Казахстан.»"
)
ARTICLE_99 = (
    "ст. 99: «Срок рассмотрения жалобы составляет двадцать рабочих дней со дня "
    "регистрации жалобы… Продление срока рассмотрения жалобы не допускается, за "
    "исключением случаев, установленных законами Республики Казахстан.»"
)


@lru_cache(maxsize=1)
def holidays() -> frozenset[date]:
    """Load editable ISO holiday dates; ignore the explicit TODO marker."""
    values = json.loads(HOLIDAYS_PATH.read_text(encoding="utf-8"))
    return frozenset(date.fromisoformat(value) for value in values if value[:4].isdigit())


def is_working_day(value: date) -> bool:
    return value.weekday() < 5 and value not in holidays()


def next_working_day(value: date) -> date:
    candidate = value
    while not is_working_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def register_date(received_at: datetime) -> date:
    """Apply АППК ст. 64(3) to determine the registration date."""
    received_date = received_at.date()
    if not is_working_day(received_date) or received_at.time() > WORKDAY_END:
        return next_working_day(received_date + timedelta(days=1))
    return received_date


def add_working_days(start: date, n: int) -> date:
    """Return the date N working days after start, excluding start itself."""
    if n < 0:
        raise ValueError("n must be non-negative")
    candidate = start
    remaining = n
    while remaining:
        candidate += timedelta(days=1)
        if is_working_day(candidate):
            remaining -= 1
    return next_working_day(candidate)


def working_days_between(a: date, b: date) -> int:
    """Signed count of working days after a through b, inclusive of b."""
    if a == b:
        return 0
    direction = 1 if b > a else -1
    current = a
    count = 0
    while current != b:
        current += timedelta(days=direction)
        if is_working_day(current):
            count += direction
    return count


def deadline_for(appeal_type: str, registered: date) -> date:
    """Return the statutory deadline, rolled forward per АППК ст. 76(5)."""
    rule = DEADLINES[appeal_type]
    if rule["unit"] == "calendar":
        return next_working_day(registered + timedelta(days=int(rule["days"])))
    return add_working_days(registered, int(rule["days"]))


def notification_deadline(extension_decision_date: date) -> date:
    """Three-working-day applicant notification sub-deadline, ст. 76(3)."""
    return add_working_days(extension_decision_date, 3)


def extension_deadline(extension_decision_date: date) -> date:
    """Two calendar months from the decision, rolled to a working day."""
    month_index = extension_decision_date.month - 1 + 2
    year = extension_decision_date.year + month_index // 12
    month = month_index % 12 + 1
    month_lengths = [31, 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
                     31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day = min(extension_decision_date.day, month_lengths[month - 1])
    return next_working_day(date(year, month, day))


def legal_tooltip(appeal_type: str) -> str:
    return ARTICLE_99 if appeal_type == "жалоба" else ARTICLE_76
