"""Single calculation clock for live and frozen demo processing."""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date
from pathlib import Path
from typing import Iterator


RESULTS_PATH = Path(__file__).resolve().parents[1] / "data" / "demo_results.json"
_FROZEN_MODE: ContextVar[bool] = ContextVar("jauap_frozen_clock", default=False)


def _frozen_reference_date() -> date:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    return date.fromisoformat(results["as_of_date"])


def demo_now() -> date:
    """Return the committed demo date in frozen mode and today's date live."""
    if _FROZEN_MODE.get():
        return _frozen_reference_date()
    return date.today()


@contextmanager
def clock_mode(*, frozen: bool) -> Iterator[None]:
    """Scope the calculation clock without leaking mode between requests."""
    token = _FROZEN_MODE.set(frozen)
    try:
        yield
    finally:
        _FROZEN_MODE.reset(token)
