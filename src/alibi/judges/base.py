from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from alibi.types import Question


class Judge(Protocol):
    """Pluggable judge backend. Callers depend on this, never on an implementation."""

    def evaluate(self, state: str, questions: list[Question]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class CallRecord:
    """Per-call cost/latency, so judge backends stay comparable apples-to-apples."""

    model: str
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    cost: float | None
