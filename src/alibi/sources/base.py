from __future__ import annotations

from typing import Protocol

from alibi.types import Step


class TraceSource(Protocol):
    """Adapter over a tracing backend. All trace ingestion goes through this."""

    def get_trace(self, trace_id: str) -> list[Step]: ...
