"""LangSmith implementation of TraceSource. The only module that talks to LangSmith."""

from __future__ import annotations

from typing import Any

from alibi.types import Step


class LangSmithTraceSource:
    def __init__(self, client: Any | None = None, project_name: str | None = None) -> None:
        if client is None:
            from langsmith import Client

            from alibi.config import load_settings

            client = Client(api_key=load_settings().langsmith_api_key)
        self._client = client
        self._project_name = project_name

    def get_trace(self, trace_id: str) -> list[Step]:
        kwargs: dict[str, Any] = {"trace_id": trace_id}
        if self._project_name:
            kwargs["project_name"] = self._project_name
        runs = list(self._client.list_runs(**kwargs))
        # dotted_order encodes execution order across the run tree; fall back to start_time.
        runs.sort(key=lambda r: (r.dotted_order or "", r.start_time or 0))
        return [_to_step(r) for r in runs]


def _to_step(run: Any) -> Step:
    return Step(
        step_id=str(run.id),
        type=run.run_type,
        timestamp=run.start_time,
        inputs=dict(run.inputs or {}),
        outputs=dict(run.outputs) if run.outputs is not None else None,
        error=run.error,
        name=run.name,
    )
