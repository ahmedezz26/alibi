"""TraceSource over local JSON files: one file per trace, a list of step objects."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from alibi.types import Step


class JsonFileTraceSource:
    """Reads ``<directory>/<trace_id>.json``, or a single file when trace_id is a path."""

    def __init__(self, directory: str | Path = ".") -> None:
        self._directory = Path(directory)

    def get_trace(self, trace_id: str) -> list[Step]:
        path = Path(trace_id)
        if not path.is_file():
            path = self._directory / f"{trace_id}.json"
        return [_to_step(i, raw) for i, raw in enumerate(json.loads(path.read_text()))]


def _to_step(position: int, raw: dict[str, Any]) -> Step:
    ts = raw.get("timestamp")
    return Step(
        step_id=str(raw.get("step_id", position)),
        type=raw.get("type", "unknown"),
        timestamp=datetime.fromisoformat(ts) if ts else None,
        inputs=raw.get("inputs") or {},
        outputs=raw.get("outputs"),
        error=raw.get("error"),
        name=raw.get("name"),
    )
