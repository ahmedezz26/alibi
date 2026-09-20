"""Pick a TraceSource for one trace given as a file path or a LangSmith trace id."""

from __future__ import annotations

import os
from pathlib import Path

from alibi.types import Step

# One routing table: the guard below and the dispatch both read it, so adding a format
# cannot leave a new extension falling through to LangSmith.
BY_SUFFIX = {".jsonl": "claude-code", ".json": "json"}
SOURCES = ("auto", *dict.fromkeys(BY_SUFFIX.values()), "langsmith")


def load_steps(path_or_id: str, source: str = "auto", project: str | None = None) -> list[Step]:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    if source != "langsmith":
        # MCP clients pass the path verbatim, so "~/..." arrives unexpanded.
        path_or_id = str(Path(path_or_id).expanduser())
        _check_exists(path_or_id, source)
    if source == "auto":
        source = BY_SUFFIX.get(Path(path_or_id).suffix, "langsmith")
    if source == "claude-code":
        from alibi.sources.claudecode import ClaudeCodeTraceSource

        return ClaudeCodeTraceSource().get_trace(path_or_id)
    if source == "json":
        from alibi.sources.jsonfile import JsonFileTraceSource

        return JsonFileTraceSource().get_trace(path_or_id)
    from alibi.sources.langsmith import LangSmithTraceSource

    return LangSmithTraceSource(project_name=project).get_trace(path_or_id)


def _check_exists(path_or_id: str, source: str) -> None:
    """A path-shaped argument that is not there is a typo, not a LangSmith id: say so rather
    than posting the path to the LangSmith API and failing on a UUID check. ``JsonFileTraceSource``
    also resolves a bare ``<dir>/<id>`` to ``<dir>/<id>.json``, so that form is left alone."""
    separators = (os.sep, os.altsep) if os.altsep else (os.sep,)
    looks_like_a_path = Path(path_or_id).suffix in BY_SUFFIX or any(
        s in path_or_id for s in separators
    )
    if not looks_like_a_path:
        return
    candidates = [Path(path_or_id)]
    if source in ("auto", "json") and Path(path_or_id).suffix not in BY_SUFFIX:
        candidates.append(Path(f"{path_or_id}.json"))
    if not any(c.is_file() for c in candidates):
        raise FileNotFoundError(f"no such trace file: {path_or_id}")
