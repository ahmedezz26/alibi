"""Pick a TraceSource for one trace given as a file path or a LangSmith trace id."""

from __future__ import annotations

import os
from pathlib import Path

from alibi.types import Step

SOURCES = ("auto", "json", "claude-code", "langsmith")
TRACE_SUFFIXES = (".json", ".jsonl")


def load_steps(path_or_id: str, source: str = "auto", project: str | None = None) -> list[Step]:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    # A path that looks like a trace file but is not there is a typo, not a LangSmith id:
    # say so, rather than posting the path to the LangSmith API and failing on a UUID check.
    looks_like_a_path = path_or_id.endswith(TRACE_SUFFIXES) or os.sep in path_or_id
    if source != "langsmith" and looks_like_a_path and not Path(path_or_id).is_file():
        raise FileNotFoundError(f"no such trace file: {path_or_id}")
    if source == "auto":
        if path_or_id.endswith(".jsonl"):
            source = "claude-code"
        elif path_or_id.endswith(".json"):
            source = "json"
        else:
            source = "langsmith"
    if source == "claude-code":
        from alibi.sources.claudecode import ClaudeCodeTraceSource

        return ClaudeCodeTraceSource().get_trace(path_or_id)
    if source == "json":
        from alibi.sources.jsonfile import JsonFileTraceSource

        return JsonFileTraceSource().get_trace(path_or_id)
    from alibi.sources.langsmith import LangSmithTraceSource

    return LangSmithTraceSource(project_name=project).get_trace(path_or_id)
