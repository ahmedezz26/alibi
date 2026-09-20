"""Pick a TraceSource for one trace given as a file path or a LangSmith trace id."""

from __future__ import annotations

import os
from pathlib import Path

from alibi.types import Step

# One routing table: resolution and dispatch both read it, so a new format cannot be added
# to one and forgotten in the other.
BY_SUFFIX = {".jsonl": "claude-code", ".json": "json"}
SOURCES = ("auto", "json", "claude-code", "langsmith")


class TraceFormatError(ValueError):
    """The file was found and read, but it is not a trace this adapter understands."""


def load_steps(path_or_id: str, source: str = "auto", project: str | None = None) -> list[Step]:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    if source != "langsmith":
        path = _resolve(path_or_id, source)
        if path is not None:
            # Route off the file that was actually found, so a resolved "<dir>/<id>" cannot
            # be treated as a LangSmith id just because the argument had no suffix.
            path_or_id = str(path)
            if source == "auto":
                source = BY_SUFFIX.get(path.suffix, "json")
    if source in ("auto", "langsmith"):
        from alibi.sources.langsmith import LangSmithTraceSource

        return LangSmithTraceSource(project_name=project).get_trace(path_or_id)
    if source == "claude-code":
        from alibi.sources.claudecode import ClaudeCodeTraceSource

        adapter = ClaudeCodeTraceSource()
    else:
        from alibi.sources.jsonfile import JsonFileTraceSource

        adapter = JsonFileTraceSource()
    try:
        return adapter.get_trace(path_or_id)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError) as e:
        raise TraceFormatError(f"could not read {path_or_id}: {e}") from e


def _resolve(path_or_id: str, source: str) -> Path | None:
    """The file this argument names, or None when it is not a path at all (a LangSmith id).

    MCP clients pass the path verbatim, so ``~`` arrives unexpanded. A path-shaped argument
    that is not there raises here rather than being POSTed to the LangSmith API as an id.
    ``JsonFileTraceSource`` also resolves a bare ``<dir>/<id>`` to ``<dir>/<id>.json``.
    """
    path = Path(path_or_id).expanduser()
    separators = (os.sep, os.altsep) if os.altsep else (os.sep,)
    if path.suffix not in BY_SUFFIX and not any(s in path_or_id for s in separators):
        return None if source == "auto" else _must_exist(path, path_or_id)
    if path.is_file():
        return path
    if path.suffix not in BY_SUFFIX:
        with_json = path.with_name(f"{path.name}.json")
        if with_json.is_file():
            return with_json
    raise FileNotFoundError(f"no such trace file: {path_or_id}")


def _must_exist(path: Path, given: str) -> Path:
    """An explicit --source json/claude-code names a file, even without a separator."""
    if path.is_file():
        return path
    with_json = path.with_name(f"{path.name}.json")
    if with_json.is_file():
        return with_json
    raise FileNotFoundError(f"no such trace file: {given}")
