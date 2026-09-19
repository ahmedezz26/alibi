"""Best-effort TraceSource for Claude Code session transcripts
(``~/.claude/projects/<project>/<session-id>.jsonl``).

The JSONL format is internal to Claude Code and may change between versions
(https://code.claude.com/docs/en/sessions.md). Unknown lines and block types are skipped.
Thinking blocks are skipped: they are often redacted and are not the agent's actions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alibi.types import Step


class ClaudeCodeTraceSource:
    def get_trace(self, trace_id: str) -> list[Step]:
        steps: list[Step] = []
        for line in Path(trace_id).read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("type") not in ("user", "assistant"):
                continue
            for kind, name, inputs in _blocks(entry):
                steps.append(Step(str(len(steps)), kind, None, inputs, name=name))
        return steps


def _blocks(entry: dict[str, Any]):
    content = (entry.get("message") or {}).get("content")
    role = entry["type"]
    if isinstance(content, str):
        yield role, None, {"content": content}
        return
    for block in content or []:
        t = block.get("type")
        if t == "text":
            yield role, None, {"content": block.get("text", "")}
        elif t == "tool_use":
            yield "assistant", block.get("name"), block.get("input") or {}
        elif t == "tool_result":
            yield "tool", None, {"content": _text(block.get("content"))}


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content)
    return "" if content is None else str(content)
