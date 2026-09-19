"""V7 router: decide what kind of work a trace is before localizing (plain code, no model).

Looks only at the start of the trace (like a product would, from the first steps):
- ``code``: a real share of the agent's actions are shell commands or file edits;
- ``conversation``: the agent asks a person questions and gets replies (customer service);
- ``other``: anything else (agents planning among themselves, web tasks); plain V2.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from alibi.types import Step

TraceKind = Literal["code", "conversation", "other"]

HEAD_STEPS = 60  # decide from the start of the run
CODE_SHARE = 0.2  # share of agent steps that run commands or edit files
MIN_CUSTOMER_TURNS = 2

_CODE_ACTION = re.compile(
    r"```(?:bash|sh|python|str_replace|create|insert|view)\b|execute_bash|str_replace_editor"
    r"|\bgit (?:diff|apply|checkout|status)\b|\bpytest\b|\bsed -i\b"
)
_TOOL_CALL = re.compile(r"tool_calls|Tool calls?:|```|\w+\(\{")
_TOOL_RESULT = re.compile(r"^\s*(?:Tool result|\{|\[|Here's|Command |Observation)")
_QUESTION = re.compile(r"[?？]")


def _text(step: Step) -> str:
    return str(step.inputs.get("content", step.inputs))


def _is_agent(step: Step) -> bool:
    return step.type == "assistant"


def trace_kind(steps: Sequence[Step]) -> TraceKind:
    head = list(steps[:HEAD_STEPS])
    agent = [s for s in head if _is_agent(s)]
    if agent and sum(bool(_CODE_ACTION.search(_text(s))) for s in agent) / len(agent) >= CODE_SHARE:
        return "code"
    turns = 0
    for prev, cur in zip(head, head[1:], strict=False):
        asks = _is_agent(prev) and not _TOOL_CALL.search(_text(prev))
        asks = asks and bool(_QUESTION.search(_text(prev)))
        reply = cur.type in ("user", "tool") and not _TOOL_RESULT.search(_text(cur))
        turns += asks and reply
    return "conversation" if turns >= MIN_CUSTOMER_TURNS else "other"
