"""Alibi as an MCP server: one tool that runs V2 (Jev) on a long agent trace.

Run: ``alibi-mcp`` (stdio). Needs ALIBI_JUDGE_BACKEND=typesafe, TYPESAFE_API_KEY and
ALIBI_ALLOW_PAID_MODELS=1 for traces above the length gate (Jev is a paid API).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from alibi.config import load_settings
from alibi.judges import Judge
from alibi.localize import (
    Diagnosis,
    JudgeUnavailable,
    build_judge,
    diagnose,
)
from alibi.sources.auto import TraceFormatError, load_steps

Source = Literal["auto", "json", "claude-code", "langsmith"]


def build_server(judge_factory: Callable[[], Judge] | None = None) -> MCPServer:
    server = MCPServer("alibi")

    @server.tool()
    def diagnose_trace(
        trace: str, source: Source = "auto", project: str | None = None
    ) -> Diagnosis:
        """Find where a long AI-agent run went wrong. Reads the trace chapter by chapter
        (never all at once), raises a CUSUM alarm, looks back, and returns the 3 steps to
        read first. `trace`: path to a .json trace or a Claude Code session .jsonl, or a
        LangSmith trace id. A trace under the length gate (default 50K tokens) or over the
        cost ceiling (default 250K tokens) is returned with `gated` true and a message
        saying which: it is not analysed and nothing is spent."""
        settings = load_settings()
        try:
            steps = load_steps(trace, source, project)
        except (FileNotFoundError, TraceFormatError, OSError) as e:
            raise ToolError(str(e)) from e

        try:
            # The factory runs only if this trace is analysed, so a refusal needs no key.
            # build_judge checks the backend first: without it the trace would go to whatever
            # backend the environment names, including free endpoints that may log prompts.
            return diagnose(
                steps,
                settings=settings,
                judge_factory=lambda: build_judge(settings, judge_factory),
            )
        except JudgeUnavailable as e:
            raise ToolError(str(e)) from e

    return server


def main() -> None:
    build_server().run("stdio")
