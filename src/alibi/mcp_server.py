"""Alibi as an MCP server: one tool that runs V2 (Jev) on a long agent trace.

Run: ``alibi-mcp`` (stdio). Needs ALIBI_JUDGE_BACKEND=typesafe, TYPESAFE_API_KEY and
ALIBI_ALLOW_PAID_MODELS=1 for traces above the length gate (Jev is a paid API).
"""

from __future__ import annotations

from collections.abc import Callable

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from alibi.config import load_settings
from alibi.judges import Judge, make_judge
from alibi.localize import Diagnosis, diagnose, needs_judge
from alibi.sources.auto import load_steps


def build_server(judge_factory: Callable[[], Judge] | None = None) -> MCPServer:
    server = MCPServer("alibi")

    @server.tool()
    def diagnose_trace(trace: str, source: str = "auto", project: str | None = None) -> Diagnosis:
        """Find where a long AI-agent run went wrong. Reads the trace chapter by chapter
        (never all at once), raises a CUSUM alarm, looks back, and returns the 3 steps to
        read first. `trace`: path to a .json trace or a Claude Code session .jsonl, or a
        LangSmith trace id. A trace under the length gate (default 50K tokens) or over the
        cost ceiling (default 250K tokens) is returned with `gated` true and a message
        saying which: it is not analysed and nothing is spent."""
        settings = load_settings()
        try:
            steps = load_steps(trace, source, project)
        except FileNotFoundError as e:
            raise ToolError(str(e)) from e
        judge = None
        if needs_judge(steps, settings):
            # Jev is the only supported sensor. Without this the trace would be sent to
            # whatever backend the environment names, including free endpoints that may log.
            if settings.judge_backend != "typesafe":
                raise ToolError(
                    "Alibi needs the Jev backend: set ALIBI_JUDGE_BACKEND=typesafe and "
                    "TYPESAFE_API_KEY. Nothing was sent."
                )
            judge = (judge_factory or (lambda: make_judge(settings)))()
        return diagnose(steps, judge, settings)

    return server


def main() -> None:
    build_server().run("stdio")
