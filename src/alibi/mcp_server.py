"""Alibi as an MCP server: one tool that runs V2 (Jev) on a long agent trace.

Run: ``alibi-mcp`` (stdio). Needs ALIBI_JUDGE_BACKEND=typesafe, TYPESAFE_API_KEY and
ALIBI_ALLOW_PAID_MODELS=1 for traces above the length gate (Jev is a paid API).
"""

from __future__ import annotations

from collections.abc import Callable

from mcp.server import MCPServer

from alibi.chunking import trace_tokens
from alibi.config import load_settings
from alibi.judges import Judge, make_judge
from alibi.localize import Diagnosis, diagnose
from alibi.sources.auto import load_steps


def build_server(judge_factory: Callable[[], Judge] | None = None) -> MCPServer:
    server = MCPServer("alibi")

    @server.tool()
    def diagnose_trace(trace: str, source: str = "auto", project: str | None = None) -> Diagnosis:
        """Find where a long AI-agent run went wrong. Reads the trace chapter by chapter
        (never all at once), raises a CUSUM alarm, looks back, and returns the 3 steps to
        read first. `trace`: path to a .json trace or a Claude Code session .jsonl, or a
        LangSmith trace id. Traces under the length gate (default 50K tokens) are not
        analysed: a direct read is enough."""
        settings = load_settings()
        steps = load_steps(trace, source, project)
        judge = None
        # Only between the gates is a judge needed; outside them diagnose() answers for free.
        if settings.min_trace_tokens <= trace_tokens(steps) <= settings.max_trace_tokens:
            judge = (judge_factory or (lambda: make_judge(settings)))()
        return diagnose(steps, judge, settings)

    return server


def main() -> None:
    build_server().run("stdio")
