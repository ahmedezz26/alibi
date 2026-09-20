"""Alibi as an MCP server: one tool that runs V2 (Jev) on a long agent trace.

Run: ``alibi-mcp`` (stdio). Needs ALIBI_JUDGE_BACKEND=typesafe, TYPESAFE_API_KEY and
ALIBI_ALLOW_PAID_MODELS=1 for traces above the length gate (Jev is a paid API).
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable
from typing import Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from alibi.config import load_settings
from alibi.judges import Judge
from alibi.localize import (
    AnalysisFailed,
    Diagnosis,
    JudgeUnavailable,
    SettingsError,
    build_judge,
    diagnose,
)
from alibi.sources.auto import TraceFormatError, load_steps

Source = Literal["auto", "json", "claude-code", "langsmith"]


def build_server(judge_factory: Callable[[], Judge] | None = None) -> MCPServer:
    """The server. ``judge_factory`` lets a host wire up its own Jev client; it still goes
    through ``build_judge``, so the backend check and the spend guard apply to it.

    Return a judge per call, or one that is not concurrently in another diagnosis: the cost
    reported to the client is this trace's share of the judge's own call log. A judge that
    keeps no ``calls`` log is fine, and its cost is reported as null rather than as zero.
    """
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
        except (TraceFormatError, OSError) as e:  # TraceFormatError is a ValueError
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
        except (JudgeUnavailable, SettingsError) as e:  # nothing built: nothing sent or spent
            raise ToolError(str(e)) from e
        except AnalysisFailed as e:
            # Said plainly, because unlike every other error here this one can cost money.
            # The cause's own message is left out: it comes from the judge's client, which
            # may quote the chapter it was given, and this string goes to a model that will
            # relay it. It goes to the server's stderr instead, which stays on this machine.
            # The traceback, not just the repr: a bug in Alibi must not be indistinguishable
            # from a Jev outage. stderr is the server's own, so it stays on this machine.
            traceback.print_exception(e.cause, file=sys.stderr)
            raise ToolError(
                f"{e.surface_message(cause_detail=False)} The cause is on the alibi-mcp "
                "server's stderr, which stays on this machine."
            ) from e

    return server


def main() -> None:
    build_server().run("stdio")
