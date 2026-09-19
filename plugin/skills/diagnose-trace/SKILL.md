---
description: Find where a long AI-agent run went wrong. Use when the user asks why an agent run, a LangSmith trace, or a Claude Code session failed, or which step caused a failure, and the trace is long (tens of thousands of tokens or more).
---

# Diagnose a long agent trace with Alibi

1. Identify the trace: a `.json` trace file, a LangSmith trace id, or a Claude Code
   session file (`~/.claude/projects/<project>/<session-id>.jsonl`; newest file = latest
   session).
2. Call the `diagnose_trace` tool of the `alibi` MCP server with that path or id.
3. If the result is `gated`, tell the user the trace is short enough to read directly, and
   read it yourself.
4. Otherwise, open the 3 suspect steps in order, with a few steps of context around each,
   and explain which one most plausibly caused the failure and why. Present them as
   "read these first", not as a certain verdict: exact-step accuracy is modest (see the
   project README).
5. Mention the cost and time reported in the result.

Requirements: `TYPESAFE_API_KEY` set in the environment (Jev, TypeSafe AI). Traces are sent
to TypeSafe's API for analysis.
