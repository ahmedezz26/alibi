---
description: Find where a long AI-agent run went wrong. Use when the user asks why an agent run, a LangSmith trace, or a Claude Code session failed, or which step caused a failure, and the trace is long (tens of thousands of tokens or more).
---

# Diagnose a long agent trace with Alibi

1. Identify the trace: a `.json` trace file, a LangSmith trace id, or a Claude Code
   session file (`~/.claude/projects/<project>/<session-id>.jsonl`; newest file = latest
   session).
2. Tell the user how large the trace is and what analysing it will cost before you call
   anything: Jev bills input tokens and each one is read about twice, which is roughly
   $0.10 per million trace tokens (about $0.01 for a 100K-token trace).
3. Call the `diagnose_trace` tool of the `alibi` MCP server with that path or id.
4. If the result is `gated`, it was not analysed and nothing was spent. Either the trace is
   short enough to read directly, or it is over the cost ceiling (default 250K tokens): the
   message says which. Read it yourself, or tell the user they can raise
   `ALIBI_MAX_TRACE_TOKENS` if they want it analysed anyway.
5. Otherwise, open the 3 suspect steps in order, with a few steps of context around each,
   and explain which one most plausibly caused the failure and why. Present them as
   "read these first", not as a certain verdict: exact-step accuracy is modest (see the
   project README).
6. Report the cost and time the result records, against the estimate you gave.

Requirements: `TYPESAFE_API_KEY` set in the environment (Jev, TypeSafe AI). Traces are sent
to TypeSafe's API for analysis.
