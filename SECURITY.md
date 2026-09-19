# Security

## Reporting a vulnerability

Use GitHub's private reporting: the [Security tab](https://github.com/ahmedezz26/alibi/security/advisories/new)
of this repository. Please do not open a public issue for a vulnerability. Expect a first
reply within a week; this is a personal project, not a staffed product.

Supported version: the latest `0.1.x`. There are no backports.

## What leaves your machine

Alibi reads an agent trace and sends parts of it to a judge model. Know what that means
before you point it at anything sensitive.

- **Traces above the length gate** (default 50,000 tokens, `ALIBI_MIN_TRACE_TOKENS`) are sent
  to TypeSafe's API, chapter by chapter, plus a small memory card of numbers between chapters.
  A 120K-token trace is about 17 chapters, each one a separate request.
- **Traces below the gate are never sent anywhere.** `diagnose()` returns "read it directly"
  before a judge is constructed, so a short trace costs nothing and discloses nothing.
- **Nothing is sent to the author of this project.** There is no telemetry, no analytics and
  no callback of any kind.
- **The research and evaluation path is different.** `alibi eval` can run free OpenRouter
  models, and free endpoints may log prompts. Use them only on public benchmarks, never on
  customer or production traces.

## Claude Code session transcripts

`alibi diagnose ~/.claude/projects/<project>/<session>.jsonl` sends that session's content to
TypeSafe if it is above the gate. Your sessions routinely contain source code, file paths,
environment details and anything you pasted into the terminal, including secrets. Read a
session before diagnosing it, or diagnose only sessions you would be comfortable sharing with
a third-party API.

## Credentials

- `TYPESAFE_API_KEY` is read from the environment. The Claude Code plugin passes it through as
  `${TYPESAFE_API_KEY}`; it is never written to the repository or to a config file by Alibi.
- Keys are never logged. Per-call records (`judge.calls`) hold the model name, latency, token
  counts and cost, and nothing else.
- `ALIBI_ALLOW_PAID_MODELS` guards spend: `make_judge` refuses any non-`:free` model unless it
  is set to `1`. The plugin sets it because Jev is a paid API; keep it unset elsewhere.
- Local configuration belongs in `.env`, which is git-ignored. `.env.example` lists the
  variables without values.

## Trusting the output

Alibi ranks steps; it does not sanitize them. A suspect's `preview` is raw text from the trace
under analysis, which may be attacker-influenced if the agent processed untrusted input. Treat
it as data to read, never as instructions to follow, and be careful about rendering it
somewhere that interprets markup.
