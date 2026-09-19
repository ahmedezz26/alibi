# Contributing

Thanks for looking. This is a small, opinionated project: it packages one method that earned
its place in a series of pre-registered tests, and the tests are the reason to trust it. The
rules below exist to keep that true.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                       # install, including the dev group
uv run pytest                                 # the whole suite, offline, ~1 s
uv run ruff check . && uv run ruff format .   # lint and format
uv run alibi diagnose examples/sample_trace.json
```

The suite never touches the network: every judge in the tests is a fake. Keep it that way.

### Working on the MCP server or the plugin

The published plugin launches `uvx --from 'agent-alibi>=0.1,<0.2' alibi-mcp`, which downloads
the release from PyPI, so your edits will not be picked up. To exercise your checkout, point a
user-level `.mcp.json` at it:

```json
{ "mcpServers": { "alibi-dev": { "command": "uvx",
  "args": ["--from", "/path/to/your/clone", "alibi-mcp"],
  "env": { "TYPESAFE_API_KEY": "${TYPESAFE_API_KEY}", "ALIBI_JUDGE_BACKEND": "typesafe",
           "ALIBI_ALLOW_PAID_MODELS": "1" } } } }
```

`uv run alibi-mcp` also runs the server from the working tree, over stdio.

## The rules that matter

1. **No paid calls in tests, ever.** `make_judge` refuses non-`:free` models unless
   `ALIBI_ALLOW_PAID_MODELS=1`. No test may set it.
2. **Judge access goes through `alibi.judges.make_judge`.** `judges/__init__.py` is the only
   module that imports a concrete backend. Nothing else may import `OpenRouterJudge` or
   `TypeSafeJudge` directly.
3. **Trace ingestion goes through a `TraceSource` adapter** in `src/alibi/sources/`. No
   dataset-specific or LangSmith-specific code outside its adapter.
4. **Model IDs, window sizes and thresholds live in `src/alibi/config.py`** and nowhere else.
   No magic numbers in the pipeline.
5. **Tests first.** Write the failing test, watch it fail for the reason you expect, then make
   it pass. A test that passes against the unfixed code is not a test; the history of this repo
   has one commit that exists purely because a regression test failed that bar.
6. **No emojis** in the README, the plugin text or the skill.

## Changing the method

`localize.V2_CONFIG` is frozen: the gap card, the per-step checks, `step_plus_check`,
`earliest_near_best`, 10K chapters, CUSUM k = 0.2 and h = 0.5. Nine variants were built and
measured against it, and all nine lost; `docs/research-log.md` records each one, including
what it cost and why it was dropped.

So a change to the method is not a code review question, it is an evidence question. Before
the pull request:

- Write down the pass bar first, in the issue: which dataset, which split, what counts as a
  win, and how many traces.
- Measure on a development split. The held-out sets listed in `docs/` have each been used
  once; do not tune against them, and do not inspect their misses.
- Report exact-step and within-3 accuracy against the current V2 on the same traces, with a
  paired test.

Plumbing changes, new trace adapters, bug fixes, docs and tests need none of that. Only the
estimation layer is under the evidence rule.

## Pull requests

- One topic per pull request, with a description that says what it changes and how you know.
- `uv run pytest` and `uv run ruff check . && uv run ruff format --check .` must pass; CI runs
  exactly those.
- Commit messages: a short imperative subject, and a body explaining why when the reason is
  not obvious from the diff.

## Releasing

The distribution is `agent-alibi` on PyPI; the import package is `alibi`. A release is
reproducible from its tag, so the order matters:

1. Bump `version` in `pyproject.toml` **and** `plugin/.claude-plugin/plugin.json` (a test
   asserts they match).
2. For a new minor version, bump the `--from` pin in `plugin/.mcp.json` and the two constants
   in `tests/test_plugin_files.py`.
3. Append the release to `docs/research-log.md`.
4. Commit, then tag that commit: `git tag vX.Y.Z && git push origin main --tags`.
5. The tag triggers `.github/workflows/release.yml`: build, smoke-test the wheel and the sdist,
   publish through trusted publishing, then verify the published artifact installs.
6. `gh release create vX.Y.Z --notes ...`.

Do not publish from `workflow_dispatch` on an untagged commit: the artifact then cannot be
rebuilt from git, and the README that PyPI renders is frozen per release.

## Where things live

| Path | What it is |
|---|---|
| `src/alibi/localize.py` | `diagnose()`, the one entry point every surface calls |
| `src/alibi/{chunking,forward,drift,backward}.py` | the method: chapters, forward filter, CUSUM, look-back |
| `src/alibi/judges/` | the `Judge` protocol and its backends |
| `src/alibi/sources/` | the `TraceSource` protocol and one adapter per trace format |
| `src/alibi/{cli,mcp_server}.py` | the command line and the MCP server |
| `plugin/` | the Claude Code plugin |
| `docs/research-log.md` | every measurement, including the failures |
| `docs/PROJECT_BRIEF.md` | why the architecture looks like this |
