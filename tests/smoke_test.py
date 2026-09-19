"""Smoke test for a built distribution: the package imports and gates a short trace.

Run against a wheel or sdist without the project checkout, so it catches packaging mistakes
the test suite cannot see (a missing module, a wrong ``module-name``, a broken entry point):

    uv run --isolated --no-project --with dist/*.whl tests/smoke_test.py
"""

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from alibi.config import load_settings
from alibi.localize import diagnose
from alibi.mcp_server import build_server
from alibi.sources.auto import load_steps


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        trace = Path(tmp) / "t.json"
        trace.write_text(json.dumps([{"type": "user", "inputs": {"content": "hi"}}] * 3))
        steps = load_steps(str(trace))
        d = diagnose(steps, judge=None, settings=load_settings())
        assert d.gated, "a 3-step trace must be gated"
        assert d.judge_calls == 0, "a gated trace must not call the judge"

        out = subprocess.run(
            [sys.executable, "-m", "alibi.cli", "diagnose", str(trace)],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "direct read" in out.stdout, out.stdout

    tools = asyncio.run(build_server().list_tools())
    assert [t.name for t in tools] == ["diagnose_trace"], tools

    print("smoke test passed")


if __name__ == "__main__":
    main()
