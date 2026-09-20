import json

import pytest
from mcp import Client

from alibi.mcp_server import build_server


def trace_file(tmp_path, n=3):
    p = tmp_path / "t.json"
    p.write_text(json.dumps([{"type": "user", "inputs": {"q": "hi"}}] * n))
    return str(p)


@pytest.mark.anyio
async def test_lists_one_diagnose_tool():
    async with Client(build_server()) as client:
        tools = await client.list_tools()
    assert [t.name for t in tools.tools] == ["diagnose_trace"]


@pytest.mark.anyio
async def test_trace_over_the_cost_ceiling_never_builds_a_judge(tmp_path, monkeypatch):
    """Over the ceiling the answer is free, so it must not need a key or a backend."""
    monkeypatch.setenv("ALIBI_MIN_TRACE_TOKENS", "0")
    monkeypatch.setenv("ALIBI_MAX_TRACE_TOKENS", "1")

    def factory():
        raise AssertionError("no judge above the cost ceiling")

    async with Client(build_server(judge_factory=factory)) as client:
        result = await client.call_tool("diagnose_trace", {"trace": trace_file(tmp_path)})
    assert result.structured_content["gated"] is True
    assert "ceiling" in result.structured_content["message"]


@pytest.mark.anyio
async def test_short_trace_is_gated_and_never_builds_a_judge(tmp_path):
    def factory():
        raise AssertionError("no judge for gated traces")

    async with Client(build_server(judge_factory=factory)) as client:
        result = await client.call_tool("diagnose_trace", {"trace": trace_file(tmp_path)})
    assert result.structured_content["gated"] is True
    assert result.structured_content["suspects"] == []


@pytest.mark.anyio
async def test_a_long_trace_refuses_a_non_jev_backend(tmp_path, monkeypatch):
    """Without this guard a private trace would be POSTed to a free OpenRouter endpoint."""
    monkeypatch.setenv("ALIBI_JUDGE_BACKEND", "openrouter")
    monkeypatch.setenv("ALIBI_MIN_TRACE_TOKENS", "0")

    def factory():
        raise AssertionError("no judge may be built for the wrong backend")

    async with Client(build_server(judge_factory=factory)) as client:
        result = await client.call_tool("diagnose_trace", {"trace": trace_file(tmp_path)})
    assert result.is_error
    assert "ALIBI_JUDGE_BACKEND=typesafe" in result.content[0].text
    assert "Nothing was sent" in result.content[0].text
