import json

import pytest
from mcp import Client

from alibi.mcp_server import build_server


class StepSevenJudge:
    """Health jumps at the last chapter; step 7 is the likeliest cause."""

    calls = ()

    def evaluate(self, state, questions):
        out = {}
        for q in questions:
            if q.answer_type == "score":
                out[q.id] = 1.0 if "[step 8]" in state else 0.0
            elif q.answer_type == "choice":
                first = next(iter(q.criteria))
                out[q.id] = {"choice": first, "probabilities": {first: 1.0}, "confidence": 1.0}
            elif q.id == "s7":
                out[q.id] = 0.9
            else:
                out[q.id] = 0.1
        return out


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


@pytest.mark.anyio
async def test_a_missing_key_says_so_instead_of_a_generic_error(tmp_path, monkeypatch):
    """The likeliest first run: the plugin passes ${TYPESAFE_API_KEY} through unset."""
    monkeypatch.setenv("ALIBI_JUDGE_BACKEND", "typesafe")
    monkeypatch.setenv("ALIBI_MIN_TRACE_TOKENS", "0")

    async with Client(build_server()) as client:
        result = await client.call_tool("diagnose_trace", {"trace": trace_file(tmp_path)})
    assert result.is_error
    assert "TYPESAFE_API_KEY" in result.content[0].text
    assert "Nothing was sent" in result.content[0].text


@pytest.mark.anyio
async def test_an_unreadable_trace_says_what_is_wrong(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    async with Client(build_server()) as client:
        result = await client.call_tool("diagnose_trace", {"trace": str(bad)})
    assert result.is_error
    assert "could not read" in result.content[0].text


@pytest.mark.anyio
async def test_the_source_argument_is_constrained_in_the_schema():
    async with Client(build_server()) as client:
        tools = await client.list_tools()
    source = tools.tools[0].input_schema["properties"]["source"]
    assert set(source.get("enum", [])) == {"auto", "json", "claude-code", "langsmith"}


@pytest.mark.anyio
async def test_a_long_trace_is_analysed_with_the_injected_judge(tmp_path, monkeypatch):
    """The success path: nothing else in the suite proves the factory's judge is used."""
    monkeypatch.setenv("ALIBI_JUDGE_BACKEND", "typesafe")
    monkeypatch.setenv("ALIBI_MIN_TRACE_TOKENS", "0")
    monkeypatch.setenv("ALIBI_JUDGE_WINDOW_TOKENS", "300")

    kinds = ["user"] + ["assistant", "tool"] * 9
    trace = tmp_path / "t.json"
    trace.write_text(
        json.dumps([{"type": kinds[i], "inputs": {"content": "x" * 400}} for i in range(9)])
    )
    async with Client(build_server(judge_factory=StepSevenJudge)) as client:
        result = await client.call_tool("diagnose_trace", {"trace": str(trace)})
    assert not result.is_error
    assert result.structured_content["gated"] is False
    assert result.structured_content["suspects"][0]["step"] == 7
