import json

import pytest
from fakes import FailingAfter, TypedJudge, step_dicts
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


@pytest.mark.anyio
async def test_a_missing_key_says_so_instead_of_a_generic_error(tmp_path, monkeypatch):
    """The likeliest first run: the plugin allows the spend and passes ${TYPESAFE_API_KEY}
    through unset."""
    monkeypatch.setenv("ALIBI_JUDGE_BACKEND", "typesafe")
    monkeypatch.setenv("ALIBI_ALLOW_PAID_MODELS", "1")
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


def long_trace(tmp_path, monkeypatch):
    """A trace above the length gate, with chapters small enough to need several."""
    monkeypatch.setenv("ALIBI_JUDGE_BACKEND", "typesafe")
    monkeypatch.setenv("ALIBI_ALLOW_PAID_MODELS", "1")
    monkeypatch.setenv("ALIBI_MIN_TRACE_TOKENS", "0")
    monkeypatch.setenv("ALIBI_JUDGE_WINDOW_TOKENS", "300")
    trace = tmp_path / "t.json"
    trace.write_text(json.dumps(step_dicts()))
    return str(trace)


@pytest.mark.anyio
async def test_a_long_trace_is_analysed_with_the_injected_judge(tmp_path, monkeypatch):
    """The success path: nothing else in the suite proves the factory's judge is used, or
    that its call log reaches the Diagnosis the client reads."""
    trace = long_trace(tmp_path, monkeypatch)
    async with Client(build_server(judge_factory=TypedJudge)) as client:
        result = await client.call_tool("diagnose_trace", {"trace": trace})
    assert not result.is_error
    out = result.structured_content
    assert out["gated"] is False
    assert out["suspects"][0]["step"] == 7
    assert out["judge_calls"] > 0
    assert out["cost_usd"] == pytest.approx(0.002 * out["judge_calls"])
    assert out["judge_seconds"] == pytest.approx(0.5 * out["judge_calls"], abs=0.05)


@pytest.mark.anyio
async def test_an_injected_judge_cannot_sidestep_the_spend_guard(tmp_path, monkeypatch):
    """A host wiring up its own Jev client still spends the user's money, so the guard is
    checked before the factory runs, not inside whatever it builds."""
    trace = long_trace(tmp_path, monkeypatch)
    monkeypatch.delenv("ALIBI_ALLOW_PAID_MODELS")
    async with Client(build_server(judge_factory=TypedJudge)) as client:
        result = await client.call_tool("diagnose_trace", {"trace": trace})
    assert result.is_error
    assert "ALIBI_ALLOW_PAID_MODELS=1" in result.content[0].text
    assert "Nothing was sent" in result.content[0].text


@pytest.mark.anyio
async def test_a_judge_failing_mid_analysis_says_what_was_billed(tmp_path, monkeypatch):
    """Every other error here is free. This one is not, and the client - a model that will
    relay it - is told exactly how much of it was not."""
    trace = long_trace(tmp_path, monkeypatch)
    async with Client(build_server(judge_factory=lambda: FailingAfter(2))) as client:
        result = await client.call_tool("diagnose_trace", {"trace": trace})
    assert result.is_error
    assert "2 Jev call(s) completed and are billed" in result.content[0].text


@pytest.mark.anyio
async def test_the_cause_of_a_failure_is_not_relayed_to_the_client(tmp_path, monkeypatch):
    """The judge client's own message may quote the chapter it was given, and this string
    goes to a model. The kind of failure is named; the trace's content is not."""
    trace = long_trace(tmp_path, monkeypatch)

    class Leaky:
        calls = ()

        def evaluate(self, state, questions):
            raise RuntimeError(f"400 rejected: input_value={state[:120]!r}")

    async with Client(build_server(judge_factory=Leaky)) as client:
        result = await client.call_tool("diagnose_trace", {"trace": trace})
    assert result.is_error
    text = result.content[0].text
    assert "xxxx" not in text and "input_value" not in text
    assert "RuntimeError" in text  # what kind of failure it was still gets through


@pytest.mark.anyio
async def test_a_bad_window_size_is_not_reported_as_a_spend(tmp_path, monkeypatch):
    """`chunk_trace` refuses a zero window before the judge is built: free, and said so."""
    trace = long_trace(tmp_path, monkeypatch)
    monkeypatch.setenv("ALIBI_JUDGE_WINDOW_TOKENS", "0")

    def factory():
        raise AssertionError("a settings error must not build a judge")

    async with Client(build_server(judge_factory=factory)) as client:
        result = await client.call_tool("diagnose_trace", {"trace": trace})
    assert result.is_error
    assert "ALIBI_JUDGE_WINDOW_TOKENS" in result.content[0].text
    assert "billed" not in result.content[0].text
