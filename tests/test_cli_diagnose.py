import json
import subprocess
import sys

import pytest
from fakes import FailingAfter, step_dicts

from alibi import cli, localize


def trace_file(tmp_path, n=3):
    p = tmp_path / "t.json"
    p.write_text(json.dumps([{"type": "user", "inputs": {"q": "hi"}}] * n))
    return str(p)


def no_judge_allowed(monkeypatch, reason):
    """Let every policy check pass, then fail if a judge is actually built.

    The patch must be on ``localize``: that is where ``build_judge`` looks ``make_judge`` up.
    Patching ``cli`` intercepts nothing, so the guard it certifies would be imaginary.
    """
    monkeypatch.setenv("ALIBI_JUDGE_BACKEND", "typesafe")
    monkeypatch.setenv("ALIBI_ALLOW_PAID_MODELS", "1")
    monkeypatch.setenv("TYPESAFE_API_KEY", "not-used")

    def boom(*_args, **_kwargs):
        raise AssertionError(reason)

    monkeypatch.setattr(localize, "make_judge", boom)


def test_short_trace_prints_the_gate_message_without_a_judge(tmp_path, capsys, monkeypatch):
    no_judge_allowed(monkeypatch, "a gated trace must not build a judge")
    assert cli.main(["diagnose", trace_file(tmp_path)]) == 0
    assert "direct read" in capsys.readouterr().out


def test_json_output_is_a_diagnosis(tmp_path, capsys):
    assert cli.main(["diagnose", trace_file(tmp_path), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["gated"] is True and out["suspects"] == []


def test_max_tokens_overrides_the_cost_ceiling(tmp_path, capsys):
    """The ceiling is overridable from the command line, not only through the environment."""
    trace = trace_file(tmp_path, n=3)
    assert cli.main(["diagnose", trace, "--min-trace-tokens", "0", "--max-trace-tokens", "1"]) == 0
    out = capsys.readouterr().out
    assert "ceiling" in out and "ALIBI_MAX_TRACE_TOKENS" in out


def test_a_missing_trace_file_is_a_clean_error(tmp_path, capsys):
    assert cli.main(["diagnose", str(tmp_path / "gone.jsonl")]) == 2
    assert "no such trace file" in capsys.readouterr().err


def test_an_empty_trace_needs_no_judge(tmp_path, capsys, monkeypatch):
    """diagnose() answers an empty trace for free, so the CLI must not demand a backend."""
    no_judge_allowed(monkeypatch, "an empty trace must not build a judge")
    p = tmp_path / "empty.json"
    p.write_text("[]")
    assert cli.main(["diagnose", str(p), "--min-trace-tokens", "0"]) == 0
    assert "no steps" in capsys.readouterr().out


def test_a_malformed_trace_file_is_a_clean_error(tmp_path, capsys):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    assert cli.main(["diagnose", str(p)]) == 2
    assert "could not read" in capsys.readouterr().err


def test_the_flag_spellings_from_0_1_2_still_work(tmp_path, capsys):
    """--min-tokens/--max-tokens shipped in 0.1.2 and 0.1.3 (as aliases); scripts using them
    must keep working, even though the same spelling means a window size on `score`."""
    trace = trace_file(tmp_path, n=3)
    assert cli.main(["diagnose", trace, "--min-tokens", "0", "--max-tokens", "1"]) == 0
    assert "ceiling" in capsys.readouterr().out


def test_the_backend_message_names_every_variable_it_needs(tmp_path, capsys, monkeypatch):
    """Following the message must be enough: the paid-model guard is the third variable."""
    monkeypatch.setenv("ALIBI_JUDGE_BACKEND", "openrouter")
    assert cli.main(["diagnose", trace_file(tmp_path), "--min-trace-tokens", "0"]) == 2
    err = capsys.readouterr().err
    for var in ("ALIBI_JUDGE_BACKEND=typesafe", "TYPESAFE_API_KEY", "ALIBI_ALLOW_PAID_MODELS=1"):
        assert var in err


def test_a_missing_key_is_a_clean_error(tmp_path, capsys, monkeypatch):
    """The plugin's own configuration: the spend is allowed, the key is passed through unset."""
    monkeypatch.setenv("ALIBI_JUDGE_BACKEND", "typesafe")
    monkeypatch.setenv("ALIBI_ALLOW_PAID_MODELS", "1")
    assert cli.main(["diagnose", trace_file(tmp_path), "--min-trace-tokens", "0"]) == 2
    assert "TYPESAFE_API_KEY" in capsys.readouterr().err


def test_a_long_trace_refuses_to_spend_without_the_paid_guard(tmp_path, capsys, monkeypatch):
    """Jev has no free tier, so analysing bills: without the guard nothing may be built."""
    monkeypatch.setenv("ALIBI_JUDGE_BACKEND", "typesafe")
    monkeypatch.setenv("TYPESAFE_API_KEY", "would-have-worked")
    assert cli.main(["diagnose", trace_file(tmp_path), "--min-trace-tokens", "0"]) == 2
    err = capsys.readouterr().err
    assert "ALIBI_ALLOW_PAID_MODELS=1" in err and "Nothing was sent" in err
    # This check runs before the key check, so following it must not hit a second failure.
    assert "TYPESAFE_API_KEY" in err


def test_a_judge_failing_mid_analysis_is_not_reported_as_a_free_refusal(
    tmp_path, capsys, monkeypatch
):
    """Exit 2 means nothing was spent. A judge that dies part-way through has already sent
    the trace and billed some of it, so it gets its own code and the real count."""
    monkeypatch.setenv("ALIBI_JUDGE_BACKEND", "typesafe")
    monkeypatch.setenv("ALIBI_ALLOW_PAID_MODELS", "1")
    monkeypatch.setenv("ALIBI_JUDGE_WINDOW_TOKENS", "300")
    monkeypatch.setattr(localize, "make_judge", lambda *a, **k: FailingAfter(2))
    trace = tmp_path / "long.json"
    trace.write_text(json.dumps(step_dicts()))
    assert cli.main(["diagnose", str(trace), "--min-trace-tokens", "0"]) == 3
    err = capsys.readouterr().err
    assert "2 Jev call(s) completed and are billed" in err
    assert "connection reset" in err


def test_a_bad_window_size_is_a_free_error_not_a_spend_report(tmp_path, capsys, monkeypatch):
    """A zero window makes `chunk_trace` refuse. That happens before the judge exists, so it
    is exit 2 and must not tell the user their calls were billed."""
    no_judge_allowed(monkeypatch, "a settings error must not build a judge")
    monkeypatch.setenv("ALIBI_JUDGE_WINDOW_TOKENS", "0")
    assert cli.main(["diagnose", trace_file(tmp_path), "--min-trace-tokens", "0"]) == 2
    err = capsys.readouterr().err
    assert "ALIBI_JUDGE_WINDOW_TOKENS" in err  # the setting the user can actually change
    assert "billed" not in err


def test_option_abbreviations_still_work(tmp_path, capsys):
    """argparse abbreviations worked on `diagnose` in 0.1.3 and work on every other
    subcommand; only the two gate flags are ambiguous, each keeping its older spelling."""
    assert cli.main(["diagnose", trace_file(tmp_path), "--js"]) == 0
    assert json.loads(capsys.readouterr().out)["gated"] is True
    with pytest.raises(SystemExit):
        cli.main(["diagnose", trace_file(tmp_path), "--min-t", "0"])


def test_a_bad_timestamp_is_a_clean_error(tmp_path, capsys):
    p = tmp_path / "ts.json"
    p.write_text(json.dumps([{"type": "user", "timestamp": "not-a-date", "inputs": {}}]))
    assert cli.main(["diagnose", str(p)]) == 2
    assert "could not read" in capsys.readouterr().err


def test_module_entry_point_runs(tmp_path):
    """`python -m alibi.cli` must define every handler before main() dispatches."""
    out = subprocess.run(
        [sys.executable, "-m", "alibi.cli", "diagnose", trace_file(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    assert "direct read" in out.stdout


def test_long_trace_needs_the_jev_backend(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("ALIBI_JUDGE_BACKEND", "openrouter")
    assert cli.main(["diagnose", trace_file(tmp_path), "--min-trace-tokens", "0"]) == 2
    assert "ALIBI_JUDGE_BACKEND=typesafe" in capsys.readouterr().err


def test_a_mid_analysis_failure_prints_a_traceback_for_the_bug_report(
    tmp_path, capsys, monkeypatch
):
    """A bug in Alibi must not be indistinguishable from a Jev outage; the cost stays last."""
    monkeypatch.setenv("ALIBI_JUDGE_BACKEND", "typesafe")
    monkeypatch.setenv("ALIBI_ALLOW_PAID_MODELS", "1")
    monkeypatch.setenv("ALIBI_JUDGE_WINDOW_TOKENS", "300")
    monkeypatch.setattr(localize, "make_judge", lambda *a, **k: FailingAfter(2))
    trace = tmp_path / "long.json"
    trace.write_text(json.dumps(step_dicts()))
    assert cli.main(["diagnose", str(trace), "--min-trace-tokens", "0"]) == 3
    err = capsys.readouterr().err
    assert "Traceback" in err and "connection reset" in err
    assert err.strip().splitlines()[-1].endswith("completed and are billed.")


def test_a_window_too_small_to_group_steps_is_a_free_error(tmp_path, capsys, monkeypatch):
    no_judge_allowed(monkeypatch, "a degenerate window must not build a judge")
    monkeypatch.setenv("ALIBI_JUDGE_WINDOW_TOKENS", "10")
    trace = tmp_path / "long.json"
    trace.write_text(json.dumps(step_dicts()))
    assert cli.main(["diagnose", str(trace), "--min-trace-tokens", "0"]) == 2
    err = capsys.readouterr().err
    assert "ALIBI_JUDGE_WINDOW_TOKENS" in err and "billed" not in err
