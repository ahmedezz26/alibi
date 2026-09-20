"""``alibi score``: fetch a trace, chunk it, and score each chunk with the judge."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from alibi.backward import backward_pass
from alibi.chunking import chunk_trace
from alibi.config import load_settings
from alibi.drift import DEFAULT_BASELINE, DEFAULT_H, DEFAULT_K, cusum_path, detect_drift
from alibi.forward import forward_pass
from alibi.judges import make_judge
from alibi.sources import TraceSource
from alibi.sources.auto import SOURCES


def _make_source(args: argparse.Namespace) -> TraceSource:
    if args.source == "langsmith":
        from alibi.sources.langsmith import LangSmithTraceSource

        return LangSmithTraceSource(project_name=args.project)
    from alibi.sources.jsonfile import JsonFileTraceSource

    return JsonFileTraceSource()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alibi")
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score", help="per-chunk forward scoring of one trace")
    score.add_argument("trace_id", help="LangSmith trace ID, or path to a JSON trace file")
    score.add_argument("--source", choices=["json", "langsmith"], default="json")
    score.add_argument("--project", help="LangSmith project name")
    score.add_argument("--max-tokens", type=int, default=28_000)
    score.add_argument("--overlap-tokens", type=int, default=2_000)
    score.add_argument(
        "--no-state", action="store_true", help="stateless per-chunk scoring (ablation)"
    )
    score.add_argument("--no-backward", action="store_true", help="skip the backward pass")
    score.add_argument("--cusum-k", type=float, default=DEFAULT_K)
    score.add_argument("--cusum-h", type=float, default=DEFAULT_H)
    score.add_argument("--cusum-baseline", type=float, default=DEFAULT_BASELINE)

    ev = sub.add_parser(
        "eval", aliases=["eval-agentrx"], help="localize annotated critical steps on a benchmark"
    )
    ev.add_argument(
        "--benchmark",
        choices=["agentrx", "agentracer", "trajerrbench", "longrca"],
        default="agentrx",
    )
    ev.add_argument("--data-dir", help="default: the benchmark's data directory")
    ev.add_argument(
        "--subset",
        default="all",
        help="agentrx: tau_retail|magentic_one|all; agentracer: agentic|coding|math|all; "
        "trajerrbench: tau2bench|swebenchpro|all; longrca: swe_bench_pro|terminal_bench_2|"
        "travelplanner|vitabench|webarena_verified|all",
    )
    ev.add_argument("--split", default="test", help="agentracer: train|test")
    ev.add_argument(
        "--ids",
        help="JSON file of trace ids (a list, or {'ids': [...]}); overrides other selection",
    )
    ev.add_argument("--min-tokens", type=int, help="only traces longer than this (approx tokens)")
    ev.add_argument("--sample", type=int, help="evenly spaced sample of N traces (pilot)")
    ev.add_argument("--limit", type=int, help="first N trajectories only (cost check)")
    ev.add_argument(
        "--mode",
        choices=["hybrid", "windowed", "whole"],
        default="hybrid",
        help="hybrid: one call if the trace fits the judge's reliable length, else windowed",
    )
    ev.add_argument("--max-tokens", type=int, help="window size (default: judge window tokens)")
    ev.add_argument("--overlap-tokens", type=int, help="window overlap (default: 1/8 of window)")
    ev.add_argument("--route-threshold", type=int, help="default: judge reliable tokens")
    ev.add_argument("--no-state", action="store_true")
    ev.add_argument(
        "--pipeline",
        choices=["text", "typed"],
        help="typed: System One judge (Jev) pipeline; default follows ALIBI_JUDGE_BACKEND",
    )
    ev.add_argument(
        "--combine-rule",
        default="chapter_x_step",
        help="typed pipeline: fixed rule from backward.COMBINE_RULES",
    )
    ev.add_argument("--card", choices=["symptom", "gap"], default="symptom")
    ev.add_argument(
        "--checks", action="store_true", help="typed: ask per-step quiet-mistake checks"
    )
    ev.add_argument(
        "--commit", action="store_true", help="typed (V7): ask a commit-point Noul per agent step"
    )
    ev.add_argument(
        "--reach-after", type=int, default=0, help="typed (V8): chapters after the alarm to check"
    )
    ev.add_argument("--pick", choices=["best", "earliest_near_best"], default="best")
    ev.add_argument("--workers", type=int, default=8)
    ev.add_argument("--out", required=True, help="JSONL results file (resumable)")

    fin = sub.add_parser(
        "finalists", help="V3: add the finalist round to saved typed-pipeline results"
    )
    fin.add_argument("--benchmark", choices=["agentrx", "agentracer", "trajerrbench", "longrca"])
    fin.add_argument("--data-dir")
    fin.add_argument("--subset", default="all")
    fin.add_argument("--split", default="test")
    fin.add_argument("--in", dest="inp", required=True, help="typed-pipeline JSONL results")
    fin.add_argument("--out", required=True, help="JSONL with finalist picks (resumable)")
    fin.add_argument("--k", type=int, default=8, help="shortlist size")
    fin.add_argument("--rule", default="step_plus_check", help="rule used to shortlist")
    fin.add_argument("--context", type=int, default=0, help="neighbour steps shown per side")
    fin.add_argument("--pick", choices=["best", "earliest_near_best"], default="best")
    fin.add_argument("--workers", type=int, default=8)

    dl = sub.add_parser("download-agentrx", help="fetch the gated dataset using HF_TOKEN")
    dl.add_argument("--data-dir", default="data/agentrx")

    diag = sub.add_parser(
        "diagnose", help="V2 + Jev: the 3 steps of a long agent trace to read first"
    )
    diag.add_argument("trace", help="path to a .json / Claude Code .jsonl trace, or a LangSmith id")
    diag.add_argument("--source", choices=list(SOURCES), default="auto")
    diag.add_argument("--project", help="LangSmith project name")
    # --min-tokens/--max-tokens are the 0.1.2 and 0.1.3 spellings, kept so existing scripts
    # keep working. They mean the window size on `score` and `eval`, which is why the
    # --*-trace-tokens names are the documented ones here. Having both makes "--min-t" an
    # ambiguous prefix; argparse says so.
    diag.add_argument(
        "--min-trace-tokens",
        "--min-tokens",
        type=int,
        help="override the length gate (ALIBI_MIN_TRACE_TOKENS)",
    )
    diag.add_argument(
        "--max-trace-tokens",
        "--max-tokens",
        type=int,
        help="override the cost ceiling (ALIBI_MAX_TRACE_TOKENS)",
    )
    diag.add_argument("--json", action="store_true", help="print the Diagnosis as JSON")

    args = parser.parse_args(argv)
    if args.command == "diagnose":
        return _diagnose(args)
    if args.command in ("eval", "eval-agentrx"):
        return _eval(args)
    if args.command == "download-agentrx":
        return _download_agentrx(args)
    if args.command == "finalists":
        return _finalists(args)

    steps = _make_source(args).get_trace(args.trace_id)
    chunks = chunk_trace(steps, args.max_tokens, args.overlap_tokens)
    judge = make_judge()
    scores = forward_pass(judge, chunks, propagate_state=not args.no_state)

    series = [s.score for s in scores]
    drift_args = {"k": args.cusum_k, "baseline": args.cusum_baseline}
    change_points = detect_drift(series, h=args.cusum_h, **drift_args)

    localization = None
    if not args.no_backward and scores:
        localization = backward_pass(judge, chunks, scores, [cp.alarm for cp in change_points])

    calls = getattr(judge, "calls", [])
    json.dump(
        {
            "trace_id": args.trace_id,
            "steps": len(steps),
            "chunks": [s.__dict__ for s in scores],
            "cusum": cusum_path(series, **drift_args).round(4).tolist(),
            "change_points": [cp.__dict__ for cp in change_points],
            "localization": (
                {
                    "failure_chunk": localization.failure_chunk,
                    "critical_step": localization.critical_step,
                    "precursors": [p.__dict__ for p in localization.precursors],
                }
                if localization
                else None
            ),
            "cost": sum(c.cost or 0 for c in calls),
            "latency_s": sum(c.latency_s for c in calls),
        },
        sys.stdout,
        indent=2,
    )
    print()
    return 0


def _diagnose(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from alibi.localize import (
        AnalysisFailed,
        JudgeUnavailable,
        SettingsError,
        build_judge,
        diagnose,
    )
    from alibi.sources.auto import TraceFormatError, load_steps

    settings = load_settings()
    if args.min_trace_tokens is not None:
        settings = replace(settings, min_trace_tokens=args.min_trace_tokens)
    if args.max_trace_tokens is not None:
        settings = replace(settings, max_trace_tokens=args.max_trace_tokens)
    try:
        steps = load_steps(args.trace, args.source, args.project)
    except (TraceFormatError, OSError) as e:  # TraceFormatError is a ValueError
        print(e, file=sys.stderr)
        return 2

    try:
        # The factory runs only if this trace is actually analysed, so a refusal needs no key.
        d = diagnose(steps, settings=settings, judge_factory=lambda: build_judge(settings))
    except (JudgeUnavailable, SettingsError) as e:
        # No judge was built, so nothing was sent and nothing was spent. SettingsError is a
        # setting the pipeline rejects before the judge exists, e.g. a zero window size; it
        # is its own type so an internal ValueError still surfaces as the bug it is.
        print(e, file=sys.stderr)
        return 2
    except AnalysisFailed as e:
        # Distinct from exit 2 on purpose: the trace was already being sent, so this is not
        # a free refusal and a script must not treat it as one. The traceback goes first so
        # a bug in Alibi is reportable rather than indistinguishable from a Jev outage, and
        # the cost is the last line, where a user looks.
        traceback.print_exception(e.cause, file=sys.stderr)
        print(e.surface_message(), file=sys.stderr)
        return 3
    if args.json:
        print(d.model_dump_json(indent=2))
        return 0
    print(d.message)
    if not d.gated:
        spend = (
            f"{d.judge_calls} Jev calls, {d.judge_seconds:.0f} s, ${d.cost_usd:.4f}"
            if d.judge_calls is not None
            else "this judge logs no calls, so the cost is unknown"
        )
        print(
            f"{d.trace_tokens:,} tokens, {d.n_chapters} chapters, alarm at chapter "
            f"{d.alarm_chapter}; {spend}"
        )
        for rank, s in enumerate(d.suspects, 1):
            print(
                f"{rank}. step {s.step} ({s.step_type}{' ' + s.name if s.name else ''}), "
                f"chapter {s.chapter}, score {s.score:.3f}\n   {s.preview[:160]!r}"
            )
    return 0


def _load_benchmark(args: argparse.Namespace) -> tuple[list, TraceSource]:
    if args.benchmark == "agentracer":
        from alibi.sources import agentracer

        data_dir = args.data_dir or agentracer.DEFAULT_DATA_DIR
        domains = agentracer.DOMAINS if args.subset == "all" else (args.subset,)
        return (
            agentracer.load_annotations(data_dir, domains, args.split),
            agentracer.AgenTracerTraceSource(data_dir),
        )
    if args.benchmark == "trajerrbench":
        from alibi.sources import trajerrbench

        data_dir = args.data_dir or trajerrbench.DEFAULT_DATA_DIR
        subsets = trajerrbench.SUBSETS if args.subset == "all" else (args.subset,)
        return (
            trajerrbench.load_annotations(data_dir, subsets),
            trajerrbench.TrajErrBenchTraceSource(data_dir),
        )
    if args.benchmark == "longrca":
        from alibi.sources import longrca

        data_dir = args.data_dir or longrca.DEFAULT_DATA_DIR
        subsets = longrca.SUBSETS if args.subset == "all" else (args.subset,)
        return longrca.load_annotations(data_dir, subsets), longrca.LongRCATraceSource(data_dir)
    from alibi.sources import agentrx

    data_dir = args.data_dir or agentrx.DEFAULT_DATA_DIR
    splits = tuple(agentrx.SPLITS) if args.subset == "all" else (args.subset,)
    return agentrx.load_annotations(data_dir, splits), agentrx.AgentRxTraceSource(data_dir)


def select_ids(annotations: list, path: Path) -> list:
    """Exactly the listed traces, in file order (fixed sets, e.g. the locked held-out)."""
    data = json.loads(Path(path).read_text())
    ids = data["ids"] if isinstance(data, dict) else data
    by_id = {a.trace_id: a for a in annotations}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        raise ValueError(f"trace ids not in this benchmark/split: {unknown[:5]}")
    return [by_id[i] for i in ids]


def _eval(args: argparse.Namespace) -> int:
    from pathlib import Path

    from alibi.chunking import trace_tokens
    from alibi.evaluate import EvalConfig, run, summarize

    annotations, source = _load_benchmark(args)
    if args.ids:
        annotations = select_ids(annotations, Path(args.ids))
    elif args.min_tokens:
        annotations = [
            a for a in annotations if trace_tokens(source.get_trace(a.trace_id)) > args.min_tokens
        ]
    if not args.ids and args.sample and args.sample < len(annotations):
        stride = len(annotations) / args.sample
        annotations = [annotations[int(i * stride)] for i in range(args.sample)]
    annotations = annotations[: args.limit]
    print(f"{len(annotations)} traces selected", file=sys.stderr)
    settings = load_settings()
    pipeline = args.pipeline or ("typed" if settings.judge_backend == "typesafe" else "text")
    window = args.max_tokens or settings.judge_window_tokens
    cfg = EvalConfig(
        window,
        args.overlap_tokens if args.overlap_tokens is not None else window // 8,
        propagate_state=not args.no_state,
        mode=args.mode,
        route_threshold=args.route_threshold or settings.judge_reliable_tokens,
        pipeline=pipeline,
        combine_rule=args.combine_rule,
        card=args.card,
        checks=args.checks,
        pick=args.pick,
        commit=args.commit,
        reach_after=args.reach_after,
    )

    def progress(row: dict) -> None:
        if "error" in row:
            print(f"ERROR {row['trace_id']}: {row['error']}", file=sys.stderr)
        else:
            print(
                f"{row['trace_id'][:24]:24} gold={row['gold']:>3} pred={row['pred']!s:>4} "
                f"chunks={row['n_chunks']:>2} ${row['cost']:.4f}",
                file=sys.stderr,
            )

    rows = run(
        annotations,
        make_judge,
        cfg,
        Path(args.out),
        source,
        workers=args.workers,
        on_result=progress,
    )
    json.dump(summarize(rows), sys.stdout, indent=2)
    print()
    return 0


def _finalists(args: argparse.Namespace) -> int:
    from concurrent.futures import ThreadPoolExecutor

    from alibi.evaluate import add_finalists, summarize

    _, source = _load_benchmark(args)
    budget = load_settings().judge_window_tokens  # one chapter: known to fit the judge
    rows = [json.loads(line) for line in Path(args.inp).read_text().splitlines()]
    rows = [r for r in rows if "error" not in r]
    out = Path(args.out)
    done = {}
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                done[r["trace_id"]] = r
    todo = [r for r in rows if r["trace_id"] not in done]

    def one(row: dict) -> dict:
        try:
            steps = source.get_trace(row["trace_id"])
            return add_finalists(
                row, steps, make_judge(), args.k, budget, args.rule, args.pick, args.context
            )
        except Exception as e:  # recorded and retried on resume
            return {"trace_id": row["trace_id"], "error": repr(e)[:500]}

    with ThreadPoolExecutor(args.workers) as pool, out.open("w") as f:
        for r in done.values():
            f.write(json.dumps(r) + "\n")
        for r in pool.map(one, todo):
            f.write(json.dumps(r) + "\n")
            f.flush()
            done[r["trace_id"]] = r
    json.dump(summarize(list(done.values())), sys.stdout, indent=2)
    print()
    return 0


def _download_agentrx(args: argparse.Namespace) -> int:
    from pathlib import Path

    import requests

    from alibi.sources.agentrx import SPLITS

    token = load_settings().hf_token
    if not token:
        print("HF_TOKEN is not set (see .env.example)", file=sys.stderr)
        return 1
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    base = "https://huggingface.co/datasets/microsoft/AgentRx/resolve/main/"
    for ann_file, traj_file, _ in SPLITS.values():
        for name in (ann_file, traj_file):
            resp = requests.get(
                base + name, headers={"Authorization": f"Bearer {token}"}, timeout=60
            )
            resp.raise_for_status()
            (data_dir / name).write_bytes(resp.content)
            print(f"{name}: {len(resp.content)} bytes", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
