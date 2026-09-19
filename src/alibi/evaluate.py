"""Critical-step localization evaluation (AgentRx, AgenTracer, ...): does the pipeline find it?

Each trajectory has exactly one annotated root-cause step and the pipeline predicts at
most one, so precision = hits / predictions and recall = hits / trajectories.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from alibi.backward import (
    OUTCOME_FAILURE,
    ChunkEvidence,
    backward_pass,
    backward_pass_typed,
    combine_scores,
    finalist_round,
    pick_step,
    suspect_probs,
)
from alibi.chunking import chunk_trace, trace_tokens
from alibi.drift import detect_drift
from alibi.forward import forward_pass, forward_pass_typed
from alibi.judges import Judge
from alibi.routing import trace_kind
from alibi.sources import TraceSource
from alibi.types import Annotation, Question

TOLERANCES = (0, 1, 3)


@dataclass(frozen=True)
class EvalConfig:
    max_tokens: int
    overlap_tokens: int
    propagate_state: bool = True
    # "hybrid": whole-trace call if the trace fits route_threshold tokens, else windowed.
    # "whole": always one call (baseline). "windowed": always the windowed pipeline.
    mode: str = "hybrid"
    route_threshold: int = 10_000
    cusum_k: float = 0.2
    cusum_h: float = 0.5
    # "text": free-form LLM judge (written summary). "typed": System One judge (Jev):
    # memory card of probabilities, per-step Nouls, P(chunk) x P(step). Windowed only.
    pipeline: str = "text"
    combine_rule: str = "chapter_x_step"  # typed pipeline: see backward.COMBINE_RULES
    card: str = "symptom"  # typed: "gap" also shows the request and how the run ended
    checks: bool = False  # typed: ask backward.STEP_CHECKS per step
    commit: bool = False  # typed (V7): ask backward.COMMIT_QUESTION per agent step
    reach_after: int = 0  # typed (V8): also look back at this many chunks after the alarm
    pick: str = "best"  # typed: "best" | "earliest_near_best"


WHOLE_TRACE_QUESTIONS = [
    Question(
        "critical_step",
        "The [step N] number of the root cause: the earliest step where the AGENT itself went "
        "wrong (a bad decision, tool call, argument, or claim) that made the failure "
        "inevitable. Correct tool outputs or observations are not root causes, even if the "
        "agent later mishandles them. Always name one step, even if uncertain.",
        "integer",
    ),
    Question("confidence", "0-1 confidence that this is the root-cause step.", "number"),
    Question(
        "rationale", "One sentence explaining the causal link, citing step numbers.", "string"
    ),
]


def evaluate_whole_trace(
    ann: Annotation, source: TraceSource, judge: Judge, cfg: EvalConfig
) -> dict[str, Any]:
    """Baseline: the judge reads the entire trace in one call, told only that the run failed."""
    steps = source.get_trace(ann.trace_id)
    [chunk] = chunk_trace(steps, max_tokens=10**9, overlap_tokens=0)
    answers = judge.evaluate(
        f"Known outcome:\n{OUTCOME_FAILURE}\n\nFull trace:\n{chunk.text}", WHOLE_TRACE_QUESTIONS
    )
    raw = int(answers["critical_step"])
    calls = getattr(judge, "calls", [])
    return {
        "trace_id": ann.trace_id,
        "split": ann.split,
        "category": ann.category,
        "n_steps": len(steps),
        "n_chunks": 1,
        "trace_tokens": chunk.tokens,
        "gold": ann.critical_step,
        "pred": min(max(raw, 0), len(steps) - 1),
        "pred_valid": 0 <= raw < len(steps),
        "confidence": answers["confidence"],
        "rationale": answers["rationale"],
        "failure_chunk_contains_gold": True,
        "gold_before_failure_chunk": False,
        "anchor": "whole_trace",
        "cost": sum(c.cost or 0 for c in calls),
        "latency_s": sum(c.latency_s for c in calls),
        "n_calls": len(calls),
        "config": asdict(cfg),
        "judge_model": getattr(judge, "model", None),
        "judge_reasoning": getattr(judge, "reasoning", None),
    }


def row_scores(row: dict[str, Any], rule: str) -> dict[int, float]:
    """Per-step scores of a typed-pipeline row under a fixed rule, from stored evidence."""
    evidence = [
        ChunkEvidence(
            e["chunk"],
            e["p_chunk"],
            {int(k): v for k, v in e["p_steps"].items()},
            {int(k): v for k, v in (e.get("p_checks") or {}).items()},
            {int(k): v for k, v in (e.get("p_commit") or {}).items()},
        )
        for e in row["backward_evidence"]
    ]
    suspect = [suspect_probs(a) for a in row["forward_answers"]]
    return combine_scores(evidence, row["forward_scores"], suspect, rule)


V2_RULE = "step_plus_check"
# V7: the router's trace kind picks the branch. Conversation (z) has no branch yet.
V7_RULES = {"code": "commit_x_step_plus_check"}


def v7_rule(row: dict[str, Any]) -> str:
    return V7_RULES.get(row.get("trace_kind", ""), V2_RULE)


def rescore(row: dict[str, Any], rule: str, pick: str = "best") -> int:
    """Re-pick a typed-pipeline row's step under another fixed rule, without new calls."""
    return pick_step(row_scores(row, rule), pick)


def add_finalists(
    row: dict[str, Any],
    steps: list,
    judge: Judge,
    k: int,
    budget_tokens: int,
    rule: str,
    pick: str = "best",
    context: int = 0,
) -> dict[str, Any]:
    """V3: shortlist the top-k steps by ``rule`` and let one Choice compare them side by
    side. Keeps the previous pick as ``pred_v2``; ``pred`` becomes the finalist pick."""
    scores = row_scores(row, rule)
    shortlist = sorted(scores, key=lambda i: (-scores[i], i))[:k]
    probs = finalist_round(judge, steps, shortlist, budget_tokens, context)
    chosen = pick_step(probs, pick)
    calls = getattr(judge, "calls", [])
    return {
        **row,
        "pred_v2": row["pred"],
        "pred": chosen,
        "finalists": {
            "pick": pick,
            "context": context,
            "candidates": sorted(shortlist),
            "probs": {str(i): p for i, p in sorted(probs.items())},
            "gold_in_shortlist": row["gold"] in shortlist,
            "cost": sum(c.cost or 0 for c in calls),
            "latency_s": sum(c.latency_s for c in calls),
        },
    }


def evaluate_one(
    ann: Annotation, source: TraceSource, judge: Judge, cfg: EvalConfig
) -> dict[str, Any]:
    steps = source.get_trace(ann.trace_id)
    n_tokens = trace_tokens(steps)
    if cfg.pipeline == "typed" and cfg.mode != "windowed":
        raise ValueError("the typed pipeline never sends the whole trace; use mode='windowed'")
    if cfg.mode == "whole" or (cfg.mode == "hybrid" and n_tokens <= cfg.route_threshold):
        return {**evaluate_whole_trace(ann, source, judge, cfg), "route": "whole"}
    return {
        **evaluate_windowed(ann, steps, judge, cfg),
        "route": "windowed",
        "trace_tokens": n_tokens,
    }


def evaluate_windowed(
    ann: Annotation, steps: list, judge: Judge, cfg: EvalConfig
) -> dict[str, Any]:
    chunks = chunk_trace(steps, cfg.max_tokens, cfg.overlap_tokens)
    typed = cfg.pipeline == "typed"
    if typed:
        fwd = forward_pass_typed(judge, chunks, steps, propagate_state=cfg.propagate_state)
    else:
        fwd = forward_pass(judge, chunks, propagate_state=cfg.propagate_state)
    alarms = [cp.alarm for cp in detect_drift([s.score for s in fwd], cfg.cusum_k, cfg.cusum_h)]
    # Benchmarks contain only failed runs, so known_failed=True.
    if typed:
        loc = backward_pass_typed(
            judge,
            chunks,
            steps,
            fwd,
            alarms,
            known_failed=True,
            rule=cfg.combine_rule,
            card=cfg.card,
            checks=cfg.checks,
            pick=cfg.pick,
            commit=cfg.commit,
            reach_after=cfg.reach_after,
        )
    else:
        loc = backward_pass(judge, chunks, fwd, alarms, known_failed=True)

    failed = chunks[loc.failure_chunk]
    calls = getattr(judge, "calls", [])
    return {
        "trace_id": ann.trace_id,
        "split": ann.split,
        "category": ann.category,
        "n_steps": len(steps),
        "n_chunks": len(chunks),
        "gold": ann.critical_step,
        "pred": loc.critical_step,
        "forward_scores": [s.score for s in fwd],
        "alarms": alarms,
        "failure_chunk": loc.failure_chunk,
        "anchor": loc.anchor,
        "failure_chunk_contains_gold": failed.start <= ann.critical_step < failed.end,
        "gold_before_failure_chunk": ann.critical_step < failed.start,
        "precursors": [
            {"chunk": p.chunk_index, "score": p.score, "step": p.step, "valid": p.step_valid}
            for p in loc.precursors
        ],
        "pipeline": cfg.pipeline,
        "trace_kind": trace_kind(steps),  # V7 router (plain code)
        "ranked": [list(r) for r in loc.ranked],
        "forward_answers": [s.answers for s in fwd] if typed else None,
        "combine_rule": cfg.combine_rule if typed else None,
        "backward_evidence": [
            {
                "chunk": e.chunk_index,
                "p_chunk": e.p_chunk,
                "p_steps": e.p_steps,
                "p_checks": e.p_checks,
                "p_commit": e.p_commit,
            }
            for e in loc.evidence
        ],
        "cost": sum(c.cost or 0 for c in calls),
        "latency_s": sum(c.latency_s for c in calls),
        "n_calls": len(calls),
        "config": asdict(cfg),
        "judge_model": getattr(judge, "model", None),
        "judge_reasoning": getattr(judge, "reasoning", None),
    }


def run(
    annotations: list[Annotation],
    make_judge: Callable[[], Judge],
    cfg: EvalConfig,
    out_path: Path,
    source: TraceSource,
    workers: int = 8,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate in parallel, appending one JSON line per trajectory; resumes from out_path.

    Rows that errored are retried on resume; the latest row per trajectory wins.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            row = json.loads(line)
            if "error" not in row:
                done[row["trace_id"]] = row
    todo = [a for a in annotations if a.trace_id not in done]

    with ThreadPoolExecutor(max_workers=workers) as pool, out_path.open("a") as out:
        futures = {pool.submit(evaluate_one, a, source, make_judge(), cfg): a for a in todo}
        for fut in as_completed(futures):
            ann = futures[fut]
            try:
                row = fut.result()
            except Exception as e:  # keep going; record the failure
                row = {"trace_id": ann.trace_id, "split": ann.split, "error": repr(e)[:500]}
            out.write(json.dumps(row) + "\n")
            out.flush()
            done[row["trace_id"]] = row
            if on_result:
                on_result(row)
    wanted = {a.trace_id for a in annotations}
    return [r for tid, r in done.items() if tid in wanted]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    n = len(ok)
    predicted = [r for r in ok if r["pred"] is not None]
    summary: dict[str, Any] = {
        "trajectories": n,
        "errors": len(rows) - n,
        "predictions": len(predicted),
    }
    for tol in TOLERANCES:
        hits = sum(abs(r["pred"] - r["gold"]) <= tol for r in predicted)
        summary[f"precision@±{tol}"] = round(hits / len(predicted), 3) if predicted else None
        summary[f"recall@±{tol}"] = round(hits / n, 3) if n else None
        # Random-guess baseline: expected recall of a uniform guess over the trace's steps.
        summary[f"random_recall@±{tol}"] = (
            round(sum(_uniform_hit_rate(r["gold"], r["n_steps"], tol) for r in ok) / n, 3)
            if n
            else None
        )
    if n:
        summary["failure_chunk_contains_gold"] = round(
            sum(r["failure_chunk_contains_gold"] for r in ok) / n, 3
        )
        summary["gold_before_failure_chunk"] = round(
            sum(r["gold_before_failure_chunk"] for r in ok) / n, 3
        )
        summary["forward_anchored"] = round(sum(r.get("anchor") == "forward" for r in ok) / n, 3)
        summary["mean_chunks"] = round(sum(r["n_chunks"] for r in ok) / n, 2)
    summary["total_cost"] = round(sum(r.get("cost", 0) for r in ok), 4)
    summary["mean_latency_s"] = round(sum(r.get("latency_s", 0) for r in ok) / n, 1) if n else None
    return summary


def _uniform_hit_rate(gold: int, n_steps: int, tol: int) -> float:
    lo, hi = max(0, gold - tol), min(n_steps - 1, gold + tol)
    return (hi - lo + 1) / n_steps
