"""V7: a per-step "commit point" Noul, asked inside the look-back calls, and a fixed rule."""

import pytest

from alibi.backward import (
    COMBINE_RULES,
    ChunkEvidence,
    backward_pass_typed,
    combine_scores,
    step_questions,
)
from alibi.chunking import chunk_trace
from alibi.evaluate import row_scores
from alibi.types import Step


def steps_and_chunks():
    kinds = ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    steps = [Step(str(i), k, None, {"t": i}) for i, k in enumerate(kinds)]
    return steps, chunk_trace(steps, max_tokens=30, overlap_tokens=1, count_tokens=lambda _t: 10)


def test_commit_noul_is_asked_only_for_agent_steps_and_only_when_enabled():
    _, chunks = steps_and_chunks()
    assert not any(q.id.startswith("k_") for q in step_questions(chunks[0], checks=True))
    qs = {q.id: q for q in step_questions(chunks[0], checks=True, commit=True)}
    assert {k for k in qs if k.startswith("k_")} == {"k_0", "k_1"}  # steps 0..2; 2 is a tool
    assert qs["k_1"].answer_type == "noul" and "[step 1]" in qs["k_1"].prompt


def test_commit_rule_multiplies_v2_score_by_commit_probability():
    ev = [
        ChunkEvidence(
            0, 1.0, {1: 0.6, 3: 0.6}, {1: {"request": 0.2}, 3: {"request": 0.2}}, {1: 0.9, 3: 0.1}
        )
    ]
    v2 = combine_scores(ev, [0.0], [{}], "step_plus_check")
    v7 = combine_scores(ev, [0.0], [{}], "commit_x_step_plus_check")
    assert v2[1] == v2[3] == pytest.approx(0.4)
    assert v7[1] == pytest.approx(0.36) and v7[3] == pytest.approx(0.04)


def test_commit_rule_scores_steps_without_a_commit_answer_as_zero():
    ev = [ChunkEvidence(0, 1.0, {2: 0.9}, {}, {})]
    assert combine_scores(ev, [0.0], [{}], "commit_x_step_plus_check")[2] == 0.0


def test_commit_rule_is_registered():
    assert "commit_x_step_plus_check" in COMBINE_RULES


class Judge:
    def evaluate(self, state, questions):
        return {q.id: (0.8 if q.id == "k_3" else 0.3) for q in questions}


def test_backward_pass_keeps_commit_answers_and_rows_rescore_from_them():
    from alibi.forward import ChunkScore

    steps, chunks = steps_and_chunks()
    fwd = [ChunkScore(c.index, c.start, c.end, 0.9 * c.index, {}) for c in chunks]
    loc = backward_pass_typed(
        Judge(), chunks, steps, fwd, known_failed=True, checks=True, commit=True, workers=1
    )
    by_chunk = {e.chunk_index: e for e in loc.evidence}
    assert by_chunk[1].p_commit == {3: 0.8, 5: 0.3}  # step 4 is a tool
    row = {
        "backward_evidence": [
            {
                "chunk": e.chunk_index,
                "p_chunk": e.p_chunk,
                "p_steps": {str(k): v for k, v in e.p_steps.items()},
                "p_checks": {str(k): v for k, v in e.p_checks.items()},
                "p_commit": {str(k): v for k, v in e.p_commit.items()},
            }
            for e in loc.evidence
        ],
        "forward_answers": [{} for _ in chunks],
        "forward_scores": [0.9 * c.index for c in chunks],
    }
    scores = row_scores(row, "commit_x_step_plus_check")
    assert max(scores, key=scores.get) == 3


def test_v7_routes_code_traces_to_the_commit_rule_and_others_to_v2():
    from alibi.evaluate import V2_RULE, v7_rule

    assert v7_rule({"trace_kind": "code"}) == "commit_x_step_plus_check"
    assert v7_rule({"trace_kind": "conversation"}) == V2_RULE == "step_plus_check"
    assert v7_rule({"trace_kind": "other"}) == V2_RULE
    assert v7_rule({}) == V2_RULE  # rows from before the router
