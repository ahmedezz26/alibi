import pytest

from alibi.backward import (
    CHAPTER_KEY,
    STEP_CHECKS,
    ChunkEvidence,
    backward_pass_typed,
    combine_scores,
    pick_step,
    step_questions,
)
from alibi.chunking import chunk_trace
from alibi.forward import HEALTH_KEY, SUSPECT_KEY, WARNING_SIGNS, ChunkScore
from alibi.types import Step


def make():
    steps = [
        Step(step_id=str(i), type="agent", timestamp=None, inputs={"text": f"action-{i}"})
        for i in range(9)
    ]
    steps[0] = Step(
        step_id="0", type="user", timestamp=None, inputs={"text": "USER-REQUEST: use TripAdvisor"}
    )
    steps[8] = Step(
        step_id="8", type="agent", timestamp=None, inputs={"text": "FINAL: gave blog hikes"}
    )
    chunks = chunk_trace(steps, max_tokens=30, overlap_tokens=1, count_tokens=lambda _t: 10)
    return steps, chunks


def forward(chunks, health):
    signs = {k: 0.1 for k in WARNING_SIGNS}
    return [
        ChunkScore(
            c.index, c.start, c.end, h, {HEALTH_KEY: h, SUSPECT_KEY: {"choice": "7"}, **signs}
        )
        for c, h in zip(chunks, health, strict=True)
    ]


class Judge:
    def __init__(self):
        self.prompts, self.questions = [], []

    def evaluate(self, state, questions):
        self.prompts.append(state)
        self.questions.append(questions)
        out = {}
        for q in questions:
            out[q.id] = 0.5 if q.id == CHAPTER_KEY else 0.2
            if q.id == "s1" or q.id == "c_request_1":
                out[q.id] = 0.9
        return out


def test_gap_card_shows_request_and_ending_even_when_forward_anchored():
    steps, chunks = make()
    judge = Judge()
    backward_pass_typed(
        judge, chunks, steps, forward(chunks, [0.0, 0.1, 0.9]), workers=1, card="gap"
    )
    chunk1_prompt = judge.prompts[1]  # chunk 1 = steps 3..5: request/ending only via the card
    assert "USER-REQUEST: use TripAdvisor" in chunk1_prompt
    assert "FINAL: gave blog hikes" in chunk1_prompt


def test_symptom_card_is_unchanged_default():
    steps, chunks = make()
    judge = Judge()
    backward_pass_typed(judge, chunks, steps, forward(chunks, [0.0, 0.1, 0.9]), workers=1)
    assert "FINAL: gave blog hikes" not in judge.prompts[1]


def test_step_checks_are_asked_per_step_when_enabled():
    _, chunks = make()
    ids = {q.id for q in step_questions(chunks[0], checks=True)}
    assert {f"c_{k}_{i}" for k in STEP_CHECKS for i in range(0, 3)} <= ids
    assert all(q.answer_type == "noul" for q in step_questions(chunks[0], checks=True))
    assert not any(i.startswith("c_") for i in {q.id for q in step_questions(chunks[0])})


def test_evidence_keeps_check_answers():
    steps, chunks = make()
    loc = backward_pass_typed(
        Judge(), chunks, steps, forward(chunks, [0.0, 0.1, 0.9]), workers=1, checks=True
    )
    assert loc.evidence[0].p_checks[1]["request"] == 0.9


def test_step_plus_check_rule_uses_strongest_check():
    ev = [ChunkEvidence(0, 0.5, {1: 0.2, 2: 0.4}, {1: {"request": 0.9, "unfounded": 0.1}, 2: {}})]
    scores = combine_scores(ev, [0.0], [{}], "step_plus_check")
    assert scores[1] == pytest.approx(0.5 * (0.2 + 0.9) / 2)
    assert scores[2] == pytest.approx(0.5 * (0.4 + 0.0) / 2)


def test_earliest_near_best_pick():
    scores = {3: 0.40, 9: 0.45, 12: 0.20}
    assert pick_step(scores, "best") == 9
    assert pick_step(scores, "earliest_near_best") == 3  # 0.40 >= 0.8 * 0.45
    assert pick_step({3: 0.30, 9: 0.45}, "earliest_near_best") == 9


def test_checks_skip_tool_and_system_steps():
    steps = [
        Step(step_id="0", type="system", timestamp=None, inputs={"t": "s"}),
        Step(step_id="1", type="assistant", timestamp=None, inputs={"t": "a"}),
        Step(step_id="2", type="tool", timestamp=None, inputs={"t": "t"}),
    ]
    [chunk] = chunk_trace(steps, max_tokens=100, overlap_tokens=1, count_tokens=lambda _t: 10)
    ids = {q.id for q in step_questions(chunk, checks=True)}
    assert {"s0", "s1", "s2"} <= ids  # root-cause question still asked for every step
    assert {f"c_{k}_1" for k in STEP_CHECKS} <= ids
    assert not any(i.startswith("c_") and i.endswith(("_0", "_2")) for i in ids)
