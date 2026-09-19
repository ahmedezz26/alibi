from alibi.backward import BACKWARD_QUESTIONS, backward_pass, failure_chunk_index
from alibi.chunking import chunk_trace
from alibi.forward import ChunkScore
from alibi.types import Step


def make_chunks(n_steps=9):
    steps = [Step(step_id=str(i), type="tool", timestamp=None) for i in range(n_steps)]
    return chunk_trace(steps, max_tokens=30, overlap_tokens=1, count_tokens=lambda _t: 10)


def forward_scores(chunks, values):
    return [
        ChunkScore(c.index, c.start, c.end, v, {"rationale": f"r{c.index}"}, state_in=f"s{c.index}")
        for c, v in zip(chunks, values, strict=True)
    ]


class ScriptedJudge:
    """Returns answers keyed by which chunk's text appears in the prompt."""

    def __init__(self, chunks, by_chunk):
        self.chunks, self.by_chunk, self.prompts = chunks, by_chunk, []

    def evaluate(self, state, questions):
        assert questions == BACKWARD_QUESTIONS
        self.prompts.append(state)
        idx = next(c.index for c in self.chunks if state.endswith(c.text))
        score, step = self.by_chunk[idx]
        return {"precursor_score": score, "critical_step": step, "rationale": "x"}


def test_failure_chunk_prefers_first_alarm_else_argmax():
    chunks = make_chunks()
    fwd = forward_scores(chunks, [0.1, 0.9, 0.9])
    assert failure_chunk_index(fwd, alarms=[2]) == 2
    assert failure_chunk_index(fwd) == 1  # tie -> earliest
    assert failure_chunk_index(forward_scores(chunks, [0.1, 0.2, 0.0])) is None


def test_localizes_earlier_root_cause_and_walks_backward():
    chunks = make_chunks()
    assert [(c.start, c.end) for c in chunks] == [(0, 3), (3, 6), (6, 9)]
    fwd = forward_scores(chunks, [0.0, 0.1, 1.0])
    judge = ScriptedJudge(chunks, {0: (0.9, 1), 1: (0.3, 4), 2: (0.6, 7)})

    loc = backward_pass(judge, chunks, fwd, workers=1)  # sequential so call order is checkable

    assert loc.failure_chunk == 2
    assert loc.critical_step == 1
    assert [p.chunk_index for p in loc.precursors] == [0, 1, 2]
    assert judge.prompts[0].endswith(chunks[2].text)  # started at the failure
    assert all("Later, in steps 6..8" in p for p in judge.prompts)
    assert "Running summary before this window:\ns1" in judge.prompts[1]


def test_ignores_steps_outside_the_chunk_and_later_chunks():
    chunks = make_chunks()
    fwd = forward_scores(chunks, [0.0, 1.0, 0.2])
    judge = ScriptedJudge(chunks, {0: (1.0, 7), 1: (0.5, 4)})  # step 7 not in chunk 0

    loc = backward_pass(judge, chunks, fwd)

    assert len(judge.prompts) == 2  # chunk 2 is after the failure, never examined
    assert (loc.precursors[0].step, loc.precursors[0].step_valid) == (2, False)  # clamped
    assert loc.critical_step == 4  # valid step beats a higher-scoring clamped one


def test_always_predicts_even_when_judge_names_no_valid_step():
    chunks = make_chunks()
    fwd = forward_scores(chunks, [1.0, 0.0, 0.0])
    loc = backward_pass(ScriptedJudge(chunks, {0: (0.0, -1)}), chunks, fwd)
    assert loc.critical_step == 0  # -1 clamped into chunk 0


def test_low_confidence_outcome_anchored_trace_still_predicts():
    chunks = make_chunks()
    fwd = forward_scores(chunks, [0.0, 0.0, 0.0])
    judge = ScriptedJudge(chunks, {0: (0.0, 1), 1: (0.2, 4), 2: (0.0, 8)})
    loc = backward_pass(judge, chunks, fwd, known_failed=True)
    assert (loc.anchor, loc.critical_step) == ("outcome", 4)


def test_no_forward_signal_known_failed_anchors_on_last_chunk():
    chunks = make_chunks()
    fwd = forward_scores(chunks, [0.0, 0.1, 0.0])
    judge = ScriptedJudge(chunks, {0: (0.2, 1), 1: (0.8, 4), 2: (0.1, -1)})

    loc = backward_pass(judge, chunks, fwd, known_failed=True)

    assert (loc.anchor, loc.failure_chunk, loc.critical_step) == ("outcome", 2, 4)
    assert len(judge.prompts) == 3  # walked every chunk
    assert all("ultimately FAILED" in p for p in judge.prompts)


def test_weak_signal_without_outcome_falls_back_to_argmax():
    chunks = make_chunks()
    fwd = forward_scores(chunks, [0.0, 0.3, 0.0])
    loc = backward_pass(ScriptedJudge(chunks, {0: (0.0, -1), 1: (0.4, 3)}), chunks, fwd)
    assert (loc.anchor, loc.failure_chunk) == ("forward", 1)
