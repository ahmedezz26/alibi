from alibi.chunking import chunk_trace
from alibi.forward import HEALTH_KEY, SUSPECT_KEY, WARNING_SIGNS, forward_pass_typed
from alibi.types import Step


def make_chunks():
    steps = [Step(step_id=str(i), type="tool", timestamp=None, inputs={"n": i}) for i in range(9)]
    return steps, chunk_trace(steps, max_tokens=30, overlap_tokens=1, count_tokens=lambda _t: 10)


class TypedJudge:
    """Answers every question by type; health rises by 0.3 per chunk."""

    def __init__(self):
        self.prompts, self.questions = [], []

    def evaluate(self, state, questions):
        self.prompts.append(state)
        self.questions.append(questions)
        n = len(self.prompts)
        out = {}
        for q in questions:
            if q.answer_type == "score":
                out[q.id] = min(1.0, 0.3 * (n - 1))
            elif q.answer_type == "noul":
                out[q.id] = 0.1 * n
            else:
                first = next(iter(q.criteria))
                out[q.id] = {"choice": first, "probabilities": {first: 1.0}, "confidence": 1.0}
        return out


def test_asks_health_score_warning_signs_and_suspect_step_per_chunk():
    steps, chunks = make_chunks()
    judge = TypedJudge()
    forward_pass_typed(judge, chunks, steps)

    qs = {q.id: q for q in judge.questions[1]}
    assert qs[HEALTH_KEY].answer_type == "score" and len(qs[HEALTH_KEY].criteria) == 4
    assert all(qs[s].answer_type == "noul" for s in WARNING_SIGNS)
    assert qs[SUSPECT_KEY].answer_type == "choice"
    assert list(qs[SUSPECT_KEY].criteria) == ["3", "4", "5"]  # chunk 1 = steps 3..5


def test_health_is_the_chunk_score_and_memory_card_carries_numbers():
    steps, chunks = make_chunks()
    judge = TypedJudge()
    scores = forward_pass_typed(judge, chunks, steps)

    assert [s.score for s in scores] == [0.0, 0.3, 0.6]
    assert scores[0].state_in == ""
    assert "how the agent is doing: 0.30" in scores[2].state_in
    assert "20%" in scores[2].state_in  # warning-sign probability from chunk 1
    assert scores[2].state_in in judge.prompts[2]
    assert chunks[2].text in judge.prompts[2]


def test_task_start_is_repeated_for_later_chunks():
    steps, chunks = make_chunks()
    judge = TypedJudge()
    forward_pass_typed(judge, chunks, steps)
    assert "Task (start of the run):\n[step 0]" in judge.prompts[1]


def test_stateless_mode_drops_the_memory_card():
    steps, chunks = make_chunks()
    judge = TypedJudge()
    scores = forward_pass_typed(judge, chunks, steps, propagate_state=False)
    assert all(s.state_in == "" for s in scores)
    assert all("Memory card" not in p for p in judge.prompts)
