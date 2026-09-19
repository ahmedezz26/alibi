from alibi.chunking import chunk_trace
from alibi.forward import STATE_KEY, forward_pass, truncate_state
from alibi.sources.jsonfile import JsonFileTraceSource


class StubJudge:
    def __init__(self, scores):
        self.scores, self.prompts, self.question_ids = iter(scores), [], []

    def evaluate(self, state, questions):
        self.prompts.append(state)
        self.question_ids.append([q.id for q in questions])
        n = len(self.prompts)
        return {"anomaly_score": next(self.scores), "rationale": "r", STATE_KEY: f"summary {n}"}


def sample_chunks():
    steps = JsonFileTraceSource().get_trace("examples/sample_trace.json")
    chunks = chunk_trace(steps, max_tokens=120, overlap_tokens=40)
    assert len(chunks) > 1
    return chunks


def test_scores_each_chunk_and_clamps():
    chunks = sample_chunks()
    scores = forward_pass(StubJudge([0.2, 1.7, -0.3, 0.5][: len(chunks)]), chunks)
    assert [s.chunk_index for s in scores] == [c.index for c in chunks]
    assert all(0.0 <= s.score <= 1.0 for s in scores)
    assert scores[1].score == 1.0


def test_state_is_carried_to_next_chunk():
    chunks = sample_chunks()
    judge = StubJudge([0.0] * len(chunks))
    scores = forward_pass(judge, chunks)

    assert STATE_KEY in judge.question_ids[0]
    assert scores[0].state_in == ""
    assert scores[1].state_in == "summary 1"
    assert judge.prompts[1].startswith("Running summary of earlier windows:\nsummary 1")
    assert chunks[1].text in judge.prompts[1]


def test_stateless_mode_sends_only_the_chunk():
    chunks = sample_chunks()
    judge = StubJudge([0.0] * len(chunks))
    forward_pass(judge, chunks, propagate_state=False)
    assert STATE_KEY not in judge.question_ids[0]
    assert judge.prompts == [f"Current window:\n{c.text}" for c in chunks]


def test_truncate_state_bounds_length():
    long = " ".join(["word"] * 500)
    out = truncate_state(long, max_tokens=20)
    assert len(out) // 4 <= 20 and out.endswith("…")
    assert truncate_state("short", max_tokens=20) == "short"


def test_json_source_reads_file_path():
    steps = JsonFileTraceSource().get_trace("examples/sample_trace.json")
    assert len(steps) == 5
    assert steps[3].name == "book_flight"
    assert steps[0].step_id == "0"
