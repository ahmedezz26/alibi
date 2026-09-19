from alibi.backward import CHAPTER_KEY, backward_pass_typed
from alibi.chunking import chunk_trace
from alibi.forward import HEALTH_KEY, SUSPECT_KEY, WARNING_SIGNS, ChunkScore
from alibi.types import Step


def make():
    steps = [
        Step(step_id=str(i), type="tool", timestamp=None, inputs={"text": f"action-{i}"})
        for i in range(9)
    ]
    chunks = chunk_trace(steps, max_tokens=30, overlap_tokens=1, count_tokens=lambda _t: 10)
    return steps, chunks


def forward(chunks, health, suspect=7, signs=None):
    signs = signs or {k: 0.1 for k in WARNING_SIGNS}
    return [
        ChunkScore(
            c.index,
            c.start,
            c.end,
            h,
            {HEALTH_KEY: h, SUSPECT_KEY: {"choice": str(suspect)}, **signs},
            state_in=f"card{c.index}",
        )
        for c, h in zip(chunks, health, strict=True)
    ]


class ChapterJudge:
    """by_chunk: chunk index -> (p_chapter, {step: p_step}); unlisted steps get 0.05."""

    def __init__(self, chunks, by_chunk):
        self.chunks, self.by_chunk, self.prompts, self.questions = chunks, by_chunk, [], []

    def evaluate(self, state, questions):
        self.prompts.append(state)
        self.questions.append(questions)
        idx = next(c.index for c in self.chunks if state.endswith(c.text))
        p_ch, p_steps = self.by_chunk[idx]
        out = {CHAPTER_KEY: p_ch}
        for q in questions:
            if q.id != CHAPTER_KEY:
                out[q.id] = p_steps.get(int(q.id[1:]), 0.05)
        return out


def test_picks_highest_chapter_times_step_probability():
    steps, chunks = make()
    fwd = forward(chunks, [0.0, 0.1, 0.9])
    judge = ChapterJudge(chunks, {0: (0.9, {1: 0.6}), 1: (0.2, {4: 0.9}), 2: (0.5, {7: 0.9})})

    loc = backward_pass_typed(judge, chunks, steps, fwd, workers=1)

    assert (loc.anchor, loc.failure_chunk) == ("forward", 2)
    assert loc.critical_step == 1  # 0.9*0.6=0.54 beats 0.5*0.9=0.45 and 0.2*0.9=0.18
    assert [s for s, _ in loc.ranked] == [1, 7, 4]
    assert loc.ranked[0][1] == 0.54


def test_asks_chapter_noul_with_criteria_and_one_noul_per_step():
    steps, chunks = make()
    judge = ChapterJudge(chunks, {0: (0.1, {}), 1: (0.9, {4: 0.8})})
    backward_pass_typed(judge, chunks, steps, forward(chunks, [0.0, 0.9, 0.0]), workers=1)

    qs = {q.id: q for q in judge.questions[0]}  # chunk 1, examined first
    assert qs[CHAPTER_KEY].answer_type == "noul" and set(qs[CHAPTER_KEY].criteria) == {
        "true",
        "false",
    }
    assert sorted(k for k in qs if k != CHAPTER_KEY) == ["s3", "s4", "s5"]
    assert all(q.answer_type == "noul" for q in qs.values())


def test_never_examines_chunks_after_the_alarm():
    steps, chunks = make()
    judge = ChapterJudge(chunks, {0: (0.3, {}), 1: (0.9, {4: 0.8})})
    loc = backward_pass_typed(judge, chunks, steps, forward(chunks, [0.0, 0.9, 0.0]))
    assert len(judge.prompts) == 2 and loc.critical_step == 4


def test_problem_card_names_the_suspect_step_and_raised_warning_signs():
    steps, chunks = make()
    signs = {k: 0.1 for k in WARNING_SIGNS} | {"misread_task": 0.85}
    fwd = forward(chunks, [0.0, 0.1, 0.9], suspect=7, signs=signs)
    judge = ChapterJudge(chunks, {i: (0.5, {}) for i in range(3)})

    backward_pass_typed(judge, chunks, steps, fwd, workers=1)

    card = judge.prompts[2]  # chunk 0 (steps 0..2): step 7 can only come from the card
    assert "action-7" in card  # the suspect step's own text
    assert WARNING_SIGNS["misread_task"] in card and "85%" in card
    assert WARNING_SIGNS["off_plan"] not in card  # 10% is not a raised sign
    assert "card1" in judge.prompts[1]  # memory card before chunk 1 is shown


def test_no_forward_signal_known_failed_uses_the_final_steps():
    steps, chunks = make()
    judge = ChapterJudge(chunks, {0: (0.2, {}), 1: (0.8, {3: 0.7}), 2: (0.1, {})})
    loc = backward_pass_typed(
        judge, chunks, steps, forward(chunks, [0.0, 0.1, 0.0]), known_failed=True
    )
    assert (loc.anchor, loc.failure_chunk, loc.critical_step) == ("outcome", 2, 3)
    assert all("ultimately FAILED" in p and "action-8" in p for p in judge.prompts)


def test_combine_rule_is_selectable_and_evidence_is_kept():
    steps, chunks = make()
    fwd = forward(chunks, [0.0, 0.1, 0.9])
    judge = ChapterJudge(chunks, {0: (0.9, {1: 0.6}), 1: (0.2, {4: 0.9}), 2: (0.5, {7: 0.8})})

    loc = backward_pass_typed(judge, chunks, steps, fwd, workers=1, rule="step_only")

    assert loc.critical_step == 4  # highest step answer, chapter answer ignored
    assert [e.chunk_index for e in loc.evidence] == [0, 1, 2]
    assert loc.evidence[0].p_chunk == 0.9 and loc.evidence[0].p_steps[1] == 0.6


def test_v8_reach_after_also_examines_chapters_after_the_alarm():
    steps, chunks = make()  # 3 chunks: steps 0-2, 3-5, 6-8
    judge = ChapterJudge(chunks, {0: (0.1, {}), 1: (0.2, {}), 2: (0.9, {7: 0.9})})
    fwd = forward(chunks, [0.0, 0.9, 0.0])  # alarm at chunk 1, the cause is in chunk 2

    v2 = backward_pass_typed(judge, chunks, steps, fwd, workers=1)
    v8 = backward_pass_typed(judge, chunks, steps, fwd, workers=1, reach_after=2)

    assert v2.critical_step != 7 and v8.critical_step == 7
    assert sorted(e.chunk_index for e in v8.evidence) == [0, 1, 2]  # clipped at the last chunk
    assert v8.failure_chunk == 1  # the anchor itself is unchanged
