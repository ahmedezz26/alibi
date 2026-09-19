from alibi.backward import FINALIST_KEY, finalist_round, fit_candidates
from alibi.chunking import approx_tokens
from alibi.types import Step


def test_fit_candidates_keeps_small_steps_whole_and_caps_the_total():
    texts = {3: "small " * 40, 9: "x" * 400_000, 12: "tiny"}  # 9 is ~100K tokens
    fitted = fit_candidates(texts, budget_tokens=2_000)
    assert fitted[3] == texts[3] and fitted[12] == "tiny"
    assert "omitted" in fitted[9]
    assert sum(approx_tokens(t) for t in fitted.values()) <= 2_000


def test_fit_candidates_leaves_everything_alone_when_it_fits():
    texts = {1: "a" * 400, 2: "b" * 400}
    assert fit_candidates(texts, budget_tokens=1_000) == texts


class ChoiceJudge:
    def __init__(self, probs):
        self.probs, self.prompts, self.questions = probs, [], []

    def evaluate(self, state, questions):
        self.prompts.append(state)
        self.questions.append(questions)
        return {
            FINALIST_KEY: {
                "choice": max(self.probs, key=self.probs.get),
                "probabilities": self.probs,
                "confidence": 0.5,
            }
        }


def steps():
    s = [
        Step(step_id=str(i), type="assistant", timestamp=None, inputs={"t": f"act-{i}"})
        for i in range(20)
    ]
    s[0] = Step(step_id="0", type="user", timestamp=None, inputs={"t": "REQUEST"})
    s[19] = Step(step_id="19", type="assistant", timestamp=None, inputs={"t": "THE-END"})
    return s


def test_finalist_round_asks_one_choice_over_candidates_in_time_order():
    judge = ChoiceJudge({"4": 0.2, "11": 0.7, "15": 0.1})
    probs = finalist_round(judge, steps(), [15, 4, 11], budget_tokens=5_000)

    [q] = judge.questions[0]
    assert q.id == FINALIST_KEY and q.answer_type == "choice"
    assert list(q.criteria) == ["4", "11", "15"]  # chronological
    prompt = judge.prompts[0]
    assert "REQUEST" in prompt and "THE-END" in prompt
    assert prompt.index("act-4") < prompt.index("act-11") < prompt.index("act-15")
    assert probs == {4: 0.2, 11: 0.7, 15: 0.1}


def test_finalist_prompt_respects_the_budget_with_a_giant_candidate():
    s = steps()
    s[7] = Step(step_id="7", type="assistant", timestamp=None, inputs={"t": "y" * 400_000})
    judge = ChoiceJudge({"7": 0.5, "8": 0.5})
    finalist_round(judge, s, [7, 8], budget_tokens=3_000)
    # candidates within budget; request and ending add at most ~2 x 1.5K on top
    assert approx_tokens(judge.prompts[0]) <= 3_000 + 3_200


def test_add_finalists_shortlists_top_k_and_records_the_pick():
    from alibi.evaluate import add_finalists

    row = {
        "trace_id": "t/1",
        "gold": 11,
        "pred": 4,
        "forward_scores": [0.1, 0.9],
        "forward_answers": [{}, {}],
        "backward_evidence": [
            {"chunk": 0, "p_chunk": 0.8, "p_steps": {"4": 0.6, "5": 0.1}, "p_checks": {}},
            {
                "chunk": 1,
                "p_chunk": 0.8,
                "p_steps": {"11": 0.5, "15": 0.4, "16": 0.05},
                "p_checks": {},
            },
        ],
    }
    judge = ChoiceJudge({"4": 0.2, "11": 0.7, "15": 0.1})
    out = add_finalists(row, steps(), judge, k=3, budget_tokens=5_000, rule="chapter_x_step")

    assert list(judge.questions[0][0].criteria) == ["4", "11", "15"]  # top-3, time order
    assert out["pred_v2"] == 4 and out["pred"] == 11
    assert out["finalists"]["probs"] == {"4": 0.2, "11": 0.7, "15": 0.1}
    assert out["finalists"]["gold_in_shortlist"] is True


def test_context_shows_neighbours_and_marks_the_candidate():
    judge = ChoiceJudge({"4": 0.5, "11": 0.5})
    finalist_round(judge, steps(), [4, 11], budget_tokens=5_000, context=1)
    prompt = judge.prompts[0]
    assert "act-3" in prompt and "act-5" in prompt and "act-10" in prompt and "act-12" in prompt
    assert ">>> CANDIDATE [step 4]" in prompt and ">>> CANDIDATE [step 11]" in prompt
    assert "act-7" not in prompt  # not a neighbour of any candidate


def test_context_blocks_still_respect_the_budget():
    s = steps()
    s[5] = Step(step_id="5", type="tool", timestamp=None, inputs={"t": "z" * 400_000})
    judge = ChoiceJudge({"4": 0.5, "11": 0.5})
    finalist_round(judge, s, [4, 11], budget_tokens=3_000, context=1)
    assert approx_tokens(judge.prompts[0]) <= 3_000 + 3_300


def test_add_finalists_can_pick_the_earliest_near_best():
    from alibi.evaluate import add_finalists

    row = {
        "trace_id": "t/1",
        "gold": 4,
        "pred": 11,
        "forward_scores": [0.1],
        "forward_answers": [{}],
        "backward_evidence": [
            {"chunk": 0, "p_chunk": 0.8, "p_steps": {"4": 0.6, "11": 0.5}, "p_checks": {}}
        ],
    }
    judge = ChoiceJudge({"4": 0.45, "11": 0.55})
    out = add_finalists(
        row,
        steps(),
        judge,
        k=2,
        budget_tokens=5_000,
        rule="chapter_x_step",
        pick="earliest_near_best",
    )
    assert out["pred"] == 4  # 0.45 >= 0.8 * 0.55
