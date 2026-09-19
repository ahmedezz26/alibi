import pytest

from alibi.backward import COMBINE_RULES, ChunkEvidence, combine_scores

EVIDENCE = [
    ChunkEvidence(0, p_chunk=0.9, p_steps={0: 0.2, 1: 0.5, 2: 0.1}),
    ChunkEvidence(1, p_chunk=0.4, p_steps={3: 0.9, 4: 0.3}),
]
HEALTH = [0.1, 0.8]  # health rose sharply in chunk 1
SUSPECT = [{0: 0.1, 1: 0.1, 2: 0.8}, {3: 0.5, 4: 0.5}]


def best(rule):
    scores = combine_scores(EVIDENCE, HEALTH, SUSPECT, rule)
    return max(scores, key=lambda s: (scores[s], -s)), scores


def test_rules_are_a_small_fixed_set():
    assert set(COMBINE_RULES) == {
        "chapter_x_step",
        "step_only",
        "rise_prior",
        "forward_suspect",
        "step_plus_check",
        "commit_x_step_plus_check",
    }


def test_chapter_x_step_is_the_original_rule():
    step, scores = best("chapter_x_step")
    assert scores[1] == pytest.approx(0.45) and scores[3] == pytest.approx(0.36)
    assert step == 1


def test_step_only_ignores_the_chapter_answer():
    assert best("step_only")[0] == 3


def test_rise_prior_weights_chunks_where_health_jumped():
    step, scores = best("rise_prior")
    # chunk 0 rise = 0.1 -> weight 0.6; chunk 1 rise = 0.7 -> weight 1.2
    assert scores[1] == pytest.approx(0.9 * 0.5 * 0.6)
    assert scores[3] == pytest.approx(0.4 * 0.9 * 1.2)
    assert step == 3


def test_forward_suspect_averages_backward_and_forward_step_evidence():
    step, scores = best("forward_suspect")
    assert scores[2] == pytest.approx(0.9 * (0.1 + 0.8) / 2)
    assert step == 2


def test_overlapping_step_keeps_the_stronger_reading():
    ev = [ChunkEvidence(0, 0.5, {2: 0.4}), ChunkEvidence(1, 0.9, {2: 0.8})]
    scores = combine_scores(ev, [0.0, 0.0], [{}, {}], "chapter_x_step")
    assert scores[2] == pytest.approx(0.72)
