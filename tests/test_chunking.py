from itertools import pairwise

import pytest

from alibi.chunking import chunk_trace
from alibi.types import Step


def make_steps(n):
    return [Step(step_id=str(i), type="tool", timestamp=None, inputs={"i": i}) for i in range(n)]


def ten_tokens_each(_text):
    return 10


def test_all_steps_covered_in_order():
    steps = make_steps(25)
    chunks = chunk_trace(steps, max_tokens=50, overlap_tokens=20, count_tokens=ten_tokens_each)
    assert chunks[0].start == 0
    assert chunks[-1].end == 25
    for a, b in pairwise(chunks):
        assert b.start < a.end  # overlapping
        assert b.start > a.start  # forward progress


def test_windows_respect_budget_and_overlap():
    chunks = chunk_trace(
        make_steps(25), max_tokens=50, overlap_tokens=20, count_tokens=ten_tokens_each
    )
    assert all(c.tokens <= 50 for c in chunks)
    for a, b in pairwise(chunks):
        assert a.end - b.start == 2  # 20 overlap tokens / 10 per step


def test_single_window_when_trace_fits():
    chunks = chunk_trace(
        make_steps(3), max_tokens=100, overlap_tokens=10, count_tokens=ten_tokens_each
    )
    assert len(chunks) == 1
    assert (chunks[0].start, chunks[0].end) == (0, 3)


def test_oversized_step_gets_own_chunk():
    chunks = chunk_trace(
        make_steps(3), max_tokens=5, overlap_tokens=2, count_tokens=ten_tokens_each
    )
    assert [(c.start, c.end) for c in chunks] == [(0, 1), (1, 2), (2, 3)]


def test_empty_trace():
    assert chunk_trace([], max_tokens=10, overlap_tokens=1) == []


def test_overlap_must_be_smaller_than_budget():
    with pytest.raises(ValueError):
        chunk_trace(make_steps(2), max_tokens=10, overlap_tokens=10)


def test_unicode_escapes_count_as_several_tokens():
    # json.dumps writes non-ASCII as \uXXXX; measured on Jev at ~4.75 real tokens each.
    from alibi.chunking import approx_tokens

    assert approx_tokens("\\u9996" * 100) == 500
    assert approx_tokens("abcd" * 100 + "\\u9996" * 10) == 150


def test_render_step_keeps_non_ascii_text_readable():
    from alibi.chunking import render_step
    from alibi.types import Step

    out = render_step(0, Step(step_id="0", type="user", timestamp=None, inputs={"q": "北京 café"}))
    assert "北京 café" in out and "\\u" not in out


def test_non_ascii_characters_count_one_token_each():
    # Measured on Jev: ~1.01 real tokens per Chinese character.
    from alibi.chunking import approx_tokens

    assert approx_tokens("北" * 400) == 400
    assert approx_tokens("abcd" * 100 + "北" * 50) == 150


def test_oversized_step_is_clipped_to_fit_the_window():
    from alibi.chunking import approx_tokens, chunk_trace
    from alibi.types import Step

    huge = Step(
        step_id="1", type="tool", timestamp=None, inputs={"out": "HEAD" + "7" * 400_000 + "TAIL"}
    )
    steps = [Step(step_id="0", type="user", timestamp=None, inputs={"q": "hi"}), huge]
    chunks = chunk_trace(steps, max_tokens=1_000, overlap_tokens=100)

    assert all(c.tokens <= 1_000 for c in chunks)
    big = next(c for c in chunks if "[step 1]" in c.text)
    assert "HEAD" in big.text and "TAIL" in big.text and "omitted" in big.text
    assert approx_tokens(big.text) <= 1_000


def test_digits_count_one_token_each():
    # Measured on Jev: a numeric tool output (70% digits) cost ~1 real token per char.
    from alibi.chunking import approx_tokens

    assert approx_tokens("12345" * 100) == 500
    assert approx_tokens("abcd" * 100 + "9" * 50) == 150
