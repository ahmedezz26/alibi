from alibi.drift import ChangePoint, cusum_detector, cusum_path, detect_drift


def test_single_spike_alarms_immediately():
    assert detect_drift([0.0, 0.0, 1.0, 0.0]) == [ChangePoint(alarm=2, onset=2)]


def test_slow_drift_below_any_threshold_is_caught():
    scores = [0.05] * 5 + [0.35] * 8  # no chunk ever exceeds 0.35
    first, second = detect_drift(scores, k=0.2, h=0.5)
    assert first == ChangePoint(alarm=8, onset=5)  # 4 chunks * 0.15 excess = 0.6 > 0.5
    assert second == ChangePoint(alarm=12, onset=9)  # drift persists after the reset


def test_noise_under_slack_never_alarms():
    assert detect_drift([0.1, 0.2, 0.15, 0.0, 0.2] * 10, k=0.2, h=0.5) == []


def test_resets_after_alarm_and_detects_again():
    assert cusum_detector([1.0, 0.0, 0.0, 1.0], k=0.2, h=0.5) == [0, 3]


def test_baseline_shifts_the_reference():
    assert detect_drift([0.5] * 5, k=0.1, h=0.5, baseline=0.4) == []


def test_path_is_nonnegative_cumulative_excess():
    assert cusum_path([0.0, 0.5, 0.5, 0.0], k=0.2).round(2).tolist() == [0.0, 0.3, 0.6, 0.4]
