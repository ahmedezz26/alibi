"""CUSUM drift detection over the per-chunk anomaly-score sequence (build-order step 5).

One-sided upper CUSUM: S_t = max(0, S_{t-1} + x_t - baseline - k); alarm when S_t > h,
then reset. Catches slow degradation that never pushes a single chunk over a threshold.
Defaults are uncalibrated placeholders: see PROJECT_BRIEF.md §8 (calibration risk).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_K = 0.2  # slack: per-chunk excess over baseline that's tolerated as noise
DEFAULT_H = 0.5  # decision threshold on the cumulative excess
DEFAULT_BASELINE = 0.0  # expected anomaly score of a normal chunk


@dataclass(frozen=True)
class ChangePoint:
    alarm: int  # chunk index where the statistic crossed h (detection time)
    onset: int  # estimated start of the drift: first chunk after S was last 0


def cusum_path(
    scores: Sequence[float], k: float = DEFAULT_K, baseline: float = DEFAULT_BASELINE
) -> np.ndarray:
    """The CUSUM statistic without alarm resets, for inspection/plotting."""
    path = np.zeros(len(scores))
    s = 0.0
    for i, x in enumerate(scores):
        s = max(0.0, s + x - baseline - k)
        path[i] = s
    return path


def detect_drift(
    scores: Sequence[float],
    k: float = DEFAULT_K,
    h: float = DEFAULT_H,
    baseline: float = DEFAULT_BASELINE,
) -> list[ChangePoint]:
    points: list[ChangePoint] = []
    s, onset = 0.0, 0
    for i, x in enumerate(scores):
        if s == 0.0:
            onset = i
        s = max(0.0, s + x - baseline - k)
        if s > h:
            points.append(ChangePoint(alarm=i, onset=onset))
            s = 0.0
    return points


def cusum_detector(scores: Sequence[float], k: float, h: float) -> list[int]:
    """Indices of detected change points (alarm times), per PROJECT_BRIEF.md §3.5."""
    return [cp.alarm for cp in detect_drift(scores, k, h)]
