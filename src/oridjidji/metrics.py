"""The handful of metrics the detectors and the calibration layer compute internally.

A leaf module on purpose: it imports nothing from the rest of the package, so the
calibration layer and the artefact loader can both depend on it without a cycle.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Scores = Sequence[float] | NDArray[np.floating[Any]]


Labels = Sequence[int] | NDArray[np.integer[Any]]


def _check(scores: Scores, labels: Labels) -> None:
    if len(scores) != len(labels):
        raise ValueError(f"scores and labels differ in length: {len(scores)} vs {len(labels)}")
    # `len(...) == 0` rather than `not scores`: the annotation above promises numpy support,
    # and a truthiness test on an array of more than one element raises "truth value is
    # ambiguous" instead. Every caller until M6 passed lists, so the promise went unchecked.
    if len(scores) == 0:
        raise ValueError("empty input")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("labels must be 0 (human) or 1 (machine)")


def auroc(positive: Scores, negative: Scores) -> float:
    """AUROC via the tie-corrected rank-sum (Mann-Whitney U) identity.

    Equals P(random machine score > random human score), with ties counted as half.
    Returns 0.5 for an empty group: no evidence is chance, not a perfect score.
    """
    # `len(...) == 0` rather than `not positive`: numpy arrays raise "truth value of an
    # array is ambiguous" on the latter, and every detector in `detectors/` scores into
    # arrays. The emptiness check is what was meant in either case.
    if len(positive) == 0 or len(negative) == 0:
        return 0.5
    combined = sorted([(v, 1) for v in positive] + [(v, 0) for v in negative])
    ranks = [0.0] * len(combined)
    index = 0
    while index < len(combined):
        stop = index
        while stop + 1 < len(combined) and combined[stop + 1][0] == combined[index][0]:
            stop += 1
        average_rank = (index + stop) / 2 + 1
        for position in range(index, stop + 1):
            ranks[position] = average_rank
        index = stop + 1
    positive_rank_sum = sum(r for r, (_, label) in zip(ranks, combined, strict=True) if label == 1)
    n_pos, n_neg = len(positive), len(negative)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def brier(scores: Scores, labels: Labels) -> float:
    """Mean squared error of calibrated probabilities.

    Rewards being right *and* being appropriately unsure, which is exactly the
    behaviour this project's abstention design depends on.
    """
    _check(scores, labels)
    return float(sum((s - y) ** 2 for s, y in zip(scores, labels, strict=True)) / len(scores))


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    mean_accuracy: float

    @property
    def gap(self) -> float:
        return abs(self.mean_accuracy - self.mean_confidence)


def calibration_bins(scores: Scores, labels: Labels, n_bins: int = 15) -> list[CalibrationBin]:
    """Equal-width confidence bins — the raw material of ECE and the reliability diagram.

    Returned rather than hidden inside `ece`, because the the specification ship-gate ("ECE > 0.10 ⇒
    fail loudly") is far easier to argue about when the bins are inspectable, and the
    reliability diagram plots exactly these.
    """
    _check(scores, labels)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for score, label in zip(scores, labels, strict=True):
        clamped = min(max(float(score), 0.0), 1.0)
        index = min(int(clamped * n_bins), n_bins - 1)
        buckets[index].append((clamped, label))

    bins: list[CalibrationBin] = []
    for index, bucket in enumerate(buckets):
        lower, upper = index / n_bins, (index + 1) / n_bins
        if not bucket:
            bins.append(CalibrationBin(lower, upper, 0, 0.0, 0.0))
            continue
        confidences = [c for c, _ in bucket]
        outcomes = [y for _, y in bucket]
        bins.append(
            CalibrationBin(
                lower=lower,
                upper=upper,
                count=len(bucket),
                mean_confidence=sum(confidences) / len(confidences),
                mean_accuracy=sum(outcomes) / len(outcomes),
            )
        )
    return bins


def ece(scores: Scores, labels: Labels, n_bins: int = 15) -> float:
    """Expected calibration error: count-weighted mean |accuracy - confidence|.

    15 bins by default, matching the specification's gate. Empty bins contribute nothing rather
    than counting as perfectly calibrated.
    """
    bins = calibration_bins(scores, labels, n_bins)
    total = sum(b.count for b in bins)
    if total == 0:
        return 0.0
    return sum(b.count * b.gap for b in bins) / total
