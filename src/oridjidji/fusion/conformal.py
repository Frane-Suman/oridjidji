"""Split conformal prediction with class-conditional (Mondrian) quantiles.

This module owns the one decision the system is forbidden to make by hand. The specification: the
inconclusive zone is *"not an arbitrary 0.4-0.6 band"* — it comes from a conformal quantile
that carries a finite-sample coverage guarantee, and that guarantee is what entitles the
system to abstain instead of guessing.

**What conformal gives, in one paragraph.** Fit on a calibration split the model never
trained on. For each class, measure how *nonconforming* its own members look, and take the
quantile that leaves at most a fraction alpha of them outside. A new text is then assigned
the set of labels it is not too nonconforming for. The guarantee is
``P(Y in C(X) | Y = k) >= 1 - alpha`` for each class k — distribution-free, finite-sample,
and conditional on the class rather than averaged over both. Averaged coverage is what lets
a detector hold 90% overall while covering the human class at 83%, and the human class is
the one where under-coverage means a person is falsely accused.

**Why the prediction set is the primitive and the verdict is a reading of it.** A set of
both labels means the evidence supports neither side; the empty set means the text is
atypical of both classes at once. Both abstain, but they are different facts, and
`is_atypical` keeps them distinguishable for the UI (the specification refuses rather than overclaims).

Implemented explicitly rather than imported: the specification names *conformal quantile computation and
prediction-set construction* as thesis-core.

COMPLEXITY BUDGET (CLAUDE.md, 307 lines). The quantiles, the prediction set, the versioned
artefact and the coverage reporting. They are one file because the guarantee only means
something end to end: a quantile without its coverage check is a number, and the specification needs the
abstention zone to carry evidence that it holds.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oridjidji.schemas import Verdict

Scores = NDArray[np.float64]
Labels = NDArray[np.int_]

#: Pre-registered abstention levels. Fitting at an unregistered alpha after seeing
#: the numbers would be choosing the abstention rate to suit the result.
ALPHAS: tuple[float, ...] = (0.05, 0.10, 0.20)

HUMAN, MACHINE = 0, 1


def _nonconformity(scores: Scores, label: int) -> Scores:
    """How poorly each score conforms to ``label``, on a scale where larger is worse.

    The score is P(machine). A machine text should score high, so its nonconformity is
    ``1 - p``; a human text should score low, so its nonconformity is ``p``. Nothing deeper
    is going on — conformal only needs an ordering, and this is the ordering the detector
    already provides.
    """
    return (1.0 - scores) if label == MACHINE else scores.copy()


def _quantile_index(n: int, alpha: float) -> int:
    """The ``ceil((n + 1)(1 - alpha))``-th smallest value, as a 1-based rank.

    **The +1 is the whole finite-sample correction and it fails silently without it.**
    Dropping it returns the plain empirical quantile: a slightly smaller threshold, a
    slightly narrower inconclusive zone, and coverage below the alpha that was promised.
    The artefact still looks entirely ordinary — which is why the rank is computed here,
    in one place, with a test pinning the arithmetic.
    """
    return math.ceil((n + 1) * (1.0 - alpha))


@dataclass(frozen=True)
class ConformalQuantiles:
    """Fitted Mondrian quantiles, and the identity that makes them mean something.

    A bare quantile is a number without units: the same value means different things under
    a different detector, a different calibration set or a different alpha. The M5 threshold
    artefact carries its model pair for the same reason, and the specification puts both into one
    versioned bundle.
    """

    alpha: float
    q_human: float
    q_machine: float
    n_human: int
    n_machine: int
    calibration_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def underpowered(self) -> tuple[int, ...]:
        """Classes whose calibration set was too small to support ``alpha`` at all.

        ``ceil((n+1)(1-alpha)) > n`` has no finite solution among the observed values, so the
        quantile is infinite and that label is always in the prediction set. Reporting which
        classes rather than a bare flag is what lets a caller say *why* everything came back
        inconclusive.
        """
        return tuple(
            label
            for label, q in ((HUMAN, self.q_human), (MACHINE, self.q_machine))
            if math.isinf(q)
        )

    def is_atypical(self, score: float) -> bool:
        """True when the prediction set is empty: the text fits neither class.

        Distinct from an ambiguous score, which fits both. Both abstain; only this one says
        the input is unlike the calibration data.
        """
        return not prediction_set(score, self)

    def to_json(self) -> str:
        payload = {
            "alpha": self.alpha,
            "q_human": None if math.isinf(self.q_human) else self.q_human,
            "q_machine": None if math.isinf(self.q_machine) else self.q_machine,
            "n_human": self.n_human,
            "n_machine": self.n_machine,
            "calibration_hash": self.calibration_hash,
            "created_at": self.created_at,
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> ConformalQuantiles:
        """Rebuild from disk, refusing a file that is not what it claims to be.

        ``null`` is how an infinite quantile survives JSON, and it round-trips back to
        infinity rather than to a default: an underpowered fit must stay underpowered.
        Unknown keys are ignored so a later addition is not a breaking change for anything
        already serving.
        """
        data: dict[str, Any] = json.loads(payload)
        required = ("alpha", "q_human", "q_machine", "n_human", "n_machine")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"conformal artefact is missing {', '.join(missing)}")
        return cls(
            alpha=float(data["alpha"]),
            q_human=math.inf if data["q_human"] is None else float(data["q_human"]),
            q_machine=math.inf if data["q_machine"] is None else float(data["q_machine"]),
            n_human=int(data["n_human"]),
            n_machine=int(data["n_machine"]),
            calibration_hash=str(data.get("calibration_hash", "")),
            created_at=str(data.get("created_at", "")),
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> ConformalQuantiles:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def fit_mondrian(
    scores: Scores,
    labels: Labels,
    *,
    alpha: float,
    calibration_hash: str = "",
) -> ConformalQuantiles:
    """Fit class-conditional quantiles on a calibration split.

    Never on train (the model has seen it, so nonconformity is optimistic) and never on
    test (the only data left that nothing has touched). An earlier decision names fitting an operating
    point on test as the anti-pattern; this is the same rule for the abstention zone.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if alpha not in ALPHAS:
        raise ValueError(f"alpha={alpha} is not one of the registered levels {ALPHAS}")
    if len(scores) != len(labels):
        raise ValueError(f"{len(scores)} scores against {len(labels)} labels")
    if not (np.any(labels == HUMAN) and np.any(labels == MACHINE)):
        raise ValueError("calibration needs both classes: a quantile per class is the point")

    quantiles: dict[int, float] = {}
    counts: dict[int, int] = {}
    for label in (HUMAN, MACHINE):
        own = _nonconformity(scores[labels == label], label)
        n = len(own)
        counts[label] = n
        rank = _quantile_index(n, alpha)
        # rank > n means no observed value is large enough to give the guarantee. Infinity
        # is the honest answer: that label is always plausible, so nothing is ever confident.
        quantiles[label] = math.inf if rank > n else float(np.sort(own)[rank - 1])

    return ConformalQuantiles(
        alpha=alpha,
        q_human=quantiles[HUMAN],
        q_machine=quantiles[MACHINE],
        n_human=counts[HUMAN],
        n_machine=counts[MACHINE],
        calibration_hash=calibration_hash,
    )


def prediction_set(score: float, quantiles: ConformalQuantiles) -> frozenset[int]:
    """Labels this score is not too nonconforming for — the conformal primitive.

    Four outcomes, and all four are meaningful: one label is a side, both labels are an
    ambiguous text, and the empty set is a text atypical of everything the detector was
    calibrated on.
    """
    included = set()
    if float(score) <= quantiles.q_human:
        included.add(HUMAN)
    if 1.0 - float(score) <= quantiles.q_machine:
        included.add(MACHINE)
    return frozenset(included)


def verdict_for(score: float, quantiles: ConformalQuantiles) -> Verdict:
    """Read a prediction set as one of the specification's three states.

    Anything but a singleton abstains. The system never emits an absolute verdict, and a set
    that holds both labels — or neither — is not evidence for a side.
    """
    labels = prediction_set(score, quantiles)
    if labels == frozenset({MACHINE}):
        return Verdict.LIKELY_AI
    if labels == frozenset({HUMAN}):
        return Verdict.LIKELY_HUMAN
    return Verdict.INCONCLUSIVE


def empirical_coverage(
    scores: Scores, labels: Labels, quantiles: ConformalQuantiles
) -> dict[int, float]:
    """Share of each class whose true label was in its prediction set.

    The number M6 is accepted on (the specification: within +-0.03 of nominal). Conformal guarantees
    coverage at or above 1-alpha, so the failure that matters is *under*-coverage: a
    detector abstaining less than it promised is over-confident on exactly the texts where
    someone would be accused.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    covered: dict[int, float] = {}
    for label in (HUMAN, MACHINE):
        own = scores[labels == label]
        if not len(own):
            covered[label] = float("nan")
            continue
        hits = sum(1 for score in own if label in prediction_set(float(score), quantiles))
        covered[label] = hits / len(own)
    return covered


@dataclass(frozen=True)
class CoveragePoint:
    """One row of the coverage-accuracy trade-off."""

    alpha: float
    committed_fraction: float
    accuracy_when_committed: float
    empirical_coverage: dict[int, float]


def coverage_accuracy_curve(
    scores: Scores, labels: Labels, alphas: tuple[float, ...] = ALPHAS
) -> list[CoveragePoint]:
    """Abstention rate against accuracy on what remains, at each registered alpha.

    The specification requires both axes because abstention is only worth its cost if accuracy on the
    committed subset beats accuracy on everything. A layer that abstains and is no more
    accurate for it is spending recall on nothing.

    Fitted and evaluated on the same array here — the caller passes a held-out split when
    the number is going in a report. This function computes the trade-off; it does not
    police which data it was handed.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    curve: list[CoveragePoint] = []
    for alpha in alphas:
        quantiles = fit_mondrian(scores, labels, alpha=alpha)
        committed = [
            (float(score), int(label))
            for score, label in zip(scores, labels, strict=True)
            if len(prediction_set(float(score), quantiles)) == 1
        ]
        if committed:
            correct = sum(
                1
                for score, label in committed
                if verdict_for(score, quantiles)
                is (Verdict.LIKELY_AI if label == MACHINE else Verdict.LIKELY_HUMAN)
            )
            accuracy = correct / len(committed)
        else:
            accuracy = 0.0
        curve.append(
            CoveragePoint(
                alpha=alpha,
                committed_fraction=len(committed) / len(scores),
                accuracy_when_committed=accuracy,
                empirical_coverage=empirical_coverage(scores, labels, quantiles),
            )
        )
    return curve
