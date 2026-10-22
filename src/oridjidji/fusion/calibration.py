"""Probability calibration: temperature, Platt, isotonic.

A detector score is not a probability. The encoder emits a softmax that reads 0.98 on texts
it is right about 90% of the time, and the specification promises the user a *calibrated* probability with
a conformal three-state verdict. Closing that gap is this module's whole job, and it sits between the fused score
and the conformal layer: calibration makes the number mean something, conformal decides when
even a calibrated number is not enough to commit.

**Why selection uses two metrics.** A constant predictor that always answers the base rate is
*perfectly calibrated* — ECE near zero, no information whatsoever. Ranking on ECE alone ships
it. Brier is what separates calibrated-and-informative from calibrated-and-empty, because it
punishes the missing sharpness. So the registered rule is: among methods that pass the ECE
ceiling, take the lowest Brier (the specification's "select by 15-bin ECE + Brier").

ECE binning and the Brier score are thesis-core and already implemented explicitly in
`evaluation.metrics`; this module fits the transforms and applies the selection rule, and
does not reimplement the metrics.

COMPLEXITY BUDGET (392 lines). Three fitted transforms, their shared artefact,
and the out-of-fold selection that keeps the specification's ECE gate able to fire at all. The three
belong together because they are chosen against each other — separating them would hide the
comparison that is the module's actual output.
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

from oridjidji.metrics import brier, calibration_bins, ece

Scores = NDArray[np.float64]
Labels = NDArray[np.int_]

#: The three registered transforms, fitted and reported together.
METHODS: tuple[str, ...] = ("temperature", "platt", "isotonic")

#: the specification's hard gate. Above this the artefact is refused rather than shipped.
ECE_CEILING = 0.10

_EPS = 1e-6


class CalibrationError(RuntimeError):
    """Raised when no calibrator clears the ECE ceiling.

    An exception rather than a warning because the specification says *refuse to ship*: a bundle carrying
    a bad calibrator hands the user a confident-looking probability that is not one, and
    nothing downstream can tell the difference.
    """


def _logit(p: Scores) -> Scores:
    clipped = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(x: Scores) -> Scores:
    """The logistic function, written so it cannot overflow.

    `1/(1 + exp(-x))` overflows for large negative x. The value it returns is right — inf in
    the denominator gives 0 — but a RuntimeWarning in the seam that turns scores into
    probabilities is noise in exactly the place where real numerical trouble would announce
    itself the same way. The branch below evaluates the tail that cannot overflow.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    tail = np.exp(x[~positive])
    out[~positive] = tail / (1.0 + tail)
    return out


def _check(scores: Scores, labels: Labels) -> tuple[Scores, Labels]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if len(scores) != len(labels):
        raise ValueError(f"{len(scores)} scores against {len(labels)} labels")
    if not (np.any(labels == 0) and np.any(labels == 1)):
        raise ValueError("calibration needs both classes present")
    return scores, labels


@dataclass(frozen=True)
class Calibrator:
    """A fitted transform from detector score to probability, and its identity.

    ``params`` holds whatever the method needs: one temperature, two Platt coefficients, or
    the isotonic step function as parallel x/y arrays. Keeping them in one JSON-serialisable
    dict is what lets all three live in the versioned bundle the specification requires without a
    per-method serialisation format.
    """

    method: str
    params: dict[str, Any]
    ece: float
    brier: float
    n_calibration: int
    calibration_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def certainty_floor(self) -> float:
        """The smallest probability this calibration set can justify, and its mirror.

        **Derived, not chosen.** After observing `n` calibration points that all fall on one
        side, a Beta(1, 1) posterior puts the next observation at `1/(n + 2)` — so `n` points
        cannot support a claim finer than that, and asserting 0 or 1 claims infinite
        evidence from a finite sample. With 634 records the floor is 0.0016.

        The specification's first non-negotiable is that the system never emits an absolute verdict, and a
        probability of exactly 1 is one: the API reported `p_ai=1.0000` with the interval
        `(1.0000, 1.0000)` before this existed. Isotonic is what reaches the endpoints — it
        interpolates between observed frequencies, and on a near-separable calibration set
        those frequencies *are* 0 and 1 — but the floor applies to every method, because a
        saturating sigmoid gets there too and the guarantee should not depend on which
        method won.
        """
        return 1.0 / (max(self.n_calibration, 1) + 2)

    def apply(self, scores: Scores) -> Scores:
        """Map raw scores to calibrated probabilities, never to certainty."""
        values = np.asarray(scores, dtype=np.float64)
        if self.method == "temperature":
            calibrated = _sigmoid(_logit(values) / float(self.params["temperature"]))
        elif self.method == "platt":
            a, b = float(self.params["a"]), float(self.params["b"])
            calibrated = _sigmoid(a * _logit(values) + b)
        elif self.method == "isotonic":
            xs = np.asarray(self.params["x"], dtype=np.float64)
            ys = np.asarray(self.params["y"], dtype=np.float64)
            # np.interp clamps outside the fitted range, which is the correct extrapolation
            # for a monotone step function: beyond the calibration data the honest answer is
            # the nearest thing that was observed, not a linear continuation off the end.
            calibrated = np.interp(values, xs, ys)
        else:
            raise ValueError(f"unknown calibration method {self.method!r}")
        floor = self.certainty_floor
        return np.clip(calibrated, floor, 1.0 - floor)

    def to_json(self) -> str:
        payload = {
            "method": self.method,
            "params": {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in self.params.items()
            },
            "ece": self.ece,
            "brier": self.brier,
            "n_calibration": self.n_calibration,
            "calibration_hash": self.calibration_hash,
            "created_at": self.created_at,
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> Calibrator:
        data: dict[str, Any] = json.loads(payload)
        required = ("method", "params", "ece", "brier", "n_calibration")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"calibrator is missing {', '.join(missing)}")
        return cls(
            method=str(data["method"]),
            params=dict(data["params"]),
            ece=float(data["ece"]),
            brier=float(data["brier"]),
            n_calibration=int(data["n_calibration"]),
            calibration_hash=str(data.get("calibration_hash", "")),
            created_at=str(data.get("created_at", "")),
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> Calibrator:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def _fit_temperature(scores: Scores, labels: Labels) -> dict[str, Any]:
    """One parameter: divide the logits by T before the sigmoid.

    Fitted by minimising negative log-likelihood over a bounded scalar. T > 1 softens an
    over-confident detector, T < 1 sharpens an under-confident one. Monotone, so it never
    changes the ranking — only how far from the middle the numbers sit.
    """
    from scipy.optimize import minimize_scalar

    logits = _logit(scores)
    targets = labels.astype(np.float64)

    def nll(temperature: float) -> float:
        probabilities = np.clip(_sigmoid(logits / temperature), _EPS, 1.0 - _EPS)
        return float(
            -np.mean(targets * np.log(probabilities) + (1 - targets) * np.log(1 - probabilities))
        )

    best = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return {"temperature": float(best.x)}


def _fit_platt(scores: Scores, labels: Labels) -> dict[str, Any]:
    """Two parameters: a logistic regression on the logit, so a slope and an intercept.

    Strictly more general than temperature scaling, which is the a-only special case with
    b = 0. Also monotone.
    """
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(_logit(scores).reshape(-1, 1), labels)
    return {"a": float(model.coef_[0][0]), "b": float(model.intercept_[0])}


def _fit_isotonic(scores: Scores, labels: Labels) -> dict[str, Any]:
    """A monotone step function, free to bend where the parametric forms cannot.

    The most flexible of the three and the readiest to overfit a small calibration split —
    which is why all three are fitted and compared rather than isotonic being assumed best.
    Stored as parallel x/y arrays so it serialises without pickling an estimator.
    """
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(scores, labels)
    xs = np.unique(scores)
    return {"x": xs.tolist(), "y": model.predict(xs).tolist()}


_FITTERS = {
    "temperature": _fit_temperature,
    "platt": _fit_platt,
    "isotonic": _fit_isotonic,
}


def fit_calibrator(
    scores: Scores,
    labels: Labels,
    *,
    method: str,
    calibration_hash: str = "",
) -> Calibrator:
    """Fit one transform on the calibration split and measure what it achieved.

    Fitted on calibration, never on train (the model has seen it, so its confidence there is
    not the confidence it will have in the wild) and never on test.
    """
    if method not in _FITTERS:
        raise ValueError(f"unknown calibration method {method!r}; registered: {METHODS}")
    scores, labels = _check(scores, labels)
    params = _FITTERS[method](scores, labels)
    fitted = Calibrator(
        method=method,
        params=params,
        ece=0.0,
        brier=0.0,
        n_calibration=len(scores),
        calibration_hash=calibration_hash,
    )
    calibrated = fitted.apply(scores)
    return Calibrator(
        method=method,
        params=params,
        ece=ece(calibrated, labels),
        brier=brier(calibrated, labels),
        n_calibration=len(scores),
        calibration_hash=calibration_hash,
    )


@dataclass(frozen=True)
class CalibrationChoice:
    """The selected calibrator and every candidate it beat.

    Reporting the losers is the same habit L3 applies to threshold rules: a reader should be
    able to see that isotonic won by 0.002 rather than by a mile, because that changes how
    much weight the choice carries.
    """

    selected: Calibrator
    considered: tuple[Calibrator, ...]
    #: The selected method's out-of-fold calibrated probabilities, one per calibration
    #: record, in input order. **The conformal layer must fit on these, not on the
    #: calibrator's in-sample output.** Fitting both stages on the same predictions lets the
    #: quantiles inherit the calibrator's optimism, and the coverage guarantee is then
    #: promised against data the calibrator had already seen — which is the one thing
    #: conformal's finite-sample argument assumes did not happen.
    out_of_fold: Scores
    #: Committed fraction at `INTERIOR_ALPHA` per method that cleared the ECE gate. Reported
    #: rather than merely used: a reader should be able to see that isotonic was excluded for
    #: committing on everything, not for being badly calibrated — it was the best calibrated.
    committed: dict[str, float] = field(default_factory=dict)
    interior_ceiling: float = 1.0

    @property
    def interior_satisfiable(self) -> bool:
        """Whether any candidate left an inconclusive zone. False means the selection fell
        back to *abstains most* and the artefact carries a caveat."""
        return any(v <= self.interior_ceiling for v in self.committed.values())


def _out_of_fold(
    scores: Scores, labels: Labels, method: str, folds: int
) -> tuple[float, float, Scores]:
    """Estimate a method's ECE and Brier on data it was not fitted on, and return the
    predictions that produced them.

    **Selecting on in-sample numbers would hand the choice to isotonic every time.** It is
    the most flexible of the three and, fitted and scored on the same split, it can drive
    ECE to almost zero by memorising the calibration set — which also makes the specification's ECE gate
    toothless, since the gate would never fire. Stratified k-fold keeps each fold's class
    balance, so a fold cannot end up with one class and an undefined fit.
    """
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    predictions = np.zeros(len(scores), dtype=np.float64)
    for train_index, test_index in splitter.split(scores.reshape(-1, 1), labels):
        fitted = fit_calibrator(scores[train_index], labels[train_index], method=method)
        predictions[test_index] = fitted.apply(scores[test_index])
    return ece(predictions, labels), brier(predictions, labels), predictions


#: The alpha the interior criterion is judged at, and how far past nominal a calibrator may
#: commit before it is disqualified. The specification's tolerance, reused rather than reinvented.
INTERIOR_ALPHA = 0.10
INTERIOR_TOLERANCE = 0.03


def committed_fraction(probabilities: Scores, labels: Labels, alpha: float) -> float:
    """Share of records the conformal layer would commit on, at ``alpha``.

    A separating detector commits on roughly ``1 - alpha``: the records it does not commit on
    are the ones its own class quantile excludes. A calibrator that commits on *everything*
    has produced a score with no interior, and the abstention zone the specification pays for is empty.
    """
    from oridjidji.fusion.conformal import fit_mondrian, prediction_set

    quantiles = fit_mondrian(probabilities, labels, alpha=alpha)
    singletons = sum(1 for p in probabilities if len(prediction_set(float(p), quantiles)) == 1)
    return singletons / len(probabilities)


def select_calibrator(
    scores: Scores,
    labels: Labels,
    *,
    ece_ceiling: float = ECE_CEILING,
    calibration_hash: str = "",
    folds: int = 5,
) -> CalibrationChoice:
    """Fit all three, gate on out-of-fold ECE, take the lowest out-of-fold Brier.

    The gate comes first and is absolute: if nothing clears the ceiling the detector
    is not calibratable on this data and the artefact is refused rather than shipped with a
    caveat nobody downstream can read.

    Every reported metric is **out-of-fold**, and the returned calibrator is then refitted on
    the whole split — the standard arrangement, and the only one where the numbers describe
    what the calibrator will do to text it has not seen. `fit_calibrator` on its own still
    reports in-sample metrics; that is what it is for, and why selection does not use it.
    """
    scores, labels = _check(scores, labels)
    scored: list[Calibrator] = []
    predictions: dict[str, Scores] = {}
    for method in METHODS:
        out_ece, out_brier, out_predictions = _out_of_fold(scores, labels, method, folds)
        predictions[method] = out_predictions
        refitted = fit_calibrator(scores, labels, method=method, calibration_hash=calibration_hash)
        scored.append(
            Calibrator(
                method=method,
                params=refitted.params,
                ece=out_ece,
                brier=out_brier,
                n_calibration=len(scores),
                calibration_hash=calibration_hash,
                created_at=refitted.created_at,
            )
        )
    candidates: tuple[Calibrator, ...] = tuple(scored)
    eligible = [candidate for candidate in candidates if candidate.ece <= ece_ceiling]
    if not eligible:
        best = min(candidates, key=lambda candidate: candidate.ece)
        raise CalibrationError(
            f"no calibrator reached the ECE ceiling {ece_ceiling}: best was "
            f"{best.method} at ECE {best.ece:.4f}. The specification refuses to ship this artefact — "
            "a probability that is not calibrated is not a probability."
        )

    # --- the interior criterion (an earlier decision, amending the specification) -------------------------------
    #
    # **ECE and Brier both reward a step function, and neither can see what it destroys.**
    # ECE alone is fooled by a constant predictor at the base rate; Brier catches that by
    # punishing lost sharpness. But a staircase is *maximally* sharp, so it wins on both —
    # and on the fused score isotonic won by 0.0006 of Brier while leaving 0 of 634 records
    # with 0.05 < p < 0.95 and committing on 100% of them. The specification buys the right to abstain
    # with a coverage guarantee; that guarantee still held at 0.997 against a nominal 0.90
    # while abstention never once fired. Compliant, and hollow.
    #
    # So a third criterion, applied to every layer: a calibrator must leave an inconclusive
    # zone behind. Judged on each method's OWN out-of-fold predictions, because the quantity
    # is about generalisation, not about how tightly a method can memorise its fold.
    commits = {
        candidate.method: committed_fraction(predictions[candidate.method], labels, INTERIOR_ALPHA)
        for candidate in eligible
    }
    ceiling = (1.0 - INTERIOR_ALPHA) + INTERIOR_TOLERANCE
    preserving = [c for c in eligible if commits[c.method] <= ceiling]
    if preserving:
        selected = min(preserving, key=lambda candidate: candidate.brier)
    else:
        # Nothing left an interior. Refusing outright would leave the system with no
        # probability at all, which is worse than a flagged one — so take the calibrator that
        # abstains MOST and let the report say the criterion was unsatisfiable. Silently
        # falling back to lowest-Brier would reinstate exactly the behaviour being amended.
        selected = min(eligible, key=lambda candidate: commits[candidate.method])

    return CalibrationChoice(
        selected=selected,
        considered=candidates,
        out_of_fold=predictions[selected.method],
        committed=dict(commits),
        interior_ceiling=ceiling,
    )


@dataclass(frozen=True)
class ReliabilityRow:
    """One bin of the reliability diagram."""

    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_rate: float

    @property
    def gap(self) -> float:
        """Predicted minus observed — positive means over-confident."""
        return self.mean_predicted - self.observed_rate


def reliability_table(scores: Scores, labels: Labels, n_bins: int = 15) -> list[ReliabilityRow]:
    """The reliability diagram as a table, which is what a thesis can print.

    Empty bins are dropped rather than reported as zero: a bin nothing landed in says
    nothing about calibration, and drawing it at the origin invents a point.
    """
    return [
        ReliabilityRow(
            lower=bin_.lower,
            upper=bin_.upper,
            count=bin_.count,
            mean_predicted=bin_.mean_confidence,
            observed_rate=bin_.mean_accuracy,
        )
        for bin_ in calibration_bins(scores, labels, n_bins=n_bins)
        if bin_.count > 0
    ]


def is_well_calibrated(scores: Scores, labels: Labels, ceiling: float = ECE_CEILING) -> bool:
    """The specification's acceptance check, as one call: ECE at or under the ceiling."""
    return bool(ece(np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=int)) <= ceiling)


__all__ = [
    "ECE_CEILING",
    "METHODS",
    "CalibrationChoice",
    "CalibrationError",
    "Calibrator",
    "ReliabilityRow",
    "fit_calibrator",
    "is_well_calibrated",
    "math",
    "reliability_table",
    "select_calibrator",
]
