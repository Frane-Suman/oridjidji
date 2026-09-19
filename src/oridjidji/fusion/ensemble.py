"""Fuse several agents into one score.

Each agent answers on its own scale and in its own direction — the encoder emits a softmax,
Binoculars a ratio where **lower** means machine, Fast-DetectGPT a z-statistic where **higher**
does, A4 a logistic output. Averaging those is not a probability, it is a number with no
units, which is why `_document_score` abstains on more than one agent until this exists.

**Orientation happens before anything else, and it is the part most likely to be got wrong.**
A detector entering the ensemble with its sign inverted does not fail; it drags the fusion
toward the opposite answer while every test still passes. The specification calls the flipped sign the
classic bug in this domain, and this is its third possible entry point after the score itself
and the threshold artefact.

**Fitted on calibration, not train, and that is a deviation with a reason.** the specification says train.
But the shipped encoder trained on pooled train+dev, so its scores there are optimistic while
the zero-shot agents' are not — a regression fitted on train would learn *trust A1 completely*,
which is a fact about the split rather than about the detectors. Calibration is the only split
no agent has seen.

**Missing agents are mean-imputed with an indicator**, the second of the two options the specification
offers. Refitting per subset would mean a different model for every combination of failures,
and the specification promises the ensemble keeps answering when an agent dies — including combinations
never seen during fitting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from oridjidji.schemas import ScoreDirection

Matrix = NDArray[np.float64]

#: Value written into the indicator column when an agent could not score.
MISSING = float("nan")


#: Agents that answer and are shown, but do not enter the fused score.
#:
#: **This is E5's registered decision rule applied, not a preference.** E5 fixed in advance:
#: *"the linguistic agent contributes to the fused score only if the CI on Δ excludes 0 in its
#: favour. Otherwise it ships as an explainability layer only."* Measured on both splits, the
#: CI spans zero — +0.0017 [-0.0010, +0.0062] on calibration, +0.0013 [-0.0056, +0.0085] on
#: dev — so A4 is displayed and explained but not fused.
#:
#: **E5b then found harm, not just absence of evidence.** Applied unchanged to 329
#: unseen-generator records, the same ensemble loses **0.0111 AUROC** against the encoder
#: alone, CI [-0.0190, -0.0046], clear of zero. In distribution A4 looked harmless and mildly
#: positive; outside it, it degrades the system exactly where detection is hardest.
#:
#: So this set now rests on a measurement rather than on a null result, and the reversal is
#: no longer one line: a later roster (A2 and A3 are pending) would have to clear E5b's
#: out-of-distribution comparison, not merely E5's in-distribution one. Fusion weights fitted
#: on in-distribution data inherit that distribution's failure modes and the ensemble has no
#: mechanism to notice.
EXPLAINABILITY_ONLY: frozenset[str] = frozenset({"linguistic"})


@dataclass(frozen=True)
class AgentColumn:
    """One agent's place in the feature vector, and how to read its raw score."""

    name: str
    direction: ScoreDirection
    mean: float
    scale: float

    def oriented(self, score: float) -> float:
        """Raw score on a shared *higher means machine* axis, then standardised.

        The orientation flip happens here rather than at the caller so there is exactly one
        place it can be wrong, and one place a test can pin it.
        """
        value = -score if self.direction is ScoreDirection.LOWER_IS_AI else score
        centre = -self.mean if self.direction is ScoreDirection.LOWER_IS_AI else self.mean
        return (value - centre) / (self.scale or 1.0)


@dataclass(frozen=True)
class Ensemble:
    """Fitted fusion weights, and everything needed to apply them to a live request."""

    columns: tuple[AgentColumn, ...]
    coefficients: dict[str, float]
    #: One per agent, multiplying its "this agent was unavailable" indicator.
    missing_coefficients: dict[str, float]
    intercept: float
    n_calibration: int
    calibration_hash: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def agent_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def score(self, scores: dict[str, float | None]) -> float:
        """Fuse one document's agent scores into a single machine-probability.

        An agent absent from ``scores``, or present with ``None``, contributes its mean —
        which after standardising is zero — and turns on its indicator. So a failed agent
        moves the answer by exactly the amount the fit learned failure is worth, rather than
        by whatever its last value happened to be.
        """
        total = self.intercept
        for column in self.columns:
            value = scores.get(column.name)
            if value is None:
                total += self.missing_coefficients.get(column.name, 0.0)
                continue
            total += self.coefficients.get(column.name, 0.0) * column.oriented(float(value))
        return float(1.0 / (1.0 + np.exp(-np.clip(total, -60.0, 60.0))))

    def contributions(self, scores: dict[str, float | None]) -> list[tuple[str, float]]:
        """Signed per-agent contribution to this document's score, largest first.

        The specification requires the interpretation of fusion coefficients to be this project's own
        work, and for a standardised linear model the contribution is exact: coefficient
        times oriented, standardised value. This is what the UI's per-agent breakdown reads.
        """
        out = []
        for column in self.columns:
            value = scores.get(column.name)
            if value is None:
                out.append((column.name, self.missing_coefficients.get(column.name, 0.0)))
                continue
            out.append(
                (
                    column.name,
                    self.coefficients.get(column.name, 0.0) * column.oriented(float(value)),
                )
            )
        return sorted(out, key=lambda pair: -abs(pair[1]))

    def to_json(self) -> str:
        return json.dumps(
            {
                "columns": [
                    {
                        "name": c.name,
                        "direction": str(c.direction),
                        "mean": c.mean,
                        "scale": c.scale,
                    }
                    for c in self.columns
                ],
                "coefficients": self.coefficients,
                "missing_coefficients": self.missing_coefficients,
                "intercept": self.intercept,
                "n_calibration": self.n_calibration,
                "calibration_hash": self.calibration_hash,
                "created_at": self.created_at,
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str) -> Ensemble:
        data: dict[str, Any] = json.loads(payload)
        required = ("columns", "coefficients", "intercept", "n_calibration")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"ensemble is missing {', '.join(missing)}")
        return cls(
            columns=tuple(
                AgentColumn(
                    name=str(c["name"]),
                    direction=ScoreDirection(c["direction"]),
                    mean=float(c["mean"]),
                    scale=float(c["scale"]),
                )
                for c in data["columns"]
            ),
            coefficients=dict(data["coefficients"]),
            missing_coefficients=dict(data.get("missing_coefficients", {})),
            intercept=float(data["intercept"]),
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
    def load(cls, path: str | Path) -> Ensemble:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def fit(
    rows: list[dict[str, float | None]],
    labels: list[int],
    directions: dict[str, ScoreDirection],
    *,
    calibration_hash: str = "",
) -> Ensemble:
    """Fit fusion weights on the calibration split.

    ``rows`` maps agent name to score per record, with ``None`` where an agent could not
    answer. ``directions`` says which way each agent points; it is required rather than
    inferred, because inferring it from the data would silently "fix" a genuinely inverted
    detector and hide the bug the specification warns about.
    """
    from sklearn.linear_model import LogisticRegression

    if len(rows) != len(labels):
        raise ValueError(f"{len(rows)} rows against {len(labels)} labels")
    if len(set(labels)) < 2:
        raise ValueError("fusion needs both classes present")
    names = sorted({name for row in rows for name in row})
    if not names:
        raise ValueError("no agent columns to fuse")
    unknown = [name for name in names if name not in directions]
    if unknown:
        raise ValueError(f"no direction given for {', '.join(unknown)} — it must not be guessed")

    # Statistics per agent, over the records that agent could actually score. A mean taken
    # across missing values would be a mean of a different population.
    columns = []
    for name in names:
        observed = [float(row[name]) for row in rows if row.get(name) is not None]  # type: ignore[arg-type]
        if not observed:
            continue
        raw = np.asarray(observed, dtype=np.float64)
        oriented = -raw if directions[name] is ScoreDirection.LOWER_IS_AI else raw
        columns.append(
            AgentColumn(
                name=name,
                direction=directions[name],
                mean=float(raw.mean()),
                scale=float(oriented.std()) or 1.0,
            )
        )

    design = []
    for row in rows:
        features: list[float] = []
        indicators: list[float] = []
        for column in columns:
            value = row.get(column.name)
            features.append(0.0 if value is None else column.oriented(float(value)))
            indicators.append(1.0 if value is None else 0.0)
        design.append(features + indicators)

    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(np.asarray(design, dtype=np.float64), labels)
    weights = model.coef_[0].tolist()
    n = len(columns)
    return Ensemble(
        columns=tuple(columns),
        coefficients=dict(zip((c.name for c in columns), weights[:n], strict=True)),
        missing_coefficients=dict(zip((c.name for c in columns), weights[n:], strict=True)),
        intercept=float(model.intercept_[0]),
        n_calibration=len(rows),
        calibration_hash=calibration_hash,
    )


def bootstrap_coefficients(
    rows: list[dict[str, float | None]],
    labels: list[int],
    directions: dict[str, ScoreDirection],
    *,
    resamples: int = 500,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """95% intervals for each agent's coefficient, by resampling records.

    The specification asks for coefficients with bootstrap CIs, and the reason is interpretive rather than
    decorative: a coefficient whose interval spans zero is an agent the fusion does not
    actually rely on, which is E5's question asked one agent at a time.
    """
    rng = np.random.default_rng(seed)
    collected: dict[str, list[float]] = {}
    indices = np.arange(len(rows))
    for _ in range(resamples):
        picked = rng.choice(indices, size=len(indices), replace=True)
        sample_labels = [labels[i] for i in picked]
        if len(set(sample_labels)) < 2:
            continue
        try:
            fitted = fit([rows[i] for i in picked], sample_labels, directions)
        except ValueError:
            continue
        for name, value in fitted.coefficients.items():
            collected.setdefault(name, []).append(value)
    return {
        name: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        for name, values in collected.items()
        if values
    }


def out_of_fold_scores(
    rows: list[dict[str, float | None]],
    labels: list[int],
    directions: dict[str, ScoreDirection],
    *,
    folds: int = 5,
    seed: int = 42,
) -> list[float]:
    """Fused scores for records the fusion weights were not fitted on.

    **Without this the conformal layer downstream promises coverage it does not have.** The
    ensemble fitted on all 634 calibration records scores those same records optimistically;
    nonconformity measured on optimistic scores is too small, the quantile comes out too
    tight, and coverage on new text falls below the alpha that was advertised. The artefact
    looks entirely ordinary either way — which is the failure mode `calibrate.py` already
    describes for the calibrator, one layer further down.

    Stratified so every fold holds both classes: `fit` refuses a single-class fold, and a
    silently skipped fold would leave those records with no score at all.
    """
    if len(rows) != len(labels):
        raise ValueError(f"{len(rows)} rows against {len(labels)} labels")
    rng = np.random.default_rng(seed)
    assignment = np.empty(len(rows), dtype=int)
    for label in set(labels):
        members = np.flatnonzero(np.asarray(labels) == label)
        shuffled = rng.permutation(members)
        assignment[shuffled] = np.arange(len(shuffled)) % folds

    scores = [float("nan")] * len(rows)
    for fold in range(folds):
        held_out = np.flatnonzero(assignment == fold)
        kept = np.flatnonzero(assignment != fold)
        fitted = fit([rows[i] for i in kept], [labels[i] for i in kept], directions)
        for i in held_out:
            scores[i] = fitted.score(rows[i])
    return scores
