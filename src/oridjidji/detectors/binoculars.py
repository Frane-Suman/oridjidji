"""Binoculars — the "dvostruki obzervator" (the specification, Hans et al. 2024, arXiv:2401.12070).

Implemented explicitly rather than imported, because the specification names it as thesis-core: the
author has to defend this arithmetic at an oral exam, and a function call into someone
else's package cannot be defended.

**The idea in one paragraph.** A text that is merely *predictable* is not necessarily
machine-written — a weather report is predictable. What distinguishes machine text is that
it is predictable *in the particular way a language model is predictable*. So Binoculars
divides one model's surprise at the text by how much two closely-related models disagree
about it. Human text surprises the observer more than the two models disagree with each
other, so the ratio is high. Machine text sits where both models already agree it should
sit, so the ratio is low.

    log_ppl = mean per-token cross-entropy of the OBSERVER against the actual tokens
    x_ppl   = mean over tokens of CrossEntropy(observer_t, performer_t)
    B       = log_ppl / x_ppl

**LOWER B means MORE LIKELY MACHINE.** The spec calls a flipped sign "the classic bug
here" and it is: nothing crashes, the AUROC simply comes out as 1 - AUROC and looks
like a mediocre detector rather than an inverted one. `test_binoculars.py` asserts the
direction on constructed inputs, and `score_direction()` states it in code.

**Which model is the observer.** The paper defines M1 = observer (perplexity of the text)
and M2 = performer (supplies the next-token distributions); the reference implementation
makes the **base** model the observer and the **instruct** model the performer. Earlier
research notes in this project had that reversed. This module follows the official
convention, and E4 evaluates both orderings on a Croatian dev split before either is
trusted.

Thresholds are **never** hard-coded here: the specification forbids copying a threshold from a paper,
and `calibrate()` derives one from Croatian data, returning an artefact that records what
produced it.
"""

from __future__ import annotations

import json
from dataclasses import MISSING, asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from oridjidji.detectors.backbone import Array, PerTokenScores

# Below this many tokens the mean of a per-token quantity is dominated by whichever few
# tokens happened to occur, so the ratio is noise. Reported as UNSCORABLE rather than
# returned as a number nobody should act on (the specification: never a verdict we cannot support).
MIN_TOKENS = 32


def log_perplexity(scores: PerTokenScores) -> float:
    """Mean per-token cross-entropy of the observer against the tokens that appeared.

    This is the numerator: how surprised one model is by the text. Cross-entropy of the
    realised token is just its negative log-probability, averaged.
    """
    if scores.n_tokens == 0:
        raise ValueError("Cannot compute perplexity of an empty token sequence.")
    return float(-np.mean(scores.realised_logprobs))


def cross_perplexity(scores: PerTokenScores) -> float:
    """Mean over tokens of CrossEntropy(observer distribution, performer distribution).

    This is the denominator: how much the two models disagree about what should come
    next, averaged over positions. Note it compares the **full distributions**, not the
    realised token — that is what makes it a measure of disagreement rather than a second
    perplexity.

        CE(p, q) = -sum_v p(v) * log q(v)

    with p the observer's distribution and q the performer's. Computed in log space
    throughout: the vocabulary is ~250k entries and exponentiating first loses precision
    on exactly the small probabilities that dominate the sum.
    """
    per_token = scores.cross_entropies
    if per_token is None:
        raise ValueError(
            "Cross-perplexity needs both models. These scores carry only the observer — "
            "use Fast-DetectGPT for single-model scoring."
        )
    if scores.n_tokens == 0:
        raise ValueError("Cannot compute cross-perplexity of an empty token sequence.")

    # The vocabulary sum, and the 0*log(0) convention it needs, live in `cross_entropies`
    # on whichever representation was handed in. Reading the reduction rather than the
    # distributions is what lets a backbone collapse the vocabulary axis during its forward
    # pass and hand over ~3 KB instead of ~195 MB.
    value = float(np.mean(per_token))
    if not np.isfinite(value):
        raise ValueError(
            "Cross-perplexity is not finite: the performer assigns zero probability to "
            "something the observer considers possible. That is a degenerate pair, not a "
            "very confident one."
        )
    return value


def binoculars_score(scores: PerTokenScores) -> float:
    """B = log_ppl / x_ppl. **Lower means more likely machine.**

    The division is what makes this more than a perplexity detector: it normalises away
    "this text is inherently predictable" and leaves "this text is predictable in the way
    a language model is".
    """
    denominator = cross_perplexity(scores)
    if denominator <= 0.0:
        # Two models that agree perfectly at every position. Real models never do; this
        # means a degenerate input or the same model loaded twice, and dividing would
        # produce inf rather than a score.
        raise ValueError(
            "Cross-perplexity is zero: the observer and performer agree exactly at every "
            "position. That means the same model was loaded twice, or the input is "
            "degenerate — either way the ratio is undefined, not infinite."
        )
    return log_perplexity(scores) / denominator


def score_direction() -> str:
    """States the sign convention in code, so a test can assert it."""
    return "lower_is_machine"


@dataclass(frozen=True)
class BinocularsThresholds:
    """A calibrated operating point, with everything needed to know what it means.

    A bare number is useless: a Binoculars threshold depends on the model pair, the
    tokenizer's fertility on the language, and the calibration set. The specification requires all of
    it stored alongside, the same discipline the feature norms get in `features.norms`.
    """

    threshold: float
    model_pair: str
    calibration_hash: str
    n_calibration: int
    achieved_auroc: float
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    selection_rule: str = "youden_j"
    # Which side of the threshold is machine. **Not a default to be ignored.** M5 ships two
    # detectors with opposite conventions — Binoculars is lower-is-machine, Fast-DetectGPT
    # is higher-is-machine — through one artefact type, so an artefact that assumed a
    # direction would invert one of them silently: no crash, just a detector that reports
    # every human text as machine and every machine text as human.
    lower_is_machine: bool = True
    detector: str = "binoculars"
    # WHICH statistic within that family. `binoculars/official` and `binoculars/swapped`
    # are different numbers on the same model pair — official spans 0.686-1.009 on dev,
    # swapped 0.942-1.708 — so a threshold fitted for one is an arbitrary number for the
    # other, and applying it would classify almost everything as machine with no error
    # raised. Same failure class as `lower_is_machine` above, missed for the same reason:
    # two statistics sharing a family name look like one thing.
    #
    # Empty means "recorded before this field existed"; `configuration_is_recorded` says
    # so rather than letting an unlabelled artefact pass as a labelled one.
    configuration: str = ""

    @property
    def configuration_is_recorded(self) -> bool:
        """False for artefacts written before the field existed. Not the same as valid."""
        return bool(self.configuration)

    def predict_machine(self, score: float) -> bool:
        """True when the score falls on the machine side, whichever side that is."""
        return score <= self.threshold if self.lower_is_machine else score >= self.threshold

    def to_json(self) -> str:
        """The versioned artefact E4 registers.

        A threshold travels with everything needed to know what it means: the model pair
        that produced it, the calibration set's identity, the rule that picked it, and the
        AUROC it achieved. A bare number would be unusable — the same threshold means
        different things under a different tokenizer's fertility.
        """
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> BinocularsThresholds:
        data = json.loads(payload)
        missing = {f.name for f in fields(cls) if f.name not in data}
        # Only fields with defaults may be absent; a missing threshold or model pair means
        # the artefact is not what it claims to be, and guessing would ship a wrong number.
        required = {
            f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING
        }
        if missing & required:
            raise ValueError(f"threshold artefact is missing {sorted(missing & required)}")
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}})

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> BinocularsThresholds:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def calibrate(
    machine_scores: Array,
    human_scores: Array,
    *,
    model_pair: str,
    calibration_hash: str,
    rule: str = "youden_j",
    lower_is_machine: bool = True,
    detector: str = "binoculars",
    configuration: str = "",
    target_fpr: float = 0.01,
) -> BinocularsThresholds:
    """Pick a threshold from **Croatian** scores. Never copy one from a paper.

    Three rules, all reported. L3 registers two — Youden's J (maximise TPR - FPR) and
    closest-to-(0,1) on the ROC — which disagree when the classes are imbalanced or the
    score distributions are skewed, and which one a deployment should use is a values
    question rather than a statistical one, because the false-positive cost falls on a
    student accused of cheating.

    **`fpr_constrained` is added because the other two do not answer that values question.**
    This project reports **TPR@1%FPR** as its headline, and both registered rules select an
    operating point at whatever false-positive rate happens to maximise their own criterion.
    A system that reports one operating point and ships another is describing something it
    does not do. This rule takes the most permissive threshold whose false-positive rate on
    the calibration set stays at or under ``target_fpr``, so the shipped point and the
    reported number mean the same thing.

    It is an *addition*, not a replacement: L3's two are still computed and still reported,
    and nothing is chosen here on the author's behalf.
    """
    from oridjidji.metrics import auroc

    if len(machine_scores) == 0 or len(human_scores) == 0:
        raise ValueError("Calibration needs both classes present.")

    # The candidate rule follows the detector's own convention: `score <= t` means machine
    # for Binoculars and `score >= t` means machine for Fast-DetectGPT. Hardcoding one of
    # them would pick a threshold that maximises TPR - FPR for the *other* detector
    # inverted, which is a real number describing the wrong thing.
    candidates = np.unique(np.concatenate([machine_scores, human_scores]))
    best_value, best_threshold = -np.inf, float(candidates[0])
    for threshold in candidates:
        if lower_is_machine:
            tpr = float(np.mean(machine_scores <= threshold))
            fpr = float(np.mean(human_scores <= threshold))
        else:
            tpr = float(np.mean(machine_scores >= threshold))
            fpr = float(np.mean(human_scores >= threshold))
        if rule == "youden_j":
            value = tpr - fpr
        elif rule == "fpr_constrained":
            # Maximise TPR among thresholds that keep FPR within budget; one that breaks the
            # budget scores -inf rather than competing on TPR.
            value = tpr if fpr <= target_fpr else -np.inf
        else:
            value = -((1 - tpr) ** 2 + fpr**2)
        if value > best_value:
            best_value, best_threshold = value, float(threshold)

    if rule == "fpr_constrained" and best_value == -np.inf:
        # No threshold meets the budget: the detector cannot operate at this false-positive
        # rate on this data. The safe answer for a tool that can accuse a student is to
        # accuse nobody, so the threshold goes outside the observed range and the agent
        # classifies nothing as machine. Falling back to the least-bad candidate instead
        # would ship an operating point at an FPR nobody chose.
        best_threshold = (
            float(candidates[0]) - 1.0 if lower_is_machine else float(candidates[-1]) + 1.0
        )

    # `auroc` expects higher-is-positive, so a lower-is-machine detector's scores are
    # negated. Flipping here rather than in `auroc` keeps that helper honest for every
    # other caller.
    achieved = (
        auroc(-machine_scores, -human_scores)
        if lower_is_machine
        else auroc(machine_scores, human_scores)
    )

    return BinocularsThresholds(
        configuration=configuration,
        threshold=best_threshold,
        model_pair=model_pair,
        calibration_hash=calibration_hash,
        n_calibration=len(machine_scores) + len(human_scores),
        achieved_auroc=achieved,
        selection_rule=rule,
        lower_is_machine=lower_is_machine,
        detector=detector,
    )
