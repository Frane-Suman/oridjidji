"""Binoculars and Fast-DetectGPT as served agents.

`binoculars.py` and `fast_detect_gpt.py` hold the arithmetic; `hf_backbone.py` holds the
weights. This is the thin layer between them and the API — one class that turns a backbone,
a configuration and a calibrated threshold into the `AgentResult` the ring draws.

**Two agents share one backbone.** That is the specification's whole point: Binoculars needs both
models, Fast-DetectGPT needs one, and on a 16 GB laptop with no discrete GPU three
detectors each loading their own copy is the difference between the Advanced mode being
usable and not. `for_pair` builds both from a single `HuggingFaceBackbone`.

**Direction travels with the score.** Binoculars is lower-is-machine and Fast-DetectGPT is
higher-is-machine, and the fusion layer in M6 will read both. An agent that reported the
wrong `ScoreDirection` would not fail — it would invert one detector inside the ensemble,
which is the specification's "classic bug" arriving through the serving layer rather than the maths.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

from oridjidji.detectors import aggregation, binoculars, fast_detect_gpt
from oridjidji.schemas import AgentResult, AgentStatus, ScoreDirection

if TYPE_CHECKING:  # pragma: no cover - typing only
    from oridjidji.detectors.backbone import ReducedScores
    from oridjidji.detectors.binoculars import BinocularsThresholds


class ScoringBackbone(Protocol):
    """What an agent needs from the model layer. Narrower than `Backbone` on purpose."""

    @property
    def model_pair(self) -> str: ...

    def load(self) -> None: ...

    def score_tokens(self, text: str) -> ReducedScores: ...


@dataclass(frozen=True)
class _Spec:
    """The parts of a detector that differ between the two agents."""

    name: str
    direction: ScoreDirection
    rationale_hr: str


BINOCULARS = _Spec(
    name="binoculars",
    direction=ScoreDirection.LOWER_IS_AI,
    rationale_hr=(
        "Omjer iznenađenja dvaju jezičnih modela; niža vrijednost znači vjerojatniji stroj."
    ),
)
BINOCULARS_V4 = _Spec(
    name="binoculars_v4",
    direction=ScoreDirection.LOWER_IS_AI,
    rationale_hr=(
        "Omjer iznenađenja s većom težinom na granicama hrvatskih riječi; niža "
        "vrijednost znači vjerojatniji stroj."
    ),
)
FAST_DETECT_GPT = _Spec(
    name="fast_detect_gpt",
    direction=ScoreDirection.HIGHER_IS_AI,
    rationale_hr=(
        "Zakrivljenost uvjetne vjerojatnosti jednog modela; viša vrijednost znači "
        "vjerojatniji stroj."
    ),
)


class StatisticalDetector:
    """One zero-shot agent over a shared backbone.

    Same shape as `EncoderDetector`: `name`, `score(text) -> AgentResult`, and a version
    string the API can report. The ring does not know or care which kind of detector it is
    holding.
    """

    def __init__(
        self,
        spec: _Spec,
        backbone: ScoringBackbone,
        thresholds: BinocularsThresholds | None = None,
    ) -> None:
        self._spec = spec
        self.backbone = backbone
        self.thresholds = thresholds

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def version(self) -> str:
        """Which models produced the score, and whether it has a Croatian threshold.

        A statistical detector without a calibrated threshold still produces a number, and
        that number is meaningless as a probability — the specification forbids importing one from a
        paper, so the version string says `uncalibrated` rather than implying otherwise.
        """
        state = (
            f"@{self.thresholds.selection_rule}" if self.thresholds is not None else "@uncalibrated"
        )
        return f"{self.backbone.model_pair}{state}"

    def load(self) -> None:
        self.backbone.load()

    @property
    def prepare_key(self) -> str:
        """Request-local evidence key shared by A2 and A3."""
        return f"statistical:{id(self.backbone)}"

    def prepare(self, text: str) -> ReducedScores:
        """Run the expensive backbone once; the orchestrator fans this out."""
        return self.backbone.score_tokens(text)

    def score_prepared(self, scores: ReducedScores) -> AgentResult:
        """Turn shared token reductions into this agent's raw score."""
        started = time.perf_counter()
        try:
            if self._spec is BINOCULARS:
                value = binoculars.binoculars_score(scores)
            elif self._spec is BINOCULARS_V4:
                if scores.cross_entropies is None:
                    raise ValueError("Binoculars V4 needs the performer cross-entropy")
                if scores.word_starts is None:
                    raise ValueError("Binoculars V4 needs tokenizer offset word boundaries")
                value = aggregation.boundary_entropy(
                    -np.asarray(scores.realised_logprobs, dtype=np.float64),
                    np.asarray(scores.cross_entropies, dtype=np.float64),
                    np.asarray(scores.word_starts, dtype=np.bool_),
                    -np.asarray(scores.expected_logprobs, dtype=np.float64),
                )
            else:
                value = fast_detect_gpt.curvature(scores)
        except Exception as exc:  # noqa: BLE001 - same truthful degrade path as score()
            return AgentResult(
                name=self.name,
                version=self.version,
                status=AgentStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        return AgentResult(
            name=self.name,
            version=self.version,
            status=AgentStatus.OK,
            score=value,
            direction=self._spec.direction,
            latency_ms=int((time.perf_counter() - started) * 1000),
            rationale_hr=self._spec.rationale_hr,
            token_data={
                "window_policy": scores.window_policy,
                "total_tokens": scores.total_tokens or scores.n_tokens,
                "scored_tokens": scores.scored_tokens or scores.n_tokens,
                "coverage": scores.coverage,
                "scored_chars": getattr(scores, "scored_chars", None),
            },
        )

    def score(self, text: str) -> AgentResult:
        """Score one document. Never raises — a dead detector is a reportable state.

        A 3.4 GB pair that will not load is exactly the case this exists for: the UI draws
        that band dashed and labels it `nedostupno`, and the remaining detectors still
        answer. The error text travels in the result instead of out of the request.
        """
        started = time.perf_counter()
        try:
            scores = self.prepare(text)
            result = self.score_prepared(scores)
        except Exception as exc:  # noqa: BLE001 — see docstring: this is the degrade path
            return AgentResult(
                name=self.name,
                version=self.version,
                status=AgentStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        return result.model_copy(update={"latency_ms": int((time.perf_counter() - started) * 1000)})

    def predicts_machine(self, score: float) -> bool | None:
        """Which side of the calibrated threshold this score falls on, or None.

        `None` when the detector has no Croatian threshold — the honest answer, and the
        one M6's conformal layer needs rather than a guess. The specification refuses rather than
        overclaims, and that starts here.
        """
        if self.thresholds is None:
            return None
        return self.thresholds.predict_machine(score)


def for_pair(
    backbone: ScoringBackbone,
    *,
    binoculars_variant: Literal["V0", "V4"] = "V0",
    binoculars_thresholds: BinocularsThresholds | None = None,
    curvature_thresholds: BinocularsThresholds | None = None,
) -> list[StatisticalDetector]:
    """Both agents over one backbone — the the specification arrangement, in one call.

    Built together so a caller cannot accidentally give each its own copy of 3.4 GB of
    weights. The thresholds are separate because they are separate artefacts: two
    detectors with opposite conventions cannot share one.
    """
    if binoculars_variant not in {"V0", "V4"}:
        raise ValueError(f"Unknown Binoculars variant: {binoculars_variant}")
    binoculars_spec = BINOCULARS if binoculars_variant == "V0" else BINOCULARS_V4
    return [
        StatisticalDetector(binoculars_spec, backbone, binoculars_thresholds),
        StatisticalDetector(FAST_DETECT_GPT, backbone, curvature_thresholds),
    ]
