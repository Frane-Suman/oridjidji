"""Turn raw agent results into calibrated Basic or Advanced document results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from oridjidji.api.runtime_loading import FusionLayer
from oridjidji.i18n import get_message
from oridjidji.schemas import AgentResult, AgentStatus, Verdict

# Honest abstention when no compatible probability layer is available.
ABSTAIN: tuple[float, tuple[float, float]] = (0.5, (0.15, 0.85))


@dataclass(frozen=True)
class DocumentScore:
    """A document probability and the calibration that produced it.

    The layer travels with the number because the verdict is read from *its* conformal
    quantiles. A fused probability judged against the encoder's quantiles would be a number
    compared to a threshold measured on a different axis -- 8.8's classic bug, one layer up.
    """

    p_ai: float
    interval: tuple[float, float]
    warnings: list[str]
    layer: Any | None = None
    operating_decision_ai: bool = False
    probability_source: Literal["none", "basic", "advanced"] = "none"
    fallback_used: bool = False


def _operating_decision_ai(calibrated: float, layer: Any) -> bool:
    """Apply the layer's frozen operating threshold, independently of conformal verdict."""
    threshold = getattr(layer, "operating_threshold", None)
    return threshold is not None and calibrated >= float(threshold)


def _commit(
    calibrated: float, quantiles: Any, floor: float = 0.0
) -> tuple[float, tuple[float, float], list[str]]:
    """Turn a calibrated probability into a point, an interval, and any warning.

    Shared by the single-agent and fused paths so the abstention arithmetic exists once.

    ``floor`` is the calibration set's certainty floor, and the interval respects it for the
    same reason the point estimate does: an interval ending at exactly 1.0 says the text
    could be certainly machine, which the specification forbids as plainly as `p_ai=1.0` does. Clipping
    the point alone left `(0.9969, 1.0000)` on the wire.
    """
    from oridjidji.fusion.conformal import prediction_set

    labels = prediction_set(calibrated, quantiles)
    if len(labels) == 1:
        # Committed: a band around the calibrated probability, CLIPPED to what the
        # calibration set supports rather than shrunk symmetrically to fit inside it.
        # Shrinking collapsed the interval to zero width exactly at the floor — (0.9984,
        # 0.9984), an interval asserting no uncertainty at all, which is the same the specification
        # violation as p_ai=1.0 wearing a different shape. Asymmetry near the ends is the
        # honest cost: there is less room above a high probability than below it.
        low = max(floor, calibrated - 0.10)
        high = min(1.0 - floor, calibrated + 0.10)
        return calibrated, (low, high), []
    # Abstaining, but still reporting the calibrated probability -- so the interval has to
    # CONTAIN it. Pairing a point estimate of 0.97 with a fixed (0.15, 0.85) band produced a
    # response whose own schema rejected it: "p_ai lies outside its own interval". Widening
    # to span both the point and the abstention band keeps the invariant and says the honest
    # thing: this is the probability, and it is not enough to commit.
    low, high = ABSTAIN[1]
    # An EMPTY set and a set holding BOTH labels are different facts, and conformal.py keeps
    # them separable so the UI does not flatten them. Empty means the text is nonconforming
    # to both classes — unlike the calibration data — which is not the same as "no calibration
    # exists", and the message said the latter for both.
    warnings = [get_message("atypical_notice")] if not labels else []
    return calibrated, (min(calibrated, low), max(calibrated, high)), warnings


def _document_score(
    agents: list[AgentResult],
    layer: Any | None = None,
    alpha: float = 0.10,
    fusion: FusionLayer | None = None,
) -> DocumentScore:
    """Fold agent scores into a document probability, an interval, and any warnings.

    **With a fitted layer this maps; without one it still declines.** The map is the
    calibrator: a detector score is not a probability, and 2.4 forbids importing the
    transform from a paper, so it has to have been fitted on Croatian calibration data.
    Until `train calibrate` has run, `uncalibrated_notice` says a score was computed but
    is not being presented as a probability.

    With more than one agent answering, 8.9's fusion combines them -- and only fusion may.
    An unweighted mean of detectors with different scales and opposite directions is not a
    probability, so without a fitted ensemble this still abstains rather than averaging.
    """
    p_ai, interval = ABSTAIN
    if not agents:
        return DocumentScore(p_ai, interval, [])
    usable = [
        agent for agent in agents if agent.status is AgentStatus.OK and agent.score is not None
    ]
    if not usable:
        return DocumentScore(p_ai, interval, [get_message("ui_error_model_unavailable")])

    from oridjidji.fusion.ensemble import EXPLAINABILITY_ONLY

    selected = set(fusion.roster) if fusion is not None else None
    selected_member_failed = selected is not None and any(
        agent.status is AgentStatus.FAILED and agent.name in selected for agent in agents
    )
    if selected is not None:
        present = {agent.name for agent in usable}
        selected_member_failed = selected_member_failed or bool(selected - present)

    # An agent can answer, be shown, and still not enter the score. E5 registered that rule
    # in advance and A4 fell on the wrong side of it: the CI on its contribution spans zero,
    # so it is displayed and explained but not fused. Partitioning here rather than
    # dropping the agent keeps it visible to the user, which is the whole point of shipping
    # it as an explainability layer.
    scoring = [
        agent
        for agent in usable
        if (selected is not None and agent.name in selected)
        or (selected is None and agent.name not in EXPLAINABILITY_ONLY)
    ]
    if not scoring:
        return DocumentScore(p_ai, interval, [get_message("uncalibrated_notice")])

    if selected_member_failed:
        encoder = next(
            (agent for agent in scoring if agent.name == "encoder" and agent.score is not None),
            None,
        )
        if encoder is None or layer is None:
            return DocumentScore(
                p_ai,
                interval,
                [get_message("advanced_degraded_notice"), get_message("uncalibrated_notice")],
            )
        calibrated = float(layer.calibrator.apply([encoder.score])[0])
        quantiles = layer.quantiles.get(alpha) or next(iter(layer.quantiles.values()))
        point, span, warnings = _commit(calibrated, quantiles, layer.calibrator.certainty_floor)
        warnings.append(get_message("advanced_degraded_notice"))
        return DocumentScore(
            point,
            span,
            warnings,
            layer,
            _operating_decision_ai(calibrated, layer),
            "basic",
            True,
        )

    if fusion is not None and fusion.roster == ("encoder",):
        encoder = next(
            (agent for agent in scoring if agent.name == "encoder" and agent.score is not None),
            None,
        )
        if encoder is None:
            return DocumentScore(p_ai, interval, [get_message("ui_error_model_unavailable")])
        calibrated = float(fusion.layer.calibrator.apply([encoder.score])[0])
        quantiles = fusion.layer.quantiles.get(alpha) or next(iter(fusion.layer.quantiles.values()))
        point, span, warnings = _commit(
            calibrated, quantiles, fusion.layer.calibrator.certainty_floor
        )
        return DocumentScore(
            point,
            span,
            warnings,
            fusion.layer,
            _operating_decision_ai(calibrated, fusion.layer),
            "advanced",
        )

    if len(scoring) > 1:
        if fusion is None or fusion.ensemble is None:
            return DocumentScore(p_ai, interval, [get_message("uncalibrated_notice")])
        row: dict[str, float | None] = {agent.name: agent.score for agent in scoring}
        fused = fusion.ensemble.score(row)
        calibrated = float(fusion.layer.calibrator.apply([fused])[0])
        quantiles = fusion.layer.quantiles.get(alpha) or next(iter(fusion.layer.quantiles.values()))
        point, span, warnings = _commit(
            calibrated, quantiles, fusion.layer.calibrator.certainty_floor
        )
        return DocumentScore(
            point,
            span,
            warnings,
            fusion.layer,
            _operating_decision_ai(calibrated, fusion.layer),
            "advanced",
        )

    if layer is None:
        return DocumentScore(p_ai, interval, [get_message("uncalibrated_notice")])
    calibrated = float(layer.calibrator.apply([scoring[0].score])[0])
    quantiles = layer.quantiles.get(alpha) or next(iter(layer.quantiles.values()))
    point, span, warnings = _commit(calibrated, quantiles, layer.calibrator.certainty_floor)
    return DocumentScore(
        point,
        span,
        warnings,
        layer,
        _operating_decision_ai(calibrated, layer),
        "basic",
    )


def score_for_mode(
    agents: list[AgentResult],
    *,
    mode: Literal["osnovni", "napredni"],
    basic_layer: Any | None,
    fusion: FusionLayer | None,
    advanced_configured: bool,
) -> DocumentScore:
    """Apply exactly the probability contract selected by the request.

    An incomplete Advanced installation is an unavailable mode, not a failed Advanced
    inference.  Only a configured fusion whose selected statistical agents then fail may
    return the independently calibrated Basic result with ``fallback_used=True``.
    """
    if mode == "osnovni":
        return _document_score(agents, basic_layer)
    if not advanced_configured or fusion is None:
        return DocumentScore(
            *ABSTAIN,
            [get_message("ui_error_model_unavailable")],
        )
    return _document_score(agents, basic_layer, fusion=fusion)


def _verdict_from(
    p_ai: float,
    interval: tuple[float, float],
    layer: Any | None = None,
    alpha: float = 0.10,
) -> Verdict:
    """Three-state outcome (2.1, 2.3).

    With a fitted layer the boundary is the conformal prediction set, which is what 2.3
    requires: a set holding both labels means the evidence supports neither side, an empty
    set means the text is atypical of both, and only a singleton commits.

    The layer passed here must be the one that produced ``p_ai`` -- `DocumentScore` carries
    it for exactly that reason.

    Without one the rule falls back to the interval spanning 0.5 -- provisional, and marked
    as such, but the same shape: an interval that spans the middle cannot support a side.
    """
    if layer is not None:
        from oridjidji.fusion.conformal import verdict_for

        quantiles = layer.quantiles.get(alpha) or next(iter(layer.quantiles.values()))
        return verdict_for(p_ai, quantiles)
    low, high = interval
    if low <= 0.5 <= high:
        return Verdict.INCONCLUSIVE
    return Verdict.LIKELY_AI if p_ai > 0.5 else Verdict.LIKELY_HUMAN


def _prediction_labels_from(
    p_ai: float,
    layer: Any | None = None,
    alpha: float = 0.10,
) -> list[str]:
    """Serialize the same conformal set that determines the three-state verdict."""
    if layer is None:
        return ["human", "ai"]
    from oridjidji.fusion.conformal import prediction_set

    quantiles = layer.quantiles.get(alpha) or next(iter(layer.quantiles.values()))
    labels = prediction_set(p_ai, quantiles)
    return [name for label, name in ((0, "human"), (1, "ai")) if label in labels]
