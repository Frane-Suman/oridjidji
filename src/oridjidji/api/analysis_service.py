"""Turn raw agent results into calibrated Basic or Advanced document results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from oridjidji.api.runtime_loading import FusionLayer
from oridjidji.i18n import get_message
from oridjidji.schemas import AgentResult, AgentStatus, Verdict

# Honest abstention when no compatible probability layer is available: the neutral midpoint
# is reported, and without a layer `_verdict_from` returns INCONCLUSIVE.
ABSTAIN_P_AI = 0.5


@dataclass(frozen=True)
class DocumentScore:
    """A document probability and the calibration that produced it.

    The layer travels with the number because the verdict is read from *its* conformal
    quantiles. A fused probability judged against the encoder's quantiles would be a number
    compared to a threshold measured on a different axis -- 8.8's classic bug, one layer up.
    """

    p_ai: float
    warnings: list[str]
    layer: Any | None = None
    operating_decision_ai: bool = False
    probability_source: Literal["none", "basic", "advanced"] = "none"
    fallback_used: bool = False


def _operating_decision_ai(calibrated: float, layer: Any) -> bool:
    """Apply the layer's frozen operating threshold, independently of conformal verdict."""
    threshold = getattr(layer, "operating_threshold", None)
    return threshold is not None and calibrated >= float(threshold)


def _commit(calibrated: float, quantiles: Any) -> tuple[float, list[str]]:
    """Return the calibrated probability and any warning its conformal set implies.

    Shared by the single-agent and fused paths so the abstention arithmetic exists once.
    The probability is reported as calibrated; whether it commits to a side is decided by
    the class-conditional conformal set in `_verdict_from`, never by a band around it
    (there is no heuristic display interval).
    """
    from oridjidji.fusion.conformal import prediction_set

    labels = prediction_set(calibrated, quantiles)
    # An EMPTY set and a set holding BOTH labels are different facts, and conformal.py keeps
    # them separable so the UI does not flatten them. Empty means the text is nonconforming
    # to both classes — unlike the calibration data — which is not the same as "no calibration
    # exists", and the message said the latter for both.
    warnings = [get_message("atypical_notice")] if not labels else []
    return calibrated, warnings


def _document_score(
    agents: list[AgentResult],
    layer: Any | None = None,
    alpha: float = 0.10,
    fusion: FusionLayer | None = None,
) -> DocumentScore:
    """Fold agent scores into a document probability and any warnings.

    **With a fitted layer this maps; without one it still declines.** The map is the
    calibrator: a detector score is not a probability, and 2.4 forbids importing the
    transform from a paper, so it has to have been fitted on Croatian calibration data.
    Until `train calibrate` has run, `uncalibrated_notice` says a score was computed but
    is not being presented as a probability.

    With more than one agent answering, 8.9's fusion combines them -- and only fusion may.
    An unweighted mean of detectors with different scales and opposite directions is not a
    probability, so without a fitted ensemble this still abstains rather than averaging.
    """
    p_ai = ABSTAIN_P_AI
    if not agents:
        return DocumentScore(p_ai, [])
    usable = [
        agent for agent in agents if agent.status is AgentStatus.OK and agent.score is not None
    ]
    if not usable:
        return DocumentScore(p_ai, [get_message("ui_error_model_unavailable")])

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
        return DocumentScore(p_ai, [get_message("uncalibrated_notice")])

    if selected_member_failed:
        encoder = next(
            (agent for agent in scoring if agent.name == "encoder" and agent.score is not None),
            None,
        )
        if encoder is None or layer is None:
            return DocumentScore(
                p_ai,
                [get_message("advanced_degraded_notice"), get_message("uncalibrated_notice")],
            )
        calibrated = float(layer.calibrator.apply([encoder.score])[0])
        quantiles = layer.quantiles.get(alpha) or next(iter(layer.quantiles.values()))
        point, warnings = _commit(calibrated, quantiles)
        warnings.append(get_message("advanced_degraded_notice"))
        return DocumentScore(
            point,
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
            return DocumentScore(p_ai, [get_message("ui_error_model_unavailable")])
        calibrated = float(fusion.layer.calibrator.apply([encoder.score])[0])
        quantiles = fusion.layer.quantiles.get(alpha) or next(iter(fusion.layer.quantiles.values()))
        point, warnings = _commit(calibrated, quantiles)
        return DocumentScore(
            point,
            warnings,
            fusion.layer,
            _operating_decision_ai(calibrated, fusion.layer),
            "advanced",
        )

    if len(scoring) > 1:
        if fusion is None or fusion.ensemble is None:
            return DocumentScore(p_ai, [get_message("uncalibrated_notice")])
        row: dict[str, float | None] = {agent.name: agent.score for agent in scoring}
        fused = fusion.ensemble.score(row)
        calibrated = float(fusion.layer.calibrator.apply([fused])[0])
        quantiles = fusion.layer.quantiles.get(alpha) or next(iter(fusion.layer.quantiles.values()))
        point, warnings = _commit(calibrated, quantiles)
        return DocumentScore(
            point,
            warnings,
            fusion.layer,
            _operating_decision_ai(calibrated, fusion.layer),
            "advanced",
        )

    if layer is None:
        return DocumentScore(p_ai, [get_message("uncalibrated_notice")])
    calibrated = float(layer.calibrator.apply([scoring[0].score])[0])
    quantiles = layer.quantiles.get(alpha) or next(iter(layer.quantiles.values()))
    point, warnings = _commit(calibrated, quantiles)
    return DocumentScore(
        point,
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
        return DocumentScore(ABSTAIN_P_AI, [get_message("ui_error_model_unavailable")])
    return _document_score(agents, basic_layer, fusion=fusion)


def _verdict_from(
    p_ai: float,
    layer: Any | None = None,
    alpha: float = 0.10,
) -> Verdict:
    """Three-state outcome (2.1, 2.3).

    With a fitted layer the boundary is the class-conditional conformal prediction set,
    which is what 2.3 requires: a set holding both labels means the evidence supports
    neither side, an empty set means the text is atypical of both, and only a singleton
    commits.

    The layer passed here must be the one that produced ``p_ai`` -- `DocumentScore` carries
    it for exactly that reason.

    Without a layer there is no calibrated probability to judge, so the outcome is
    INCONCLUSIVE; every layer-less path reports `ABSTAIN_P_AI`.
    """
    if layer is None:
        return Verdict.INCONCLUSIVE
    from oridjidji.fusion.conformal import verdict_for

    quantiles = layer.quantiles.get(alpha) or next(iter(layer.quantiles.values()))
    return verdict_for(p_ai, quantiles)


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
