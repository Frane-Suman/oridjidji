"""View-model construction for the web layer.

Everything here exists so the templates can stay declarative. Croatian number
formatting, the verdict → zone mapping and the ring geometry are all things that can be
got wrong silently, so they live in Python where a test can reach them rather than in a
Jinja expression where one cannot.

The rule this module enforces on behalf of the templates: **no model facts in the
frontend**. Nothing here compares a probability to a threshold or
decides a verdict. `zone_for` maps a verdict the API already decided onto a CSS class,
and that is the whole of the frontend's involvement in classification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from oridjidji.i18n import MESSAGES_HR, NOTICE_HR, VERDICT_MESSAGE_KEYS, get_message

# verdict → the style vocabulary in app.css. Three zones, three fills; the mapping is
# the template's only "decision" and it is a lookup, not a comparison.
ZONE_FOR_VERDICT: dict[str, str] = {
    "LIKELY_HUMAN": "low",
    "INCONCLUSIVE": "inconclusive",
    "LIKELY_AI": "high",
}


def hr_number(value: float, decimals: int = 1) -> str:
    """Croatian decimal formatting: comma, not point.

    `78.4` becomes `78,4`. Used for every number the user reads, so a percentage never
    renders in the English convention next to Croatian prose.
    """
    return f"{value:.{decimals}f}".replace(".", ",")


def hr_percent(fraction: float, decimals: int = 1) -> str:
    """A probability in [0, 1] as a Croatian percentage figure, without the sign.

    The sign is added by the template with a non-breaking space (`78,4&nbsp;%`), which
    is the Croatian convention and also stops the number wrapping away from its unit.
    """
    return hr_number(fraction * 100.0, decimals)


def zone_for(verdict: str) -> str:
    """CSS zone for a verdict the API decided. Unknown verdicts fail loudly.

    A silent fallback would render an unrecognised verdict in the low zone — the
    calmest-looking one — which is the wrong direction to guess in.
    """
    if verdict not in ZONE_FOR_VERDICT:
        raise ValueError(f"Unknown verdict {verdict!r}; expected one of {list(ZONE_FOR_VERDICT)}")
    return ZONE_FOR_VERDICT[verdict]


def headline_for(verdict: str) -> str:
    """The Croatian headline. Never says "human" — see the note in `i18n.py`."""
    return MESSAGES_HR[VERDICT_MESSAGE_KEYS[verdict]]


@dataclass(frozen=True)
class Dot:
    """One halftone dot in a ring band."""

    cx: float
    cy: float
    r: float
    fill: str
    opacity: float


def ring_dots(band_index: int, band_count: int, per_band: int = 48) -> list[Dot]:
    """Halftone dots for one detector band, outermost band first.

    Bands are laid out from the outside in, because the API order is the reporting
    order and the outermost ring is what the eye reads first. Dots start sparse and pale
    and are filled by SSE events — the initial `r` and opacity here are the *empty*
    state, never a partially-filled one, since a ring that starts half full implies
    progress that has not happened.

    Colour walks the red → orange gradient across bands. That gradient appears in
    exactly two places in the product, the logo and the ring, and this is the
    second.
    """
    if band_count <= 0:
        return []
    if not 0 <= band_index < band_count:
        raise ValueError(f"band_index {band_index} out of range for {band_count} bands")

    outer, inner = 190.0, 90.0
    step = (outer - inner) / max(band_count, 1)
    radius = outer - step * band_index - step / 2
    fill = "var(--signal-red)" if band_index % 2 == 0 else "var(--signal-orange)"

    dots = []
    for i in range(per_band):
        angle = 2.0 * math.pi * i / per_band
        dots.append(
            Dot(
                cx=round(200.0 + radius * math.cos(angle), 2),
                cy=round(200.0 + radius * math.sin(angle), 2),
                r=1.5,  # empty state: sparse and small
                fill=fill,
                opacity=0.18,  # pale until this detector reports
            )
        )
    return dots


def result_context(
    *,
    verdict: str,
    p_ai: float,
    n_words: int,
    agents_used: int,
    agents_total: int,
    limitations: list[str],
    prediction_set: list[str] | None = None,
    requested_mode: str | None = None,
    effective_mode: str | None = None,
    runtime_status: str | None = None,
    warnings: list[str] | None = None,
    agent_evidence: list[dict[str, object]] | None = None,
    linguistic_evidence: list[dict[str, object]] | None = None,
    interpretation_hr: str | None = None,
    scope_hr: str | None = None,
) -> dict[str, object]:
    """Everything `assessment.html` needs, with nothing left for the template to compute.

    The rail marks only the calibrated estimate. No display band is drawn: it would not be a statistically grounded uncertainty estimate.
    """
    headline = headline_for(verdict)
    prediction_count = len(prediction_set or [])
    if prediction_count == 1:
        prediction_explanation = MESSAGES_HR["ui_result_prediction_single"]
    elif prediction_count > 1:
        prediction_explanation = MESSAGES_HR["ui_result_prediction_ambiguous"]
    else:
        prediction_explanation = MESSAGES_HR["ui_result_prediction_empty"]
    mode_labels = {
        "osnovni": MESSAGES_HR["ui_mode_basic"],
        "napredni": MESSAGES_HR["ui_mode_advanced"],
    }
    requested_mode_label = mode_labels.get(requested_mode or "", "")
    effective_mode_label = mode_labels.get(effective_mode or "", "")
    interpretation_keys = {
        "LIKELY_AI": "ui_interpretation_ai",
        "LIKELY_HUMAN": "ui_interpretation_human",
        "INCONCLUSIVE": "ui_interpretation_inconclusive",
    }
    return {
        "zone": zone_for(verdict),
        "headline": headline,
        "estimate": hr_percent(p_ai),
        "estimate_pct": round(p_ai * 100.0, 2),
        "rail_label": get_message(
            "ui_result_rail_label",
            estimate=hr_percent(p_ai),
            verdict=headline,
        ),
        "basis": get_message(
            "ui_result_basis", used=agents_used, total=agents_total, n_words=n_words
        )
        if agents_total
        else "",
        "scope_hr": scope_hr or "",
        "inconclusive_body": MESSAGES_HR["ui_result_inconclusive_body"],
        "prediction_explanation": prediction_explanation,
        "interpretation_hr": interpretation_hr or get_message(interpretation_keys[verdict]),
        "requested_mode_label": requested_mode_label,
        "effective_mode_label": effective_mode_label,
        "requested_mode_line": get_message("ui_result_requested_mode", mode=requested_mode_label)
        if requested_mode_label
        else "",
        "effective_mode_line": get_message("ui_result_executed_mode", mode=effective_mode_label)
        if effective_mode_label
        else "",
        "runtime_status": runtime_status or "",
        "warnings": warnings or [],
        "agent_evidence": agent_evidence or [],
        "linguistic_evidence": linguistic_evidence or [],
        "notice": NOTICE_HR,
        "limitations": limitations,
        "explanations": {
            "token_heatmap": [],
            "top_features": [],
            "shap": [],
        },
    }
