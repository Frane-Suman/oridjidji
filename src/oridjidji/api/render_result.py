"""Render a finished analysis into the HTML the live page shows.

**The bug this closes.** `app.js:render()` set a run id, dispatched `oridjidji:result`, and
stopped. Nothing listened for that event. An analysis completed and the panel stayed empty —
the demo could run and show nothing, which is a larger M8 gap than the missing Dockerfile.

**Why the server renders it.** `app.js` says so in its own comment: *"The result markup is
built server-side so the review page and the live page cannot drift apart."* `/pregled` renders
`components/assessment.html` from fixtures; the live page now renders the same component from a
real response. One template, so a change to the result panel cannot silently apply to only one
of them.

**No Croatian lives here.** Every string comes from `web.render.result_context`, which reads
`i18n`. The brandbook is explicit that the frontend may never invent or hard-code a label
, and a renderer that assembled its own sentences would be the frontend by another
name.

The fragment travels **inside the terminal SSE event**, beside the JSON. That avoids a second
round trip and avoids the server holding per-run state — the alternative was an endpoint that
accepts a whole result as input, which is awkward to validate and easy to abuse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from oridjidji.i18n import MESSAGES_HR
from oridjidji.web import render

#: Where the Jinja templates live, mirroring `api/app.py`'s own resolution.
WEB_ROOT = Path(__file__).resolve().parents[1] / "web"


def _value(value: Any) -> Any:
    """Return the serial value of an enum while leaving ordinary values alone."""
    return getattr(value, "value", value)


def _field(row: Any, name: str, default: Any = None) -> Any:
    """Read one field from a Pydantic model or a history-style dictionary."""
    return row.get(name, default) if isinstance(row, dict) else getattr(row, name, default)


def _operating_view(rows: Any) -> list[dict[str, str]]:
    """Format optional standalone operating evidence without changing its meaning."""
    return [
        {
            "target_hr": render.hr_percent(float(_field(row, "target_fpr"))) + " %",
            "value_hr": render.hr_number(float(_field(row, "raw_threshold")), 4),
            "fpr_hr": render.hr_percent(float(_field(row, "achieved_fpr"))) + " %",
            "tpr_hr": render.hr_percent(float(_field(row, "observed_tpr"))) + " %",
        }
        for row in rows or []
    ]


def _agent_view(row: Any) -> dict[str, object]:
    """Turn a typed evidence row into display-only Croatian values."""
    name = str(_field(row, "name", ""))
    status = str(_value(_field(row, "status", "failed"))).lower()
    raw_score = _field(row, "raw_score", _field(row, "score"))
    contribution = _field(row, "fusion_contribution")
    latency = _field(row, "latency_ms")
    direction = str(_value(_field(row, "direction", "HIGHER_IS_AI"))).lower()
    used = bool(_field(row, "used_in_assessment", name != "linguistic"))
    coverage = _field(row, "token_coverage", _field(row, "coverage"))
    label = _field(row, "label_hr") or MESSAGES_HR.get(f"agent_{name}", name)
    return {
        "name": name,
        "label_hr": label,
        "status": status,
        "status_hr": MESSAGES_HR["ui_agent_ok"]
        if status in {"ok", "success", "succeeded"}
        else MESSAGES_HR["ui_agent_failed"],
        "raw_score_hr": render.hr_number(float(raw_score), 4) if raw_score is not None else "—",
        "fusion_contribution_hr": (
            f"{float(contribution):+.4f}".replace(".", ",") if contribution is not None else "—"
        ),
        "contribution_sign": (
            "positive" if contribution is not None and contribution >= 0 else "negative"
        )
        if contribution is not None
        else "neutral",
        "latency_hr": f"{int(latency)} ms" if latency is not None else "—",
        "direction_hr": (
            MESSAGES_HR["ui_agent_direction_higher"]
            if direction.endswith("higher_is_ai")
            else MESSAGES_HR["ui_agent_direction_lower"]
        ),
        "participation_hr": MESSAGES_HR["ui_yes"] if used else MESSAGES_HR["ui_no"],
        "coverage_hr": render.hr_percent(float(coverage)) + " %" if coverage is not None else "",
        "rationale_hr": _field(row, "rationale_hr") or MESSAGES_HR["ui_agent_no_rationale"],
        "interpretation_hr": _field(row, "interpretation_hr")
        or _field(row, "rationale_hr")
        or MESSAGES_HR["ui_agent_no_rationale"],
        "operating_evidence": _operating_view(_field(row, "standalone_thresholds", [])),
        "used_in_assessment": used,
    }


def result_fragment(response: Any, templates: Any) -> str:
    """The `assessment.html` fragment for a finished `AnalyzeResponse`.

    ``templates`` is the app's own `Jinja2Templates`, passed in rather than built here so the
    fragment is rendered by the same environment — and with the same `t()` global — that the
    full pages use. Constructing a second environment is how two renderers start to drift.
    """
    source_evidence = list(getattr(response, "agent_evidence", []) or response.agents)
    evidence = [_agent_view(agent) for agent in source_evidence]
    assessment = [row for row in evidence if row["used_in_assessment"]]
    linguistic = [row for row in evidence if not row["used_in_assessment"]]
    usable = [row for row in assessment if row["status"] in {"ok", "success", "succeeded"}]
    requested_mode = str(getattr(response, "requested_mode", "osnovni"))
    effective_mode = str(getattr(response, "effective_mode", requested_mode))
    runtime_status = str(_value(response.runtime_identity.runtime_status))
    context = render.result_context(
        verdict=str(_value(response.verdict)),
        p_ai=response.p_ai,
        n_words=response.gates.n_words,
        agents_used=len(usable),
        agents_total=len(assessment),
        limitations=list(response.limitations_hr),
        prediction_set=list(response.prediction_set),
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        runtime_status=runtime_status,
        warnings=list(response.warnings),
        agent_evidence=assessment,
        linguistic_evidence=linguistic,
        interpretation_hr=response.interpretation_hr,
        scope_hr=getattr(response, "scope_hr", ""),
    )
    explanations = response.explanations.model_dump(mode="json")
    explanations["token_heatmap"] = [
        {
            **row,
            "weight_hr": render.hr_percent(float(row.get("weight", 0.0))),
            "strength": _signal_strength(float(row.get("weight", 0.0))),
        }
        for row in explanations.get("token_heatmap", [])
    ]
    context["explanations"] = explanations
    template = templates.env.get_template("components/assessment.html")
    return str(template.render(**context))


def _signal_strength(weight: float) -> str:
    """Map a relative token rank to one of four presentation-only intensity levels."""
    if weight >= 0.75:
        return "strong"
    if weight >= 0.5:
        return "medium"
    if weight >= 0.25:
        return "soft"
    return "faint"
