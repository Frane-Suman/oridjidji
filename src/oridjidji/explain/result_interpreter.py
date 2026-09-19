"""Concise Croatian interpretation of detector evidence without changing the score."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from oridjidji.api.schemas import AgentEvidence
from oridjidji.i18n import get_message

LOGGER = logging.getLogger(__name__)


class CompletionRuntime(Protocol):
    def complete(self, prompt: str, *, max_tokens: int = 120) -> str: ...


class InterpretationBundle(BaseModel):
    """Validated text returned to the response renderer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary_hr: str = Field(min_length=1, max_length=360)
    agents_hr: dict[str, str] = Field(default_factory=dict)
    source: Literal["deterministic", "gemma2_2b_it"] = "deterministic"


class _GeneratedPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary_hr: str
    agents_hr: dict[str, str] = Field(default_factory=dict)


def _short_sentence(value: str, *, limit: int) -> str:
    clean = " ".join(value.replace("•", " ").replace("*", " ").split()).strip("-–— ")
    words = clean.split()
    if len(words) > limit:
        clean = " ".join(words[:limit]).rstrip(".,;:") + "…"
    if clean and clean[-1] not in ".!?…":
        clean += "."
    return clean


def _hr_percent(value: float) -> str:
    return f"{value * 100.0:.1f}".replace(".", ",")


def _hr_decimal(value: float) -> str:
    return f"{value:.4f}".replace(".", ",")


def _acceptable_summary(value: str, *, required_numbers: tuple[str, ...] = ()) -> bool:
    folded = value.casefold()
    forbidden = (
        "ai je autor",
        "ai je većina",
        "je većina",
        "interval zaključaka",
        "tekst je napisao",
        "sigurno",
        "{",
        "}",
    )
    normalized = value.replace("\u00a0", " ")
    return (
        bool(value)
        and not any(fragment in folded for fragment in forbidden)
        and all(number in normalized for number in required_numbers)
    )


def _agent_fallback(row: AgentEvidence) -> str:
    if getattr(row.status, "value", row.status) != "ok":
        return get_message("ui_agent_interpretation_failed")
    if not row.used_in_assessment:
        return get_message("ui_agent_interpretation_explanation")
    if row.raw_score is None:
        return get_message("ui_agent_no_rationale")

    direction = str(getattr(row.direction, "value", row.direction)).lower()
    direction_key = (
        "ui_agent_score_higher_explanation"
        if direction.endswith("higher_is_ai")
        else "ui_agent_score_lower_explanation"
    )
    sentences = [get_message(direction_key, score=_hr_decimal(row.raw_score))]

    if row.fusion_contribution is None:
        sentences.append(get_message("ui_agent_score_basic_explanation"))
    else:
        contribution = row.fusion_contribution
        if contribution > 0:
            contribution_key = "ui_agent_contribution_raises_explanation"
        elif contribution < 0:
            contribution_key = "ui_agent_contribution_lowers_explanation"
        else:
            contribution_key = "ui_agent_contribution_neutral_explanation"
        signed = f"{contribution:+.4f}".replace(".", ",")
        sentences.append(get_message(contribution_key, contribution=signed))

    if row.standalone_thresholds:
        point = min(row.standalone_thresholds, key=lambda item: item.target_fpr)
        threshold_key = (
            "ui_agent_threshold_ai_explanation"
            if point.detector_indicates_ai
            else "ui_agent_threshold_human_explanation"
        )
        sentences.append(
            get_message(
                threshold_key,
                threshold=_hr_decimal(point.raw_threshold),
                target_fpr=_hr_percent(point.target_fpr),
            )
        )
    return " ".join(sentences)


def deterministic_interpretation(
    *, verdict: str, p_ai: float, interval: tuple[float, float], evidence: list[AgentEvidence]
) -> InterpretationBundle:
    """Safe fallback used by Basic and whenever Gemma cannot return valid Croatian JSON."""
    low, high = interval
    width = high - low
    estimate_hr = _hr_percent(p_ai)
    low_hr = _hr_percent(low)
    high_hr = _hr_percent(high)
    if verdict == "LIKELY_AI":
        message_key = "ui_interpretation_precise_ai"
    elif verdict == "LIKELY_HUMAN":
        message_key = "ui_interpretation_precise_human"
    else:
        message_key = "ui_interpretation_precise_inconclusive"
    summary = get_message(
        message_key,
        estimate=estimate_hr,
        low=low_hr,
        high=high_hr,
    )
    if width >= 0.5:
        summary += " " + get_message("ui_interpretation_wide_interval")
    return InterpretationBundle(
        summary_hr=summary,
        agents_hr={row.name: _agent_fallback(row) for row in evidence},
    )


def _prompt(
    *,
    verdict: str,
    p_ai: float,
    interval: tuple[float, float],
    evidence: list[AgentEvidence],
) -> str:
    rows = [
        {
            "name": row.name,
            "label": row.label_hr,
            "status": getattr(row.status, "value", str(row.status)),
            "raw_score": row.raw_score,
            "direction": getattr(row.direction, "value", str(row.direction)),
            "fusion_contribution": row.fusion_contribution,
            "used": row.used_in_assessment,
        }
        for row in evidence
    ]
    payload = json.dumps(
        {
            "verdict": verdict,
            "p_ai": round(p_ai, 4),
            "interval": [round(interval[0], 4), round(interval[1], 4)],
            "methods": rows,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    draft = deterministic_interpretation(
        verdict=verdict,
        p_ai=p_ai,
        interval=interval,
        evidence=evidence,
    ).summary_hr
    return (
        "Preoblikuj nacrt u najviše dvije kratke i jasne hrvatske rečenice razumljive osobi "
        "bez tehničkog znanja. Obavezno zadrži sva tri navedena postotka i smisao zaključka. "
        "Ne koristi riječi signal, fuzija, kalibracija ili konformalno. Ne tvrdi da je autorstvo "
        "dokazano ili sigurno. Vrati samo rečenice, bez uvoda, popisa ili JSON-a. Nacrt: "
        + draft
        + " Podaci za provjeru: "
        + payload
    )


def _json_object(text: str) -> dict[str, Any]:
    start, stop = text.find("{"), text.rfind("}")
    if start < 0 or stop <= start:
        raise ValueError("interpretation does not contain a JSON object")
    value = json.loads(text[start : stop + 1])
    if not isinstance(value, dict):
        raise ValueError("interpretation JSON is not an object")
    return value


def _partial_json_string(text: str, key: str) -> str:
    """Recover one complete JSON string when generation stopped before the final brace."""
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"', text)
    if match is None:
        return ""
    try:
        value = json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        return ""
    return value if isinstance(value, str) else ""


def interpret_with_gemma(
    runtime: CompletionRuntime,
    *,
    verdict: str,
    p_ai: float,
    interval: tuple[float, float],
    evidence: list[AgentEvidence],
) -> InterpretationBundle:
    """Ask Gemma to verbalize fixed numbers; invalid output falls back deterministically."""
    fallback = deterministic_interpretation(
        verdict=verdict, p_ai=p_ai, interval=interval, evidence=evidence
    )
    required_numbers = (_hr_percent(p_ai), _hr_percent(interval[0]), _hr_percent(interval[1]))
    try:
        raw = runtime.complete(
            _prompt(verdict=verdict, p_ai=p_ai, interval=interval, evidence=evidence),
            max_tokens=120,
        )
    except Exception as exc:  # noqa: BLE001 - interpretation must not invalidate scoring
        LOGGER.warning("Gemma interpretation fell back to deterministic copy: %s", exc)
        return fallback
    try:
        generated = _GeneratedPayload.model_validate(_json_object(raw))
    except (json.JSONDecodeError, TypeError, ValueError):
        recovered = _partial_json_string(raw, "summary_hr")
        summary = _short_sentence(
            recovered or (raw if not raw.lstrip().startswith("{") else ""), limit=32
        )
        if not _acceptable_summary(summary, required_numbers=required_numbers):
            return fallback
        agents = dict(fallback.agents_hr)
        for row in evidence:
            recovered_agent = _short_sentence(_partial_json_string(raw, row.name), limit=20)
            if recovered_agent:
                agents[row.name] = recovered_agent
        return InterpretationBundle(
            summary_hr=summary,
            agents_hr=agents,
            source="gemma2_2b_it",
        )
    summary = _short_sentence(generated.summary_hr, limit=32)
    if not _acceptable_summary(summary, required_numbers=required_numbers):
        return fallback
    agents = dict(fallback.agents_hr)
    for row in evidence:
        generated_text = generated.agents_hr.get(row.name, "")
        sentence = _short_sentence(generated_text, limit=20)
        if sentence:
            agents[row.name] = sentence
    return InterpretationBundle(
        summary_hr=summary,
        agents_hr=agents,
        source="gemma2_2b_it",
    )
