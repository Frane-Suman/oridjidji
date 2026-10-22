"""Efficient Advanced segment scoring from one request-scoped causal-model pass."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Sequence
from typing import Protocol, cast

import numpy as np

from oridjidji.detectors.backbone import ReducedScores
from oridjidji.detectors.windowing import slice_reduced_words
from oridjidji.pipeline.orchestrator import PipelineOutcome, PreparedDetector
from oridjidji.schemas import AgentResult, AgentStatus
from oridjidji.segmentation.aggregate import SpanScore, aggregate_windows
from oridjidji.segmentation.windows import Window, build_windows


class Detector(Protocol):
    name: str

    def score(self, text: str) -> AgentResult: ...


SegmentFinalizer = Callable[[list[AgentResult]], float | None]


def _evidence_offsets(
    text: str, tokenizer: object, evidence: ReducedScores, count: int
) -> list[tuple[int, int]]:
    """Recover exact source spans for the prefix evidence without retokenizing heuristics."""
    if not callable(tokenizer) or not str(evidence.window_policy).startswith("prefix"):
        return []
    encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [(int(start), int(stop)) for start, stop in encoded["offset_mapping"]]
    target = [int(value) for value in evidence.token_ids[:count]]
    for start in range(max(0, len(ids) - len(target) + 1)):
        if ids[start : start + len(target)] == target:
            return offsets[start : start + len(target)]
    return []


def _rank_weights(values: list[float]) -> list[float]:
    """Map local evidence to stable relative ranks while assigning ties the same weight."""
    if not values:
        return []
    if len(values) == 1:
        return [0.5]
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    denominator = float(len(values) - 1)
    return [
        float(
            (
                np.searchsorted(ordered, value, side="left")
                + np.searchsorted(ordered, value, side="right")
                - 1
            )
            / (2.0 * denominator)
        )
        for value in values
    ]


def token_heatmap(
    text: str, outcome: PipelineOutcome, detectors: Sequence[Detector], *, limit: int = 512
) -> list[dict[str, object]]:
    """Map Fast-DetectGPT's token contributions back onto the exact submitted words.

    The weights are relative ranks inside this document. They support interpretation but
    are deliberately not presented as per-word authorship probabilities.
    """
    statistical = next(
        (
            cast(PreparedDetector, detector)
            for detector in detectors
            if detector.name in {"binoculars", "binoculars_v4"} and hasattr(detector, "prepare_key")
        ),
        None,
    )
    if statistical is None:
        return []
    evidence = (outcome.prepared_evidence or {}).get(statistical.prepare_key)
    if not isinstance(evidence, ReducedScores) or not evidence.n_tokens:
        return []
    backbone = getattr(statistical, "backbone", None)
    tokenizer = getattr(backbone, "tokenizer", None)
    if tokenizer is None:
        return []
    count = min(evidence.n_tokens, limit)
    offsets = _evidence_offsets(text, tokenizer, evidence, count)
    if len(offsets) != count:
        return []
    variance = np.maximum(np.asarray(evidence.logprob_variances[:count], dtype=np.float64), 1e-8)
    local = (
        np.asarray(evidence.realised_logprobs[:count], dtype=np.float64)
        - np.asarray(evidence.expected_logprobs[:count], dtype=np.float64)
    ) / np.sqrt(variance)
    rows: list[dict[str, object]] = []
    scored_values: list[float] = []
    scored_indexes: list[int] = []
    for match in re.finditer(r"\s+|\S+", text):
        start, stop = match.span()
        overlapping = [
            float(local[index])
            for index, (token_start, token_stop) in enumerate(offsets)
            if token_stop > start and token_start < stop and token_stop > token_start
        ]
        row: dict[str, object] = {
            "text": match.group(0),
            "start": start,
            "end": stop,
            "scored": bool(overlapping) and not match.group(0).isspace(),
            "weight": 0.0,
        }
        rows.append(row)
        if row["scored"]:
            scored_indexes.append(len(rows) - 1)
            scored_values.append(float(np.mean(overlapping)))
    for index, weight in zip(scored_indexes, _rank_weights(scored_values), strict=True):
        rows[index]["weight"] = weight
    return rows


async def score_advanced_segments(
    text: str,
    *,
    outcome: PipelineOutcome,
    detectors: Sequence[Detector],
    finalizer: SegmentFinalizer,
    min_words: int,
) -> list[SpanScore]:
    """Score windows while reusing A2/A3 evidence and rerunning only the small A1 encoder."""
    windows, scores = await score_advanced_window_probabilities(
        text,
        outcome=outcome,
        detectors=detectors,
        finalizer=finalizer,
        min_words=min_words,
    )
    if not windows:
        return []
    return aggregate_windows(
        windows,
        scores,
        total_words=len(text.split()),
    )


async def score_advanced_window_probabilities(
    text: str,
    *,
    outcome: PipelineOutcome,
    detectors: Sequence[Detector],
    finalizer: SegmentFinalizer,
    min_words: int,
) -> tuple[list[Window], list[float | None]]:
    """Return window probabilities from one document-level causal-model pass."""
    windows = build_windows(text, min_words=min_words)
    if not windows:
        return [], []
    encoder = next((detector for detector in detectors if detector.name == "encoder"), None)
    statistical = [
        cast(PreparedDetector, detector)
        for detector in detectors
        if detector.name in {"binoculars", "fast_detect_gpt"}
        and all(hasattr(detector, name) for name in ("prepare_key", "score_prepared"))
    ]
    if encoder is None or {detector.name for detector in statistical} != {
        "binoculars",
        "fast_detect_gpt",
    }:
        return windows, [None] * len(windows)
    prepared = outcome.prepared_evidence or {}
    evidence = prepared.get(statistical[0].prepare_key)
    if not isinstance(evidence, ReducedScores):
        return windows, [None] * len(windows)

    scores: list[float | None] = []
    for window in windows:
        try:
            reduced = slice_reduced_words(
                evidence, start_word=window.start_word, end_word=window.end_word
            )
            encoder_result = await asyncio.to_thread(encoder.score, window.text)
            results = [
                encoder_result,
                *[detector.score_prepared(reduced) for detector in statistical],
            ]
            if any(result.status is not AgentStatus.OK for result in results):
                scores.append(None)
                continue
            scores.append(finalizer(results))
        except (ValueError, TypeError):
            scores.append(None)
    return windows, scores
