"""Long-lived detector runtime shared by the web API and native evaluation.

This module stops the scientific evaluator at the same boundary as the web
application: preprocessing, detector execution, fusion, calibration and conformal
prediction are shared, while HTTP admission, history and Croatian presentation stay
in the API composition root.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

from oridjidji.api.analysis_service import DocumentScore, score_for_mode
from oridjidji.api.runtime_loading import (
    FusionLayer,
    load_detectors,
    load_fusion,
    load_probability_layer,
    verify_bundle,
)
from oridjidji.config import AppConfig
from oridjidji.pipeline import Orchestrator, PipelineOutcome
from oridjidji.preprocessing import preprocess
from oridjidji.schemas import AgentResult, Document

RuntimeMode = Literal["osnovni", "napredni"]
InputForm = Literal["raw", "normalized"]


class EvaluationInputError(ValueError):
    """An evaluator-only input violates its registered text contract."""


@dataclass(frozen=True)
class RuntimeAnalysis:
    """Model-facing result before API presentation and persistence."""

    document: Document
    outcome: PipelineOutcome
    score: DocumentScore
    elapsed_ms: int


@dataclass(frozen=True)
class PairedRuntimeAnalysis:
    """Basic and Advanced results derived from one safe Advanced roster pass."""

    basic: RuntimeAnalysis
    advanced: RuntimeAnalysis
    shared_encoder_pass: bool


class AnalysisRuntime:
    """Own one loaded detector roster and the layers that interpret its scores."""

    def __init__(
        self,
        detectors: list[Any],
        *,
        basic_layer: Any | None,
        fusion: FusionLayer | None,
        timeout_seconds: float,
        advanced_execution_policy: Literal["sequential", "concurrent"] = "concurrent",
        warmup: bool = False,
        score_function: Callable[..., DocumentScore] = score_for_mode,
    ) -> None:
        self.detectors = detectors
        self.basic_layer = basic_layer
        self.fusion = fusion
        self._score_function = score_function
        names = {detector.name for detector in detectors}
        self.advanced_configured = fusion is not None and set(fusion.roster).issubset(names)
        self.orchestrators = {
            "osnovni": Orchestrator(
                [detector for detector in detectors if detector.name == "encoder"],
                timeout_s=timeout_seconds,
                execution_policy="sequential",
            ),
            "napredni": Orchestrator(
                detectors,
                timeout_s=timeout_seconds,
                execution_policy=advanced_execution_policy,
            ),
        }
        self._closed = False
        if warmup:
            self._warmup()

    @classmethod
    def from_config(cls, cfg: AppConfig, *, warmup: bool = False) -> AnalysisRuntime:
        """Verify and load one configured runtime without constructing an HTTP app."""
        verify_bundle(cfg)
        policy = str(cfg.inference.get("advanced_execution_policy", "concurrent"))
        if policy not in {"sequential", "concurrent"}:
            raise ValueError("advanced_execution_policy must be sequential or concurrent")
        return cls(
            load_detectors(cfg),
            basic_layer=load_probability_layer(cfg),
            fusion=load_fusion(cfg),
            timeout_seconds=float(cfg.inference.get("agent_timeout_s", 30.0)),
            advanced_execution_policy=cast(Literal["sequential", "concurrent"], policy),
            warmup=warmup,
        )

    @property
    def can_share_pair(self) -> bool:
        """Whether one Advanced pass contains every score needed by both modes."""
        return self.advanced_configured and any(
            detector.name == "encoder" for detector in self.detectors
        )

    def _warmup(self) -> None:
        loaded_resources: set[int] = set()
        for detector in self.detectors:
            resource = getattr(detector, "backbone", detector)
            if id(resource) in loaded_resources:
                continue
            loaded_resources.add(id(resource))
            resource.load()

    def score_results(self, mode: RuntimeMode, results: list[AgentResult]) -> DocumentScore:
        """Apply the exact Basic or Advanced probability layer used by the API."""
        return self._score_function(
            results,
            mode=mode,
            basic_layer=self.basic_layer,
            fusion=self.fusion,
            advanced_configured=self.advanced_configured,
        )

    @staticmethod
    def prepare_evaluation_text(
        text: str,
        *,
        minimum_words: int,
        input_form: InputForm,
        check_language: bool = True,
    ) -> tuple[Document, str]:
        """Prepare registered evaluator input without exposing a public bypass.

        The public API still enforces its configured minimum. M9 can explicitly ask for
        the registered 50-word length bucket because this method is package-internal and
        has no route or request-header equivalent. ``raw`` exists solely for E3's paired
        raw-versus-normalized robustness measurement.
        """
        if minimum_words < 1:
            raise EvaluationInputError("minimum_words must be positive")
        document = preprocess(text, check_language=check_language)
        if document.n_words < minimum_words:
            raise EvaluationInputError(
                f"evaluation text has {document.n_words} words; needs {minimum_words}"
            )
        if check_language and document.language_accepted is False:
            raise EvaluationInputError(
                f"evaluation text failed the Croatian language gate: {document.language}"
            )
        return document, document.raw if input_form == "raw" else document.normalized

    async def analyze_evaluation_text(
        self,
        text: str,
        mode: RuntimeMode,
        *,
        minimum_words: int,
        input_form: InputForm = "normalized",
        check_language: bool = True,
    ) -> RuntimeAnalysis:
        """Run one evaluator input through shared preprocessing and model execution."""
        document, scored_text = self.prepare_evaluation_text(
            text,
            minimum_words=minimum_words,
            input_form=input_form,
            check_language=check_language,
        )
        return await self.analyze_document(document, scored_text, mode)

    async def analyze_document(
        self, document: Document, scored_text: str, mode: RuntimeMode
    ) -> RuntimeAnalysis:
        """Run a prepared document; callers own gates and presentation."""
        started = time.perf_counter()
        outcome = await self.orchestrators[mode].run(
            scored_text,
            finalizer=lambda results: self.score_results(mode, results),
        )
        scored = outcome.final
        if not isinstance(scored, DocumentScore):
            raise RuntimeError("pipeline finalizer did not return a document score")
        return RuntimeAnalysis(
            document=document,
            outcome=outcome,
            score=scored,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    async def analyze_pair_evaluation_text(
        self,
        text: str,
        *,
        minimum_words: int,
        input_form: InputForm = "normalized",
        check_language: bool = True,
    ) -> PairedRuntimeAnalysis:
        """Score both modes, sharing A1 when the Advanced roster safely contains it."""
        document, scored_text = self.prepare_evaluation_text(
            text,
            minimum_words=minimum_words,
            input_form=input_form,
            check_language=check_language,
        )
        if not self.can_share_pair:
            basic = await self.analyze_document(document, scored_text, "osnovni")
            advanced = await self.analyze_document(document, scored_text, "napredni")
            return PairedRuntimeAnalysis(basic, advanced, False)

        started = time.perf_counter()
        outcome = await self.orchestrators["napredni"].run(scored_text)
        encoder = [result for result in outcome.agents if result.name == "encoder"]
        basic_score = self.score_results("osnovni", encoder)
        advanced_score = self.score_results("napredni", outcome.agents)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return PairedRuntimeAnalysis(
            RuntimeAnalysis(document, outcome, basic_score, elapsed_ms),
            RuntimeAnalysis(document, outcome, advanced_score, elapsed_ms),
            True,
        )

    def close(self) -> None:
        """Close each shared native resource exactly once."""
        if self._closed:
            return
        self._closed = True
        closed: set[int] = set()
        for detector in self.detectors:
            resource = getattr(detector, "backbone", detector)
            if id(resource) in closed:
                continue
            closed.add(id(resource))
            close = getattr(resource, "close", None)
            if callable(close):
                close()
