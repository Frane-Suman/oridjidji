"""FastAPI application.

The same application exposes two first-class analysis modes. Basic runs the calibrated
BERTic encoder. Advanced runs the calibrated A1+A2+A3 fusion and uses A4 only for
linguistic explanation. A research qualification in the runtime identity describes the
evidence behind Advanced mode; it never disables the user's choice.

The stub-first discipline paid off exactly as an earlier decision predicted for generation: the
gates, the ethics furniture and the `INCONCLUSIVE` path were built and tested against
`_stub_scores`, and swapping in a real detector touched two functions and no endpoint.

**What a loaded detector does and does not change.** It changes what runs: a real
score, a real band in the ring, real latency. Whether it changes the *answer* depends on
a second artefact. A softmax output is not a probability — nothing has mapped it onto
observed frequencies of being right — and the specification requires the abstention zone to come from
split conformal prediction rather than a band someone picked.

`train calibrate` fits both on the calibration split and `load_probability_layer` finds
them. **With that layer present the API commits**: the calibrator supplies the
probability and a conformal prediction set decides whether the evidence supports a side.
**Without it the API still abstains** and `uncalibrated_notice` says why — a machine with
weights but no calibration has no map from score to probability, and inventing one is the
exact failure this project exists to avoid.

Advanced fusion consumes the agents' raw, direction-checked scores. If its statistical
backbone fails, the response explicitly reports `degraded_basic` and returns the
independently calibrated Basic result rather than pretending that partial evidence is a
complete Advanced analysis.

The web layer (brandbook, `oridjidji.web`) is served from here too. It can be built
against the stub precisely because the brandbook forbids the frontend holding model
facts — no thresholds, no detector names, no length limits — so the UI is decoupled from
detector readiness by construction rather than by luck.

The token and linguistic panels render only evidence produced by the current request.
SQLite history is wired with opt-in text retention. Neither surface invents a score when
an artefact is absent.

COMPLEXITY BUDGET (AGENTS.md, roughly 1,100 lines). Artifact discovery now lives in
`runtime_loading.py`, probability/verdict calculation in `analysis_service.py`, process
controls in `runtime_control.py`, and demo verification in `preflight.py`. This remaining
file is the composition root: its route closures deliberately share one set of loaded CPU
models, admission control, history, runtime identities and web mounts. The length is wiring;
detector mathematics and reusable decisions remain in separately tested modules.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from oridjidji import __version__
from oridjidji.api.analysis_runtime import AnalysisRuntime
from oridjidji.api.analysis_service import (
    DocumentScore,
    _prediction_labels_from,
    _verdict_from,
    score_for_mode,
)
from oridjidji.api.analysis_service import (
    _document_score as _document_score,
)
from oridjidji.api.render_result import result_fragment
from oridjidji.api.runtime_control import (
    AdvancedAdmission,
    AdvancedBusy,
    RequestRateLimiter,
    rss_mb,
)
from oridjidji.api.runtime_loading import (
    FusionLayer as FusionLayer,
)
from oridjidji.api.runtime_loading import (
    hash_verification_disabled,
    load_fusion,
    verify_bundle,
)
from oridjidji.api.runtime_loading import (
    load_detectors as load_detectors,
)
from oridjidji.api.runtime_loading import (
    load_probability_layer as load_probability_layer,
)
from oridjidji.api.runtime_loading import (
    load_statistical_detectors as load_statistical_detectors,
)
from oridjidji.api.schemas import (
    AgentEvidence,
    AnalyzeRequest,
    AnalyzeResponse,
    ConfigResponse,
    Explanations,
    FeedbackRequest,
    Gates,
    HealthResponse,
    Mode,
    RunStatus,
    RuntimeCapabilities,
    RuntimeIdentity,
    RuntimePolicy,
    StandaloneThresholdEvidence,
    StreamEvent,
)
from oridjidji.artefacts import StatisticalThresholdReport
from oridjidji.config import AppConfig, load_config
from oridjidji.explain.result_interpreter import (
    InterpretationBundle,
    deterministic_interpretation,
    interpret_with_gemma,
)
from oridjidji.i18n import (
    LIMITATION_KEYS,
    MESSAGES_HR,
    NOTICE_HR,
    VERDICT_MESSAGE_KEYS,
    get_message,
)
from oridjidji.pipeline import Orchestrator, token_heatmap
from oridjidji.preprocessing import preprocess
from oridjidji.schemas import AgentResult, AgentStatus
from oridjidji.storage import HistoryStore
from oridjidji.web import render

MODES = ["osnovni", "napredni"]

# A single analysis is bounded because inference is CPU-only on the same laptop that
# renders the page. Served to the UI rather than known by it.
MAX_WORDS = 5000
MAX_BODY_BYTES = 128 * 1024

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"


def _limitations(min_words: int) -> list[str]:
    """The permanent panel, rendered in display order."""
    return [get_message(key, min_words=min_words) for key in LIMITATION_KEYS]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_compatible_statistical_thresholds(
    candidate_path: Path, candidate: Any
) -> dict[str, StatisticalThresholdReport]:
    """Load reporting-only A2/A3 thresholds after checking every serving identity.

    These files are deliberately outside the Advanced fusion manifest because they were
    fitted later and do not affect its score. Their own index binds them back to the exact
    candidate, pair and calibration cohort before the browser may display them.
    """
    directory = candidate_path.parent / "statistical_thresholds"
    index_path = directory / "statistical_thresholds.index.json"
    if not index_path.is_file():
        return {}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("purpose") != "reporting_only_raw_fusion_unchanged":
        raise ValueError("statistical thresholds are not marked reporting-only")
    if index.get("candidate_manifest_sha256") != _sha256(candidate_path):
        raise ValueError("statistical thresholds reference a different Advanced candidate")
    pair = json.loads(Path(candidate.gguf_pair_manifest).read_text(encoding="utf-8"))
    expected_formulas = {
        "binoculars": candidate.binoculars_formula_hash,
        "fast_detect_gpt": candidate.fast_detect_gpt_formula_hash,
    }
    reports: dict[str, StatisticalThresholdReport] = {}
    for detector, item in dict(index.get("reports", {})).items():
        if detector not in expected_formulas or not isinstance(item, dict):
            raise ValueError(f"unknown statistical threshold report: {detector}")
        report_path = directory / str(item["path"])
        if _sha256(report_path) != item.get("sha256"):
            raise ValueError(f"statistical threshold report hash mismatch: {detector}")
        report = StatisticalThresholdReport.model_validate_json(
            report_path.read_text(encoding="utf-8")
        )
        identities_match = (
            report.detector == detector
            and report.report_identity == item.get("identity")
            and report.evidence_sha256 == index.get("evidence_sha256")
            and report.repository_hash == candidate.repository_hash
            and report.selection_hash == candidate.selection_hash
            and report.preprocessing_hash == candidate.preprocessing_hash
            and report.calibration_role_hash == candidate.role_hashes["calibration_fit"]
            and report.backbone_hash == candidate.statistical_backbone_hash
            and report.pair_name == candidate.statistical_pair_name
            and report.quantization == candidate.statistical_quantization
            and report.window_policy == candidate.window_policy
            and report.formula_hash == expected_formulas[detector]
            and report.tokenizer_mapping_hash == pair.get("tokenizer_mapping_sha256")
            and report.observer_weights_hash == pair.get("observer_sha256")
            and report.performer_weights_hash == pair.get("performer_sha256")
        )
        if not identities_match:
            raise ValueError(f"statistical threshold identities mismatch: {detector}")
        reports[detector] = report
    return reports


def _effective_mode(requested_mode: Mode, scored: DocumentScore) -> Mode:
    if scored.probability_source == "advanced":
        return "napredni"
    if scored.probability_source == "basic":
        return "osnovni"
    return requested_mode


def _threshold_evidence(
    report: StatisticalThresholdReport, raw_score: float
) -> list[StandaloneThresholdEvidence]:
    evidence = []
    for threshold in report.thresholds:
        overall = next(metric for metric in threshold.metrics if metric.domain == "all")
        oriented = -raw_score if report.direction == "lower_is_ai" else raw_score
        evidence.append(
            StandaloneThresholdEvidence(
                target_fpr=threshold.target_fpr,
                raw_threshold=threshold.raw_threshold,
                detector_indicates_ai=oriented >= threshold.oriented_threshold,
                achieved_fpr=overall.achieved_fpr,
                observed_tpr=overall.tpr,
                report_identity=report.report_identity,
            )
        )
    return evidence


def _build_agent_evidence(
    agents: list[AgentResult],
    *,
    requested_mode: Mode,
    scored: DocumentScore,
    fusion: FusionLayer | None,
    labels_hr: dict[str, str],
    thresholds: dict[str, StatisticalThresholdReport],
) -> tuple[list[AgentEvidence], int, int]:
    effective_mode = _effective_mode(requested_mode, scored)
    if requested_mode == "osnovni":
        assessment_names = {"encoder"}
    elif fusion is not None:
        assessment_names = set(fusion.roster)
    else:
        assessment_names = {agent.name for agent in agents if agent.name != "linguistic"}
    active_names = (
        {"encoder"}
        if effective_mode == "osnovni" and scored.probability_source == "basic"
        else assessment_names
        if scored.probability_source == "advanced"
        else set()
    )
    contributions: dict[str, float] = {}
    if (
        scored.probability_source == "advanced"
        and fusion is not None
        and fusion.ensemble is not None
    ):
        raw = {agent.name: agent.score for agent in agents}
        contributions = dict(fusion.ensemble.contributions(raw))
    rows = []
    for agent in agents:
        participates = (
            agent.name in active_names
            and agent.status is AgentStatus.OK
            and agent.score is not None
        )
        coverage_value = (agent.token_data or {}).get("coverage")
        coverage = float(coverage_value) if isinstance(coverage_value, (int, float)) else None
        report = thresholds.get(agent.name)
        rows.append(
            AgentEvidence(
                name=agent.name,
                label_hr=labels_hr[agent.name],
                rationale_hr=(
                    agent.rationale_hr
                    or get_message(
                        "ui_agent_unavailable"
                        if agent.status is AgentStatus.FAILED
                        else "ui_method_step_agents"
                    )
                ),
                raw_score=agent.score,
                direction=agent.direction,
                status=agent.status,
                latency_ms=agent.latency_ms,
                used_in_assessment=participates,
                fusion_contribution=contributions.get(agent.name),
                token_coverage=coverage,
                standalone_thresholds=(
                    _threshold_evidence(report, agent.score)
                    if report is not None and agent.score is not None
                    else []
                ),
            )
        )
    total = sum(agent.name in assessment_names for agent in agents)
    used_count = sum(row.used_in_assessment for row in rows)
    return rows, used_count, total


def _copy_report_hr(
    *,
    effective_mode: Mode,
    verdict: str,
    p_ai: float,
    interval: tuple[float, float],
    agents_used: int,
    agents_total: int,
    n_words: int,
    warnings_hr: list[str],
    interpretation_hr: str,
) -> str:
    mode_label = get_message("ui_mode_basic" if effective_mode == "osnovni" else "ui_mode_advanced")
    low, high = interval
    lines = [
        f"{get_message('ui_mode_label')}: {mode_label}",
        f"{get_message('ui_result_label')}: {get_message(VERDICT_MESSAGE_KEYS[verdict])}",
        f"{render.hr_number(p_ai * 100, decimals=1)} %",
        get_message(
            "ui_result_interval",
            low=render.hr_number(low * 100, decimals=1),
            high=render.hr_number(high * 100, decimals=1),
        ),
        interpretation_hr,
        get_message("ui_result_basis", used=agents_used, total=agents_total, n_words=n_words),
        *warnings_hr,
        get_message("disclaimer_short"),
        *NOTICE_HR,
    ]
    return "\n".join(lines)


def get_app_config() -> AppConfig:
    """Overridable in tests; the real one reads the selected local profile."""
    return load_config(os.environ.get("ORIDJIDJI_PROFILE", "lite"))


# Complexity justification: ``create_app`` is the single composition root for the HTTP
# contract. Its route closures share loaded model objects, admission control, history and
# runtime identities; moving those closures into independent modules would hide that they
# operate on the same long-lived CPU resources. Detector mathematics remains outside here.
def create_app(
    detectors: list[Any] | None = None,
    *,
    warmup: bool = False,
    history: HistoryStore | None = None,
    enforce_rate_limit: bool = True,
    persist_history: bool = True,
) -> FastAPI:
    """Build the app. Pass ``detectors`` to override discovery (tests, benchmarks).

    Discovery runs once, here, rather than per request: which detectors exist is a property
    of the machine, and re-scanning a directory on every analysis would let the ring change
    width mid-session.

    ``warmup`` builds the ONNX sessions now instead of on first use. It costs seconds of
    startup and saves the **first user** from paying them — measured at 27.9 s on the target
    laptop. It defaults to False so that constructing the app stays cheap for tests and for
    a health check that may never score anything; `serve` turns it on.
    """
    app = FastAPI(
        title="Oriđiđi",
        version=__version__,
        summary="Procjena vjerojatnosti da je hrvatski tekst strojno generiran.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def limit_analysis_body(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.method == "POST" and request.url.path in {
            "/api/v1/analyze",
            "/api/v1/analyze/stream",
        }:
            declared = request.headers.get("content-length")
            if declared is not None and int(declared) > MAX_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": MESSAGES_HR["ui_error_analysis_failed"]},
                )
            body = await request.body()
            if len(body) > MAX_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": MESSAGES_HR["ui_error_analysis_failed"]},
                )
        return await call_next(request)

    startup_cfg = get_app_config()
    verify_bundle(startup_cfg)
    loaded = load_detectors(startup_cfg) if detectors is None else detectors
    # Discovered once, like the detectors: whether calibration exists is a property of
    # the machine, and re-reading it per request would let the verdict rule change
    # mid-session.
    layer = load_probability_layer(startup_cfg)
    # Fusion is discovered once alongside the layer: which artefacts exist is a property of
    # the machine, and re-reading them per request would let the answer change mid-session.
    fusion = load_fusion(startup_cfg)
    timeout_s = float(startup_cfg.inference.get("agent_timeout_s", 30.0))
    runtime = AnalysisRuntime(
        loaded,
        basic_layer=layer,
        fusion=fusion,
        timeout_seconds=timeout_s,
        advanced_execution_policy=cast(
            Literal["sequential", "concurrent"],
            startup_cfg.inference.get("advanced_execution_policy", "concurrent"),
        ),
        score_function=score_for_mode,
    )
    agent_names = [detector.name for detector in loaded]
    agent_labels = {name: get_message(f"agent_{name}") for name in agent_names}
    agents_by_mode = {
        "osnovni": [name for name in agent_names if name == "encoder"],
        "napredni": agent_names,
    }

    # Built here rather than beside the web routes because the SSE stream renders the result
    # fragment with it too. One environment, so the live page and /pregled cannot end up with
    # two different `t()` globals or two different template search paths.
    templates = Jinja2Templates(directory=str(WEB_ROOT / "templates"))
    templates.env.globals["t"] = lambda key, **kw: get_message(key, **kw)
    templates.env.globals["ring_dots"] = render.ring_dots
    model_versions = {}
    for detector in loaded:
        provenance_version = getattr(getattr(detector, "provenance", None), "version", None)
        model_versions[detector.name] = str(
            provenance_version or getattr(detector, "version", "unknown")
        )
    if not model_versions:
        model_versions = {"detectors": "none — no exported model found (run `train export`)"}

    basic_candidate: Any | None = None
    advanced_candidate: Any | None = None
    skip_verify = hash_verification_disabled()
    if startup_cfg.inference.get("encoder_candidate_manifest"):
        from oridjidji.artefacts import EncoderCandidateManifest

        try:
            basic_candidate = EncoderCandidateManifest.load(
                Path(str(startup_cfg.inference["encoder_candidate_manifest"]))
            )
        except (OSError, ValueError):
            if not skip_verify:
                raise
            warnings.warn(
                "The Basic candidate manifest did not verify; provenance will be "
                "incomplete because hash verification is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )
    if startup_cfg.inference.get("advanced_candidate_manifest"):
        from oridjidji.fusion.advanced_bundle import AdvancedCandidateManifest

        try:
            advanced_candidate = AdvancedCandidateManifest.load(
                Path(str(startup_cfg.inference["advanced_candidate_manifest"]))
            )
        except (OSError, ValueError):
            if not skip_verify:
                raise
            warnings.warn(
                "The Advanced candidate manifest did not verify; provenance will be "
                "incomplete because hash verification is disabled.",
                RuntimeWarning,
                stacklevel=2,
            )

    statistical_thresholds: dict[str, StatisticalThresholdReport] = {}
    advanced_manifest_path = startup_cfg.inference.get("advanced_candidate_manifest")
    if advanced_candidate is not None and advanced_manifest_path:
        try:
            statistical_thresholds = _load_compatible_statistical_thresholds(
                Path(str(advanced_manifest_path)), advanced_candidate
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            warnings.warn(
                f"Standalone statistical threshold evidence was ignored: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    failed_agent_names: set[str] = set()
    advanced_installation_complete = fusion is not None and set(fusion.roster).issubset(
        set(agent_names)
    )

    def capabilities_snapshot() -> dict[str, RuntimeCapabilities]:
        """Report what this process can execute after the latest load or score attempt."""
        available = set(agent_names) - failed_agent_names
        encoder_available = "encoder" in available
        basic = RuntimeCapabilities(
            document_detection=encoder_available and layer is not None,
        )
        has_binoculars = bool({"binoculars", "binoculars_v4"} & available)
        has_fast_detect_gpt = "fast_detect_gpt" in available
        roster_available = fusion is not None and set(fusion.roster).issubset(available)
        advanced = RuntimeCapabilities(
            document_detection=encoder_available and roster_available,
            statistical_evidence=has_binoculars and has_fast_detect_gpt,
        )
        return {"osnovni": basic, "napredni": advanced}

    def runtime_identity_for(mode: Literal["osnovni", "napredni"]) -> RuntimeIdentity:
        capabilities = capabilities_snapshot()[mode]
        basic_probability = getattr(layer, "calibration_hash", None)
        if basic_candidate is not None:
            encoder_identity = basic_candidate.weights_hash
            repository = basic_candidate.repository_hash
            preprocessing = basic_candidate.preprocessing_hash
        else:
            encoder_identity = model_versions.get("encoder")
            repository = None
            preprocessing = None
        if mode == "osnovni":
            return RuntimeIdentity(
                encoder=encoder_identity,
                calibration=basic_probability,
                conformal=basic_probability,
                repository=repository,
                preprocessing=preprocessing,
                runtime_status="basic",
                probability_device="cpu",
                capabilities=capabilities,
            )
        if advanced_candidate is None:
            return RuntimeIdentity(
                encoder=encoder_identity,
                calibration=(
                    getattr(fusion.layer, "calibration_hash", None) if fusion is not None else None
                ),
                conformal=(
                    getattr(fusion.layer, "calibration_hash", None) if fusion is not None else None
                ),
                repository=repository,
                preprocessing=preprocessing,
                runtime_status=(
                    "legacy_unvalidated"
                    if advanced_installation_complete and capabilities.document_detection
                    else "advanced_unavailable"
                ),
                probability_device="cpu" if advanced_installation_complete else None,
                capabilities=capabilities,
            )
        ensemble_hash = (
            advanced_candidate.files.get(advanced_candidate.ensemble)
            if advanced_candidate.ensemble is not None
            else None
        )
        return RuntimeIdentity(
            encoder=encoder_identity,
            causal_pair=advanced_candidate.statistical_backbone_hash,
            fusion=ensemble_hash,
            calibration=advanced_candidate.advanced_probability_identity,
            conformal=advanced_candidate.advanced_probability_identity,
            repository=advanced_candidate.repository_hash,
            preprocessing=advanced_candidate.preprocessing_hash,
            runtime_status=(
                advanced_candidate.runtime_status
                if advanced_candidate.version == "advanced-candidate-v5"
                and capabilities.document_detection
                else "legacy_unvalidated"
            )
            if capabilities.document_detection
            else "advanced_unavailable",
            device_transfer_status=(
                advanced_candidate.device_transfer_status
                if advanced_candidate.version == "advanced-candidate-v5"
                else "not_checked"
            ),
            probability_device=(
                advanced_candidate.probability_device
                if advanced_candidate.version == "advanced-candidate-v5"
                else None
            ),
            limitations=[
                limitation
                for limitation in advanced_candidate.limitations
                if "segment" not in limitation.lower()
            ],
            capabilities=capabilities,
        )

    def effective_identity_for(
        requested_mode: Literal["osnovni", "napredni"], scored: DocumentScore
    ) -> RuntimeIdentity:
        """Report the probability layer that produced the returned assessment."""
        if requested_mode == "napredni" and scored.probability_source == "advanced":
            return runtime_identity_for("napredni")
        basic_identity = runtime_identity_for("osnovni")
        if requested_mode == "napredni" and scored.fallback_used:
            return basic_identity.model_copy(
                update={
                    "runtime_status": "degraded_basic",
                    "limitations": [get_message("advanced_degraded_notice")],
                    "capabilities": capabilities_snapshot()["osnovni"],
                }
            )
        if requested_mode == "napredni":
            return runtime_identity_for("napredni").model_copy(
                update={"runtime_status": "advanced_unavailable"}
            )
        return basic_identity

    history_store = history
    if history_store is None and detectors is None and persist_history:
        history_store = HistoryStore(
            Path(startup_cfg.inference.get("history_db", "data/processed/history.sqlite3"))
        )

    if warmup:
        loaded_resources: set[int] = set()
        for detector in loaded:
            # A2 and A3 point at the same object. Loading the resource rather than scoring
            # each agent avoids two warm-up forward passes while preserving graceful start.
            resource = getattr(detector, "backbone", detector)
            if id(resource) in loaded_resources:
                continue
            loaded_resources.add(id(resource))
            try:
                resource.load()
            except Exception as exc:  # noqa: BLE001 - startup remains an abstaining state
                failed_agent_names.update(
                    candidate.name
                    for candidate in loaded
                    if getattr(candidate, "backbone", candidate) is resource
                )
                warnings.warn(
                    f"Detector warm-up failed for {detector.name}: {type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def update_agent_health(results: list[AgentResult]) -> None:
        """Bind capability flags to the latest real result from each loaded resource."""
        by_name = {result.name: result for result in results}
        resource_groups: dict[int, list[str]] = {}
        for detector in loaded:
            resource = getattr(detector, "backbone", detector)
            resource_groups.setdefault(id(resource), []).append(detector.name)
        for names in resource_groups.values():
            observed = [by_name[name] for name in names if name in by_name]
            if not observed:
                continue
            if any(result.status is AgentStatus.FAILED for result in observed):
                failed_agent_names.update(names)
            elif len(observed) == len(names):
                failed_agent_names.difference_update(names)

    def score_results(
        mode: Literal["osnovni", "napredni"], results: list[AgentResult]
    ) -> DocumentScore:
        """One scoring path shared by direct POST and SSE."""
        update_agent_health(results)
        return runtime.score_results(mode, results)

    def gemma_interpreter_runtime() -> Any | None:
        """Return the already-loaded Gemma instruct runtime without loading another model."""
        for detector in loaded:
            if detector.name not in {"binoculars", "binoculars_v4"}:
                continue
            performer = getattr(getattr(detector, "backbone", None), "performer", None)
            if callable(getattr(performer, "complete", None)):
                return performer
        return None

    async def interpret_result(
        *, requested_mode: Mode, scored: DocumentScore, agents: list[AgentResult]
    ) -> InterpretationBundle:
        """Explain fixed scores; interpretation never enters calibration or fusion."""
        verdict = str(_verdict_from(scored.p_ai, scored.interval, scored.layer))
        evidence, _, _ = _build_agent_evidence(
            agents,
            requested_mode=requested_mode,
            scored=scored,
            fusion=fusion,
            labels_hr=agent_labels,
            thresholds=statistical_thresholds,
        )
        fallback = deterministic_interpretation(
            verdict=verdict,
            p_ai=scored.p_ai,
            interval=scored.interval,
            evidence=evidence,
        )
        if requested_mode != "napredni" or scored.probability_source != "advanced":
            return fallback
        runtime = gemma_interpreter_runtime()
        if runtime is None:
            return fallback
        return await asyncio.to_thread(
            interpret_with_gemma,
            runtime,
            verdict=verdict,
            p_ai=scored.p_ai,
            interval=scored.interval,
            evidence=evidence,
        )

    # The product modes are execution contracts. Basic is the supervised Croatian
    # encoder only; Advanced runs every available evidence agent. Keeping separate
    # orchestrators also makes SSE event counts truthful for the selected mode.
    orchestrators = runtime.orchestrators
    advanced_admission = AdvancedAdmission()
    deferred_releases: set[asyncio.Task[None]] = set()
    rate_limiter = RequestRateLimiter(
        int(startup_cfg.inference.get("analysis_requests_per_minute", 120))
    )

    async def check_rate(request: Request) -> None:
        # The native M9 evaluator constructs this same application and calls it through
        # an in-process TestClient.  It must exercise the production preprocessing,
        # orchestration, calibration and response contract, but a dataset evaluation is
        # not an interactive HTTP client and must not trip the per-address web limiter.
        if not enforce_rate_limit:
            return
        address = request.client.host if request.client is not None else "local"
        await rate_limiter.check(address)

    async def acquire_advanced(mode: Literal["osnovni", "napredni"]) -> bool:
        if mode != "napredni":
            return False
        try:
            await advanced_admission.acquire()
        except AdvancedBusy as exc:
            raise HTTPException(
                status_code=429,
                detail=MESSAGES_HR["ui_error_analysis_failed"],
                headers={"Retry-After": str(max(1, int(timeout_s)))},
            ) from exc
        return True

    async def release_advanced(orchestrator: Orchestrator, acquired: bool) -> None:
        if not acquired:
            return
        if orchestrator.has_pending_native_work:

            async def drain_and_release() -> None:
                await orchestrator.wait_until_native_idle()
                await advanced_admission.release()

            task = asyncio.create_task(drain_and_release())
            deferred_releases.add(task)
            task.add_done_callback(deferred_releases.discard)
            return
        await advanced_admission.release()

    def close_models() -> None:
        runtime.close()

    app.router.add_event_handler("shutdown", close_models)

    def advanced_explanations(
        text: str,
        mode: str,
        outcome: Any,
    ) -> Explanations:
        """Build Advanced explanation panels from the existing request evidence."""
        if mode != "napredni":
            return Explanations()
        heatmap = token_heatmap(text, outcome, loaded)
        top_features = [
            feature
            for agent in outcome.agents
            if agent.name == "linguistic" and agent.token_data
            for feature in agent.token_data.get("top_features", [])
            if isinstance(feature, dict)
        ]
        return Explanations(token_heatmap=heatmap, top_features=top_features)

    def prepare_input(text: str, cfg: AppConfig) -> tuple[Any, list[str], str | None]:
        """Run the identical gates and warnings for direct and streaming requests."""
        document = preprocess(text)
        response_warnings: list[str] = []
        if document.language_message_hr:
            response_warnings.append(document.language_message_hr)
        if document.n_homoglyphs_normalized:
            response_warnings.append(get_message("homoglyph_warning"))
        refusal: str | None = None
        if document.language_accepted is False:
            refusal = get_message("lang_rejected", lang=document.language)
        elif document.n_words < cfg.min_word_count:
            refusal = get_message("too_short", min_words=cfg.min_word_count)
        elif document.n_words > MAX_WORDS:
            refusal = MESSAGES_HR["ui_error_too_long"]
        return document, response_warnings, refusal

    def build_response(
        *,
        run_id: str,
        requested_mode: Mode,
        cfg: AppConfig,
        document: Any,
        outcome: Any,
        scored: DocumentScore,
        effective_identity: RuntimeIdentity,
        explanations: Explanations,
        interpretation: InterpretationBundle,
        response_warnings: list[str],
        elapsed_ms: int,
    ) -> AnalyzeResponse:
        """Construct the one final response shape used by POST and POST/SSE."""
        verdict = _verdict_from(scored.p_ai, scored.interval, scored.layer)
        effective_mode = _effective_mode(requested_mode, scored)
        evidence, agents_used, agents_total = _build_agent_evidence(
            outcome.agents,
            requested_mode=requested_mode,
            scored=scored,
            fusion=fusion,
            labels_hr=agent_labels,
            thresholds=statistical_thresholds,
        )
        evidence = [
            row.model_copy(
                update={
                    "interpretation_hr": interpretation.agents_hr.get(row.name, row.rationale_hr)
                }
            )
            for row in evidence
        ]
        return AnalyzeResponse(
            run_id=run_id,
            requested_mode=requested_mode,
            effective_mode=effective_mode,
            verdict=verdict,
            p_ai=scored.p_ai,
            operating_decision_ai=scored.operating_decision_ai,
            interval=scored.interval,
            prediction_set=_prediction_labels_from(scored.p_ai, scored.layer),
            agents=outcome.agents,
            agent_evidence=evidence,
            assessment_agents_used=agents_used,
            assessment_agents_total=agents_total,
            interpretation_hr=interpretation.summary_hr,
            interpretation_source=interpretation.source,
            explanations=explanations,
            gates=Gates(
                min_length_ok=True,
                n_words=document.n_words,
                language=document.language,
                language_confidence=document.language_confidence,
                language_accepted=document.language_accepted,
                is_bcs=document.language_is_bcs,
            ),
            warnings=response_warnings,
            disclaimer_hr=MESSAGES_HR["disclaimer_short"],
            limitations_hr=_limitations(cfg.min_word_count),
            copy_report_hr=_copy_report_hr(
                effective_mode=effective_mode,
                verdict=str(verdict),
                p_ai=scored.p_ai,
                interval=scored.interval,
                agents_used=agents_used,
                agents_total=agents_total,
                n_words=document.n_words,
                warnings_hr=response_warnings,
                interpretation_hr=interpretation.summary_hr,
            ),
            model_versions=model_versions,
            runtime_identity=effective_identity,
            elapsed_ms=elapsed_ms,
        )

    @app.get("/api/v1/health")
    def health() -> HealthResponse:
        health_mode: Literal["osnovni", "napredni"] = (
            "napredni" if advanced_candidate is not None else "osnovni"
        )
        health_capabilities = capabilities_snapshot()[health_mode]
        health_status = "ok" if health_capabilities.document_detection else "degraded"
        if (
            health_capabilities.document_detection
            and advanced_candidate is not None
            and (
                advanced_candidate.version != "advanced-candidate-v5"
                or advanced_candidate.runtime_status == "research_only"
            )
        ):
            health_status = "research"
        return HealthResponse(
            status=health_status,
            version=__version__,
            detectors_loaded=agent_names,
            bundle_version=", ".join(sorted(model_versions.values())),
            rss_mb=rss_mb(),
            runtime_identity=runtime_identity_for(health_mode),
            runtime_policy=RuntimePolicy(
                min_words=startup_cfg.min_word_count,
                advanced_timeout_seconds=int(timeout_s),
                roster=(
                    tuple(advanced_candidate.roster)
                    if advanced_candidate is not None
                    else ("encoder",)
                ),
                basic_threshold_identity=(
                    _sha256(Path(basic_candidate.probability_layer))
                    if basic_candidate is not None
                    else None
                ),
                advanced_threshold_identity=(
                    _sha256(Path(advanced_candidate.advanced_probability_layer))
                    if advanced_candidate is not None
                    else None
                ),
            ),
        )

    @app.get("/api/v1/config")
    def config(cfg: Annotated[AppConfig, Depends(get_app_config)]) -> ConfigResponse:
        capabilities_by_mode = capabilities_snapshot()
        return ConfigResponse(
            modes=MODES,
            min_words=cfg.min_word_count,
            max_words=MAX_WORDS,
            # Band count comes from here: the ring draws exactly the detectors
            # the API says it has, so a machine with no weights renders no bands rather
            # than placeholder ones.
            agents=agent_names,
            agents_by_mode=agents_by_mode,
            capabilities_by_mode=capabilities_by_mode,
            agent_labels_hr=agent_labels,
            disclaimer_hr=MESSAGES_HR["disclaimer_short"],
            limitations_hr=_limitations(cfg.min_word_count),
            verdict_labels_hr={
                verdict: MESSAGES_HR[key] for verdict, key in VERDICT_MESSAGE_KEYS.items()
            },
        )

    @app.post("/api/v1/analyze")
    async def analyze(
        payload: AnalyzeRequest,
        http_request: Request,
        cfg: Annotated[AppConfig, Depends(get_app_config)],
    ) -> AnalyzeResponse:
        await check_rate(http_request)
        started = time.perf_counter()
        run_id = str(uuid.uuid4())
        identity = runtime_identity_for(payload.mode)
        if history_store is not None:
            history_store.begin(
                run_id,
                submitted_text=payload.text,
                store_text=payload.store,
                mode=payload.mode,
                domain_hint=payload.domain_hint,
                runtime_identity=identity.model_dump(mode="json"),
            )
        document, response_warnings, refusal_detail = prepare_input(payload.text, cfg)

        # Refuse rather than guess. A short or non-Croatian input has no defensible
        # answer, and returning one anyway is the failure these rules exist to stop.
        if refusal_detail is not None:
            if history_store is not None:
                history_store.finish(
                    {
                        "run_id": run_id,
                        "runtime_identity": identity.model_dump(mode="json"),
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    },
                    status="refused",
                    error={"status_code": 422, "detail": refusal_detail},
                )
            raise HTTPException(status_code=422, detail=refusal_detail)

        # Normalised, not raw: the specification's normalisation is the homoglyph defence, and the
        # training corpus had no homoglyphs in it — so this is simultaneously the input
        # closest to what the model learned and the one an attacker cannot dress up.
        # E3 measures whether that holds.
        try:
            acquired = await acquire_advanced(payload.mode)
        except HTTPException as exc:
            if history_store is not None:
                history_store.finish(
                    {
                        "run_id": run_id,
                        "runtime_identity": identity.model_dump(mode="json"),
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    },
                    status="refused",
                    error={"status_code": exc.status_code, "detail": exc.detail},
                )
            raise
        orchestrator = orchestrators[payload.mode]
        try:
            outcome = await orchestrator.run(
                document.normalized,
                finalizer=lambda results: score_results(payload.mode, results),
            )
        except asyncio.CancelledError:
            if history_store is not None:
                history_store.finish(
                    {
                        "run_id": run_id,
                        "runtime_identity": identity.model_dump(mode="json"),
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    },
                    status="failed",
                    error={"type": "CancelledError", "detail": "request disconnected"},
                )
            await release_advanced(orchestrator, acquired)
            raise
        except Exception as exc:
            if history_store is not None:
                history_store.finish(
                    {
                        "run_id": run_id,
                        "runtime_identity": identity.model_dump(mode="json"),
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    },
                    status="failed",
                    error={"type": type(exc).__name__, "detail": str(exc)[:300]},
                )
            await release_advanced(orchestrator, acquired)
            raise
        try:
            scored = outcome.final
            if not isinstance(scored, DocumentScore):
                raise RuntimeError("pipeline finalizer did not return a document score")
            effective_identity = effective_identity_for(payload.mode, scored)
            response_warnings += scored.warnings
            explanations = advanced_explanations(document.normalized, payload.mode, outcome)
            interpretation = await interpret_result(
                requested_mode=payload.mode, scored=scored, agents=outcome.agents
            )
            response = build_response(
                run_id=run_id,
                requested_mode=payload.mode,
                cfg=cfg,
                document=document,
                outcome=outcome,
                scored=scored,
                effective_identity=effective_identity,
                explanations=explanations,
                interpretation=interpretation,
                response_warnings=response_warnings,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
            if history_store is not None:
                history_store.finish(response.model_dump(mode="json"))
            return response
        except asyncio.CancelledError:
            if history_store is not None:
                history_store.finish(
                    {
                        "run_id": run_id,
                        "runtime_identity": identity.model_dump(mode="json"),
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    },
                    status="failed",
                    error={"type": "CancelledError", "detail": "request disconnected"},
                )
            raise
        except Exception as exc:
            if history_store is not None:
                history_store.finish(
                    {
                        "run_id": run_id,
                        "runtime_identity": identity.model_dump(mode="json"),
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    },
                    status="failed",
                    error={"type": type(exc).__name__, "detail": str(exc)[:300]},
                )
            raise
        finally:
            await release_advanced(orchestrator, acquired)

    @app.get("/api/v1/runs")
    def runs(
        verdict: str | None = None,
        domain_hint: str | None = None,
        status: RunStatus | None = None,
        mode: Mode | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        if history_store is None:
            return []
        return history_store.list(
            verdict=verdict, domain_hint=domain_hint, status=status, mode=mode, limit=limit
        )

    @app.get("/api/v1/runs.csv")
    def runs_csv(
        verdict: str | None = None,
        domain_hint: str | None = None,
        status: RunStatus | None = None,
        mode: Mode | None = None,
    ) -> Response:
        body = (
            history_store.export_csv(
                verdict=verdict, domain_hint=domain_hint, status=status, mode=mode
            )
            if history_store
            else ""
        )
        return Response(
            body,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="oridjidji-history.csv"'},
        )

    @app.get("/api/v1/runs/{run_id}")
    def run_by_id(run_id: str) -> dict[str, Any]:
        row = history_store.get(run_id) if history_store else None
        if row is None:
            raise HTTPException(status_code=404, detail=get_message("ui_history_not_found"))
        return row

    @app.post("/api/v1/runs/{run_id}/feedback")
    def feedback(run_id: str, request: FeedbackRequest) -> dict[str, str]:
        if history_store is None or not history_store.add_feedback(
            run_id, request.expected_label, request.comment
        ):
            raise HTTPException(status_code=404, detail=get_message("ui_history_not_found"))
        return {"status": get_message("ui_feedback_saved")}

    @app.post("/api/v1/analyze/stream")
    async def analyze_stream(
        payload: AnalyzeRequest,
        http_request: Request,
        cfg: Annotated[AppConfig, Depends(get_app_config)],
    ) -> StreamingResponse:
        """Stream truthful transitions for the same typed request as ``analyze``."""
        await check_rate(http_request)
        started = time.perf_counter()
        run_id = str(uuid.uuid4())
        identity = runtime_identity_for(payload.mode)
        if history_store is not None:
            history_store.begin(
                run_id,
                submitted_text=payload.text,
                store_text=payload.store,
                mode=payload.mode,
                domain_hint=payload.domain_hint,
                runtime_identity=identity.model_dump(mode="json"),
            )
        document, response_warnings, refusal = prepare_input(payload.text, cfg)

        acquired = False
        if refusal is None:
            try:
                acquired = await acquire_advanced(payload.mode)
            except HTTPException as exc:
                if history_store is not None:
                    history_store.finish(
                        {
                            "run_id": run_id,
                            "runtime_identity": identity.model_dump(mode="json"),
                            "elapsed_ms": int((time.perf_counter() - started) * 1000),
                        },
                        status="refused",
                        error={"status_code": exc.status_code, "detail": exc.detail},
                    )
                raise

        def elapsed() -> int:
            return int((time.perf_counter() - started) * 1000)

        async def events() -> AsyncIterator[str]:
            def send(event: StreamEvent) -> str:
                return f"event: {event.event}\ndata: {event.model_dump_json()}\n\n"

            orchestrator = orchestrators[payload.mode]
            try:
                yield send(
                    StreamEvent(
                        event="step",
                        step_hr=MESSAGES_HR["ui_progress_preparing"],
                        elapsed_ms=elapsed(),
                    )
                )
                if refusal is not None:
                    if history_store is not None:
                        history_store.finish(
                            {
                                "run_id": run_id,
                                "runtime_identity": identity.model_dump(mode="json"),
                                "elapsed_ms": elapsed(),
                            },
                            status="refused",
                            error={"status_code": 422, "detail": refusal},
                        )
                    yield send(StreamEvent(event="error", step_hr=refusal, elapsed_ms=elapsed()))
                    return

                queue: asyncio.Queue[StreamEvent] = asyncio.Queue()

                async def pipeline_event(event: Any) -> None:
                    if event.event == "run_started":
                        label = get_message("ui_progress_pipeline_started")
                    elif event.event == "agent_started":
                        label = get_message(
                            "ui_progress_agent_started",
                            agent=agent_labels.get(event.agent, event.agent),
                        )
                    elif event.event == "agent_failed":
                        label = get_message(
                            "ui_progress_agent_failed",
                            agent=agent_labels.get(event.agent, event.agent),
                        )
                    elif event.event == "agent_finished":
                        label = get_message(
                            "ui_progress_agent_finished",
                            agent=agent_labels.get(event.agent, event.agent),
                        )
                    elif event.event == "fusion_started":
                        label = get_message("ui_progress_fusion_started")
                    else:
                        label = get_message("ui_progress_pipeline_finished")
                    await queue.put(
                        StreamEvent(
                            event=event.event,
                            step_hr=label,
                            agent=event.agent,
                            index=event.index,
                            of=event.of,
                            elapsed_ms=event.elapsed_ms,
                        )
                    )

                pipeline_task = asyncio.create_task(
                    orchestrator.run(
                        document.normalized,
                        on_event=pipeline_event,
                        finalizer=lambda results: score_results(payload.mode, results),
                        emit_run_finished=False,
                    )
                )
                while not pipeline_task.done() or not queue.empty():
                    try:
                        yield send(await asyncio.wait_for(queue.get(), timeout=0.1))
                    except TimeoutError:
                        continue
                outcome = await pipeline_task
                scored = outcome.final
                if not isinstance(scored, DocumentScore):
                    raise RuntimeError("pipeline finalizer did not return a document score")
                effective_identity = effective_identity_for(payload.mode, scored)
                response_warnings.extend(scored.warnings)

                explanations = advanced_explanations(document.normalized, payload.mode, outcome)
                yield send(
                    StreamEvent(
                        event="step",
                        step_hr=get_message("ui_progress_interpretation"),
                        elapsed_ms=elapsed(),
                    )
                )
                interpretation = await interpret_result(
                    requested_mode=payload.mode, scored=scored, agents=outcome.agents
                )
                yield send(
                    StreamEvent(
                        event="run_finished",
                        step_hr=get_message("ui_progress_pipeline_finished"),
                        of=len(outcome.agents),
                        elapsed_ms=elapsed(),
                    )
                )

                response = build_response(
                    run_id=run_id,
                    requested_mode=payload.mode,
                    cfg=cfg,
                    document=document,
                    outcome=outcome,
                    scored=scored,
                    effective_identity=effective_identity,
                    explanations=explanations,
                    interpretation=interpretation,
                    response_warnings=response_warnings,
                    elapsed_ms=elapsed(),
                )
                if history_store is not None:
                    history_store.finish(response.model_dump(mode="json"))
                body = json.dumps(
                    {
                        "response": json.loads(response.model_dump_json()),
                        "html": result_fragment(response, templates),
                    },
                    ensure_ascii=False,
                )
                yield f"event: result\ndata: {body}\n\n"
            except asyncio.CancelledError:
                if history_store is not None:
                    history_store.finish(
                        {
                            "run_id": run_id,
                            "runtime_identity": identity.model_dump(mode="json"),
                            "elapsed_ms": elapsed(),
                        },
                        status="failed",
                        error={"type": "CancelledError", "detail": "request disconnected"},
                    )
                raise
            except Exception as exc:
                if history_store is not None:
                    history_store.finish(
                        {
                            "run_id": run_id,
                            "runtime_identity": identity.model_dump(mode="json"),
                            "elapsed_ms": elapsed(),
                        },
                        status="failed",
                        error={"type": type(exc).__name__, "detail": str(exc)[:300]},
                    )
                yield send(
                    StreamEvent(
                        event="error",
                        step_hr=MESSAGES_HR["ui_error_analysis_failed"],
                        elapsed_ms=elapsed(),
                    )
                )
            finally:
                await release_advanced(orchestrator, acquired)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------ web layer
    app.mount("/static", StaticFiles(directory=str(WEB_ROOT / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    def home(request: Request, cfg: Annotated[AppConfig, Depends(get_app_config)]) -> HTMLResponse:
        ui_config = {
            "minWords": cfg.min_word_count,
            "maxWords": MAX_WORDS,
            "agents": agent_names,
            "agentsByMode": agents_by_mode,
            "defaultMode": "osnovni",
            "preparing": MESSAGES_HR["ui_progress_preparing"],
            "remaining": MESSAGES_HR["ui_editor_remaining"],
            "ready": MESSAGES_HR["ui_editor_ready"],
            "overLimit": MESSAGES_HR["ui_editor_over_limit"],
            "disabled": {
                "empty": MESSAGES_HR["ui_editor_disabled_empty"],
                "short": MESSAGES_HR["ui_editor_disabled_short"],
                "long": MESSAGES_HR["ui_editor_disabled_long"],
            },
            "agentStates": {
                state: get_message(f"ui_agent_state_{state}")
                for state in ("pending", "started", "finished", "failed")
            },
            "tokenNumericShow": MESSAGES_HR["ui_token_numeric_show"],
            "tokenNumericHide": MESSAGES_HR["ui_token_numeric_hide"],
            "copied": MESSAGES_HR["ui_result_copied"],
            "copyFailed": MESSAGES_HR["ui_result_copy_failed"],
            "errors": {
                "noText": MESSAGES_HR["ui_error_no_text"],
                "tooShort": MESSAGES_HR["ui_error_too_short"],
                "tooLong": MESSAGES_HR["ui_error_too_long"],
                "tooLarge": MESSAGES_HR["ui_error_too_large"],
                "rateLimited": MESSAGES_HR["ui_error_rate_limited"],
                "invalid": MESSAGES_HR["ui_error_invalid_request"],
                "failed": MESSAGES_HR["ui_error_analysis_failed"],
                "dropped": MESSAGES_HR["ui_error_stream_dropped"],
            },
            "wordCount": MESSAGES_HR["ui_editor_wordcount"],
        }
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "agents": agent_names,
                "agent_labels": agent_labels,
                "mode_options": [
                    ("osnovni", get_message("ui_mode_basic")),
                    ("napredni", get_message("ui_mode_advanced")),
                ],
                "status": MESSAGES_HR["ui_progress_preparing"],
                "config_json": json.dumps(ui_config, ensure_ascii=False),
            },
        )

    def history_page_config() -> dict[str, object]:
        """Labels and endpoint behavior for history pages, sourced from Croatian i18n."""
        return {
            "modeLabels": {
                "osnovni": get_message("ui_mode_basic"),
                "napredni": get_message("ui_mode_advanced"),
            },
            "domainLabels": {
                "academic": get_message("ui_domain_academic"),
                "news": get_message("ui_domain_news"),
                "": get_message("ui_domain_auto"),
            },
            "verdictLabels": {
                verdict: get_message(key) for verdict, key in VERDICT_MESSAGE_KEYS.items()
            },
            "statusLabels": {
                status: get_message(f"ui_history_status_{status}")
                for status in ("started", "succeeded", "refused", "failed")
            },
            "failed": get_message("ui_history_failed"),
            "loadFailed": get_message("ui_history_failed"),
            "feedbackFailed": get_message("ui_feedback_failed"),
            "unknown": get_message("ui_history_unknown"),
            "feedbackSaved": get_message("ui_feedback_saved"),
            "runTitle": get_message("ui_history_run", run_id="{run_id}"),
        }

    @app.get("/povijest", response_class=HTMLResponse)
    def history_page(request: Request) -> HTMLResponse:
        """Render the local result-history shell; rows are fetched through the typed API."""
        status_options = [
            (status, get_message(f"ui_history_status_{status}"))
            for status in ("started", "succeeded", "refused", "failed")
        ]
        verdict_options = [
            (verdict, get_message(key)) for verdict, key in VERDICT_MESSAGE_KEYS.items()
        ]
        return templates.TemplateResponse(
            request=request,
            name="povijest.html",
            context={
                "history_config_json": json.dumps(history_page_config(), ensure_ascii=False),
                "status_options": status_options,
                "verdict_options": verdict_options,
            },
        )

    @app.get("/povijest/{run_id}", response_class=HTMLResponse)
    def history_detail_page(request: Request, run_id: str) -> HTMLResponse:
        """Render one history-detail shell without embedding retained text in HTML."""
        return templates.TemplateResponse(
            request=request,
            name="povijest_detail.html",
            context={
                "run_id_json": json.dumps(run_id, ensure_ascii=False),
                "history_config_json": json.dumps(history_page_config(), ensure_ascii=False),
            },
        )

    @app.get("/pregled", response_class=HTMLResponse)
    def review(
        request: Request, cfg: Annotated[AppConfig, Depends(get_app_config)]
    ) -> HTMLResponse:
        """Every result and error state side by side, from fixtures.

        A development route, and labelled as one on the page. It exists because the
        self-critique gate requires the three result states screenshotted together, and
        the stub only ever returns INCONCLUSIVE — so the states cannot otherwise be
        compared, which is exactly the comparison the specification says decides whether the design
        works.
        """
        limitations = _limitations(cfg.min_word_count)
        fixtures = [
            ("INCONCLUSIVE", 0.5, (0.15, 0.85), "osnovni", "osnovni", ""),
            ("LIKELY_AI", 0.784, (0.641, 0.882), "osnovni", "osnovni", ""),
            ("LIKELY_HUMAN", 0.21, (0.09, 0.34), "osnovni", "osnovni", ""),
            (
                "INCONCLUSIVE",
                0.56,
                (0.28, 0.79),
                "napredni",
                "napredni",
                "research_only",
            ),
            (
                "INCONCLUSIVE",
                0.47,
                (0.15, 0.85),
                "napredni",
                "osnovni",
                "degraded_basic",
            ),
            (
                "INCONCLUSIVE",
                0.5,
                (0.15, 0.85),
                "napredni",
                "napredni",
                "advanced_unavailable",
            ),
        ]
        results = [
            render.result_context(
                verdict=verdict,
                p_ai=p_ai,
                interval=interval,
                n_words=312,
                agents_used=3,
                agents_total=3,
                limitations=limitations,
                requested_mode=requested_mode,
                effective_mode=effective_mode,
                runtime_status=runtime_status,
            )
            for (
                verdict,
                p_ai,
                interval,
                requested_mode,
                effective_mode,
                runtime_status,
            ) in fixtures
        ]
        errors = [
            MESSAGES_HR[key]
            for key in (
                "ui_error_no_text",
                "ui_error_too_short",
                "ui_error_too_long",
                "ui_error_model_unavailable",
                "ui_error_analysis_failed",
                "ui_error_stream_dropped",
            )
        ]
        return templates.TemplateResponse(
            request=request,
            name="pregled.html",
            context={"results": results, "errors": errors},
        )

    @app.get("/metoda", response_class=HTMLResponse)
    def method_page(
        request: Request, cfg: Annotated[AppConfig, Depends(get_app_config)]
    ) -> HTMLResponse:
        """Explain the running bundle without copying experiment numbers into the UI."""
        return templates.TemplateResponse(
            request=request,
            name="metoda.html",
            context={
                "min_words": cfg.min_word_count,
                "agents": [agent_labels.get(name, name) for name in agent_names],
                "model_versions": model_versions,
                "method_steps": [
                    get_message("ui_method_step_preprocess"),
                    get_message("ui_method_step_agents"),
                    get_message("ui_method_step_calibrate"),
                    get_message("ui_method_step_conformal"),
                ],
                "limitations": _limitations(cfg.min_word_count),
            },
        )

    return app
