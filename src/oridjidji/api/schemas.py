"""The HTTP contract, typed.

Kept apart from `oridjidji.schemas`, which holds the domain models — a corpus record
and a preprocessed document exist whether or not anything is served over HTTP. This
module is the wire format, and the split keeps either file explainable in a sentence.

The typed response keeps the serving contract stable across real and stub runtimes.

Two rules from the specification are enforced here rather than left to the UI to remember:

* **No absolute verdict.** `p_ai` is a calibrated probability that always travels with
  the class-conditional conformal `prediction_set` and the three-state `verdict` derived
  from it, so `INCONCLUSIVE` is a value the type system knows about rather than an error
  path someone has to think of. No heuristic range is reported.
* **The disclaimer is not optional.** `disclaimer_hr` and `limitations_hr` are
  required fields with no defaults on the response, so a caller cannot construct a
  result that omits them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from oridjidji.schemas import AgentResult, AgentStatus, ScoreDirection, Verdict

Mode = Literal["osnovni", "napredni"]
DomainHint = Literal["academic", "news"]
RunStatus = Literal["started", "succeeded", "refused", "failed"]


class RuntimeCapabilities(BaseModel):
    """Features the selected mode can actually produce in this process.

    Scientific status and product availability are deliberately independent.  A
    ``research_only`` Advanced candidate may still be selectable and execute document
    detection; these booleans tell the client which evidence panels it may render.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_detection: bool = False
    statistical_evidence: bool = False


class RuntimeIdentity(BaseModel):
    """Exact model/data identities used by the running process.

    Missing optional artifacts remain explicit ``null`` values.  A client therefore
    sees a degraded runtime as a typed state rather than having to infer it from an
    absent dictionary key.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    encoder: str | None = None
    causal_pair: str | None = None
    fusion: str | None = None
    calibration: str | None = None
    conformal: str | None = None
    repository: str | None = None
    preprocessing: str | None = None
    runtime_status: Literal[
        "basic",
        "validated_candidate",
        "research_only",
        "legacy_unvalidated",
        "advanced_unavailable",
        "degraded_basic",
    ] = "basic"
    device_transfer_status: Literal["passed", "failed", "not_checked"] | None = None
    probability_device: Literal["cpu", "cuda"] | None = None
    limitations: list[str] = Field(default_factory=list)
    capabilities: RuntimeCapabilities = Field(default_factory=RuntimeCapabilities)


class RuntimePolicy(BaseModel):
    """Frozen serving choices needed to prove an M9 scorer matches the web runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_words: int = Field(ge=1)
    advanced_timeout_seconds: int = Field(ge=1)
    roster: tuple[str, ...]
    basic_threshold_identity: str | None = Field(default=None, min_length=64, max_length=64)
    advanced_threshold_identity: str | None = Field(default=None, min_length=64, max_length=64)


class Gates(BaseModel):
    """What the input checks decided, and why.

    Carried in the response because a refusal has to be explainable: "too short" and
    "not Croatian" are different answers and the user is owed the difference.
    """

    min_length_ok: bool
    n_words: int
    language: str
    language_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    language_accepted: bool | None = None
    is_bcs: bool | None = None


class AnalysedScope(BaseModel):
    """Which part of the submitted text the estimate is about.

    The encoder reads only the opening `max_tokens` and the statistical detectors only their
    prefix window, so a long document is judged on its beginning. Counts are words as the
    editor counts them; `None` means that detector did not take part or read a
    non-contiguous sample.
    """

    n_words: int = Field(ge=0)
    encoder_words: int | None = Field(default=None, ge=0)
    statistical_words: int | None = Field(default=None, ge=0)
    statistical_used: bool = False
    whole_text: bool


class Explanations(BaseModel):
    """Material the UI turns into a heatmap, a feature list and SHAP bars."""

    token_heatmap: list[dict[str, object]] = Field(default_factory=list)
    top_features: list[dict[str, object]] = Field(default_factory=list)
    shap: list[dict[str, object]] = Field(default_factory=list)


class StreamEvent(BaseModel):
    """One server-sent event from a running analysis.

    The ring is defined as one band per detector, each filling when *that detector's*
    event arrives — so progress has to be a stream of real transitions, not a timer. The
    brandbook is explicit that the frontend may never invent or hard-code a step label
    and never fake progress, which means the Croatian label travels **in
    the event**. `step_hr` is that label.

    `index` and `of` let the ring say "3 of 5" without the frontend counting detectors
    itself, and let a band be addressed before its result exists.
    """

    event: str = Field(
        pattern=(
            "^(step|agent|run_started|agent_started|agent_finished|agent_failed|"
            "fusion_started|run_finished|result|error)$"
        )
    )
    step_hr: str
    agent: str | None = None
    index: int = 0
    of: int = 0
    elapsed_ms: int = 0


class AnalyzeRequest(BaseModel):
    """A request to analyse one text.

    `store` defaults to False: the specification requires that no text is retained without an
    explicit opt-in, so the safe value is the one you get by saying nothing.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    mode: Mode = "osnovni"
    domain_hint: DomainHint | None = None
    store: bool = False


class FeedbackRequest(BaseModel):
    """Optional user correction attached to one stored result."""

    model_config = ConfigDict(extra="forbid")

    expected_label: Literal["human", "ai", "edited", "unknown"]
    comment: str | None = Field(default=None, max_length=1000)


class StandaloneThresholdEvidence(BaseModel):
    """One Croatian operating point shown as context for A2 or A3.

    This evidence is descriptive. The fused assessment continues to consume the raw
    agent score and never substitutes this standalone decision for the fitted ensemble.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_fpr: float = Field(gt=0.0, lt=1.0)
    raw_threshold: float
    detector_indicates_ai: bool
    achieved_fpr: float = Field(ge=0.0, le=1.0)
    observed_tpr: float = Field(ge=0.0, le=1.0)
    report_identity: str = Field(min_length=64, max_length=64)


class AgentEvidence(BaseModel):
    """Presentation-safe interpretation of one raw detector result.

    ``AgentResult`` remains the stable model-facing contract. This companion tells the
    browser how the result participated in this particular assessment without making it
    infer detector roles, score direction or fusion mathematics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    label_hr: str
    rationale_hr: str
    interpretation_hr: str = ""
    raw_score: float | None = None
    direction: ScoreDirection
    status: AgentStatus
    latency_ms: int | None = None
    used_in_assessment: bool = False
    fusion_contribution: float | None = None
    token_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    standalone_thresholds: list[StandaloneThresholdEvidence] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    """The analysis result.

    `p_ai` is the calibrated probability; `verdict` is decided by the class-conditional
    conformal `prediction_set`, and `operating_decision_ai` is the separate frozen binary
    operating threshold. The former heuristic `interval` (a ±0.10 display band) is no longer
    part of the response; stored history responses that still contain it
    validate because unknown keys are ignored.
    """

    run_id: str
    requested_mode: Mode
    effective_mode: Mode
    verdict: Verdict
    p_ai: float = Field(ge=0.0, le=1.0)
    operating_decision_ai: bool
    prediction_set: list[str]
    gates: Gates
    # Optional so history rows stored earlier still validate.
    scope: AnalysedScope | None = None
    scope_hr: str = ""
    agents: list[AgentResult] = Field(default_factory=list)
    agent_evidence: list[AgentEvidence] = Field(default_factory=list)
    assessment_agents_used: int = Field(ge=0)
    assessment_agents_total: int = Field(ge=0)
    interpretation_hr: str = ""
    interpretation_source: Literal["deterministic", "gemma2_2b_it"] = "deterministic"
    explanations: Explanations = Field(default_factory=Explanations)
    warnings: list[str] = Field(default_factory=list)
    # No defaults: a response cannot be built without the ethics copy.
    disclaimer_hr: str
    limitations_hr: list[str]
    copy_report_hr: str
    model_versions: dict[str, str] = Field(default_factory=dict)
    runtime_identity: RuntimeIdentity = Field(default_factory=RuntimeIdentity)
    elapsed_ms: int = 0

    def model_post_init(self, _context: object) -> None:
        if not self.limitations_hr:
            raise ValueError("limitations_hr must not be empty — the panel is furniture")
        if self.assessment_agents_used > self.assessment_agents_total:
            raise ValueError("assessment_agents_used cannot exceed assessment_agents_total")


class HealthResponse(BaseModel):
    """Liveness plus what is actually loaded."""

    status: str
    version: str
    detectors_loaded: list[str] = Field(default_factory=list)
    bundle_version: str | None = None
    rss_mb: float | None = None
    runtime_identity: RuntimeIdentity = Field(default_factory=RuntimeIdentity)
    runtime_policy: RuntimePolicy | None = None


class ConfigResponse(BaseModel):
    """What the UI needs to render itself before any analysis has run.

    Everything the interface would otherwise have to know about the model lives here.
    The brandbook's standing rule is that the frontend holds no model facts — no
    thresholds, no detector names, no length limits — so each of those has to be
    servable, and this is where they are served from.
    """

    modes: list[Mode]
    min_words: int
    # The "too long" state needs a number, and the frontend is not
    # allowed to hold one. A single analysis is bounded because inference is CPU-only
    # on the same laptop that renders the page.
    max_words: int
    agents: list[str] = Field(default_factory=list)
    agents_by_mode: dict[str, list[str]] = Field(default_factory=dict)
    capabilities_by_mode: dict[Mode, RuntimeCapabilities] = Field(default_factory=dict)
    # Display labels for the ring legend, keyed by agent name. A detector the API
    # reports as failed renders its label with `nedostupno` beside it, so the legend
    # can say which band is dashed and why.
    agent_labels_hr: dict[str, str] = Field(default_factory=dict)
    disclaimer_hr: str
    limitations_hr: list[str]
    verdict_labels_hr: dict[str, str] = Field(default_factory=dict)
