"""Core data contracts (Pydantic v2).

These are the stable interfaces between pipeline stages.
They are deliberately minimal in M0: fields are added by the milestone that
produces them, never speculatively.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Verdict(StrEnum):
    """Three-state outcome. The system never emits an absolute verdict."""

    LIKELY_HUMAN = "LIKELY_HUMAN"
    INCONCLUSIVE = "INCONCLUSIVE"
    LIKELY_AI = "LIKELY_AI"


class AgentStatus(StrEnum):
    """Agent completion state; a failed agent degrades gracefully, never a 500."""

    OK = "ok"
    SKIPPED = "skipped"
    FAILED = "failed"


class ScoreDirection(StrEnum):
    """Which way a raw score points before normalisation.

    Binoculars is LOWER_IS_AI; most other agents are HIGHER_IS_AI. Kept
    explicit because a silently flipped sign is the classic bug in this domain.
    """

    HIGHER_IS_AI = "higher_is_ai"
    LOWER_IS_AI = "lower_is_ai"


class Domain(StrEnum):
    """Dataset domain, used for leave-one-domain-out evaluation."""

    NEWS = "news"
    ACADEMIC = "academic"
    WEB_OTHER = "web_other"
    # Edited prose from print — books and periodicals (Riznica). Deliberately not
    # folded into an existing value: it is neither crawled news nor a thesis, and a domain
    # label that lies is worse than a third value.
    #
    # It reaches only the human-only artefact, never `dataset.parquet`. Two sites fire if
    # it ever does: `runner.DOMAIN_TASK` raises `KeyError` on an unmapped domain, and
    # `equalise_by_generator` returns `[]` when a generator produced nothing for a domain.
    # Both are loud, which is the point — a literary domain in the labelled set needs a
    # generation run, not just a corpus.
    LITERARY = "literary"
    # Statute, regulation and official notice (Riznica `Zakoni`). Its own value
    # because legal Croatian is formulaic to a degree no other register approaches, which
    # is exactly why the M9 bias probe wants it: formulaic human prose is the text most at
    # risk of being called machine-written.
    LEGAL = "legal"


class TextPipeline(StrEnum):
    """How the text reached us.

    Recorded per record because *domain* and *extraction pipeline* are confounded in
    this dataset (web-crawled news vs PDF-extracted theses): making the confound a
    column keeps it measurable instead of invisible.
    """

    WEB_EXTRACTED = "web_extracted"
    PDF_EXTRACTED = "pdf_extracted"
    GENERATED = "generated"
    # Reassembled from a tokenised corpus file. Distinct from the two above because it is
    # the pipeline most likely to leave a fingerprint: `base.REJECTED_SOURCES` turns a
    # source away for exactly this. Riznica qualifies only because its vertical records
    # spacing explicitly (`<g/>`), so the reassembly is exact — but the pipeline stays a
    # column so `probe-artefacts` can keep checking that claim rather than trusting it.
    CORPUS_VERTICAL = "corpus_vertical"


class CorpusRecord(BaseModel):
    """One dataset record — the single contract every corpus adapter yields.

    Provenance fields are mandatory from day one: a record whose origin, licence and
    collection period cannot be stated is a record that cannot be defended.
    """

    record_id: str
    text: str
    label: int = Field(default=0, ge=0, le=1)  # 0 = human, 1 = machine-generated
    domain: Domain
    genre: str | None = None

    # Provenance
    source: str
    source_id: str
    source_url: str | None = None
    licence: str
    collected_period: str
    crawl_date: str | None = None
    published_year: int | None = None
    retrieved_at: str
    sampling_seed: int
    content_hash: str

    # Measurements
    title: str | None = None
    n_words: int = Field(ge=0)
    n_chars: int = Field(ge=0)
    lang: str = "hr"
    lang_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    quality_score: float | None = None
    text_pipeline: TextPipeline

    # Generation metadata — null for human records.
    # `generator_family` drives the in-family/out-of-family reporting that keeps
    # statistical detectors from being flattered by their own model family;
    # `prompt_source_record_id` is the human record whose title seeded this text,
    # which is what makes the topic-only classifier land near chance.
    generator: str | None = None
    generator_family: str | None = None
    model_version: str | None = None
    decoding_params: str | None = None
    prompt_id: str | None = None
    prompt_hash: str | None = None
    prompt_source_record_id: str | None = None
    prompt_source_content_hash: str | None = None
    generated_at: str | None = None
    cost_estimate_usd: float | None = None

    # Supplemental repository provenance. Null means legacy/unknown, never inferred fact.
    parent_document_id: str | None = None
    group_id: str | None = None
    author_id: str | None = None
    collection_cohort: str | None = None
    discipline: str | None = None
    output_register: str | None = None
    prompt_family: str | None = None
    prompt_language: str | None = None
    prompt_version: str | None = None
    preprocessing_version: str | None = None
    evaluation_role: str | None = None


class AgentRequirements(BaseModel):
    """What an agent needs before it can run; checked by the orchestrator."""

    models: list[str] = Field(default_factory=list)
    ram_mb: int = 0
    needs_gpu: bool = False
    needs_api_key: str | None = None


class Document(BaseModel):
    """Preprocessed input text. Raw and normalised text are both kept.

    The scalar fields are *measurements taken during preprocessing* (homoglyph and
    diacritic signals feed both the UI warnings and the feature layer). All new
    fields default so that a minimal ``Document(raw=..., normalized=...)`` remains
    valid.
    """

    model_config = ConfigDict(frozen=True)

    raw: str
    normalized: str
    language: str = "hr"
    words: list[str] = Field(default_factory=list)
    sentences: list[str] = Field(default_factory=list)
    paragraphs: list[str] = Field(default_factory=list)

    zero_width_count: int = 0
    n_homoglyphs_normalized: int = 0
    homoglyph_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    diacritic_density: float = Field(default=0.0, ge=0.0)  # per 1,000 characters
    diacritic_deficit: float = Field(default=0.0, ge=0.0, le=1.0)
    language_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    language_candidates: list[tuple[str, float]] = Field(default_factory=list)

    # The language gate's verdict, carried rather than recomputed. Non-negotiable the specification
    # requires rejecting non-Croatian input with a clear Croatian message, and a caller
    # holding only a Document must be able to do that without running detection again.
    # ``None`` means the gate did not run (``check_language=False``) — asserting
    # ``True`` there would fabricate a verdict that was never taken.
    language_accepted: bool | None = None
    language_is_bcs: bool | None = None
    language_message_hr: str | None = None

    @property
    def n_words(self) -> int:
        return len(self.words)

    @property
    def n_sentences(self) -> int:
        return len(self.sentences)

    @property
    def n_paragraphs(self) -> int:
        return len(self.paragraphs)


class LanguageCheck(BaseModel):
    """Result of the language gate.

    hr/sr/bs separation is genuinely unreliable; ``is_bcs`` + ``message_hr`` carry
    that honesty to the UI instead of feigning precision.
    """

    language: str
    confidence: float = Field(ge=0.0, le=1.0)
    candidates: list[tuple[str, float]] = Field(default_factory=list)
    accepted: bool
    is_bcs: bool
    message_hr: str | None = None


class FeatureInfo(BaseModel):
    """Registry metadata for one linguistic feature; shown in the UI/SHAP."""

    name: str
    family: str
    description: str


class AgentResult(BaseModel):
    """One detector agent's output.

    ``score`` is the agent's raw score in its native scale; normalisation happens
    in the fusion layer. ``rationale_hr`` is a short Croatian explanation shown in
    the UI.
    """

    name: str
    version: str
    status: AgentStatus = AgentStatus.OK
    score: float | None = None
    direction: ScoreDirection = ScoreDirection.HIGHER_IS_AI
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    latency_ms: int | None = None
    rationale_hr: str | None = None
    token_data: dict[str, Any] | None = None
    error: str | None = None
