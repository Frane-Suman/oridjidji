"""The fine-tuned encoder, loaded for serving.

E1 chose `classla/bcms-bertic` and `training.export` turned it into a quantised ONNX
graph. This module is what runs that graph when a request arrives.

It is deliberately not `export.score_texts`. That function reloads the model per call,
which is right for a one-shot measurement and wrong for a server: the load dominates,
and the number a user waits for is a single document scored by an already-warm process.
Here the session is built once and held.

**The score this returns is not a probability.** It is the softmax output of a network
trained with cross-entropy, which is a well-known overconfident quantity — and nothing
has yet mapped it onto observed frequencies of being right. The specification requires a *calibrated*
probability with an interval, and the specification requires the abstention zone to come from split
conformal prediction. Both arrive in M6. Until then this is a detector score in its own
native scale, exactly as `AgentResult.score` is documented to be, and the fusion layer
above it must refuse to present it as anything more.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oridjidji.schemas import AgentResult, AgentStatus, ScoreDirection

# Tokens the encoder actually reads, when nothing overrides it. The length curve shows
# accuracy saturating near 100 words (~140 BERTić tokens): 100w gives AUROC 0.9989 and
# TPR@5%FPR 1.000, 200w gives 1.0000 — a difference both confidence intervals swallow.
# Records arrive at a median 389 tokens, so the model was doing ~2.7x the work for no
# measurable gain while missing the specification's latency budget. The served value
# comes from `inference.max_tokens`.
MAX_TOKENS = 160

# What the model was fine-tuned at. Kept as a named constant because scoring *above* it
# would run the encoder outside its training distribution — the window may shrink, never
# grow.
TRAINED_MAX_TOKENS = 512

# Preferred first. **fp32 ONNX leads, and INT8 is the fallback** — the reverse of what
# The specification assumes about laptops, on measured evidence.
#
# Measured on the test split, same 96+96 records, same 160-token window:
#
#   variant  AUROC    TPR@1%FPR   machine score range   human median   median ms
#   onnx     0.9990   0.9792      [0.0133, 0.9980]      0.0022          472.7
#   int8     0.9594   0.7188      [0.9970, 0.9979]      0.9970          461.9
#
# INT8 collapses the output range into the fourth decimal: every document, human ones
# included, scores ~0.997. Ranking survives (AUROC 0.9594) but the probabilities do not,
# and TPR at the only threshold defensible against a person falls by **0.26**.
#
# It buys 11 ms. Both variants clear the specification's 500 ms budget at a 160-token window, so the
# trade was never worth making. `m4_export.md` accepted INT8 on an AUROC delta of 0.0041
# against the specification's 0.01 tolerance — checking the summary an earlier decision had already shown to hide
# exactly this kind of collapse.
VARIANT_PREFERENCE = ("onnx", "int8", "")


@dataclass(frozen=True)
class EncoderProvenance:
    """Where a loaded model came from, in the words of the run that produced it.

    Read from the `oridjidji_provenance.json` the export notebook writes. Served to
    `/api/v1/health` so a running instance can say which weights it holds — a demo that
    cannot name its own model is not reproducible.
    """

    candidate: str
    model_id: str
    variant: str
    cv_auroc: float | None = None
    n_train: int | None = None

    @property
    def version(self) -> str:
        """A single string for `AgentResult.version` and `model_versions`."""
        return f"{self.model_id}@{self.variant}"

    @classmethod
    def load(cls, model_dir: Path, variant: str) -> EncoderProvenance:
        path = model_dir / "oridjidji_provenance.json"
        if not path.is_file():
            # A checkpoint with no provenance still runs, but it cannot be cited. Say
            # so in the version string rather than inventing a model id.
            return cls(candidate="unknown", model_id=model_dir.name, variant=variant)
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            candidate=str(data.get("candidate", "unknown")),
            model_id=str(data.get("model_id", model_dir.name)),
            variant=variant,
            cv_auroc=data.get("cv_auroc"),
            n_train=data.get("n_train"),
        )


def find_exported_model(root: str | Path) -> tuple[Path, str] | None:
    """Locate the servable export under ``root``, preferring ONNX fp32. None if absent.

    The preference order is ``VARIANT_PREFERENCE`` above, which puts fp32 ahead of INT8
    for the reason recorded there: INT8 buys 11 ms and costs 0.26 TPR at the only
    threshold defensible against a person. This docstring said "preferring INT8"
    until the M1-M5 sweep, which was the order before that measurement.

    Returning None rather than raising is the whole point: the API has to start, serve
    the ethics furniture and refuse honestly on a machine where the notebook has not run
    yet. A missing model is a state to report, not a crash.
    """
    base = Path(root)
    if not base.is_dir():
        return None
    # Point at the export itself, not only at the directory holding exports. `eval e6-probe`
    # takes an export path while `train calibrate` takes a root, so the same-looking argument
    # meant two different things — and one of them answered "no exported model" for a path
    # that plainly was one. Accepting both removes the trap rather than documenting it.
    if (base / "config.json").is_file():
        suffix = next((s for s in VARIANT_PREFERENCE if s and base.name.endswith(f"_{s}")), "")
        return base, suffix or "fp32"
    # Newest first WITHIN a variant. The directory can hold several exports at once — a
    # retrain writes new weights beside the old ones, and nothing deletes the old ones — so
    # ties have to break somehow. Alphabetical order broke them by name, which meant an
    # obsolete `bcms_bertic_e1_winner_onnx` would outrank the model that replaced it and be
    # served silently. Recency is the only ordering with a defensible meaning here: an
    # export left behind by an earlier run is a leftover, not a candidate. Which one loaded
    # is in the detector's provenance, so the choice is visible rather than assumed.
    candidates = sorted(
        (p for p in base.iterdir() if p.is_dir() and (p / "config.json").is_file()),
        key=lambda p: (-p.stat().st_mtime, p.name),
    )
    for suffix in VARIANT_PREFERENCE:
        for path in candidates:
            if suffix and path.name.endswith(f"_{suffix}"):
                return path, suffix
    return (candidates[0], "fp32") if candidates else None


class EncoderDetector:
    """One loaded encoder, scoring one document at a time.

    COMPLEXITY BUDGET: the lazy-load dance exists because `onnxruntime` and
    `transformers` are heavy imports the fast test suite does not install, and
    because constructing the app must not pay for a model load that a health check may
    never use.
    """

    name = "encoder"

    def __init__(
        self,
        model_dir: str | Path,
        variant: str = "int8",
        *,
        max_tokens: int = MAX_TOKENS,
        intra_op_threads: int | None = None,
    ) -> None:
        if max_tokens > TRAINED_MAX_TOKENS:
            raise ValueError(
                f"max_tokens={max_tokens} exceeds the {TRAINED_MAX_TOKENS} the model was "
                "trained at; scoring a longer window runs it outside its training "
                "distribution and reports the result as the same measurement."
            )
        self.max_tokens = max_tokens
        # Threads onnxruntime may use for one graph. `None` keeps its own default, which is
        # every logical processor — and on a hybrid CPU that is measurably the wrong choice:
        # spreading a latency-sensitive graph across efficiency cores and hyperthreads costs
        # more than the parallelism returns. The value is DERIVED, by the sweep in
        # `train export`, and carried in `inference.intra_op_threads`.
        self.intra_op_threads = intra_op_threads
        self.model_dir = Path(model_dir)
        self.provenance = EncoderProvenance.load(self.model_dir, variant)
        self._tokenizer: Any = None
        self._model: Any = None

    def load(self) -> None:
        """Build the session.

        Call it at startup. Left to the lazy path in `score`, the **first user** pays the
        whole cost — measured at **27.9 s** on the target laptop, which is not a first
        impression the specification's budget survives.
        """
        if self._model is not None:
            return
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        from oridjidji.artefacts import onnx_load_kwargs

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        # Graph-name resolution and the thread-pinned session both live in
        # `onnx_load_kwargs`. They used to live here AND in `export.score_texts`, which is
        # how the two drifted far enough for the measurement path to fail on a file the
        # serving path loaded fine.
        self._model = ORTModelForSequenceClassification.from_pretrained(
            str(self.model_dir), **onnx_load_kwargs(self.model_dir, self.intra_op_threads)
        )

    def close(self) -> None:
        """Release the tokenizer and ONNX session during application shutdown."""
        self._model = None
        self._tokenizer = None

    def score(self, text: str) -> AgentResult:
        """Score one document. Never raises: a dead detector is a reportable state.

        The specification requires a failed agent to degrade the result rather than the request —
        the UI draws that band dashed and labels it `nedostupno`, and the remaining
        detectors still answer. Swallowing the exception here is what makes that
        possible, so the error text travels in the result instead.
        """
        started = time.perf_counter()
        try:
            self.load()
            score = self._forward(text)
        except Exception as exc:  # noqa: BLE001 — see docstring: this is the degrade path
            return AgentResult(
                name=self.name,
                version=self.provenance.version,
                status=AgentStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        return AgentResult(
            name=self.name,
            version=self.provenance.version,
            status=AgentStatus.OK,
            score=score,
            direction=ScoreDirection.HIGHER_IS_AI,
            latency_ms=int((time.perf_counter() - started) * 1000),
            rationale_hr=(
                "BERTićev rezultat ulazi u hrvatski kalibriranu završnu procjenu; viša "
                "vrijednost znači više strojnog signala."
            ),
        )

    def _forward(self, text: str) -> float:
        """Softmax over the two logits, machine class. Uncalibrated — see module docs."""
        import numpy as np

        encoded = self._tokenizer(
            text, truncation=True, max_length=self.max_tokens, return_tensors="pt"
        )
        logits = self._model(**encoded).logits.numpy()
        shifted = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return float((shifted / shifted.sum(axis=-1, keepdims=True))[0, 1])
