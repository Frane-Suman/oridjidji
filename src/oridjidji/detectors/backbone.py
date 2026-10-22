"""The observer/performer pair, loaded once and shared.

One module owns the language models. Binoculars needs both, Fast-DetectGPT needs one,
and the GLTR heatmap reuses log-probs the others already computed — so **no detector
loads its own copy of a model**. On a 16 GB laptop with no discrete GPU that is the
difference between the Advanced mode being usable and not.

What lives here is the *contract*. The scoring maths lives in `binoculars.py` and
`fast_detect_gpt.py` as pure functions over arrays, so it can be tested exhaustively
without loading 3.4 GB of weights — which matters, because the thesis has to defend those
formulas and a test that needs a GPU is a test nobody runs.

The concrete transformers-backed implementation arrives with E4, once the model pair is
chosen and downloaded. Until then `FakeBackbone` in the tests plays the part.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# float32, not float64. These arrays are (n_tokens, vocab): at the 160-token serving
# window that is 82 MB per model for EuroLLM's 128k vocabulary and 97 MB for Qwen3's 152k,
# doubled for a pair and quadrupled again at 512 tokens. Log-probabilities carry nowhere
# near seven significant digits of meaning, so the second half of a float64 is padding that
# costs real memory on a 16 GB laptop.
Array = NDArray[np.float32]


@dataclass(frozen=True)
class TokenScores:
    """Per-token evidence for one text, from one forward pass of each model.

    ``observer_logprobs`` and ``performer_logprobs`` are full next-token distributions,
    shape ``(n_tokens, vocab)``, already normalised to log-probabilities. Keeping the
    whole distribution rather than only the realised token's probability is what makes
    the cross-perplexity term and Fast-DetectGPT's analytic variance computable at all.

    ``token_ids`` is the actual continuation, shape ``(n_tokens,)`` — position ``t`` in
    the log-prob arrays predicts ``token_ids[t]``.
    """

    token_ids: NDArray[np.int64]
    observer_logprobs: Array
    performer_logprobs: Array | None = None

    def __post_init__(self) -> None:
        n = len(self.token_ids)
        if self.observer_logprobs.shape[0] != n:
            raise ValueError(
                f"observer_logprobs has {self.observer_logprobs.shape[0]} rows for "
                f"{n} tokens; position t must predict token_ids[t]"
            )
        if (
            self.performer_logprobs is not None
            and self.performer_logprobs.shape != self.observer_logprobs.shape
        ):
            raise ValueError(
                "observer and performer distributions must have the same shape — "
                f"got {self.observer_logprobs.shape} and {self.performer_logprobs.shape}. "
                "A mismatch means the tokenizers differ, which invalidates the "
                "cross-perplexity term rather than merely making it noisy."
            )

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def realised_logprobs(self) -> Array:
        """Log-probability the observer assigned to the token that actually appeared."""
        return np.asarray(
            self.observer_logprobs[np.arange(self.n_tokens), self.token_ids], dtype=np.float32
        )

    # --- the reductions every detector actually consumes ------------------------------
    #
    # Each collapses the vocabulary axis to one number per token. They live here rather
    # than in the detectors because a backbone can compute them *inside* the forward pass
    # and never materialise the (n_tokens, vocab) array at all — see `ReducedScores`. The
    # detectors are written against these four properties, so they work on either
    # representation without knowing which one they were handed.

    @property
    def expected_logprobs(self) -> Array:
        """Per-position mean log-probability under the observer's OWN distribution."""
        probs = np.exp(self.observer_logprobs)
        return np.asarray(np.sum(probs * self.observer_logprobs, axis=1), dtype=np.float32)

    @property
    def logprob_variances(self) -> Array:
        """Per-position variance of the log-probability under the observer's own distribution."""
        probs = np.exp(self.observer_logprobs)
        mean = np.sum(probs * self.observer_logprobs, axis=1)
        mean_square = np.sum(probs * np.square(self.observer_logprobs), axis=1)
        # Clipped at zero: the identity guarantees non-negativity, floating point does not,
        # and a variance of -1e-16 becomes NaN under a square root.
        return np.asarray(np.maximum(mean_square - np.square(mean), 0.0), dtype=np.float32)

    @property
    def cross_entropies(self) -> Array | None:
        """Per-position CE(observer, performer). None when only one model was run."""
        if self.performer_logprobs is None:
            return None
        observer_probs = np.exp(self.observer_logprobs)
        # The 0 * log(0) = 0 convention, applied explicitly. A vocabulary entry the observer
        # rules out contributes nothing regardless of what the performer thinks, but in
        # floating point `0.0 * -inf` is NaN — and a NaN here does not raise, it propagates
        # through the mean and out into a score that looks like a number. A position where
        # the observer is confident and the performer is not stays +inf, which is real
        # disagreement and is caught downstream.
        with np.errstate(invalid="ignore"):
            products = observer_probs * self.performer_logprobs
        contributions = np.where(observer_probs > 0.0, products, 0.0)
        return np.asarray(-np.sum(contributions, axis=1), dtype=np.float32)


@dataclass(frozen=True)
class ReducedScores:
    """The same four per-token reductions, without ever holding the vocabulary axis.

    A backbone reduces inside the forward pass and returns this instead of `TokenScores`,
    which turns 195 MB per text into about 3 KB. Every detector reads the same four
    property names off either type, so nothing downstream branches on which it received.

    `TokenScores` remains the representation E4's grid and the whole statistical test suite
    use: those construct distributions directly and need the vocabulary axis to exist.
    """

    token_ids: NDArray[np.int64]
    realised_logprobs: Array
    expected_logprobs: Array
    logprob_variances: Array
    cross_entropies: Array | None = None
    #: True where the token opens a whitespace-delimited word (E9). Binoculars weights a
    #: five-token word five times a one-token word, and an earlier decision measured Croatian words with a
    #: diacritic costing ~4 tokens against ~2.5 without — so which tokens open words is the
    #: quantity every aggregation variant turns on.
    word_starts: NDArray[np.bool_] | None = None
    #: `max_v p1(v)`: how peaked the observer's distribution is, given the CONTEXT ALONE.
    #: A mask built from this conditions on `x_<i` only. Masking on the observed token's own
    #: probability would select against exactly the positions carrying "the model was
    #: overconfident here" (E9's excluded variant).
    max_probs: Array | None = None
    #: The model-implicit quantile of the realised token, `P(p1(x~) <= p1(x_i))`. Under a
    #: text actually sampled from the model these are near-uniform, so rank statistics over
    #: them are invariant to the language's base entropy — which is the point for Croatian.
    pit: Array | None = None
    #: Probability assigned to output slots that the text tokenizer cannot emit. This is
    #: normally zero. Qwen3.5 exposes extra multimodal/reserved logits, so recording their
    #: mass lets the comparison detect an architecture-specific support mismatch without
    #: changing the published full-vocabulary Binoculars statistic after seeing results.
    observer_unmapped_mass: Array | None = None
    performer_unmapped_mass: Array | None = None
    #: Coverage is part of the statistical evidence identity.  It is deliberately carried
    #: beside the arrays: a score over the first 160 tokens and one over two 512-token
    #: windows are different measurements even when their final scalar happens to match.
    total_tokens: int | None = None
    scored_tokens: int | None = None
    window_policy: str = "prefix160"
    #: End character of the last scored token when the scored tokens are one prefix of the
    #: text, so the UI can name the part of a document the score is about. `None`
    #: for a discontinuous policy, which reads a beginning and an end rather than a prefix.
    scored_chars: int | None = None

    def __post_init__(self) -> None:
        n = len(self.token_ids)
        for name in ("realised_logprobs", "expected_logprobs", "logprob_variances"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} entries for {n} tokens")
        for name in (
            "cross_entropies",
            "word_starts",
            "max_probs",
            "pit",
            "observer_unmapped_mass",
            "performer_unmapped_mass",
        ):
            value = getattr(self, name)
            if value is not None and len(value) != n:
                raise ValueError(f"{name} must have one entry per token, got {len(value)}")
        total = n if self.total_tokens is None else self.total_tokens
        scored = n if self.scored_tokens is None else self.scored_tokens
        if total < n or scored != n or total < scored:
            raise ValueError(
                "coverage must satisfy total_tokens >= scored_tokens == number of score rows"
            )

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def coverage(self) -> float:
        """Share of scoreable document tokens represented by these reductions."""
        total = self.n_tokens if self.total_tokens is None else self.total_tokens
        return self.n_tokens / total if total else 0.0


@runtime_checkable
class PerTokenScores(Protocol):
    """What a detector actually reads: four scalars per token, and nothing else.

    Both `TokenScores` (which derives them from full distributions) and `ReducedScores`
    (which stores them directly) satisfy this. Annotating the detectors against it rather
    than against `TokenScores` is what makes "works on either representation" a claim mypy
    checks, instead of a comment that happens to be true.
    """

    @property
    def n_tokens(self) -> int: ...

    @property
    def realised_logprobs(self) -> Array: ...

    @property
    def expected_logprobs(self) -> Array: ...

    @property
    def logprob_variances(self) -> Array: ...

    @property
    def cross_entropies(self) -> Array | None: ...


@runtime_checkable
class Backbone(Protocol):
    """What a detector may ask of the model layer, and nothing more.

    Deliberately narrow. A detector that could reach the tokenizer or the raw model would
    grow its own loading path, and the specification exists to stop exactly that.
    """

    @property
    def model_pair(self) -> str:
        """Identifier for the loaded pair, e.g. ``EuroLLM-1.7B/EuroLLM-1.7B-Instruct``.

        Goes into every threshold artefact: a Binoculars threshold is meaningless without
        knowing which pair produced it.
        """
        ...

    def score_tokens(self, text: str) -> PerTokenScores: ...


def assert_same_tokenizer(observer_vocab: int, performer_vocab: int, pair: str) -> None:
    """Refuse a mismatched pair loudly.

    Cross-perplexity compares two distributions token by token. If the tokenizers differ,
    position ``t`` means different things to the two models and the comparison is
    meaningless — but it still produces a number, which is why this has to be an assertion
    rather than a note in a docstring.
    """
    if observer_vocab != performer_vocab:
        raise ValueError(
            f"Model pair '{pair}' has mismatched vocabularies "
            f"({observer_vocab} vs {performer_vocab}). Binoculars compares the two "
            "models' distributions position by position, so different tokenizers make "
            "the cross-perplexity term meaningless rather than merely noisy. Pick a "
            "base+instruct pair from one family."
        )
