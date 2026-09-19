"""Binoculars aggregation variants for a morphologically rich language (E9, the specification).

Binoculars scores a document as a ratio of two per-token means:

    B = (1/L · Σᵢ aᵢ) / (1/L · Σᵢ bᵢ)          aᵢ = −log p₁(xᵢ | x_<ᵢ)
                                               bᵢ = H(p₁(·|x_<ᵢ), p₂(·|x_<ᵢ))

**The identity that decides which variants can do anything.** The `1/L` cancels, so
`B = Σaᵢ / Σbᵢ`. Renormalising the denominator from token count to *word* count cancels
identically and **changes nothing** — and any variant that reduces to a strictly monotone
transform of `B` (log B, B−1, a logit) leaves AUROC and TPR@FPR mathematically unchanged,
because ROC depends only on the ordering of the samples. Spreading the axis is cosmetic.

What can move the ordering is the **weighting** or the **aggregation function**. Every variant
here does one of those.

**Why this is worth asking on Croatian.** A five-token word currently contributes five times
the weight of a one-token word, and an earlier decision measured Croatian words carrying a diacritic costing
**~4 tokens** against ~2.5 without. Word-internal continuations are near-deterministic — for
them `aᵢ ≈ bᵢ`, so the local ratio sits at 1 and drags the document score toward 1 from both
classes at once. That is a mechanism for the compressed distributions E4-G2 measured
(σ ≈ 0.05, Cohen's d = 1.229), and E9 is the registered test of it.

The specification names Binoculars as thesis-core, so these are written out rather than imported.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

#: E9's registered mask cut-off, conditioned on the CONTEXT only.
CONTEXT_MASK_MAX_PROB = 0.95

#: Trim fraction for the trimmed-mean variant.
TRIM = 0.10


def _ratio(numerator: float, denominator: float) -> float:
    """B = Σa / Σb, refusing a denominator that cannot carry a ratio.

    A zero denominator means the observer's cross-entropy vanished over every counted token —
    a degenerate text, not a score of infinity. `ValueError` matches how `score_one` already
    reports a configuration that cannot be computed for one record.
    """
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError("cross-entropy sum is zero or non-finite; this text has no score")
    return float(numerator / denominator)


def weighted(a: Array, b: Array, weights: Array) -> float:
    """The whole family in one line: `Σwᵢaᵢ / Σwᵢbᵢ`.

    Uniform weights give the baseline exactly, which is the property the test suite pins —
    a weighting scheme that cannot reproduce V0 is not a generalisation of it.
    """
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("all weights are zero; nothing was counted")
    return _ratio(float(np.sum(weights * a)), float(np.sum(weights * b)))


# ---------------------------------------------------------------- V0: the baseline


def token_ratio(a: Array, b: Array) -> float:
    """**V0.** `B = Σaᵢ / Σbᵢ` — Binoculars as published."""
    return _ratio(float(np.sum(a)), float(np.sum(b)))


# ------------------------------------------------------ V1–V4: reweighted variants


def word_mean(a: Array, b: Array, word_starts: NDArray[np.bool_]) -> float:
    """**V1.** Average *within* each word, then across words.

        B = [Σⱼ (1/kⱼ) Σ_{i∈Wⱼ} aᵢ] / [Σⱼ (1/kⱼ) Σ_{i∈Wⱼ} bᵢ]

    This is **not** the no-op renormalisation: averaging inside the word first reweights every
    token by `1/kⱼ`, so a word contributes once regardless of how many pieces the tokenizer
    cut it into. That is the whole hypothesis, expressed as weights.
    """
    return weighted(a, b, _inverse_word_length(word_starts))


def boundary_only(a: Array, b: Array, word_starts: NDArray[np.bool_]) -> float:
    """**V2.** Word-initial tokens only.

    The transition *previous word → this word* is where a lexical choice is made; the rest of
    a word is largely the tokenizer spelling it out. This is a legitimate detection statistic
    rather than a truncated perplexity: `P(next word begins with tⱼ | x_<ⱼ)` is a
    well-defined event, and E9 calibrates its null empirically anyway.
    """
    return weighted(a, b, word_starts.astype(np.float64))


def entropy_weighted(a: Array, b: Array, entropies: Array) -> float:
    """**V3.** `wᵢ ∝ Hᵢ` — positions the observer found uncertain count for more.

    Soft rather than a hard entropy threshold, so the result does not turn on where a cut-off
    was put. A near-deterministic word-internal continuation has `Hᵢ ≈ 0` and drops out on its
    own.
    """
    return weighted(a, b, np.asarray(entropies, dtype=np.float64))


def boundary_entropy(
    a: Array, b: Array, word_starts: NDArray[np.bool_], entropies: Array, lam: float = 0.25
) -> float:
    """**V4 — E9's primary hypothesis.** `wᵢ = w_boundary(i) · Hᵢ / H_max`.

    Both proposals that motivated E9 converge here: weight by word position *and* by how
    uncertain the observer was. `lam` is the weight kept for word-internal tokens; it is not
    tuned, and 0.25 is registered rather than searched, because a hyperparameter chosen after
    seeing the number is not a hyperparameter.
    """
    entropies = np.asarray(entropies, dtype=np.float64)
    peak = float(np.max(entropies)) if len(entropies) else 0.0
    scaled = entropies / peak if peak > 0 else np.zeros_like(entropies)
    boundary = np.where(word_starts, 1.0, lam)
    return weighted(a, b, boundary * scaled)


# ------------------------------------------------- V5–V8: per-word ratio statistics


def word_ratios(a: Array, b: Array, word_starts: NDArray[np.bool_]) -> Array:
    """`rₙ = Aₙ / Bₙ` per word, where `Aₙ = Σ_{i∈Wₙ} aᵢ` by the chain rule.

    Summing within a word is exact — it is `−log P(wordₙ | word_<ₙ)`, no approximation. What
    changes the statistic is what happens *next*: taking a robust centre of the `rₙ` instead
    of the ratio of their sums.
    """
    groups = _word_groups(word_starts)
    ratios = []
    for start, stop in groups:
        denominator = float(np.sum(b[start:stop]))
        if denominator > 0.0 and np.isfinite(denominator):
            ratios.append(float(np.sum(a[start:stop])) / denominator)
    if not ratios:
        raise ValueError("no word carried a usable cross-entropy sum")
    return np.asarray(ratios, dtype=np.float64)


def word_median(a: Array, b: Array, word_starts: NDArray[np.bool_]) -> float:
    """**V5.** `median(rₙ)` — one catastrophic word cannot move the document."""
    return float(np.median(word_ratios(a, b, word_starts)))


def word_trimmed(a: Array, b: Array, word_starts: NDArray[np.bool_]) -> float:
    """**V6.** 10% trimmed mean of `rₙ`, between the median and the mean."""
    ratios = np.sort(word_ratios(a, b, word_starts))
    cut = int(len(ratios) * TRIM)
    kept = ratios[cut : len(ratios) - cut] if len(ratios) - 2 * cut > 0 else ratios
    return float(np.mean(kept))


def word_quantile(a: Array, b: Array, word_starts: NDArray[np.bool_], q: float = 0.90) -> float:
    """**V7.** `Q₉₀(rₙ)` — a tail statistic, not a centre.

    Cohen's d = 1.229 describes the *middle* of the distributions and is the ceiling on any
    method that summarises the middle. Human text is expected to carry occasional bursts of
    surprise — idioms, rare cases, authorial idiosyncrasy — that machine text mostly lacks.
    Those live in the upper tail of `rₙ`, and a mean cannot see them.
    """
    return float(np.quantile(word_ratios(a, b, word_starts), q))


def word_variance(a: Array, b: Array, word_starts: NDArray[np.bool_]) -> float:
    """**V8.** `Var(rₙ)` — how unevenly surprise is distributed across the document."""
    return float(np.var(word_ratios(a, b, word_starts)))


# -------------------------------------------------------- V9: context-only masking


def context_masked(
    a: Array, b: Array, max_probs: Array, cut: float = CONTEXT_MASK_MAX_PROB
) -> float:
    """**V9.** Drop positions where the observer was near-certain **from the context alone**.

        maskᵢ = 1[ max_v p₁(v | x_<ᵢ) < cut ]

    The distinction from a filter on `p₁(xᵢ)` is not pedantic and E9 excludes that version.
    Selecting on the *observed* token removes low-surprise positions, and those carry the
    signal *the model was overconfident here* — it would discard more machine evidence than
    human and bias the null. Conditioning on a measurable function of `x_<ᵢ` leaves the
    statistic a valid conditional contribution.
    """
    mask = (np.asarray(max_probs, dtype=np.float64) < cut).astype(np.float64)
    if float(np.sum(mask)) <= 0.0:
        raise ValueError(f"every position had max probability >= {cut}; nothing was counted")
    return weighted(a, b, mask)


# ------------------------------------------------------------ V10: rank statistic


def anderson_darling(pit: Array) -> float:
    """**V10.** Anderson–Darling against uniform, on the model-implicit quantiles.

        A² = −n − (1/n) Σₖ (2k−1)[ ln u₍ₖ₎ + ln(1 − u₍ₙ₊₁₋ₖ₎) ]

    Under text the model actually generated the `uᵢ` are near-uniform; human text departs.
    Rank statistics are **invariant to the language's base entropy**, which is the property
    worth having on a language where everything is comparatively hard to predict — the
    absolute level cancels and only the shape is tested.

    Anderson–Darling rather than Kolmogorov–Smirnov because A² weights the tails, and the
    tail is where TPR@1%FPR is decided.

    Direction note: **higher A² means further from uniform, so higher is HUMAN** — the
    opposite of Binoculars. The specification calls the flipped sign the classic bug here, so the caller
    orients it explicitly rather than inferring it.
    """
    u = np.sort(np.clip(np.asarray(pit, dtype=np.float64), 1e-12, 1 - 1e-12))
    n = len(u)
    if n < 2:
        raise ValueError("Anderson-Darling needs at least two positions")
    k = np.arange(1, n + 1, dtype=np.float64)
    return float(-n - np.sum((2 * k - 1) * (np.log(u) + np.log(1 - u[::-1]))) / n)


# ------------------------------------------------------------------- word helpers


def _word_groups(word_starts: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Half-open `[start, stop)` spans, one per word.

    Tokens before the first word start belong to no word and are dropped rather than assigned
    to the first one: a truncated document can open mid-word, and silently attaching that
    fragment would make the first word's length wrong in exactly the documents most likely to
    be odd.
    """
    indices = np.flatnonzero(np.asarray(word_starts, dtype=bool))
    if not len(indices):
        return []
    bounds = list(indices) + [len(word_starts)]
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(len(indices))]


def _inverse_word_length(word_starts: NDArray[np.bool_]) -> Array:
    """`1/kⱼ` for every token in word `j`, and 0 for tokens in no word."""
    weights = np.zeros(len(word_starts), dtype=np.float64)
    for start, stop in _word_groups(word_starts):
        weights[start:stop] = 1.0 / float(stop - start)
    return weights
