"""Lexical/statistical stylometry features (family ``lexical``; the specification, Appendix A).

MTLD and MATTR are implemented explicitly (length-robust diversity; raw TTR is
deliberately absent — it is confounded with length). Values are raw statistics;
corpus-relative z-scores arrive with the M2 corpus statistics.
"""

from __future__ import annotations

import math
from collections import Counter
from statistics import fmean, pstdev, pvariance

from oridjidji.features.base import FeatureContext, feature
from oridjidji.features.wordlists import ANGLICISMS_SEED, FUNCTION_WORDS_HR
from oridjidji.preprocessing.segment import tokenize_words
from oridjidji.schemas import Document

_MTLD_THRESHOLD = 0.72
_MATTR_WINDOW = 100


def _lower_words(doc: Document) -> list[str]:
    return [w.lower() for w in doc.words]


def _mtld_pass(words: list[str], threshold: float) -> float:
    """One directional MTLD pass (McCarthy & Jarvis 2010): mean factor length."""
    factors = 0.0
    types: set[str] = set()
    token_count = 0
    ttr = 1.0
    for word in words:
        token_count += 1
        types.add(word)
        ttr = len(types) / token_count
        if ttr < threshold:
            factors += 1.0
            types.clear()
            token_count = 0
            ttr = 1.0
    if token_count > 0:
        factors += (1.0 - ttr) / (1.0 - threshold)  # partial factor
    if factors == 0.0:
        return float(len(words))  # diversity never dropped below threshold
    return len(words) / factors


@feature("lexical")
def mtld(doc: Document, ctx: FeatureContext) -> float:
    """MTLD lexical diversity (mean of forward and backward passes, threshold 0.72).

    Why: LLM text tends to keep lexical diversity steady; humans drift and repeat,
    so factor lengths differ between the classes independent of text length.
    Failure mode: unstable under ~50 words; heavy proper-noun lists inflate it.
    """
    words = _lower_words(doc)
    if not words:
        return 0.0
    forward = _mtld_pass(words, _MTLD_THRESHOLD)
    backward = _mtld_pass(list(reversed(words)), _MTLD_THRESHOLD)
    return (forward + backward) / 2


@feature("lexical")
def mattr_100(doc: Document, ctx: FeatureContext) -> float:
    """Moving-average type-token ratio over a 100-word window.

    Why: a length-robust diversity measure — AI text is often mid-range and
    uniform, human text more extreme in either direction.
    Failure mode: texts shorter than the window degrade to plain TTR, which is
    length-confounded again.
    """
    words = _lower_words(doc)
    n = len(words)
    if n == 0:
        return 0.0
    if n <= _MATTR_WINDOW:
        return len(set(words)) / n
    window: Counter[str] = Counter(words[:_MATTR_WINDOW])
    ratios_sum = float(len(window))
    count = 1
    for i in range(_MATTR_WINDOW, n):
        out_word, in_word = words[i - _MATTR_WINDOW], words[i]
        window[out_word] -= 1
        if window[out_word] == 0:
            del window[out_word]
        window[in_word] += 1
        ratios_sum += len(window)
        count += 1
    return ratios_sum / count / _MATTR_WINDOW


@feature("lexical")
def hapax_rate(doc: Document, ctx: FeatureContext) -> float:
    """Share of words occurring exactly once.

    Why: humans mint one-off words (names, typos, colloquialisms) at a higher
    rate than sampling-temperature-limited model output.
    Failure mode: strongly length-dependent; compare only within length buckets.
    """
    words = _lower_words(doc)
    if not words:
        return 0.0
    counts = Counter(words)
    return sum(1 for c in counts.values() if c == 1) / len(words)


@feature("lexical")
def zipf_slope(doc: Document, ctx: FeatureContext) -> float:
    """Least-squares slope of log(frequency) against log(rank).

    Why: human text tends toward the Zipfian slope ≈ −1; generated text often
    flattens (more mid-frequency mass) under nucleus/top-k sampling.
    Failure mode: needs enough distinct types (≥10) for a stable fit; returns
    0.0 below that.
    """
    counts = Counter(_lower_words(doc))
    freqs = sorted(counts.values(), reverse=True)
    if len(freqs) < 10:
        return 0.0
    xs = [math.log(rank) for rank in range(1, len(freqs) + 1)]
    ys = [math.log(freq) for freq in freqs]
    mean_x, mean_y = fmean(xs), fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0.0:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / denominator


@feature("lexical")
def function_word_ratio(doc: Document, ctx: FeatureContext) -> float:
    """Share of tokens that are Croatian function words.

    Why: function-word rates are a classic authorship signal largely outside
    conscious control; models imitate averages, individuals deviate.
    Failure mode: register-sensitive (dialogue vs. exposition); the word list is
    curated, not exhaustive.
    """
    words = _lower_words(doc)
    if not words:
        return 0.0
    return sum(1 for w in words if w in FUNCTION_WORDS_HR) / len(words)


@feature("lexical")
def anglicism_rate(doc: Document, ctx: FeatureContext) -> float:
    """Seed-list anglicisms per 100 words.

    Why: informal human Croatian mixes in English loans far more readily than
    LLMs writing 'proper' Croatian prose.
    Failure mode: topic-driven (tech/business texts); seed list is small by design.
    """
    words = _lower_words(doc)
    if not words:
        return 0.0
    return sum(1 for w in words if w in ANGLICISMS_SEED) / len(words) * 100


@feature("lexical")
def sentence_len_mean(doc: Document, ctx: FeatureContext) -> float:
    """Mean sentence length in words.

    Why: generators cluster near register-typical means; humans spread wider.
    Failure mode: depends on the rule-based splitter's boundary quality.
    """
    lengths = [len(tokenize_words(s)) for s in doc.sentences]
    return fmean(lengths) if lengths else 0.0


@feature("lexical")
def sentence_len_std(doc: Document, ctx: FeatureContext) -> float:
    """Population standard deviation of sentence length in words.

    Why: raw spread complements burstiness; low spread = suspiciously even prose.
    Failure mode: unstable under ~5 sentences.
    """
    lengths = [len(tokenize_words(s)) for s in doc.sentences]
    return pstdev(lengths) if len(lengths) > 1 else 0.0


@feature("lexical")
def sentence_len_burstiness(doc: Document, ctx: FeatureContext) -> float:
    """Sentence-length burstiness: σ/μ of sentence lengths.

    Why: the most replicated human-vs-AI signal — humans alternate short and long
    sentences; models keep a steadier rhythm (lower burstiness).
    Failure mode: legal/administrative human prose is also low-burstiness; short
    texts make the ratio noisy.
    """
    lengths = [len(tokenize_words(s)) for s in doc.sentences]
    if len(lengths) < 2:
        return 0.0
    mean = fmean(lengths)
    return pstdev(lengths) / mean if mean > 0 else 0.0


@feature("lexical")
def paragraph_len_variance(doc: Document, ctx: FeatureContext) -> float:
    """Population variance of paragraph lengths in words.

    Why: AI answers often ship uniformly sized paragraphs (low variance);
    human documents vary with content.
    Failure mode: single-paragraph inputs return 0.0, indistinguishable from
    genuinely uniform multi-paragraph text.
    """
    lengths = [len(tokenize_words(p)) for p in doc.paragraphs]
    return pvariance(lengths) if len(lengths) > 1 else 0.0
