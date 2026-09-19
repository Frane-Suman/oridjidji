"""Readability-adjacent features on a Croatian syllable counter (the specification, Appendix A).

Deliberately NO composite readability index in M1: published formulas carry
English-calibrated coefficients, and copying them would violate the project's
"no thresholds from papers" rule in spirit. The components (syllable and length
statistics) are the defensible signal; a Croatian-adapted composite may be fitted
later from corpus data.
"""

from __future__ import annotations

from statistics import fmean

from oridjidji.features.base import FeatureContext, feature
from oridjidji.schemas import Document

_VOWELS = frozenset("aeiouAEIOU")


def count_syllables(word: str) -> int:
    """Croatian syllables ≈ vowel groups, plus syllabic r in vowelless contexts.

    The letter j is a consonant, so *vrijeme* → v-r-i-j-e-m-e counts i/e/e = 3
    correctly. Syllabic r (prst, trg, vrt) adds a nucleus when an r has no
    adjacent vowel. Minimum 1 for any word with letters.
    """
    if not word:
        return 0
    groups = 0
    previous_vowel = False
    for ch in word:
        is_vowel = ch in _VOWELS
        if is_vowel and not previous_vowel:
            groups += 1
        previous_vowel = is_vowel
    syllabic_r = 0
    lower = word.lower()
    for i, ch in enumerate(lower):
        if ch != "r":
            continue
        left_vowel = i > 0 and lower[i - 1] in _VOWELS
        right_vowel = i < len(lower) - 1 and lower[i + 1] in _VOWELS
        if not left_vowel and not right_vowel:
            syllabic_r += 1
    return max(groups + syllabic_r, 1)


@feature("readability")
def syllables_per_word_mean(doc: Document, ctx: FeatureContext) -> float:
    """Mean syllables per word (Croatian counter incl. syllabic r).

    Why: lexical complexity proxy; generators aim at register-typical values while
    human variance is wider, and it interacts with tokenizer fertility later.
    Failure mode: heavy foreign-name content breaks the vowel heuristic.
    """
    if not doc.words:
        return 0.0
    return fmean(count_syllables(w) for w in doc.words)


@feature("readability")
def long_word_rate(doc: Document, ctx: FeatureContext) -> float:
    """Share of words with ≥4 syllables.

    Why: long-word share tracks nominal/bureaucratic style — a register where
    Croatian LLM output clusters; humans move between registers within one text.
    Failure mode: correlates with topic (administrative vocabulary) as much as
    with authorship.
    """
    if not doc.words:
        return 0.0
    return sum(1 for w in doc.words if count_syllables(w) >= 4) / len(doc.words)
