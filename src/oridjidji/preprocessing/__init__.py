"""Text preprocessing pipeline: normalise → homoglyphs → segment → measure.

``preprocess`` is the single entry point producing the frozen ``Document`` that
every downstream agent consumes. Raw text is preserved next to the normalised
form. Language identification runs by default (inject a fake detector or
pass ``check_language=False`` in unit tests); the length gate itself is enforced
at the API layer, not here — preprocessing measures, it does not refuse.
"""

from __future__ import annotations

from functools import lru_cache

from oridjidji.preprocessing.langid import DetectorLike, detect_language
from oridjidji.preprocessing.lexicon import Lexicon
from oridjidji.preprocessing.normalize import (
    HomoglyphNormalizer,
    count_invisibles,
    diacritic_density,
    expected_diacritic_deficit,
    homoglyph_ratio,
    normalize_unicode,
)
from oridjidji.preprocessing.segment import (
    canonicalise_paragraph_breaks,
    split_paragraphs,
    split_sentences,
    tokenize_words,
)
from oridjidji.schemas import Document


@lru_cache(maxsize=1)
def _homoglyph_normalizer() -> HomoglyphNormalizer:
    return HomoglyphNormalizer()


def preprocess(
    raw: str,
    *,
    lexicon: Lexicon | None = None,
    check_language: bool = True,
    detector: DetectorLike | None = None,
) -> Document:
    """Run the the specification pipeline and return a ``Document``.

    COMPLEXITY BUDGET (CLAUDE.md, >50 lines): a single fixed sequence — the the specification
    pipeline — that ends by packing its results into one ``Document``. Most of the length
    is that constructor. The order is the specification, so it stays in one place.

    ``diacritic_deficit`` is computed only when a lexicon is supplied (the
    production lexicon is fetched separately; unit tests pass a fixture) — without
    one it stays 0.0 rather than pretending to be a measurement.
    """
    zero_width = count_invisibles(raw)
    text = normalize_unicode(raw)
    text, n_homoglyphs = _homoglyph_normalizer().normalize(text)
    # Both classes, one rule: a corpus that marks paragraph breaks with a single newline
    # would otherwise read as a single paragraph and separate the classes on extraction
    # style rather than authorship. See `canonicalise_paragraph_breaks`.
    text = canonicalise_paragraph_breaks(text)

    words = tokenize_words(text)
    sentences = split_sentences(text)
    paragraphs = split_paragraphs(text)

    language = "hr"
    language_confidence: float | None = None
    language_candidates: list[tuple[str, float]] = []
    language_accepted: bool | None = None
    language_is_bcs: bool | None = None
    language_message_hr: str | None = None
    if check_language and text.strip():
        check = detect_language(text, detector)
        language = check.language
        language_confidence = check.confidence
        language_candidates = check.candidates
        language_accepted = check.accepted
        language_is_bcs = check.is_bcs
        language_message_hr = check.message_hr

    return Document(
        raw=raw,
        normalized=text,
        language=language,
        words=words,
        sentences=sentences,
        paragraphs=paragraphs,
        zero_width_count=zero_width,
        n_homoglyphs_normalized=n_homoglyphs,
        homoglyph_ratio=min(homoglyph_ratio(n_homoglyphs, text), 1.0),
        diacritic_density=diacritic_density(text),
        diacritic_deficit=(expected_diacritic_deficit(words, lexicon) if lexicon else 0.0),
        language_confidence=language_confidence,
        language_candidates=language_candidates,
        language_accepted=language_accepted,
        language_is_bcs=language_is_bcs,
        language_message_hr=language_message_hr,
    )
