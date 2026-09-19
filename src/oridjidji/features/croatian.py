"""Croatian orthographic and human-error features (the specification, Appendix A).

Ground truth for the linguistic claims: `.claude/skills/hr-linguistics/SKILL.md`.
Human-error features exploit an asymmetry: LLMs almost never make these mistakes,
so their *absence* is (weak, cumulative) evidence of machine text. All rates are
raw in M1; corpus-relative z-scores arrive with M2 statistics.

The AI-ism family lives in `aiism.py`. The split is the distinction the thesis
argues: this file measures surface form and the mistakes *humans* make, that one
measures habits *models* have.

Complexity-budget note (CLAUDE.md): this file exceeds 300 lines because every feature
carries its mandated what/Why/Failure-mode docstring — the bulk is that prose
and the compiled patterns, not logic, and no function here reaches 25 lines. The two
remaining families share the same normalised-text input and the same lexicons, so
splitting them further would spread one contract across files without removing
anything.

COMPLEXITY BUDGET (CLAUDE.md, 381 lines). One function per Croatian feature, each with the
docstring the specification requires: what it measures, why it should differ between human and machine
Croatian, and its known failure mode. The docstrings are most of the length and they are
the part that has to survive an oral defence.
"""

from __future__ import annotations

import re

from oridjidji.features.base import FeatureContext, feature
from oridjidji.features.wordlists import (
    IJE_JE_WRONG_FORMS,
    NE_VERB_FUSED_FORMS,
    NEBI_FORMS,
    SA_ALLOWED_NEXT_INITIALS,
    SA_ALLOWED_NEXT_WORDS,
)
from oridjidji.schemas import Document

_CC = frozenset("čć")
# Excludes positions after a comma/semicolon AND after sentence-enders — a
# sentence-initial "Ali ..." needs no comma and must not count as an error.
# Relative *koji* and its inflected forms belong with the conjunctions: a Croatian
# relative clause takes a comma, and omitting it is the same fast-typing slip.
# *da* is deliberately absent — a Croatian object clause takes NO comma before it
# ("Mislim da je točno"), so counting it would flag correct prose as an error.
# Longest-first alternation, or "kojima" would match as "koji" + leftover.
_REL = r"koj(?:ima|ega|emu|eg|em|oj|im|ih|ju|u|i|a|e)"
_MISSING_COMMA_CONJ = re.compile(rf"(?<![,;:.!?…–—-])\s+(jer|ali|nego|{_REL})\b", re.IGNORECASE)
_ANY_CONJ = re.compile(rf"\b(jer|ali|nego|{_REL})\b", re.IGNORECASE)
_SPACE_BEFORE_PUNCT = re.compile(r" [,.!?;:]")
_TRIPLE_DOT = re.compile(r"(?<!\.)\.\.\.(?!\.)")
# Croatian dd.mm.yyyy. dates; the day and month groups are kept so the *padding*
# style (1.1. vs 01.01.) can be compared across a text.
_DATE = re.compile(r"\b(\d{1,2})\.\s?(\d{1,2})\.\s?\d{4}\.?")


def _per_100_words(count: int, doc: Document) -> float:
    return count / doc.n_words * 100 if doc.n_words else 0.0


def _per_kchar(count: int, doc: Document) -> float:
    n = len(doc.normalized)
    return count / n * 1000 if n else 0.0


# --- orthographic ---------------------------------------------------------


@feature("orthographic")
def diacritic_density(doc: Document, ctx: FeatureContext) -> float:
    """Croatian diacritics per 1,000 characters (measured in preprocessing).

    Why: fluent written Croatian has a stable diacritic rate; 'ošišana latinica'
    (human informal typing) and some generator tokenizer artefacts push it down.
    Failure mode: all-caps or heavily numeric text lowers it without meaning.
    """
    return doc.diacritic_density


@feature("orthographic")
def diacritic_deficit(doc: Document, ctx: FeatureContext) -> float:
    """Share of words invalid as written but valid after diacritic restoration.

    Why: separates 'writes without diacritics' (human informal habit) from 'text
    simply lacks diacritic-bearing words'; LLM Croatian rarely drops diacritics.
    Failure mode: conservative lower bound — bare forms that are themselves valid
    words (kuca/kuća) never count; 0.0 when no lexicon was supplied.
    """
    return doc.diacritic_deficit


@feature("orthographic")
def homoglyph_ratio(doc: Document, ctx: FeatureContext) -> float:
    """Share of letters that were Cyrillic/Greek look-alikes before normalisation.

    Why: heavy homoglyph use is an evasion signal in itself — honest text
    in either class contains essentially none.
    Failure mode: legitimately mixed-script quotations (rare in this domain).
    """
    return doc.homoglyph_ratio


@feature("orthographic")
def zero_width_per_kchar(doc: Document, ctx: FeatureContext) -> float:
    """Invisible/zero-width characters per 1,000 characters of raw input.

    Why: invisible-character injection is a detector-evasion technique; organic
    text has none beyond stray copy-paste artefacts.
    Failure mode: text copied from PDFs/web sometimes carries innocent soft
    hyphens.
    """
    n = len(doc.raw)
    return doc.zero_width_count / n * 1000 if n else 0.0


@feature("orthographic")
def quotes_croatian_per_kchar(doc: Document, ctx: FeatureContext) -> float:
    """Croatian quotation marks („ » «) per 1,000 characters.

    Why: humans using Croatian typographic conventions signal deliberate,
    edited writing; LLM output overwhelmingly uses straight or English curly
    quotes.
    Failure mode: many human web texts also use straight quotes; the ambiguous
    right-double-quote (”) is deliberately not counted for either side.
    """
    return _per_kchar(sum(doc.normalized.count(q) for q in "„»«"), doc)


@feature("orthographic")
def quotes_straight_per_kchar(doc: Document, ctx: FeatureContext) -> float:
    """Straight double quotes (") per 1,000 characters.

    Why: complements the Croatian-quote rate; the *mix* of styles inside one text
    is an editing/generation fingerprint.
    Failure mode: quoting code or measurements inflates it.
    """
    return _per_kchar(doc.normalized.count('"'), doc)


@feature("orthographic")
def quotes_curly_english_per_kchar(doc: Document, ctx: FeatureContext) -> float:
    """English opening curly quotes (“) per 1,000 characters.

    Why: “ is the strongest single marker of English-trained generation habits in
    Croatian text — Croatian convention opens with „.
    Failure mode: text pasted through English-locale word processors.
    """
    return _per_kchar(doc.normalized.count("“"), doc)


@feature("orthographic")
def dashes_per_kchar(doc: Document, ctx: FeatureContext) -> float:
    """Em/en dashes (— –) per 1,000 characters.

    Why: dash-heavy sentence construction is a widely observed LLM stylistic tic;
    Croatian human web prose favours commas and hyphens.
    Failure mode: professionally typeset human text (books, quality journalism)
    is also dash-rich.
    """
    return _per_kchar(doc.normalized.count("—") + doc.normalized.count("–"), doc)


def _per_kchar_raw(count: int, doc: Document) -> float:
    n = len(doc.raw)
    return count / n * 1000 if n else 0.0


@feature("orthographic")
def ellipsis_unicode_per_kchar(doc: Document, ctx: FeatureContext) -> float:
    """Single-character ellipsis (…) per 1,000 characters of RAW input.

    Measured on raw text because NFKC folds … into "..." — the distinction only
    exists before normalisation (discovered by the golden test, kept deliberate).
    Why: … (vs typed ...) reflects word-processor/human editing habits and differs
    between generators.
    Failure mode: some publishing pipelines fold the character upstream of us.
    """
    return _per_kchar_raw(doc.raw.count("…"), doc)


@feature("orthographic")
def ellipsis_dots_per_kchar(doc: Document, ctx: FeatureContext) -> float:
    """Typed three-dot ellipses (...) per 1,000 characters of RAW input.

    Raw for the same NFKC-folding reason as ellipsis_unicode — on normalised text
    every … would masquerade as a typed "...".
    Why: the informal human variant of the ellipsis.
    Failure mode: numeric ranges and code fragments can false-match dots.
    """
    return _per_kchar_raw(len(_TRIPLE_DOT.findall(doc.raw)), doc)


@feature("orthographic")
def space_before_punct_per_kchar(doc: Document, ctx: FeatureContext) -> float:
    """Spaces before,. ! ?;: per 1,000 characters.

    Why: a purely human typing error; generator output is tokenizer-clean.
    Failure mode: French-convention quotations legitimately space some punctuation.
    """
    return _per_kchar(len(_SPACE_BEFORE_PUNCT.findall(doc.normalized)), doc)


@feature("orthographic")
def double_space_per_kchar(doc: Document, ctx: FeatureContext) -> float:
    """Doubled spaces per 1,000 characters (preserved by normalisation).

    Why: human typing artefact that models essentially never produce.
    Failure mode: some editors auto-collapse spaces, erasing the signal upstream.
    """
    return _per_kchar(doc.normalized.count("  "), doc)


# --- human_error ----------------------------------------------------------


@feature("human_error")
def nebi_rate(doc: Document, ctx: FeatureContext) -> float:
    """Fused negated conditionals (*nebi/nebih/nebismo/nebiste*) per 100 words.

    Why: among the most common Croatian orthography errors; LLMs virtually never
    fuse *ne bi*.
    Failure mode: sparse — most human texts also contain zero occurrences, so the
    signal is one-directional (presence ⇒ human-ish, absence ⇒ uninformative).
    """
    return _per_100_words(sum(1 for w in doc.words if w.lower() in NEBI_FORMS), doc)


@feature("human_error")
def sa_misuse_rate(doc: Document, ctx: FeatureContext) -> float:
    """Nonstandard *sa* before words not starting with s/š/z/ž, per 100 words.

    Why: overusing *sa* (\"sa bratom\") is a frequent human colloquialism; models
    trained on edited prose apply the standard rule.
    Failure mode: the standard allows *sa* before some clusters (sa mnom, sa
    psom) — the allow-list is small and regional usage varies.
    """
    words = [w.lower() for w in doc.words]
    misuses = sum(
        1
        for i, w in enumerate(words[:-1])
        if w == "sa"
        and not words[i + 1].startswith(SA_ALLOWED_NEXT_INITIALS)
        and words[i + 1] not in SA_ALLOWED_NEXT_WORDS
    )
    return _per_100_words(misuses, doc)


@feature("human_error")
def ije_je_error_rate(doc: Document, ctx: FeatureContext) -> float:
    """Known-wrong ije/je forms (seed list) per 100 words.

    Why: ije/je alternation is the hardest Croatian orthography rule for humans;
    generator output gets it right almost uniformly.
    Failure mode: seed list covers a dozen frequent errors only — a lower bound,
    extended empirically in M2.
    """
    return _per_100_words(sum(1 for w in doc.words if w.lower() in IJE_JE_WRONG_FORMS), doc)


@feature("human_error")
def missing_comma_rate(doc: Document, ctx: FeatureContext) -> float:
    """Share of *jer/ali/nego* and relative *koji* clauses missing the comma.

    Why: comma before these conjunctions and before a relative clause is prescribed;
    dropping it is a typical human fast-typing error that models rarely make.
    Failure mode: sentence-initial conjunctions need no comma and are excluded by
    the whitespace guard, but poetic inversions still false-positive; 0.0 when no
    conjunction occurs (uninformative, not evidence).
    """
    total = len(_ANY_CONJ.findall(doc.normalized))
    if total == 0:
        return 0.0
    missing = len(_MISSING_COMMA_CONJ.findall(doc.normalized))
    return missing / total


@feature("human_error")
def cc_confusion_rate(doc: Document, ctx: FeatureContext) -> float:
    """Words invalid as written but valid after swapping č↔ć, per 100 words.

    Why: č/ć confusion is the canonical Croatian spelling error; its presence is
    strong human evidence.
    Failure mode: needs a lexicon (0.0 without one); rare-word gaps in hunspell-hr
    can miss genuine confusions.
    """
    if ctx.lexicon is None:
        return 0.0
    confused = 0
    for word in doc.words:
        lower = word.lower()
        if not any(ch in _CC for ch in lower):
            continue
        if ctx.lexicon.is_word(lower):
            continue
        swapped_variants = _cc_variants(lower)
        if any(ctx.lexicon.is_word(v) for v in swapped_variants):
            confused += 1
    return _per_100_words(confused, doc)


def _cc_variants(word: str) -> list[str]:
    options = [("č", "ć") if ch in _CC else (ch,) for ch in word]
    if sum(1 for o in options if len(o) > 1) > 6:  # combinatorial guard
        return []
    variants: list[str] = [""]
    for opts in options:
        variants = [v + o for v in variants for o in opts]
    return [v for v in variants if v != word]


# --- ai_ism ---------------------------------------------------------------


@feature("human_error")
def ne_verb_fusion_rate(doc: Document, ctx: FeatureContext) -> float:
    """Fused *ne* + verb (*nezna*, *nemogu*) per 100 words, beyond *ne bi*.

    Why: Croatian writes the negative particle apart from its verb, and fusing it is
    a frequent fast-typing slip that models essentially never make — the same
    asymmetry `nebi_rate` exploits, widened past the conditional.
    Failure mode: the list is curated, so it is a lower bound and misses rarer verbs;
    correct fusions (nemam, neću, nisam, nemoj) and *ne*-prefixed adjectives
    (nemoguć, nesretan) are excluded by construction rather than by a prefix rule.
    """
    return _per_100_words(sum(1 for w in doc.words if w.lower() in NE_VERB_FUSED_FORMS), doc)


@feature("human_error")
def s_sa_underuse_rate(doc: Document, ctx: FeatureContext) -> float:
    """Bare *s* before s/š/z/ž, where the standard prescribes *sa*, per 100 words.

    Why: the mirror of `sa_misuse_rate`. That feature catches humans over-using *sa*;
    this catches the opposite slip, so the pair covers both directions of the same
    rule instead of only the commoner one.
    Failure mode: *s* is also a frequent abbreviation and appears in enumerations,
    and regional usage varies; sparse in short texts, like every error feature here.
    """
    words = [w.lower() for w in doc.words]
    underuses = sum(
        1
        for i, w in enumerate(words[:-1])
        if w == "s" and words[i + 1].startswith(SA_ALLOWED_NEXT_INITIALS)
    )
    return _per_100_words(underuses, doc)


@feature("human_error")
def proper_noun_case_inconsistency(doc: Document, ctx: FeatureContext) -> float:
    """Share of word types written both capitalised and lower-case mid-sentence.

    Why: humans drift on proper-noun capitalisation across a long text (*Zagreb* then
    *zagreb*); models are relentlessly consistent, so a non-zero value is human
    evidence and zero is uninformative.
    Failure mode: only mid-sentence tokens are considered, since a sentence-initial
    capital says nothing — which means short texts contribute almost no evidence.
    Common words that legitimately open a clause after a colon can still slip in.
    """
    seen_lower: set[str] = set()
    seen_upper: set[str] = set()
    for sentence in doc.sentences:
        words = re.findall(r"[^\W\d_]+", sentence, re.UNICODE)
        for word in words[1:]:  # skip the sentence-initial token
            (seen_upper if word[0].isupper() else seen_lower).add(word.lower())
    both = seen_lower & seen_upper
    types = seen_lower | seen_upper
    if not types:
        return 0.0
    return len(both) / len(types)


@feature("human_error")
def number_format_inconsistency(doc: Document, ctx: FeatureContext) -> float:
    """Minority share of date zero-padding styles (1.1.2024. vs 01.01.2024.).

    Why: a human writing quickly mixes padded and unpadded dates within one text;
    a model applies one convention throughout. Measured as the minority share, so a
    consistent text scores 0.0 whichever style it chose — the feature is about
    internal consistency, not about which format is correct.
    Failure mode: needs at least two dates to say anything, so most short texts
    return 0.0 for lack of evidence rather than for consistency.
    """
    dates = _DATE.findall(doc.normalized)
    if len(dates) < 2:
        return 0.0
    padded = sum(1 for day, month in dates if len(day) == 2 and len(month) == 2)
    return min(padded, len(dates) - padded) / len(dates)
