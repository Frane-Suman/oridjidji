"""Rule-based paragraph/sentence/word segmentation — the CPU fast path.

Deliberately simple: a curated Croatian abbreviation list plus ordinal-number
handling covers the news/essay register this project targets, stays inside the
<500 ms budget, and needs no model download. CLASSLA's segmenter is the
accurate alternative when the ``morphology`` extra is installed; disagreements are
a documented limitation, not a hidden one.
"""

from __future__ import annotations

import re

# Common Croatian abbreviations that end with a period mid-sentence.
_ABBREVIATIONS = frozenset(
    {
        "npr",
        "tj",
        "itd",
        "itsl",
        "sl",
        "odn",
        "tzv",
        "cca",
        "br",
        "str",
        "god",
        "st",
        "stolj",
        "čl",
        "toč",
        "sv",
        "usp",
        "engl",
        "lat",
        "dr",
        "sc",
        "mr",
        "mag",
        "prof",
        "doc",
        "dipl",
        "ing",
        "univ",
        "spec",
    }
)

# Boundary: terminator run (+ closing quotes/brackets), whitespace, then an
# optional opening quote/bracket followed by an uppercase letter or a digit.
_BOUNDARY = re.compile(
    r"(?P<punct>[.!?…]+)(?P<close>[\"'”’»«)\]]*)\s+"
    r"(?=[„“\"'«»(\[]?[A-ZČĆŽŠĐ0-9])"
)

_LAST_TOKEN = re.compile(r"([^\W_]+)$", re.UNICODE)


NEWLINE_RUN = re.compile(r"(?:[ \t]*\n[ \t]*)+")


def canonicalise_paragraph_breaks(text: str) -> str:
    """Rewrite every run of newlines as one blank line.

    This removes a near-perfect class marker, and it is the reason `split_paragraphs`
    can be trusted at all. That function splits on a BLANK line, but MaCoCu writes news
    paragraph breaks as a single ``\n``: 95.4% of human news parsed as ONE paragraph,
    while raw model output uses ``\n\n`` and 90% of AI text parsed as several.
    Paragraph count alone therefore separated the two classes almost perfectly — while
    measuring the extraction pipeline rather than the writing.

    Both classes had the structure; only the marker differed. Canonicalising it recovers
    `paragraph_len_variance` as a real feature instead of discarding it.

    It lives HERE, in `preprocess`, rather than at corpus-build time, so that it applies
    to every artefact regardless of which pipeline version wrote it — the human corpora
    predate the fix and cannot be rebuilt without invalidating the generation run that
    references their record ids.
    """
    return NEWLINE_RUN.sub("\n\n", text)


def flatten_paragraphs(text: str) -> str:
    """Join every paragraph into one, for the domain where the human side has no choice.

    ZNANJE/Dabar excerpting flattened the academic corpus outright — `excerpt` joins its
    blocks with a space, so **0.0%** of those 1,160 records contain any newline, against
    95% of generated academic text. Unlike news, that structure cannot be recovered: the
    information is gone from the source, and there is no marker left to canonicalise.

    That leaves one way to make the domain symmetric — remove the structure from the
    other side too. Transforming a single class is normally exactly how a cleaner becomes
    the signal; here it is the reverse, because the human class was already
    transformed by the extraction and this restores the balance.

    The cost is real and belongs in the limitations: "uniform paragraph lengths" is a
    genuine AI-ism and it cannot be measured on this dataset's academic domain. The
    per-domain norms artefact reports `paragraph_len_variance` as constant there, which
    is the honest way for an unavailable feature to present itself.
    """
    return " ".join(p.strip() for p in NEWLINE_RUN.split(text) if p.strip())


def split_paragraphs(text: str) -> list[str]:
    """Paragraphs = blocks separated by at least one blank line.

    Run `canonicalise_paragraph_breaks` first, or a corpus that marks breaks with a
    single newline will read as one paragraph.
    """
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def split_sentences(text: str) -> list[str]:
    """Sentence spans within ``text`` (works on a paragraph or a whole document).

    Known, accepted limitations (fast path): a sentence ending in a bare number
    ("… 2024.") does not split (Croatian ordinal dots make digit+period ambiguous),
    and unknown abbreviations followed by an uppercase word oversplit.
    """
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        before = text[: match.start("punct")]
        token_match = _LAST_TOKEN.search(before)
        token = token_match.group(1) if token_match else ""
        if token.isdigit():
            continue  # Croatian ordinal ("5. svibnja") — never a boundary here
        if token.lower() in _ABBREVIATIONS:
            continue
        if len(token) == 1 and token.isalpha() and token.isupper():
            continue  # initials: "I. Horvat"
        sentence = text[start : match.end("close")].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def tokenize_words(text: str) -> list[str]:
    """Unicode-letter word tokens (digits and punctuation excluded)."""
    return re.findall(r"[^\W\d_]+", text, re.UNICODE)
