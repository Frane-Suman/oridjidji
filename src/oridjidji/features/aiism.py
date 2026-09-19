"""AI-ism features — the register tics of generated Croatian (family ``ai_ism``, the specification).

Split out of ``croatian.py``, which had grown to hold three families at once. The
division is the one the thesis actually argues: ``croatian.py`` measures *surface form
and the mistakes humans make*, this module measures *habits models have*. Both halves
stay explainable in a sentence, which the combined file no longer was.

Everything here is a **seed**, not a finding. Appendix A's phrase list is explicitly a
starting point; the real AI-ism lexicon is discovered empirically (log-odds with an
informative Dirichlet prior over the built corpus) in M3+, and the discovered top-N is
itself a thesis result. Never hand-extend these lists silently — a list grown by
intuition would make the feature measure the author's expectations instead of the data.
"""

from __future__ import annotations

import re

from oridjidji.features.base import FeatureContext, feature
from oridjidji.features.wordlists import AI_ISM_SEED_PHRASES, CONNECTIVE_STARTERS
from oridjidji.schemas import Document

# "A, B i C" — three coordinated items, the classic rhetorical triple. Requires the
# comma so ordinary two-item coordination ("kruh i mlijeko") cannot match.
_TRICOLON = re.compile(
    r"[^\W\d_]+(?:\s+[^\W\d_]+){0,2},\s+[^\W\d_]+(?:\s+[^\W\d_]+){0,2}\s+i\s+[^\W\d_]+",
    re.UNICODE,
)
# Markdown-ish list scaffolding at the start of a line: -, *, •, or "1." / "1)".
_BULLET_LINE = re.compile(r"(?m)^\s*(?:[-*•‣▪]|\d{1,2}[.)])\s+\S")


def _per_100_words(count: int, doc: Document) -> float:
    return count / doc.n_words * 100 if doc.n_words else 0.0


@feature("ai_ism")
def ai_phrase_rate(doc: Document, ctx: FeatureContext) -> float:
    """Seed AI-ism phrases (*važno je napomenuti*, …) per 100 words.

    Why: assistant-register formulas are heavily over-represented in LLM output;
    the seed list is replaced by an empirically discovered lexicon in M3+.
    Failure mode: formal human writing (school essays!) uses some of the same
    phrases — this feature must never act alone, only inside the ensemble.
    """
    haystack = doc.normalized.lower()
    return _per_100_words(sum(haystack.count(p) for p in AI_ISM_SEED_PHRASES), doc)


@feature("ai_ism")
def connective_start_rate(doc: Document, ctx: FeatureContext) -> float:
    """Share of sentences opening with a discourse connective (Stoga, Nadalje…).

    Why: over-explicit connective scaffolding is a recognisable LLM structuring
    habit in Croatian.
    Failure mode: academic human prose also connects explicitly; short texts make
    the share jumpy.
    """
    if not doc.sentences:
        return 0.0
    starters = 0
    for sentence in doc.sentences:
        first = re.search(r"[^\W\d_]+", sentence, re.UNICODE)
        if first and first.group(0).lower() in CONNECTIVE_STARTERS:
            starters += 1
    return starters / len(doc.sentences)


@feature("ai_ism")
def tricolon_rate(doc: Document, ctx: FeatureContext) -> float:
    """Three-item coordinations ("A, B i C") per sentence (Appendix A).

    Why: the rhetorical triple is a documented LLM habit — models reach for a
    balanced list of three where human prose more often gives two items or five.
    Measured per sentence rather than per word so long sentences do not dominate.
    Failure mode: genuine enumerations (ingredients, authors, place names) match
    identically, so news and academic text carry a real baseline; this separates
    registers as much as classes and must be read against the per-domain norm.
    """
    if not doc.sentences:
        return 0.0
    return sum(len(_TRICOLON.findall(s)) for s in doc.sentences) / len(doc.sentences)


@feature("ai_ism")
def bullet_scaffold_rate(doc: Document, ctx: FeatureContext) -> float:
    """Share of lines that open as a bullet or numbered list item (Appendix A).

    Why: chat models structure answers as lists even when asked for prose, so
    scaffolding that survives into a "write an article" response is a strong tic.
    Failure mode: **the cleaning stage strips markdown scaffolding from generated
    text, so on the built dataset this should read near zero for the AI class
    by construction.** Its real use is on raw and on user-pasted input at serve time,
    where nothing has been cleaned — reporting it on cleaned training data would
    measure the cleaner, not the writer.
    """
    lines = [line for line in doc.normalized.splitlines() if line.strip()]
    if not lines:
        return 0.0
    return len(_BULLET_LINE.findall(doc.normalized)) / len(lines)
