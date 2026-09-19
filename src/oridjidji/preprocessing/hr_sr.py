"""Serbian / ekavian contamination in nominally Croatian text.

**The hole this fills.** A generator can produce Serbian-inflected Croatian and pass every
check the project runs. Measured on the Gemma 4 smoke test: a text containing `zahtevnim`,
`cev` and ekavian reflexes was scored `hr` by langid at confidence **1.000**, and its
out-of-vocabulary rate against hunspell-hr came in *below* human text's — because human OOV
is proper nouns and rare terms while a malformed word is neither. Both instruments said
"fine". A `grep` for any ekavian handling across `src/` returned nothing.

That matters because the thesis is specifically about **Croatian**. A generator writing
Serbian-tinged text would teach the detector "Serbian lexis -> machine", which is a confound
rather than authorship, and it would be invisible to every the specification check.

**How it works, in three sentences.** The jat rule asks the dictionary instead of asserting:
a word is ekavian if hunspell-hr rejects it but accepts it with `e` replaced by `je`/`ije`
(`uvek` -> `uvijek`, `cev` -> `cijev`), or with a Serbian `-ovati`/`-isati` verb ending
replaced by Croatian `-irati` (`kompromitovati` -> `kompromitirati`). A short hand list
covers the differences no jat reflex reaches — `uslov-`/`uvjet-`, `opšt-`/`opć-`. Both are
gated on the same precondition, so **nothing here can fire on a word the Croatian dictionary
accepts** (see `INVARIANT` below).

**What this is not.** `hr-linguistics` is blunt that hr/sr/bs separation is genuinely hard,
and nothing here claims to settle it. It cannot detect Serbian syntax, Serbian style, or a
Croatian text that happens to discuss Serbia — the last of which is why the human
distribution has a long tail (median 0.00, p99 10.64 per 1k on `dataset.parquet`, 9.90 on
`riznica.parquet`) and why this is a **generator** acceptance criterion, never a document
filter.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from weakref import WeakKeyDictionary

from oridjidji.preprocessing.lexicon import Lexicon

# The property that makes every marker below trustworthy, stated once so a future edit has
# to confront it: a marker fires ONLY on a surface form hunspell-hr rejects. A candidate
# that is also a Croatian word therefore cannot become a marker by accident.
#
# This is not theoretical tidiness. The first version of this module hand-listed markers and
# validated them one sample at a time, and two rounds of that shipped false positives:
# `trka`, `učesnik`, `povreda` and `posle` all look Serbian and are all in the dictionary
# (they inflated one Gemma text from 29 to 78 per 1k), and `\bvreme\w*` reached `vremena`,
# `vremenu` and `vremenskog`, which are the ordinary Croatian oblique cases of `vrijeme` —
# only the nominative differs. The invariant retires that whole class of bug.
INVARIANT = "a marker never fires on a word hunspell-hr accepts"

# Serbian ekavian collapses jat to `e` where Croatian writes `je` or `ije`. Rather than
# enumerate the pairs, put the candidate to the dictionary: substituting at each `e` costs
# at most 2n lookups on a word the dictionary has already rejected.
_JAT_SUBSTITUTIONS = ("je", "ije")

# Serbian forms verbs in `-ovati`/`-isati` where Croatian uses `-irati`, across the whole
# paradigm. A wrong entry here is harmless — it simply produces a non-word the dictionary
# rejects, so the marker does not fire.
_VERB_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("ovati", "irati"),
    ("isati", "irati"),
    ("ovanje", "iranje"),
    ("isanje", "iranje"),
    ("ovani", "irani"),
    ("isani", "irani"),
    ("ovan", "iran"),
    ("isan", "iran"),
    ("ovao", "irao"),
    ("ovala", "irala"),
    ("išu", "iraju"),
    ("iše", "ira"),
    ("uju", "iraju"),
    ("uje", "ira"),
)

# Differences the jat rule cannot reach, because they are lexical or orthographic rather
# than a jat reflex. Kept deliberately short: the rule above subsumed eleven of the fifteen
# markers this list used to carry, and a rule that consults the dictionary is easier to
# defend at a viva than a list I wrote by hand.
#
# Each is still subject to INVARIANT, which is why `tač\w*` is safe to write broadly here
# even though hunspell-hr accepts `tačke` — that form is a valid Croatian word, so it is
# skipped regardless of the pattern's reach.
#
# Three candidates were dropped on the same evidence that removed `trka`/`učesnik`/`povreda`/
# `posle` from the old list: hunspell-hr accepts `saobraćaj`, `hiljada` and `porodica`, which
# are Croatian words despite reading as Serbian. The invariant would have suppressed them
# anyway — they are gone from the list because a marker a reader cannot trust is worse than
# no marker at all. The slow test in `test_hr_sr.py` is what caught them.
LEXICAL_MARKERS: dict[str, str] = {
    r"\buslov\w*": "uvjet-",
    r"\bopšt\w*": "opć-",
    r"\btač\w*": "toč-",
    r"\bsaradn\w*": "surad-",
    r"\bsportist\w*": "sportaš-",
    r"\bpreduzeć\w*": "poduzeć-",
    r"\bpreduzetni\w*": "poduzetni-",
    r"\bbezbedn\w*": "sigurn-",
    r"\bvazduh\w*": "zrak-",
    r"\bostrv\w*": "otok-",
}

_LEXICAL_COMPILED = [
    (re.compile(pattern, re.IGNORECASE), croatian) for pattern, croatian in LEXICAL_MARKERS.items()
]

_WORD_RE = re.compile(r"[a-zA-ZčćžšđČĆŽŠĐ]+")

# Refuse a generator whose MEDIAN rate exceeds this. Derived from the human distribution,
# not from any generator: it sits at the 99th percentile of both human corpora
# (`dataset.parquet` 10.64, `riznica.parquet` 9.90) so normal Croatian — including an
# article about Serbia — passes, while a model that writes ekavian by default does not.
#
# Fixed at 10.0 before the candidate that prompted it had complete numbers, for the reason
# An earlier decision fixes thresholds in advance: a threshold chosen after seeing the score it must judge
# is not a threshold. It has since survived an instrument change — the jat rule finds strictly
# more Serbian than the hand list it replaced, and the human p99 it re-derives still lands on
# 10.0. A bar that holds when the instrument improves is evidence; moving it now would not be.
MAX_GENERATOR_RATE_PER_1K = 10.0


@dataclass(frozen=True)
class MarkerHit:
    """One Serbian form found, and the Croatian it displaces."""

    form: str
    croatian: str


_DEFAULT: Lexicon | None = None


def _default_lexicon() -> Lexicon:
    """The production hunspell-hr lexicon, built once so its per-word memo is shared."""
    global _DEFAULT
    if _DEFAULT is None:
        from oridjidji.preprocessing.lexicon import SpyllsLexicon

        _DEFAULT = SpyllsLexicon()
    return _DEFAULT


def _jat_candidates(word: str) -> list[str]:
    """Croatian forms `word` would take if its `e` were a collapsed jat, or its verb ending
    Serbian. Generated, not listed — the dictionary decides which (if any) is real."""
    candidates = [
        word[:index] + replacement + word[index + 1 :]
        for index, character in enumerate(word)
        if character == "e"
        for replacement in _JAT_SUBSTITUTIONS
    ]
    candidates.extend(
        word[: -len(suffix)] + croatian
        for suffix, croatian in _VERB_SUFFIXES
        if word.endswith(suffix)
    )
    return candidates


# Per-lexicon memo of `_croatian_for`. Keyed on the lexicon rather than shared, because two
# lexicons disagree by construction — the unit tests inject a small fixture where the
# production path uses hunspell-hr, and a shared cache would leak one's answers into the
# other. Weak keys so a discarded test lexicon does not pin its memo for the session.
#
# A `lru_cache` over `(lexicon, word)` would be shorter, but `Lexicon` is a Protocol and so
# not `Hashable` to mypy under --strict.
_MEMO: WeakKeyDictionary[Lexicon, dict[str, str | None]] = WeakKeyDictionary()


def _croatian_for(lexicon: Lexicon, word: str) -> str | None:
    """The Croatian form `word` displaces, or None if `word` is not a Serbian variant.

    Memoised because a corpus repeats its vocabulary heavily and this is the only expensive
    call in the module: an unknown word costs up to 2n dictionary lookups, and `median_rate`
    runs inside both `data.ingest` and `data.integrity`.
    """
    memo = _MEMO.setdefault(lexicon, {})
    if word in memo:
        return memo[word]
    if lexicon.is_word(word):
        answer = None  # INVARIANT: a valid Croatian word is never a marker
    else:
        answer = next((c for c in _jat_candidates(word) if lexicon.is_word(c)), None)
    memo[word] = answer
    return answer


def find_markers(text: str, lexicon: Lexicon | None = None) -> list[MarkerHit]:
    """Every Serbian/ekavian marker in `text`, in no particular order.

    Each word is tested once by the jat rule and, failing that, against the lexical list, so
    a form is counted at most once however many patterns it happens to match.
    """
    lex = lexicon if lexicon is not None else _default_lexicon()
    hits: list[MarkerHit] = []
    for word in _WORD_RE.findall(text):
        lowered = word.lower()
        croatian = _croatian_for(lex, lowered)
        if croatian is not None:
            hits.append(MarkerHit(form=word, croatian=croatian))
            continue
        if lex.is_word(lowered):
            continue  # INVARIANT, applied to the hand list as well as the rule
        for pattern, replacement in _LEXICAL_COMPILED:
            if pattern.match(lowered):
                hits.append(MarkerHit(form=word, croatian=replacement))
                break
    return hits


def serbian_marker_rate(text: str, lexicon: Lexicon | None = None) -> float:
    """Markers per 1,000 words. 0.0 for empty text.

    Per 1,000 rather than a raw count because the documents vary from ~100 to ~1,500 words,
    and a count would rank long texts as more contaminated purely for being long.
    """
    words = len(_WORD_RE.findall(text))
    return 1000.0 * len(find_markers(text, lexicon)) / words if words else 0.0


def median_rate(texts: Sequence[str], lexicon: Lexicon | None = None) -> float:
    """Median rate across texts — the statistic the acceptance criterion uses.

    Median, not mean: the first version of this measurement pooled two texts of 350 and 103
    words into one average and reported 22 per 1k as a property of the model, when the long
    text was at 5.7 and within human range. A mean over unequal texts lets the shortest one
    speak loudest; a median asks what a *typical* text from this generator looks like.
    """
    if not texts:
        return 0.0
    rates = sorted(serbian_marker_rate(text, lexicon) for text in texts)
    middle = len(rates) // 2
    if len(rates) % 2:
        return rates[middle]
    return (rates[middle - 1] + rates[middle]) / 2


def generator_is_acceptable(texts: Sequence[str], lexicon: Lexicon | None = None) -> bool:
    """Whether a generator's Croatian is clean enough for a roster slot.

    Judged on the median over its output, never per document: a single human article about
    Serbian football legitimately reaches 27 per 1k, so filtering documents would delete
    real text — and, applied to one class only, would itself become the class marker an earlier decision
    warns about.
    """
    return median_rate(texts, lexicon) <= MAX_GENERATOR_RATE_PER_1K
