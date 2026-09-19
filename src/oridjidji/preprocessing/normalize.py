"""Unicode normalisation, homoglyph mapping, and diacritic metrics.

Pipeline order: ``normalize_unicode`` (NFKC, invisible-character removal,
exotic-whitespace mapping, newline normalisation) → ``HomoglyphNormalizer`` →
diacritic analysis on the result. The raw text is kept alongside.

``normalize_unicode`` is idempotent (property-tested): homoglyph targets are ASCII
Latin while sources are Cyrillic/Greek, so no output character is ever an input of
a later pass.
"""

from __future__ import annotations

import itertools
import json
import unicodedata
from functools import lru_cache
from math import prod
from pathlib import Path

from oridjidji.preprocessing.lexicon import Lexicon

# Invisible/format characters removed outright (the specification names U+200B–200D and U+FEFF;
# soft hyphen, word joiner and bidi marks are the same evasion surface).
_INVISIBLES = "\u200b\u200c\u200d\u200e\u200f\u2060\u00ad\ufeff"

_DIACRITICS_LOWER = frozenset("čćžšđ")
_DIACRITICS_ALL = frozenset("čćžšđČĆŽŠĐ")
_STRIP_MAP = str.maketrans("čćžšđČĆŽŠĐ", "cczsdCCZSD")
# "Ošišana latinica" restoration options per bare letter.
_RESTORE_OPTIONS: dict[str, tuple[str, ...]] = {
    "c": ("č", "ć"),
    "s": ("š",),
    "z": ("ž",),
    "d": ("đ",),
}
_MAX_VARIANTS = 200


def count_invisibles(text: str) -> int:
    """Number of zero-width/invisible characters in ``text`` (evasion signal)."""
    return sum(1 for ch in text if ch in _INVISIBLES)


def normalize_unicode(text: str) -> str:
    """NFKC + newline normalisation + invisible removal + exotic-space mapping.

    Runs of ordinary spaces are deliberately preserved: doubled spaces are a
    stylometric feature, not noise.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch not in _INVISIBLES)
    return "".join(" " if (ch != " " and unicodedata.category(ch) == "Zs") else ch for ch in text)


@lru_cache(maxsize=1)
def _confusables_map() -> dict[str, str]:
    path = Path(__file__).parent / "data" / "confusables_map.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = dict(payload["map"])
    return mapping


class HomoglyphNormalizer:
    """Maps single Cyrillic/Greek look-alike characters to their Latin targets.

    The map is a committed artefact derived from Unicode UTS #39 confusables.txt
    (v17.0.0) by ``tools/build_confusables_map.py`` — never hand-written.
    Heavy homoglyph use is itself an evasion signal, surfaced via the ratio.
    """

    def __init__(self) -> None:
        self._map = _confusables_map()

    def normalize(self, text: str) -> tuple[str, int]:
        """Return (normalised text, number of characters replaced)."""
        n_replaced = sum(1 for ch in text if ch in self._map)
        if n_replaced:
            text = "".join(self._map.get(ch, ch) for ch in text)
        return text, n_replaced


def homoglyph_ratio(n_replaced: int, text: str) -> float:
    """Replaced characters per letter of ``text`` — 0.0 for letterless input."""
    n_letters = sum(1 for ch in text if ch.isalpha())
    return n_replaced / n_letters if n_letters else 0.0


def diacritic_density(text: str) -> float:
    """Croatian diacritics (č ć ž š đ) per 1,000 characters (Appendix A)."""
    if not text:
        return 0.0
    return sum(1 for ch in text if ch in _DIACRITICS_ALL) / len(text) * 1000


def strip_diacritics(text: str) -> str:
    """č/ć→c, ž→z, š→s, đ→d (both cases) — simulates "ošišana latinica"."""
    return text.translate(_STRIP_MAP)


def restoration_variants(word: str) -> list[str]:
    """All diacritic restorations of a bare lowercase word (capped, excl. itself)."""
    options = [(ch, *_RESTORE_OPTIONS.get(ch, ())) for ch in word]
    total = prod(len(o) for o in options)
    if total <= 1 or total > _MAX_VARIANTS:
        return []
    return ["".join(chars) for chars in itertools.product(*options) if "".join(chars) != word]


def expected_diacritic_deficit(words: list[str], lexicon: Lexicon) -> float:
    """Share of words invalid as written but valid once diacritics are restored.

    Measures "ošišana latinica" relative to what the lexicon expects given the
    words actually present — a plain diacritic count cannot distinguish a text
    about "dan i noć" from one that dropped its diacritics. Ambiguous bare forms
    that are themselves valid words (e.g. *kuca* = 'it ticks') do not count, which
    makes this a conservative lower bound (documented failure mode).
    """
    if not words:
        return 0.0
    deficient = 0
    for word in words:
        lower = word.lower()
        if any(ch in _DIACRITICS_LOWER for ch in lower):
            continue  # already carries diacritics
        if not any(ch in _RESTORE_OPTIONS for ch in lower):
            continue  # nothing restorable
        if lexicon.is_word(lower):
            continue  # valid as written (possibly ambiguous) — conservative skip
        if any(lexicon.is_word(v) for v in restoration_variants(lower)):
            deficient += 1
    return deficient / len(words)
