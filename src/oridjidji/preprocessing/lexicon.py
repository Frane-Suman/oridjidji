"""Croatian lexicon access behind one small protocol.

Production path: hunspell-hr read by spylls (pure Python; DECISIONS an earlier decision), fetched
into ``data/lexicons/`` by ``tools/fetch_lexicons.py``. Tests use ``WordSetLexicon``
with hand-written fixtures so unit tests never touch the network.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Lexicon(Protocol):
    """Anything that can answer "is this a valid Croatian word?"."""

    def is_word(self, word: str) -> bool: ...


class WordSetLexicon:
    """In-memory lexicon over a plain word set (test fixtures, seed lists)."""

    def __init__(self, words: Iterable[str]) -> None:
        self._words = {w.lower() for w in words}

    def is_word(self, word: str) -> bool:
        return word.lower() in self._words


class SpyllsLexicon:
    """hunspell-hr via spylls, lazily loaded from ``<base>.aff`` / ``<base>.dic``."""

    def __init__(self, base_path: str | Path = Path("data/lexicons/hr")) -> None:
        self._base = Path(base_path)
        self._dictionary: Any | None = None
        # spylls lookup does full affix analysis per call (~0.5 ms); texts repeat
        # words and the deficit check probes many variants, so memoise per word.
        self._memo: dict[str, bool] = {}

    def _load(self) -> Any:
        if self._dictionary is None:
            if not self._base.with_suffix(".aff").is_file():
                raise FileNotFoundError(
                    f"hunspell-hr dictionary not found at '{self._base}.aff/.dic'. "
                    "Fetch it with: uv run python tools/fetch_lexicons.py"
                )
            from spylls.hunspell import Dictionary

            self._dictionary = Dictionary.from_files(str(self._base))
        return self._dictionary

    def is_word(self, word: str) -> bool:
        cached = self._memo.get(word)
        if cached is not None:
            return cached
        dictionary = self._load()
        result = bool(dictionary.lookup(word)) or bool(dictionary.lookup(word.lower()))
        self._memo[word] = result
        return result
