"""Minimal string-payload disk cache keyed by content hash.

First user: CLASSLA parse results (the specification — parsing is the bottleneck, so parses are
cached by text hash + model version). Deliberately primitive: one file per key,
no eviction, no locking — a thesis prototype's cache must be explainable in one
breath and deletable with one command.
"""

from __future__ import annotations

import re
from pathlib import Path

_KEY_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class DiskCache:
    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)

    def _path(self, key: str) -> Path:
        if not _KEY_RE.match(key):
            raise ValueError(f"Cache keys must be filesystem-safe, got: {key!r}")
        return self._dir / f"{key}.json"

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def set(self, key: str, value: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path(key).write_text(value, encoding="utf-8")
