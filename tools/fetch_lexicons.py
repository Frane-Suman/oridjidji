"""Fetch the hunspell-hr dictionary into data/lexicons/ (gitignored).

Source: github.com/krunose/hunspell-hr (tri-licensed GPL-2.0/LGPL-2.1/MPL-1.1 per
its README_hr_HR.txt — consumed as data files at runtime; see docs/LICENSES.md).
File names are discovered via the GitHub contents API rather than hardcoded, and the
downloaded files are normalised to data/lexicons/hr.aff + hr.dic, with SHA-256
printed so the downloaded dictionary can be identified later.

Usage: uv run python tools/fetch_lexicons.py
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

API = "https://api.github.com/repos/krunose/hunspell-hr/contents/"
DEST = Path(__file__).resolve().parents[1] / "data" / "lexicons"


def main() -> None:
    listing = json.loads(urllib.request.urlopen(API).read().decode("utf-8"))
    wanted: dict[str, str] = {}  # extension -> download_url
    for entry in listing:
        name = entry.get("name", "")
        for ext in (".aff", ".dic"):
            if name.endswith(ext) and ext not in wanted:
                wanted[ext] = entry["download_url"]
    if set(wanted) != {".aff", ".dic"}:
        raise SystemExit(f"Expected one .aff and one .dic at repo root, found: {sorted(wanted)}")

    DEST.mkdir(parents=True, exist_ok=True)
    for ext, url in sorted(wanted.items()):
        data = urllib.request.urlopen(url).read()
        path = DEST / f"hr{ext}"
        path.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        print(f"{path}  bytes={len(data)}  sha256={digest}  source={url}")


if __name__ == "__main__":
    main()
