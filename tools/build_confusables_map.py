"""Generate the vendored homoglyph map from Unicode UTS #39 confusables.txt.

Master prompt the specification requires the homoglyph map to be *built* from confusables.txt,
never hand-written. This script downloads the current data file, keeps exactly the
entries relevant to the Latin-look-alike attack surface (single CYRILLIC or GREEK
source character whose confusable target is a pure ASCII letter/digit sequence), and
writes a committed JSON artefact with full provenance. Unit tests read the JSON and
never touch the network. Re-run this script only to refresh the Unicode
version; commit the diff and note it in the project log.

Croatian diacritics (č ć ž š đ) are Latin-script and can never appear as sources
under the CYRILLIC/GREEK name filter — the map cannot damage legitimate Croatian.

Usage: uv run python tools/build_confusables_map.py
"""

from __future__ import annotations

import datetime
import json
import unicodedata
import urllib.request
from pathlib import Path

URL = "https://www.unicode.org/Public/security/latest/confusables.txt"
OUT = Path(__file__).resolve().parents[1] / "src/oridjidji/preprocessing/data/confusables_map.json"
INCLUDE_NAME_PREFIXES = ("CYRILLIC", "GREEK")


def main() -> None:
    raw = urllib.request.urlopen(URL).read().decode("utf-8-sig")
    lines = raw.splitlines()
    header = [ln for ln in lines[:10] if ln.startswith("#") and ("Date" in ln or "Version" in ln)]

    mapping: dict[str, str] = {}
    for line in lines:
        data = line.split("#", 1)[0].strip()
        if not data:
            continue
        parts = [p.strip() for p in data.split(";")]
        if len(parts) < 3:
            continue
        src_cps = parts[0].split()
        if len(src_cps) != 1:  # only single-character sources are normalisable in place
            continue
        src = chr(int(src_cps[0], 16))
        tgt = "".join(chr(int(cp, 16)) for cp in parts[1].split())
        if not tgt or not all(c.isascii() and c.isalnum() for c in tgt):
            continue
        try:
            name = unicodedata.name(src)
        except ValueError:
            continue
        if not name.startswith(INCLUDE_NAME_PREFIXES):
            continue
        mapping[src] = tgt

    out = {
        "_meta": {
            "source": URL,
            "source_header": header,
            "generated": datetime.date.today().isoformat(),
            "rule": (
                "single-char CYRILLIC/GREEK sources with pure ASCII-alphanumeric "
                "confusable targets; licence: Unicode License"
            ),
            "entries": len(mapping),
        },
        "map": mapping,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {len(mapping)} mappings to {OUT}")
    print("\n".join(header))


if __name__ == "__main__":
    main()
