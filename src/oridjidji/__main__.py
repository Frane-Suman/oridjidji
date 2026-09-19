"""`python -m oridjidji` — the same CLI as the `oridjidji` console script.

Exists so a script can invoke the CLI through the interpreter that is already running,
rather than hoping the console script is on PATH. `scripts/regenerate_m4_results.py` runs
five long commands in sequence and must not fail on step three because a shell resolved a
different environment.

**stdout is forced to UTF-8 first.** Every user-facing string in this project is Croatian
, and a Windows console defaults to cp1252, which cannot encode `đ` — so the help
text alone raises `UnicodeEncodeError` before a command runs. Reconfiguring here fixes it
for every `python -m` invocation rather than asking each caller to remember an environment
variable.
"""

from __future__ import annotations

import sys

from oridjidji.cli import app


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    app()


if __name__ == "__main__":
    main()
