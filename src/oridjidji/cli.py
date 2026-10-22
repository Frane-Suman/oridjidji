"""Command-line interface. Serving arrives in M8; until then: version/info/inspect."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Annotated

import typer

from oridjidji import __version__
from oridjidji.config import load_config


def _configure_windows_utf8_output() -> None:
    """Keep Croatian Typer/Rich help printable in legacy Windows terminals."""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")


_configure_windows_utf8_output()

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    # Keep the terminal-only title CP-1252 safe: Rich's legacy Windows renderer encodes help
    # before writing it, and Croatian đ is not representable there. The HTML UI retains the
    # proper Oriđiđi brand through the Croatian i18n dictionary.
    help="Oridjidji - Croatian machine-generated-text detection research prototype.",
)


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


@app.command()
def info(
    profile: Annotated[str, typer.Option(help="Configuration profile (lite|full).")] = "lite",
) -> None:
    """Show the effective configuration for a profile."""
    cfg = load_config(profile)
    typer.echo(
        f"profile={cfg.profile} language={cfg.language} seed={cfg.seed} "
        f"min_word_count={cfg.min_word_count} inference={cfg.inference}"
    )


@app.command("demo-check")
def demo_check(
    profile: Annotated[
        str, typer.Option(help="Local candidate profile to verify.")
    ] = "e12-advanced-research",
    smoke: Annotated[
        Path | None,
        typer.Option(
            "--smoke",
            help="Optional UTF-8 Croatian document; runs one real Basic and Advanced request.",
        ),
    ] = None,
    candidate_root: Annotated[
        Path | None,
        typer.Option(
            "--candidate-root",
            help="Root used to resolve relative paths in a portable candidate profile.",
        ),
    ] = None,
    basic_only: Annotated[
        bool,
        typer.Option(
            "--basic-only",
            help="Require only Basic mode (Advanced has not been installed).",
        ),
    ] = False,
) -> None:
    """Verify that the localhost demo exposes working Basic and Advanced contracts."""
    from oridjidji.api.preflight import inspect_local_demo, read_smoke_text

    previous_root = os.environ.get("ORIDJIDJI_CANDIDATE_ROOT")
    try:
        if candidate_root is not None:
            os.environ["ORIDJIDJI_CANDIDATE_ROOT"] = str(candidate_root.resolve())
        result = inspect_local_demo(
            profile,
            smoke_text=read_smoke_text(smoke) if smoke is not None else None,
            require_advanced=not basic_only,
        )
    finally:
        if candidate_root is not None:
            if previous_root is None:
                os.environ.pop("ORIDJIDJI_CANDIDATE_ROOT", None)
            else:
                os.environ["ORIDJIDJI_CANDIDATE_ROOT"] = previous_root
    typer.echo(result.model_dump_json(indent=2))
    if not result.ok:
        raise typer.Exit(code=1)


def _run_fast_path(raw: str, *, langid: bool) -> tuple[dict[str, float], dict[str, float], int]:
    """One preprocess+features pass; returns (timings_ms, features, n_words)."""
    from oridjidji.features.base import extract_features
    from oridjidji.preprocessing import preprocess
    from oridjidji.preprocessing.lexicon import SpyllsLexicon

    lexicon = SpyllsLexicon() if Path("data/lexicons/hr.aff").is_file() else None
    t0 = perf_counter()
    doc = preprocess(raw, lexicon=lexicon, check_language=langid)
    t1 = perf_counter()
    features = extract_features(doc, lexicon=lexicon)
    t2 = perf_counter()
    timings = {
        "preprocess_ms": (t1 - t0) * 1000,
        "features_ms": (t2 - t1) * 1000,
        "total_ms": (t2 - t0) * 1000,
    }
    return timings, features, doc.n_words


@app.command()
def inspect(
    path: Annotated[Path | None, typer.Argument(help="UTF-8 text file to analyse.")] = None,
    text: Annotated[
        str | None, typer.Option("--text", help="Inline text instead of a file.")
    ] = None,
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
    langid: Annotated[bool, typer.Option("--langid/--no-langid")] = True,
    norms: Annotated[
        str,
        typer.Option(help="Domain whose norms to z-score against (news, academic); '' skips."),
    ] = "",
) -> None:
    """Preprocess + fast-path features with timing (dev tool; measures section 12 budget).

    Runs the pipeline twice: the first (cold) pass pays lazy model/lexicon loading,
    the second (warm) pass is the steady-state number the budget applies to.

    COMPLEXITY BUDGET (>50 lines): a dev tool that measures, formats and
    prints. It is long because it has three output shapes (timing, features, z-scores)
    and two modes (text, JSON), not because it does three things.

    With `--norms news` each rate is shown beside its z-score against that corpus, which
    is what Appendix A asks for: `nebi_rate = 0.4` is meaningless alone, and worth a lot
    against a corpus whose mean is 0.05. Norms are built by `oridjidji data build-norms`.
    """
    if (path is None) == (text is None):
        raise typer.BadParameter("Provide exactly one of PATH or --text.")
    raw = text if text is not None else Path(path).read_text(encoding="utf-8")  # type: ignore[arg-type]

    cold, _, _ = _run_fast_path(raw, langid=langid)
    warm, features, n_words = _run_fast_path(raw, langid=langid)

    scores: dict[str, float] = {}
    missing: list[str] = []
    if norms:
        from oridjidji.features.norms import load_norms, norms_path_for, zscore_vector

        # `load_norms` raises on a missing file rather than returning zeros — see its
        # module docstring. Let that propagate; a silent "exactly average" is the bug.
        scores, missing = zscore_vector(features, load_norms(norms_path_for(norms)))

    if json_out:
        payload: dict[str, object] = {
            "cold": cold,
            "warm": warm,
            "n_words": n_words,
            "features": features,
        }
        if norms:
            payload["norms_domain"] = norms
            payload["zscores"] = scores
            payload["features_without_norms"] = missing
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=1))
        return
    typer.echo(
        f"words={n_words}  cold={cold['total_ms']:.1f} ms  warm={warm['total_ms']:.1f} ms "
        f"(preprocess {warm['preprocess_ms']:.1f} + features {warm['features_ms']:.1f})"
    )
    for name in sorted(features):
        line = f"  {name:32s} {features[name]:10.4f}"
        if name in scores:
            line += f"   z={scores[name]:+7.2f}"
        typer.echo(line)
    if missing:
        typer.echo(
            f"\n  {len(missing)} feature(s) have no norm in the "
            f"{norms} artefact and are shown without a z-score: "
            f"{', '.join(missing[:6])}"
        )


@app.command()
def serve(
    host: Annotated[
        str, typer.Option(help="Bind address. Localhost by default — see below.")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    warmup: Annotated[
        bool, typer.Option(help="Build the ONNX sessions at startup instead of on first use.")
    ] = True,
    reload: Annotated[bool, typer.Option(help="Development autoreload.")] = False,
    profile: Annotated[
        str, typer.Option(help="Configuration profile, including isolated candidates.")
    ] = "lite",
    candidate_root: Annotated[
        Path | None,
        typer.Option(
            "--candidate-root",
            help="Root used to resolve paths in a relocated portable candidate.",
        ),
    ] = None,
) -> None:
    """Run the local web UI and API (M8, section 8.14).

    **Binds to localhost by default, and that is a safety default rather than a convenience
    one.** the specification and `ETHICS.md` are explicit that this tool is research and educational and
    must never be used against an individual. A detector reachable from the network is one
    somebody else can point at a student's essay, so exposing it takes a deliberate
    `--host 0.0.0.0` rather than being the path of least resistance.

    `--warmup` builds the ONNX sessions before the first request. Measured at 27.9 s on the
    target laptop, so without it the first user pays that and reads it as the tool
    being slow.
    """
    try:
        import uvicorn
    except ModuleNotFoundError:
        # Not a BadParameter: nothing the caller typed is wrong, and rich would wrap the
        # message into a usage panel that hides the one line worth reading.
        typer.echo(
            "uvicorn is not installed. It lives in the `serve` extra so the library and "
            "its tests do not carry a web server. Install it with: uv sync --extra serve"
        )
        raise typer.Exit(code=1) from None

    from oridjidji.api.app import create_app

    # The API dependency reads the same profile for health, analysis, SSE and pages. An
    # environment value also survives uvicorn's reload child process.
    os.environ["ORIDJIDJI_PROFILE"] = profile
    if candidate_root is not None:
        os.environ["ORIDJIDJI_CANDIDATE_ROOT"] = str(candidate_root.resolve())
    load_config(
        profile, candidate_root=candidate_root
    )  # fail before starting uvicorn if the profile is absent or invalid
    if host not in ("127.0.0.1", "localhost"):
        typer.echo(
            f"WARNING: binding to {host}, so this detector is reachable beyond this machine. "
            "ETHICS.md: never for grading or discipline, and never against an individual."
        )
    typer.echo(f"http://{host}:{port}  (Ctrl+C to stop)")
    # `reload` needs an import string rather than an instance, and reloading would rebuild
    # the ONNX sessions on every edit — fine in development, wrong for a demo.
    if reload:
        uvicorn.run("oridjidji.api.app:create_app", factory=True, host=host, port=port, reload=True)
    else:
        uvicorn.run(create_app(warmup=warmup), host=host, port=port)


if __name__ == "__main__":
    app()
