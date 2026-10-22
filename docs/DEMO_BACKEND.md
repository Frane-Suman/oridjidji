# Local backend checks

The archive can run interface and unit tests without models. The real `demo-check`
needs the `models/` directory that `install.ps1` downloads (or that is restored by hand).

## Model-free checks

```bash
uv sync --locked
uv run pytest -q -m "not slow"
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

Browser tests use model doubles; they do not confirm measured detection accuracy.

## Real models

Run the platform's install script first (`install.ps1` or `install.sh`). It compares
the files in `models/` with the SHA-256 digests in `release_manifest.json` and, on
Windows, can prepare Advanced mode (see [DEMO.md](DEMO.md)). `start.ps1` and
`start.sh` set `ORIDJIDJI_SKIP_BUNDLE_VERIFY=1` because the Advanced files are built
locally; set it as well when running the commands below by hand. Then run:

```bash
# Basic only (use --profile demo once Advanced has been installed):
uv run oridjidji demo-check --profile basic --basic-only --candidate-root models
uv run oridjidji serve --profile basic --candidate-root models --warmup
```

Open `http://127.0.0.1:8000` on the same machine. Basic should be available.
Advanced is available only after its own artefacts have been prepared, and it is
calibrated and evaluated on a GPU only; do not attribute the same reliability to its
CPU results. Expect about 0.12 s per Basic analysis, about 28 s per Advanced analysis,
and about 5.0 GB of memory (up to 8.1 GB after the first Advanced request). If Advanced
fails during an analysis, check that the response names the effective Basic fallback
rather than silently presenting it as Advanced.

Check a Croatian input of at least 100 words, a refused short input, the abstention
outcome (`INCONCLUSIVE`, always from an empty conformal set) and history with and
without text-retention selection.
These checks exercise serving behavior; the thesis reports the separate evaluation of
detector accuracy on the final set.
