# Local demonstration

This is a guide to running the application from the archive. The archive does not include
model weights: `install.ps1` downloads the Basic model and calibrations (25 files, about
423 MiB) into `models/` and verifies their SHA-256 digests, so Basic mode works after
installation. Gemma weights are not included and are not redistributed
(Gemma Terms of Use); Advanced mode is optional and is prepared by `install.ps1` on
Windows.

## Install and start (Windows)

Python 3.11 and `uv` are required; `install.ps1` installs `uv` through winget if it is
missing.

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
powershell -ExecutionPolicy Bypass -File start.ps1
```

`install.ps1` has six steps: check `uv`, install the Python environment, create
directories, fetch the Croatian dictionary, verify the SHA-256 digests of the files in
`models/` against `release_manifest.json`, and an optional step 6 for Advanced mode.
Step 6 needs a Hugging Face token, downloads `google/gemma-2-2b` and
`google/gemma-2-2b-it`, builds llama.cpp (tag b10516) and quantizes the models to
Q4_K_M. Before downloading it checks for git, CMake, Visual Studio Build Tools with C++
tools and at least 14 GB of free disk space. Parameters: `-PreskociNapredni` skips step
6 without asking (Basic only), and `-HfToken <token>` (or the `HF_TOKEN` variable)
supplies the token. The script can be run repeatedly.

`start.ps1` starts the server, waits for the models to load and opens
`http://127.0.0.1:8000`. Loading takes from several seconds to about 45 s, depending on the machine and on
whether Advanced is installed. On Linux/macOS use `bash install.sh` and `./start.sh`; these prepare Basic
mode only. Keep the server bound to `127.0.0.1`. If the Gemma pair is not installed,
Advanced must appear unavailable.

## Without models

A clean checkout can run the tests and the browser suite with model doubles:

```bash
uv sync --locked
uv run pytest -q -m "not slow"
```

The browser tests use the real FastAPI/Jinja interface with model doubles. Their
success checks presentation and application behavior, not detector accuracy.

## What to explain in a demonstration

Paste a Croatian text of at least 100 words. Basic uses the ONNX BERTić encoder on the
first 160 tokens and takes a fraction of a second (median about 0.12 s on the test
laptop). Advanced adds Binoculars and Fast-DetectGPT over a shared Gemma-2-2B pair and
takes about 28 s per analysis on a CPU. Advanced was calibrated and evaluated only on a
GPU and its calibration did not transfer to the CPU, so its CPU results must not be
presented as equally reliable.

Explain the calibrated estimate, the conformal set and the label separately. Abstention always comes from an empty set;
the interface then states that the estimate supports neither label, because the
analyzed part of the text resembles neither the human nor the machine texts the system
was calibrated on. A token trace is relative evidence inside that request, not a map of
authorship. The history stores input text only after an explicit selection for that
analysis.

On the final set (2,328 records), Basic incorrectly marked 30/1,000 human academic and
2/1,000 human news records as machine-written. On modern academic machine text its TPR
was 0.19 at 1% FPR. A single live example cannot replace those cohort results.

See [MODEL_CARD.md](MODEL_CARD.md), [ETHICS.md](ETHICS.md) and
[DEMO_BACKEND.md](DEMO_BACKEND.md) for mode status and backend checks.
