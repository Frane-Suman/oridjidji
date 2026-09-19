# Local backend demonstration runbook

This runbook starts the single-user thesis prototype on `localhost`. It does not publish the
service, open M9 data, retrain a model or promote an artefact.

## 1. What the two modes execute

| Website choice | Executed detector | Probability layer | Current status |
|---|---|---|---|
| **Osnovni / Basic** | E12 BERTić A1, FP32 ONNX | Basic Croatian calibration + Mondrian conformal | unpromoted candidate |
| **Napredni / Advanced** | raw A1 + Gemma-Q4 Binoculars A2 + Fast-DetectGPT A3; A4 explanations | separate A1+A2+A3 CUDA calibration + Mondrian conformal | `research_only` |

Both choices are functional and user-selectable. E11 selected A1 scientifically, but that result
does not disable Advanced. Advanced's permanent warning records that its probability was fitted
on CUDA and did not pass the CPU probability-transfer gate. If A2/A3 fails, the response is the
independently calibrated Basic result with `runtime_status=degraded_basic`.

The localhost demo intentionally has no segment-map phase. Advanced document analysis still
runs, and its token-trace panel presents relative request evidence without assigning authorship
probabilities to individual words.

## 2. Prepare the environment

From the repository root:

```powershell
uv sync --extra serve
uv run oridjidji info --profile e12-advanced-research
uv run oridjidji demo-check --profile e12-advanced-research
```

The research profile currently references BERTić artefacts on D: and the temporary owned Gemma
GGUF pair on C:. `demo-check` verifies configuration and manifest identities without requiring a
multi-document CPU scoring run.

For one real request in both modes, prepare a UTF-8 Croatian document of at least 100 words and
run:

```powershell
uv run oridjidji demo-check --profile e12-advanced-research --smoke demo.txt
```

This is a single integration smoke. Advanced can take materially longer than Basic because it
executes two 2B causal models on CPU.

To repeat the same acceptance through the installed Edge browser and save a machine-readable
report, run:

```powershell
uv run python scripts/smoke_frontend_real.py
```

The script uses the repository's fixed 120-word Croatian smoke document, a temporary history
database and `store=false`. It verifies that requested and effective modes agree and writes
`the project log`.

## 3. Start the localhost application

```powershell
uv run oridjidji serve --profile e12-advanced-research --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`. Keep the bind address on localhost. Docker and Compose are optional
portability scaffolding and are not required for the thesis demonstration.

## 4. Manual acceptance sequence

1. Confirm `/api/v1/health` reports Basic plus a research Advanced runtime and its capabilities.
2. Paste the same Croatian document, choose **Osnovni**, and run analysis.
3. Choose **Napredni** and rerun it. Confirm A1, A2 and A3 evidence is present and A4 appears only
   as explanation evidence.
4. Confirm both results contain a percentage, uncertainty interval and one of
   `LIKELY_HUMAN`, `INCONCLUSIVE` or `LIKELY_AI`.
5. Confirm Advanced shows its CUDA-calibration warning and the token-trace explanation.
6. Confirm history records both runs and does not retain submitted text unless `store=true`.
7. Exercise `/api/v1/analyze/stream` and confirm its final result matches the synchronous route.
8. Simulate statistical failure with the test profile and confirm the response is marked
   `degraded_basic`, rather than presenting Basic as a successful Advanced prediction.

## 5. Interpretation boundary

This sequence verifies the backend and the visible thesis implementation. It does not measure
final accuracy. Development reports remain evidence about consumed data. Untouched modern
Croatian AI cohorts, final humans, robustness, bias, length and Basic-versus-Advanced conclusions
remain the later M9 scientific run.
