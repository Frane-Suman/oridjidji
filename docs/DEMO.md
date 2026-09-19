# DEMO.md — running the thing

What this demonstrates: a localhost web application that estimates the probability a Croatian
text was machine-generated, shows how uncertain it is, and abstains when it cannot tell. It never
says "AI: yes".

Set up the environment first with [SETUP.md](SETUP.md); this document assumes `uv sync` has run
and the fast suite is green.

## 0. The one thing that is not in the repository

Weights are never committed (`CLAUDE.md`, *Never commit*). The demo therefore needs a **candidate
package**, and that is the only step that can stop you:

| | size | needed for |
|---|---:|---|
| `basic/` — BERTić FP32 ONNX + runtime profile | **424 MB** | Basic mode, the shipping product |
| `statistical/` — Gemma-2-2B Q4_K_M observer + performer + tokenizer | ~3.3 GB | Advanced mode (`research_only`) |
| **whole package** | **3.7 GB** | both modes |

The installer places it at `./models`.
On any other machine, point `ORIDJIDJI_CANDIDATE_ROOT` at wherever you put it; the
`e12-advanced-rc1` profile resolves every artefact relative to that root, which is what makes the
package portable.

**On the 15-minute Definition of Done:** it holds for Basic once the 424 MB `basic/` tree is
present — `uv sync` plus the checks below. It does **not** hold from a cold start on a metered
connection, because 3.7 GB has to arrive first. Copy the package from external media or Drive
rather than re-downloading it, and the criterion is comfortable; treat "fresh clone" as "fresh
clone plus the candidate package at hand".

## 1. Verify the demo without a browser

```powershell
$env:ORIDJIDJI_CANDIDATE_ROOT = '.\models'
uv run oridjidji demo-check --profile e12-advanced-rc1
```

This builds the real ONNX sessions and loads the real GGUF pair, then asserts that both mode
contracts exist and answer. Expect this on a working install:

```json
{
  "profile": "e12-advanced-rc1",
  "ok": true,
  "health_status": "research",
  "modes": ["osnovni", "napredni"],
  "capabilities_by_mode": {
    "osnovni":  {"document_detection": true, "statistical_evidence": false},
    "napredni": {"document_detection": true, "statistical_evidence": true}
  },
  "errors": []
}
```

**Measured: 111 s** on the target laptop (Intel 12 logical cores, CPU only). Most of that is
model loading, not inference. `"ok": true` with an empty `errors` list is the acceptance
condition; `health_status: research` is correct and expected — it is how the application
advertises that Advanced is `research_only`.

If `errors` is non-empty it names what is missing, which is nearly always a candidate path.

## 2. Run the application

```powershell
uv run oridjidji serve --warmup
```

Then open <http://127.0.0.1:8000>.

* `--warmup` builds the ONNX sessions before the first request. Without it the first user waits
  **27.9 s** and reads it as the tool being broken.
* It binds to **localhost deliberately**. Exposing it needs an explicit `--host 0.0.0.0`, because
  a detector reachable from the network is one somebody else can point at a student's essay
  (the specification, `ETHICS.md`).

## 3. What to show, and in what order

**Paste a Croatian text of at least 100 words.** Below that the application refuses rather than
guessing; 100 is the shipped minimum — the M9 length rule returned 50 and the product
ships the stricter value, which is worth mentioning if someone asks.

1. **The ethics banner.** Permanent UI furniture, not dismissible. Point at it first.
2. **Basic mode** (`osnovni`) — the shipping product. A calibrated probability, an uncertainty
   interval, and one of `LIKELY_HUMAN` / `INCONCLUSIVE` / `LIKELY_AI`. **Measured 0.413 s** per
   analysis on laptop CPU.
3. **Advanced mode** (`napredni`) — `research_only`, with a permanent warning of its own. Adds the
   statistical agents (Binoculars, Fast-DetectGPT). **Measured 28.17 s** on the same machine,
   which is exactly why it is not the shipping path.
4. **An inconclusive result.** Try a short-ish or heavily edited text. Abstention is the feature:
   the inconclusive zone is a split-conformal prediction set, not a hard-coded probability band
. M9 measured singleton accuracy at 0.95–0.99 when it does commit.
5. **History** — opt-in local storage, with CSV export.

A good demo ends on the abstention, not on a confident hit. The claim being defended is *"it
knows when it doesn't know"*, and that is the claim the evaluation actually supports.

## 3a. Exact texts to paste (the specification requires these, including a false positive)

**The ordinary case — a checked-in Croatian fixture, no sealed data:**

```
tests/fixtures/perf_sample_hr.txt
```

300+ words of human Croatian, already in the repository. Paste it whole. It is the same fixture
the performance benchmark times, so the 0.413 s Basic figure quoted above is this text.

**The honest case — a real false positive.** The sealed run flagged **32 of 2,000** held-out human
texts as `LIKELY_AI` in Basic mode. The most confident error is

```
znanje_dabar_hr:thesis_4849 p_ai = 0.9990
znanje_dabar_hr:thesis_69814 p_ai = 0.9990
znanje_dabar_hr:thesis_15104 p_ai = 0.9940
```

— genuine human Croatian theses the shipping detector is 99.4–99.9% certain are machine-written.
Retrieve the text by `record_id` from the frozen dataset
(`data/processed/m9/frozen/dataset_final.parquet`); it is deliberately **not** reproduced here,
because sealed evaluation text does not belong in a committed document.

Show one of these. A demo that only shows successes invites the question you cannot answer; a demo
that volunteers its worst error answers it first. The point to make is not that 3% is small, but
that this is *why* the tool reports a probability with an interval and must never be used against
an individual.

## 4. What not to claim while demonstrating

* It is **not** a plagiarism or misconduct tool. Never against an individual.
* One text is an anecdote. The evidence is `the project log`: 7,576
  predictions over 2,328 sealed records.
* **It is weakest where a demo is least likely to go.** At a 5% false-positive budget the
  shipping detector catches under half of academic Croatian from `gpt_oss_120b` (0.4912) and
  `qwen3` (0.4348). If someone wants to break it live, that is how — and saying so first is
  better than being shown.
* Academic human prose is where false positives concentrate: **3.0%** [2.0, 4.1] against 0.2%
  for news.

## 5. If it will not start

| symptom | cause |
|---|---|
| `demo-check` errors name a missing artefact | `ORIDJIDJI_CANDIDATE_ROOT` unset or pointing at the wrong tree |
| "not enough space on the disk" | the toolchain is still on C:; re-export the `UV_*`/`TEMP` variables from SETUP.md the specification |
| first request hangs ~28 s | `--warmup` was not passed |
| Advanced mode absent, Basic fine | `statistical/` missing from the candidate root; Basic alone is a valid demo |
| refuses your text | under 100 words, or not Croatian |

## 6. The no-weights fallback

If the candidate package is genuinely unavailable, the browser suite runs the real FastAPI/Jinja
application against **model doubles**:

```powershell
uv run pytest -q tests/browser
```

It exercises both mode choices, keyboard submission, opt-in storage, history and CSV, reduced
motion, 320-pixel reflow and the 200%-zoom layout, and it reported no serious or critical
WCAG 2 A/AA violations under offline axe-core 4.10.3. That demonstrates the application and its
accessibility — **not** detector accuracy. Say which one you are showing.
