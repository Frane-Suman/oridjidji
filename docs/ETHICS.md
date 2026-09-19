# ETHICS.md

Grows each milestone; the bias-probe results (M9) land here. This file is the source
of the app's permanent ethics banner and limitations panel.

## What this tool is, and is not

Oriđiđi estimates a **probability** with an uncertainty interval and an explicit
`INCONCLUSIVE` outcome. It is a research prototype for a master's thesis. It is
**not** evidence of authorship, **not** a plagiarism detector, and **must never** be
used for grading, discipline, or any decision about a person. That position is not
modesty — it is what the measured error rates of this entire field require.

**Banner text (permanent UI furniture, not dismissible):**
> *Oriđiđi procjenjuje vjerojatnost, ne utvrđuje autorstvo. Rezultat nije dokaz i ne
> smije se koristiti za ocjenjivanje ni disciplinske postupke. Pouzdanost pada za
> kratke, jednostavne i uređivane tekstove.*

## Known failure modes of the field (citable, drive the design)

- **Bias against non-native / simple writing:** Liang et al. (2023) measured a
  **61.22% average false-positive rate** on 91 TOEFL essays by non-native writers
  across 7 commercial detectors. Croatian analogue: simple/short/L2-like human
  Croatian risks elevated FPR — the M9 **bias probe** measures exactly this and the
  result goes here and into the UI limitations text, whatever it shows.
- **Withdrawn tools:** OpenAI withdrew its own AI Text Classifier on 2023-07-20 at
  26% TPR / 9% FPR. Modesty about accuracy is the honest baseline.
- **Evasion:** recursive paraphrasing collapses detectors (Sadasivan et al. 2023:
  TPR@1%FPR 99.8% → 9.7%); expert human editing evades while LLM-editing stays
  detectable (Beemo). Post-edited text is this thesis's focus, so degradation is
  measured and reported, not hidden.

## Project-specific risks and how they are handled

- **Pretraining contamination asymmetry:** BERTić-family encoders and
  Salamandra-2b were pretrained on hrWaC — the source of our human class. A detector
  or statistical backbone may "recognise" human texts it has already seen, inflating
  results. Handling: non-hrWaC human test source (MaCoCu/CLASSLA-web) in every
  evaluation; per-model reporting; stated in the limitations chapter.
- **Self-detection leakage:** statistical detectors score their own model
  family's generations too favourably; results are split in-family/out-of-family.
- **hr/sr/bs separation** is genuinely hard; the language gate says so instead of
  feigning precision.
- **Token-trace over-reading:** highlighted words show relative influence inside one request;
  they are not per-word authorship probabilities. The deferred segment map was removed from
  the localhost application in an earlier decision rather than presented without matching reliability data.
- **Choice of analysis mode:** the user may run either Basic BERTić or the complete
  A1+A2+A3 Advanced implementation. Registered E11 evidence favoured A1, but that finding
  does not hide the Advanced thesis demonstration. Advanced is permanently labelled
  `research_only` while its displayed percentage remains CUDA-calibrated and unvalidated
  on the CPU score axis.
- **Screenshots and copied reports:** the probability rail has no fixed “safe” or “danger”
  band. It shows the interval more strongly than the point estimate and names the conformal
  outcome in text. The server-generated copied report repeats the selected and effective mode,
  estimate, interval, warnings and all three permanent ethics sentences, so uncertainty is not
  lost when the result leaves the page.
- **Research and fallback states:** `research_only`, `degraded_basic` and
  `advanced_unavailable` are visible result states rather than implementation details. A failed
  Advanced request never relabels Basic output as Advanced.

## Data ethics

- Sources pass the four the specification gates or are excluded. ZNANJE is now admitted because Dabar
  OAI-PMH supplied publication years; post-2021 and undated works are excluded. This reduces
  the contamination risk but does not prove that every retained document was written without
  earlier automated assistance. Its extraction and date provenance remain explicit because
  academic source and extraction pipeline are still partially confounded.
- Per-record provenance and licence tracked from day one; share-alike obligations in
  `docs/LICENSES.md`.
- The E12 repository is additive: all **11,226 records** retain lineage and exclusions, while
  only registered views may enter training. Its 2,046 pool pairs and 256 development pairs
  remain separate from calibration and sealed final-human roles.
- The app stores no submitted text without explicit opt-in.
- Machine-translated text is never used as AI-class training data (anti-pattern the specification:
  it teaches translationese-vs-native, and Weber-Wulff et al. report *higher* FPR on
  translated human text).

## Placeholders (filled by their milestone)

- M9 bias-probe FPR table on simple/short/L2-like human Croatian: **data source
  exists** (`riznica.parquet`) and the experiment is registered as
  a probe whose decision rule was fixed in advance. The table itself is
  still pending the sealed post-E12 run. The E12 encoder candidate exists, but that does not
  open the final collection. Riznica's
  `Zakoni` subcorpus is in it deliberately:
  legal Croatian is the most formulaic human prose the project can obtain, and formulaic
  human writing is exactly what Liang et al. found being misread as machine-written.
- M9 robustness bad-news summary (paraphrase, post-edit, homoglyph): **pending**.
- Final M9 limitations text mirrored into the UI: **pending**. The current permanent
  authorship warning and Advanced research notice are implemented.
