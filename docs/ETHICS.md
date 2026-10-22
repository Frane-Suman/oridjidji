# ETHICS.md

This document records the intended use and measured limitations of the thesis prototype.
The permanent warning and limitations panel are implemented in the application.

## What this tool is, and is not

Oriđiđi displays a calibrated **estimate**, a conformal prediction set and a label
(human, machine or abstain). It is a research prototype for a master's thesis. It is
**not** evidence of authorship, **not** a plagiarism detector, and **must never** be
used for grading, discipline, or any decision about a person. That position is not
supported by its measured error rates. Abstention always results from an empty
prediction set; the application produces no two-label sets.

**Warning text (permanent UI furniture, not dismissible):**
> *Rezultat je procjena vjerojatnosti, a ne dokaz. Ne smije se koristiti za ocjenjivanje,
> stegovne postupke ni bilo kakvu odluku o pojedincu.*

## Known failure modes of the field (citable, drive the design)

- **Bias against non-native writing in prior work:** Liang et al. (2023) reported high
  false-positive rates for non-native English essays across seven commercial detectors.
  This project did not complete a corresponding per-group Croatian bias evaluation; it must
  not transfer that numerical result to Oriđiđi.
- **Withdrawn tools:** OpenAI withdrew its own AI Text Classifier on 2023-07-20 at
  26% TPR / 9% FPR. Modesty about accuracy is the honest baseline.
- **Evasion and editing:** Sadasivan et al. (2023) showed that repeated paraphrasing
  can sharply reduce one watermark detector's sensitivity. This project's final evaluation
  tested selected perturbations, but did not establish a general guarantee for
  paraphrased, translated or mixed-authorship text.

## Project-specific risks and how they are handled

- **Source and pretraining overlap:** The final human examples are new records from
  MaCoCu-Genre.hr and ZNANJE, sources also represented in development. Riznica is not
  wholly unseen by BERTić pretraining. These facts limit claims of source transfer.
- **Generator sensitivity:** Results are split by cohort and generator. A measured drop
  for Advanced on Gemma-4 academic texts does not by itself establish its cause.
- **hr/sr/bs separation** is genuinely hard; the language gate says so instead of
  feigning precision.
- **Analysed part of the text:** the Basic encoder reads only the first 160 tokens, and the
  Advanced statistical detectors read up to the first 512 tokens. The interface states
  this scope before and with every result; text added after the analysed part does not change
  the estimate.
- **Token-trace over-reading:** highlighted words show relative influence inside one request;
  they are not per-word authorship probabilities. The application returns no segment-level
  output, because no matching reliability data exist for one.
- **Choice of analysis mode:** the user may run either Basic BERTić or the Advanced mode
  (BERTić fused with Binoculars and Fast-DetectGPT). Advanced was calibrated and evaluated
  only on a GPU; transferring its calibration to the CPU failed (largest probability
  difference 0.3421), so its CPU results must not be credited with the same reliability.
  Advanced is permanently labelled `research_only`. Basic was confirmed on the CPU as well.
- **Screenshots and copied reports:** the probability rail has no fixed “safe” or “danger”
  band. Reports show the selected and effective mode, estimate, warnings, the conformal
  prediction set and the label.
- **Inconclusive outcome:** when the prediction set is empty, the interface says: “Procjena ne
  podupire nijednu oznaku: analizirani dio teksta ne nalikuje dovoljno ni ljudskim ni
  strojnim tekstovima na kojima je sustav kalibriran.” This is a statement about the text
  being outside the calibration range, not a finding about authorship.
- **Research and fallback states:** `research_only`, `degraded_basic` and
  `advanced_unavailable` are visible result states rather than implementation details. A failed
  Advanced request never relabels Basic output as Advanced.

## Data ethics

- ZNANJE records were admitted after Dabar
  OAI-PMH supplied publication years; post-2021 and undated works are excluded. This reduces
  the contamination risk but does not prove that every retained document was written without
  earlier automated assistance. Its extraction and date provenance remain explicit because
  academic source and extraction pipeline are still partially confounded.
- Per-record provenance and licence were recorded in the private research dataset;
  no corpus records are included in this archive.
- The development repository holds **11,226 records** that retain lineage and exclusions,
  while only registered views may enter training. Training, calibration, conformal fitting and
  the final set use separate records.
- The local history stores results and technical metadata. It stores the full input and
  token text only when the user chooses text retention for that analysis; the response
  shown immediately on screen can still contain token snippets.
- Machine-translated text was not used as the AI training class. The final evaluation
  did not establish performance on translated text.

## Observed false positives

On the final set (2,328 records), at the fixed decision thresholds, Basic marked 30 of
1,000 human academic records and 2 of 1,000 human news records as machine-written.
Advanced, evaluated on a GPU, marked 23 and 3 respectively.
The two domains use different source collections, so the difference cannot be assigned
to genre alone. There was no completed per-group Croatian bias study.
