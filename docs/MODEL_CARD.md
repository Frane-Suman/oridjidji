# MODEL_CARD.md — Oriđiđi detector bundle

**Status: thesis prototype.** The website exposes two document-analysis modes chosen by the
user. Basic and Advanced have separate model, calibration and conformal identities. The archive
does not contain weights; the Basic model and the calibrations of both modes (`models/`, about
423 MiB) are downloaded by `install.ps1`. The
Advanced mode additionally needs Gemma-2-2B files, which are not redistributed: `install.ps1`
(step 6) downloads them from Hugging Face with the user's token, builds llama.cpp and quantizes
the model to Q4_K_M.

## User-selectable modes

- **Basic:** `classla/bcms-bertic`, fine-tuned on 896 pairs (1,792 records). The served model is
  the FP32 ONNX export; the encoder reads the **first 160 tokens** of the text. Dynamic INT8 is
  not used: it was about 15% faster, but sensitivity at 1% FPR fell by more than 16 percentage
  points.
- **Advanced:** the raw BERTić score is fused by logistic regression with Binoculars and
  Fast-DetectGPT, both computed from one Gemma-2-2B (Q4_K_M) observer/performer pass. The
  statistical detectors read up to 512 tokens. A separate linguistic component supplies Croatian
  explanations and does not enter the probability.

Both modes return an isotonically calibrated estimate, a Mondrian conformal prediction
set and a label: human, machine or abstain. Abstention
always results from an empty prediction set; two-label sets do not occur.

| Conformal setting, α = 0.10 | q0 | q1 | Decision threshold |
|---|---:|---:|---:|
| Basic | 0.0800 | 0.0833 | 0.889 |
| Advanced | 0.0411 | 0.0536 | 0.932 |

q0 and q1 are the class-conditional quantiles of the nonconformity scores for the human and
machine class, fitted on 400 calibration records per class. The thresholds were set on human
calibration records for a 1% false-positive target.

**Advanced was calibrated and evaluated on a GPU only.** Transferring its calibration to the
CPU failed (largest probability difference 0.3421), so Advanced results on a CPU must not be
credited with the same reliability. Basic was confirmed on the CPU: the same ONNX model and
calibrator rescored all 2,328 final-set records with a largest probability difference of
3·10⁻⁵. Advanced is therefore permanently labelled `research_only`. If its statistical part
fails, the independently calibrated Basic result is returned with a degraded-mode warning.

## Intended use

Oriđiđi is a single-user localhost prototype for a master's-thesis demonstration. It estimates
signals associated with machine-generated Croatian academic and news text. It is not evidence
of authorship and must not be used for grading, discipline or decisions about a person.

The result describes only the analysed beginning of the text, and the interface states its
extent. The default configuration requires at least 100 words. The length study's registered
rule returned 50 words (the setting used in the final evaluation), but sensitivity at a strict
false-positive threshold was weaker there. Non-Croatian text, very short fragments and
strongly edited or adversarial text remain outside the demonstrated operating conditions.

## Data and evaluation evidence

The development repository contains **11,226 records** in different roles. Separate records
served calibration and conformal fitting. The final set has **2,328 records**; the aggregate per
domain consists of 836 human and 164 machine records. The final set was not used for any model,
threshold or calibration decision. This distribution intentionally omits training scripts,
source corpora and the detailed evaluation outputs; the thesis contains the methods and results.

| Basic, final-set aggregate | Academic | News |
|---|---:|---:|
| AUROC | 0.9267 | 0.9579 |
| ECE | 0.0575 | 0.0213 |
| Human false-positive rate at the fixed threshold | 30/1,000 | 2/1,000 |

On the modern-generator academic cohort Basic AUROC was 0.8759 and TPR at 1% FPR was 0.1900.
These cohort results limit how the aggregate values should be used. Advanced results in the
thesis refer to the GPU execution.

On the test laptop's CPU, a Basic analysis took a median of about 0.12 s (P90 0.129 s) and an
Advanced analysis a median of about 27.8 s (P90 30.0 s). Process memory was about 5.0 GB before
the first Advanced request and up to 8.1 GB after it.

## Calibration and abstention

Basic uses a Croatian-only isotonic probability layer and class-conditional Mondrian quantiles
fitted on a disjoint calibration role. Advanced has a separate isotonic calibrator and conformal
quantiles over the fused score. Fusion consumes the raw BERTić score, never Basic's displayed percentage.

The separate Binoculars and Fast-DetectGPT development results motivated their use as
evidence; they must not be read as final performance estimates for the complete Advanced mode.

## Known limitations

- On the final set, a baseline of logistic regression over TF-IDF features of the whole text
  reached higher discrimination than the encoder, which reads only the beginning. The added
  value of the more complex model is not confirmed, and fusion is not shown to be more accurate.
- The statistical methods are generator- and domain-sensitive, especially at a strict 1% FPR.
- There is no segment-level output; no segment fields or placeholder scores are returned, and
  the Advanced token trace is relative evidence, not a per-segment probability.
- The final evaluation compared modes, generators, lengths and selected perturbations. It did
  not run mixed-authorship, per-group bias or external-detector comparisons.
- Gemma files are governed by the Gemma Terms of Use. They are not redistributed in this
  archive and must be obtained under those terms.
