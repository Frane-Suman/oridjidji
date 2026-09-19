# MODEL_CARD.md — Oriđiđi detector bundle

**Status: local thesis-demo candidates, not promoted.** The website exposes two equally
available document-analysis modes chosen by the user. Basic and Advanced have separate model,
calibration and conformal identities. The registered E11 result favoured A1 alone, but that
scientific comparison does not disable the Advanced implementation.

## User-selectable modes

- **Basic / A1:** `classla/bcms-bertic`, maximum E12 refit over the selected
  `academic50-legacy0` recipe. FP32 ONNX is the selected CPU export. Dynamic INT8 remains an
  optional compact artefact because it failed the registered accuracy gates.
- **Advanced / A1+A2+A3:** the same raw BERTić score is fused with Binoculars V0 over
  `prefix512` and Fast-DetectGPT curvature. A2 and A3 reuse one Gemma-2-2B Q4_K_M
  observer/performer pass. A4 supplies Croatian linguistic explanations and does not enter the
  probability.

Both modes return an AI-likelihood percentage, uncertainty interval and Mondrian conformal
result: `LIKELY_HUMAN`, `INCONCLUSIVE` or `LIKELY_AI`. Advanced remains available as a
`research_only` implementation because its probability layer was fitted on CUDA and did not
pass the CPU/CUDA probability-transfer gate. A statistical failure returns the independently
calibrated Basic result with an explicit degraded-mode warning.

## Intended use

Oriđiđi is a single-user localhost prototype for a master's-thesis demonstration. It estimates
signals associated with machine-generated Croatian academic and news text. It is not evidence
of authorship and must not be used for grading, discipline or decisions about a person.

The current 100-word gate is provisional. The final value will be derived from M9's registered
length-ablation curve. Non-Croatian text, very short fragments and strongly edited or adversarial
text remain outside the demonstrated operating conditions.

## Data and development evidence

The additive E12 repository contains **11,226 records** and preserves 2,046 training-pool pairs,
256 development-challenge pairs, explicit roles, lineage and exclusions. The Basic candidate's
export, calibration and staging evidence and the Advanced backbone, E9, E11, Q4 and CUDA-fusion
reports are under `the project log*`.

These are consumed-development findings. They establish implementation behaviour and motivate
the thesis discussion; they are not final detector-accuracy claims. M9 still has to measure
Basic and Advanced on untouched prompts, modern generators, sources, perturbations, lengths and
mixed authorship, with academic and news results reported separately.

## Calibration and abstention

Basic uses a Croatian-only probability layer fitted on `calibration_fit` and class-conditional
Mondrian quantiles fitted on the disjoint `calibration_conformal` role. Advanced has a separate
Q4-fusion probability and conformal identity. Fusion consumes the raw A1 score, never Basic's
displayed percentage.

The Advanced layer is CUDA-calibrated. CPU and CUDA A2/A3 ranks were close, but the selected
isotonic mapping amplified a tail difference and failed the registered probability-transfer
limit. The UI therefore exposes Advanced for the thesis demonstration with a permanent warning,
without presenting it as a CPU-validated probability model.

Standalone reporting-only calibration on 1,000 clean `calibration_fit` rows gives Binoculars
AUROC 0.9142 and TPR 0.376 at global 1% FPR, and Fast-DetectGPT AUROC 0.9288 and TPR 0.516.
These values explain individual agent evidence; they are not final metrics and do not replace
the raw A2/A3 inputs used by fusion. The complete per-domain table and identities are in
[the standalone threshold report](results/e12_advanced_statistical_thresholds.md).

## Known limitations

- Registered E11 selected A1. A1+A2+A3 is retained as the author's explicitly labelled
  multi-method implementation, not as evidence that fusion is more accurate.
- The statistical methods are generator- and domain-sensitive, especially at a strict 1% FPR.
- There is no segment-level output. M7's segment map was formally closed outside the active
  product; no segment fields or placeholder scores are returned, and the Advanced
  token trace is relative evidence, not a per-segment probability.
- Final unseen-generator, robustness, bias, length, mixed-authorship and Basic-versus-Advanced
  conclusions remain M9 work.
- Gemma's owned Q4 files are governed by the Gemma licence and must be packaged accordingly if
  the prototype is moved or redistributed.
