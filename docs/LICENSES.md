# LICENSES.md — obligations tracking

Per-record provenance lives in the dataset itself (`licence` field, the specification). This file
tracks the artefact-level obligations. Verified 2026-08-25 (an earlier decision…an earlier decision).

## Corpora

| Source | Licence | Role and obligations |
|---|---|---|
| **MaCoCu-Genre.hr (11356/1969)** | **CC0** | **PRIMARY human source, news domain**. No obligations; attribution given as good practice |
| **ZNANJE / dabar_hr** (`procesaur/znanje`) | **CC-BY-SA-4.0** | **Academic domain, admitted** after Dabar OAI-PMH supplied publication years and post-2021/undated records were excluded. Attribution + **share-alike** |
| Dabar OAI-PMH metadata (dabar.srce.hr) | public repository metadata | publication years only; cite Dabar as the metadata source |
| hrWaC 2.1 (11356/1064) | CC-BY-SA-4.0 | **Not used for the dataset** — disqualified. E4 LM-calibration text only |
| MaCoCu-hr 2.0 (11356/1806) | CC0 | crawl-date/quality evidence probe only |
| CLASSLA-web.hr 1.0 / 2.0 | CC0 | not used (vertical format; 2.0 is a post-ChatGPT crawl) |

**Consequence for the released dataset.** The news half is CC0, but the
academic half inherits ZNANJE's **CC-BY-SA-4.0**, so the combined release is
**CC-BY-SA-4.0**. Two honest options, to be decided before publication and recorded
as a DECISIONS entry:
- release the whole dataset CC-BY-SA-4.0 (simplest, share-alike propagates), or
- release the **news half separately under CC0** and the academic half under
  CC-BY-SA-4.0, which maximises reuse of the larger, cleaner portion.

Note this is *better* than the pre-M2 position: under an earlier decision's hrWaC plan the entire
dataset was share-alike, because hrWaC is CC-BY-SA-4.0.

The current additive E12 repository has **11,226 records** and preserves the licence and
source fields of each human record and human twin. Its 2,302 supplemental AI records also
retain generator/model provenance and the applicable model or provider terms. The repository
is an internal, versioned research artefact; publishing it still requires the release-format
decision above and a separate redistribution check for Vicuna generations.

## Models

| Model | Licence | Note |
|---|---|---|
| classla/xlm-r-bertic | **CC-BY-SA-4.0** | historical E1 comparison winner; a released fine-tune of it must be share-alike |
| classla/bcms-bertic | Apache-2.0 | selected E12 supervised encoder candidate |
| Qwen/Qwen3-1.7B(-Base) | Apache-2.0 | completed owned-Q4 comparison pair; retained as research evidence and a faster non-selected alternative |
| google/gemma-2-2b(-it) | Gemma licence | selected owned-Q4 Advanced observer/performer pair; verify redistribution conditions whenever the local serving package is copied or released |
| microsoft/mdeberta-v3-base · FacebookAI/xlm-roberta-base | MIT | |
| permitt/galton-modernbertic-base · BSC-LT/mRoBERTa | Apache-2.0 | |
| EuroLLM-1.7B(-Instruct) · EuroMoE-2.6B | Apache-2.0 | |
| BSC-LT/salamandra-2b(-instruct) | Apache-2.0 | hrWaC in pretraining |
| mistralai/Mistral-7B-Instruct-v0.2 · RWKV/v5-Eagle-7B-HF | Apache-2.0 | |
| lmsys/vicuna-13b-v1.5 | **Llama 2 Community License** | check redistribution terms before publishing its generations in the dataset |
| Claude / Gemini / Groq-served models | provider ToS | holdout generations: verify each provider's output-redistribution terms in M3 before the dataset is published |

### Exact open-weight pins reserved for M9 `final_challenge`

These four repositories were read-only verified on 2026-09-13. The identities are frozen in
`configs/evaluation/m9_final_generators.yaml`; pinning a model does not grant permission to
redistribute its weights or outputs.

| Exact model identity | Licence recorded by the audited repository | M9 obligation |
|---|---|---|
| `mistralai/Mistral-7B-Instruct-v0.2@63a8b081895390a26e140280378bc85ec8bce07a` | Apache-2.0 | retain the exact model/revision provenance with each generated record |
| `lmsys/vicuna-13b-v1.5@c8327bf999adbd2efe2e75f6509fa01436100dc2` | Llama 2 Community License | review output redistribution before any final dataset release |
| `utter-project/EuroLLM-1.7B-Instruct@a25c7fa65fc2a644e6270b8940dbe295b51da681` | Apache-2.0 | retain the exact model/revision provenance with each generated record |
| `google/gemma-4-12B-it@707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` | Apache-2.0; Gemma Terms of Use also apply | review the additional terms before generation, copying weights or releasing outputs |

Only EuroLLM explicitly declares Croatian in the audited model metadata. Mistral, Vicuna and
Gemma do not make that metadata claim; their inclusion is based on project-measured E12 output,
not an inferred language-support promise. All four repositories were ungated when checked.

The `modern_final` section of the registry intentionally contains no API model IDs. Its status
is `verification_required`: the exact official catalogue identity, catalogue date,
request/response evidence and provider output terms must be frozen before any paid or free API
generation begins.

## Code we learned from (never copied)

| Repo | Licence | Our use |
|---|---|---|
| ahans30/Binoculars | BSD-3-Clause | reimplemented; cited |
| baoguangsheng/fast-detect-gpt | MIT | reimplemented; cited |
| **kinit-sk/mgt-detection-benchmark** | **GPL-3.0** | **methodology only — no code copied**; prompt template + decoding params are facts, not code |
| kinit-sk/IMGTB | MIT | cited as context; not adopted |

## Tools/data at runtime

| Artefact | Licence | Note |
|---|---|---|
| hunspell-hr dictionaries | GPL-2.0 / LGPL-2.1 / MPL-1.1 tri-licence | stated only in `README_hr_HR.txt` (repo has no LICENSE file) — cite the README; fetched as runtime data (an earlier decision, sha256 recorded), never committed |
| CLASSLA-Stanza (pip `classla` 2.2.1, optional extra) | Apache-2.0 | an earlier decision |
| UTS #39 confusables.txt (Unicode 17.0.0) | Unicode License | derived map committed at `preprocessing/data/confusables_map.json` with provenance header |
| spylls 0.1.7 | MIT | pure-Python hunspell reader |
| lingua-language-detector 2.1.1 | Apache-2.0 | language ID, offline |

## Web fonts and visual assets

| Artefact | Licence / provenance | Note |
|---|---|---|
| Archivo WOFF2, Latin and Latin Extended | SIL Open Font License 1.1 | served locally; OFL text is stored beside the font files |
| IBM Plex Sans WOFF2, Latin and Latin Extended | SIL Open Font License 1.1 | served locally; OFL text is stored beside the font files |
| IBM Plex Mono WOFF2, Latin and Latin Extended | SIL Open Font License 1.1 | served locally; OFL text is stored beside the font files |
| Oriđiđi logo animation and derived mark | thesis author asset | original 27.4 MB GIF remains an unserved, gitignored source; the web app serves derived WebP/SVG variants |

The font files come from the official Google Fonts distribution (`fonts.gstatic.com`), and
their licence texts come from the corresponding directories in the official `google/fonts`
repository. The subsets include the Croatian glyphs č, ć, đ, š and ž.

## This repository

Code licence: **author's choice, still open** — recommendation: MIT for code, with
the academic dataset component released under CC-BY-SA-4.0 (required by ZNANJE) and the
CC0 news component separable where practical. Decide before
the repo goes public; record as a DECISIONS entry.
