# LICENSES.md — obligations tracking

This archive contains application code, interface assets and tests under the project's
[MIT licence](../LICENSE), together with the fine-tuned Basic model and the calibrations of both
modes (`models/`). It contains no research corpus, generated text or Gemma weights; those are
not redistributed. The lists below record provenance and separate conditions for materials used
in the research and optional runtime; they do not grant redistribution rights to materials
absent here.

## Corpora

| Source | Licence | Role and obligations |
|---|---|---|
| **MaCoCu-Genre.hr (11356/1969)** | **CC0** | **PRIMARY human source, news domain**. No obligations; attribution given as good practice |
| **ZNANJE / dabar_hr** (`procesaur/znanje`) | **CC-BY-SA-4.0** | **Academic domain, admitted** after Dabar OAI-PMH supplied publication years and post-2021/undated records were excluded. Attribution + **share-alike** |
| Dabar OAI-PMH metadata (dabar.srce.hr) | public repository metadata | publication years only; cite Dabar as the metadata source |
| hrWaC 2.1 (11356/1064) | CC-BY-SA-4.0 | **Not used for the dataset.** Used only as language-model calibration text in development |
| MaCoCu-hr 2.0 (11356/1806) | CC0 | crawl-date/quality evidence probe only |
| CLASSLA-web.hr 1.0 / 2.0 | CC0 | not used (vertical format; 2.0 is a post-ChatGPT crawl) |

The dataset is **not** distributed with this code. A future release of corpus records needs
its own provenance, attribution and redistribution review, especially for ZNANJE and
generator outputs. The MIT licence on this repository does not apply to those records.

The private development repository has **11,226 records** and preserves the licence and
source fields of each human record and human twin. Its AI records also retain
generator/model provenance and the applicable model or provider terms. It is an internal,
versioned research artefact and is absent from this archive.

## Models

| Model | Licence | Note |
|---|---|---|
| classla/bcms-bertic | Apache-2.0 | base of the Basic encoder; the fine-tuned ONNX model in `models/basic` is derived from it |
| classla/xlm-r-bertic | **CC-BY-SA-4.0** | encoder compared in development; a released fine-tune of it must be share-alike |
| Qwen/Qwen3-1.7B(-Base) | Apache-2.0 | compared as an alternative Q4 backbone; retained as research evidence and a faster non-selected alternative |
| google/gemma-2-2b(-it) | Gemma Terms of Use | Advanced observer/performer pair (Q4_K_M). **Not redistributed**: `install.ps1` downloads the weights from Hugging Face with the user's token and quantizes them locally; the user must accept the Gemma terms |
| microsoft/mdeberta-v3-base · FacebookAI/xlm-roberta-base | MIT | |
| permitt/galton-modernbertic-base · BSC-LT/mRoBERTa | Apache-2.0 | |
| EuroLLM-1.7B(-Instruct) · EuroMoE-2.6B | Apache-2.0 | |
| BSC-LT/salamandra-2b(-instruct) | Apache-2.0 | hrWaC in pretraining |
| mistralai/Mistral-7B-Instruct-v0.2 · RWKV/v5-Eagle-7B-HF | Apache-2.0 | |
| lmsys/vicuna-13b-v1.5 | **Llama 2 Community License** | check redistribution terms before publishing its generations in the dataset |
| API-served models | provider terms | verify each provider's output-redistribution terms before any dataset is published |

### Open-weight generators of the evaluation

The generation registry and generation outputs are not part of this archive. Naming a model
does not grant permission to redistribute its weights or outputs.

| Model | Licence | Obligation |
|---|---|---|
| `mistralai/Mistral-7B-Instruct-v0.2` | Apache-2.0 | retain model provenance with each generated record |
| `lmsys/vicuna-13b-v1.5` | Llama 2 Community License | review output redistribution before any dataset release |
| `utter-project/EuroLLM-1.7B-Instruct` | Apache-2.0 | retain model provenance with each generated record |
| `google/gemma-4-12B-it` | Apache-2.0 | retain model provenance; this archive contains no generated output |

Only EuroLLM explicitly declares Croatian in its model metadata. Mistral, Vicuna and
Gemma do not make that claim; their inclusion is based on inspection of their outputs, not on
an inferred language-support promise.

The final set also contains texts from three API-served models (Claude Haiku 4.5, Qwen3.8-27B
and GPT-OSS-120B). Those records are not included in this archive.

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
| llama.cpp (ggml-org) | MIT | cloned and built by `install.ps1` for Q4_K_M quantization and inference of the Advanced mode; not included in the archive |
| hunspell-hr dictionaries | GPL-2.0 / LGPL-2.1 / MPL-1.1 tri-licence | stated only in `README_hr_HR.txt` (repo has no LICENSE file) — cite the README; fetched as runtime data (SHA-256 printed on download), never committed |
| CLASSLA-Stanza (pip `classla`, optional extra) | Apache-2.0 | optional morphology extra |
| UTS #39 confusables.txt (Unicode 17.0.0) | Unicode License | derived map committed at `preprocessing/data/confusables_map.json` with provenance header |
| spylls | MIT | pure-Python hunspell reader |
| lingua-language-detector | Apache-2.0 | language ID, offline |

## Web fonts and visual assets

| Artefact | Licence / provenance | Note |
|---|---|---|
| Archivo WOFF2, Latin and Latin Extended | SIL Open Font License 1.1 | served locally; OFL text is stored beside the font files |
| IBM Plex Sans WOFF2, Latin and Latin Extended | SIL Open Font License 1.1 | served locally; OFL text is stored beside the font files |
| IBM Plex Mono WOFF2, Latin and Latin Extended | SIL Open Font License 1.1 | served locally; OFL text is stored beside the font files |
| Oriđiđi logo animation and derived mark | thesis author asset | the original GIF is not included; the web app serves derived WebP/SVG variants |

The font files come from the official Google Fonts distribution (`fonts.gstatic.com`), and
their licence texts come from the corresponding directories in the official `google/fonts`
repository. The subsets include the Croatian glyphs č, ć, đ, š and ž.

## This repository

The application code is licensed under **MIT**; see [LICENSE](../LICENSE). The project
licence does not replace licences of fonts, third-party libraries, model weights
or research data.
