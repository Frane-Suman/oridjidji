"""Load and verify long-lived detector artifacts for the localhost API.

COMPLEXITY BUDGET (AGENTS.md, roughly 340 lines). The loaders are kept together because
they enforce one trust boundary: Basic, the shared causal pair, fusion, calibration and
segment files must resolve to compatible identities before any request is served. Each
function still performs one small, independently tested loading decision.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from oridjidji.config import AppConfig
from oridjidji.detectors.encoder import MAX_TOKENS, EncoderDetector, find_exported_model


def _statistical_contract(
    cfg: AppConfig,
) -> tuple[Literal["V0", "V4"], Literal["prefix512", "coverage1024"]]:
    """Resolve the one registered statistical feature the served fusion expects."""
    aggregation = str(cfg.inference.get("statistical_aggregation", "V0"))
    window_policy = str(cfg.inference.get("statistical_window_policy", "prefix512"))
    candidate_path = cfg.inference.get("advanced_candidate_manifest")
    if candidate_path:
        from oridjidji.fusion.advanced_bundle import AdvancedCandidateManifest

        candidate = AdvancedCandidateManifest.load(Path(str(candidate_path)))
        if "statistical_aggregation" in cfg.inference and aggregation != candidate.aggregation:
            raise ValueError("Configured statistical aggregation disagrees with Advanced candidate")
        if (
            "statistical_window_policy" in cfg.inference
            and window_policy != candidate.window_policy
        ):
            raise ValueError("Configured statistical window disagrees with Advanced candidate")
        aggregation = candidate.aggregation
        window_policy = candidate.window_policy
    if aggregation not in {"V0", "V4"}:
        raise ValueError(f"Unknown statistical aggregation: {aggregation}")
    expected_policy = "prefix512" if aggregation == "V0" else "coverage1024"
    if window_policy != expected_policy:
        raise ValueError(
            f"Statistical {aggregation} requires {expected_policy}, got {window_policy}"
        )
    return cast(Literal["V0", "V4"], aggregation), cast(
        Literal["prefix512", "coverage1024"], window_policy
    )


def resolve_n_gpu_layers(cfg: AppConfig) -> int:
    """llama.cpp layers to offload: none unless the profile or the environment asks.

    The laptop stays CPU-only by construction. The final-set scoring run happens on a
    Colab GPU, where the two Gemma Q4_K_M models are the whole cost, so that run sets
    `ORIDJIDJI_N_GPU_LAYERS=-1`. Reading it here keeps the profile file untouched.
    """
    import os

    configured = cfg.inference.get("n_gpu_layers")
    if configured is not None:
        return int(configured)
    return int(os.environ.get("ORIDJIDJI_N_GPU_LAYERS", "0"))


def _load_gguf_backbone(
    cfg: AppConfig, window_policy: Literal["prefix512", "coverage1024"]
) -> Any | None:
    """Build the lazy owned-GGUF backend, or report an incomplete candidate."""
    manifest_path = Path(str(cfg.inference.get("statistical_pair_manifest", "")))
    tokenizer_path = Path(str(cfg.inference.get("statistical_tokenizer", "")))
    if not manifest_path.is_file() or not tokenizer_path.exists():
        warnings.warn(
            "GGUF Advanced backend is selected but its pair manifest or tokenizer is missing",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    import json

    from transformers import AutoTokenizer

    from oridjidji.detectors.llama_cpp_backbone import GgufPair, LlamaCppBackbone

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    pair = GgufPair(
        name=str(payload["name"]),
        observer_path=root / str(payload["observer"]),
        performer_path=root / str(payload["performer"]),
        observer_revision=str(payload["observer_revision"]),
        performer_revision=str(payload["performer_revision"]),
        tokenizer_hash=str(payload["tokenizer_mapping_sha256"]),
        llama_cpp_revision=str(payload["llama_cpp_revision"]),
        observer_sha256=str(payload["observer_sha256"]),
        performer_sha256=str(payload["performer_sha256"]),
        add_bos=bool(payload["add_bos"]),
        quantization=str(payload["quantization"]),
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=True)
    return LlamaCppBackbone(
        pair,
        tokenizer,
        threads=int(cfg.inference.get("intra_op_threads", 8)),
        window_policy=window_policy,
        n_gpu_layers=resolve_n_gpu_layers(cfg),
    )


def load_statistical_detectors(cfg: AppConfig) -> list[Any]:
    """Binoculars and Fast-DetectGPT over ONE shared backbone, when it is available.

    Absent by default. The pair is ~14 GB on disk and E4 has not yet chosen which one
    ships, so `inference.backbone` is unset until it does — and an unset backbone means an
    empty list, not an error. The Basic mode is the encoder alone; these are the Advanced
    mode's agents.

    Both agents share one `HuggingFaceBackbone` by construction. Two detectors each loading
    their own copy of a 1.7B pair is what the specification exists to prevent on a 16 GB laptop.
    """
    name = cfg.inference.get("backbone")
    if not name:
        return []
    from oridjidji.detectors.statistical import for_pair

    aggregation, window_policy = _statistical_contract(cfg)

    if cfg.inference.get("causal_backend") == "gguf-q4":
        gguf = _load_gguf_backbone(cfg, window_policy)
        return [] if gguf is None else list(for_pair(gguf, binoculars_variant=aggregation))

    from oridjidji.artefacts import E4_BACKBONE_PAIRS
    from oridjidji.detectors.hf_backbone import BackbonePair, HuggingFaceBackbone

    candidate = next((c for c in E4_BACKBONE_PAIRS if c.name == name), None)
    if candidate is None:
        # A misconfigured name is a startup-time fact, not a per-request failure — but it
        # must not stop the API serving the ethics furniture and the encoder.
        #
        # It must also not look like the unset case. Both return an empty list, and without
        # this warning a typo in `inference.backbone` is indistinguishable from "E4 has not
        # chosen a pair yet": Advanced mode is silently missing and nothing says why. This
        # project has now written the same swallowed-degrade bug three times,
        # and each one was silent rather than wrong.
        warnings.warn(
            f"inference.backbone={name!r} matches no pair in E4_BACKBONE_PAIRS "
            f"({', '.join(c.name for c in E4_BACKBONE_PAIRS)}) — serving the encoder alone",
            RuntimeWarning,
            stacklevel=2,
        )
        return []
    observer, performer = candidate.resolve()
    threads = cfg.inference.get("intra_op_threads")
    backbone = HuggingFaceBackbone(
        BackbonePair(
            candidate.name,
            observer,
            performer,
            observer_revision=(
                None if observer != candidate.observer_id else candidate.observer_revision
            ),
            performer_revision=(
                None if performer != candidate.performer_id else candidate.performer_revision
            ),
        ),
        max_tokens=int(cfg.inference.get("statistical_max_tokens", 512)),
        intra_op_threads=int(threads) if threads else None,
        window_policy=window_policy,
    )
    return list(for_pair(backbone, binoculars_variant=aggregation))


def load_detectors(cfg: AppConfig) -> list[Any]:
    """Every detector servable on this machine, in ring order.

    An empty list is a correct answer, not a failure: weights are not in git, so a fresh
    clone has none and the API still has to start, serve the ethics panel and refuse
    honestly. Discovery is a directory scan; the model itself loads on first use.
    """
    statistical = load_statistical_detectors(cfg)
    root = cfg.inference.get("model_dir")
    if not root:
        return statistical
    found = find_exported_model(root)
    if not found:
        return statistical
    # The window is a served parameter, not a constant: an earlier decision derived it from the length
    # curve, and a machine with a different latency budget should be able to change it
    # without touching code.
    window = int(cfg.inference.get("max_tokens", MAX_TOKENS))
    # Same reasoning for the thread count: measured on this machine by `train export`, not
    # a constant, because the right value depends on the CPU that serves.
    threads = cfg.inference.get("intra_op_threads")
    # Encoder first: it is the Basic mode's primary agent.
    return [
        EncoderDetector(
            *found,
            max_tokens=window,
            intra_op_threads=int(threads) if threads else None,
        ),
        *load_linguistic_detector(cfg),
        *statistical,
    ]


def hash_verification_disabled() -> bool:
    """True when the operator has accepted unverified artefacts.

    Set ORIDJIDJI_SKIP_BUNDLE_VERIFY=1 when the weights were built locally rather than
    unpacked from the release: locally quantised GGUF files cannot reproduce the hashes
    recorded when the originals were packaged, so the check would refuse every such install.
    """
    import os

    return os.environ.get("ORIDJIDJI_SKIP_BUNDLE_VERIFY", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def verify_bundle(cfg: AppConfig) -> None:
    """Refuse to start if the fitted artefacts were not fitted together.

    Serving a probability and an abstention zone that were measured on different data is the
    failure this prevents. Missing artefacts are not a failure - a fresh clone must still
    start, serve the ethics furniture and abstain honestly; mismatched ones are.
    """
    import warnings

    from oridjidji.fusion.bundle import BUNDLE_PATH, DetectorBundle

    if hash_verification_disabled():
        warnings.warn(
            "Artefact hash verification is disabled by ORIDJIDJI_SKIP_BUNDLE_VERIFY. "
            "The server will start even if the weights do not match the hashes recorded "
            "when they were packaged.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    path = Path(cfg.inference.get("bundle_path", BUNDLE_PATH))
    if path.is_file():
        DetectorBundle.load(path)  # raises BundleMismatch, which must reach the caller
    candidate = cfg.inference.get("encoder_candidate_manifest")
    if candidate:
        from oridjidji.artefacts import EncoderCandidateManifest

        EncoderCandidateManifest.load(Path(candidate))
    advanced = cfg.inference.get("advanced_candidate_manifest")
    if advanced:
        from oridjidji.fusion.advanced_bundle import AdvancedCandidateManifest

        AdvancedCandidateManifest.load(Path(advanced))


def load_linguistic_detector(cfg: AppConfig) -> list[Any]:
    """A4, when it has been fitted, for the complete Advanced evidence set.

    It is regex and lexicon work over a preprocessed document, with no model to load and no
    weights on disk beyond a few kilobytes of coefficients. The current product contract
    keeps Basic strictly encoder-only even though the original specification allowed A4.

    Absent when unfitted, like every other agent: a fresh clone has no artefacts and the API
    still starts and refuses honestly.
    """
    from oridjidji.detectors.linguistic import LinguisticDetector, find_linguistic_model

    root = cfg.inference.get("linguistic_dir", "data/processed/linguistic")
    model = find_linguistic_model(root)
    return [LinguisticDetector(model)] if model is not None else []


def load_probability_layer(cfg: AppConfig) -> Any | None:
    """The fitted calibrator and conformal quantiles for a SINGLE detector.

    Absent by default, and absent is a correct state: a fresh clone has no weights, so it
    has no calibration either, and the API still has to start and refuse honestly. What it
    must not do is invent a probability to fill the gap (2.4).

    **The fusion layer is excluded here on purpose.** It calibrates a fused score, which
    lives on a different axis from any single detector's, and picking it up by accident
    would apply the wrong map -- `find_exported_model` already taught this project what an
    alphabetical sort does to artefact discovery.
    """
    directory = Path(cfg.inference.get("calibration_dir", "data/processed/calibration"))
    if not directory.is_dir():
        return None
    from oridjidji.artefacts import FUSION_DETECTOR, ProbabilityLayer

    indexes = [
        path
        for path in sorted(directory.glob("*.layer.json"))
        if path.name != f"{FUSION_DETECTOR}.layer.json"
    ]
    if not indexes:
        return None
    # A layer whose parts disagree about their calibration set raises rather than serving a
    # probability and an abstention zone that were never measured together (8.9).
    return ProbabilityLayer.load(indexes[0])


@dataclass(frozen=True)
class FusionLayer:
    """The selected Advanced score recipe and its independent probability layer.

    E11 may legitimately select A1 alone. In that case ``ensemble`` is absent, but the
    Advanced probability layer remains distinct from Basic calibration and the roster
    records that statistical agents are evidence only.
    """

    ensemble: Any | None
    layer: Any
    roster: tuple[str, ...]


def load_fusion(cfg: AppConfig) -> FusionLayer | None:
    """Fitted fusion, when `train fusion` has run and its layer was fitted with it.

    Both halves or neither. An ensemble without its calibrator produces a logistic decision
    value that nothing downstream can read as a probability; a fusion layer without its
    ensemble has nothing to calibrate. Returning one alone would invite the caller to pair
    it with the encoder's map, which is the units error this codebase keeps catching.
    """
    from oridjidji.artefacts import FUSION_DETECTOR, ProbabilityLayer
    from oridjidji.fusion.ensemble import Ensemble

    candidate_path = cfg.inference.get("advanced_candidate_manifest")
    if candidate_path:
        from oridjidji.fusion.advanced_bundle import AdvancedCandidateManifest

        candidate = AdvancedCandidateManifest.load(Path(str(candidate_path)))
        layer = ProbabilityLayer.load(candidate.advanced_probability_layer)
        if candidate.ensemble is None:
            if candidate.roster != ("encoder",):
                raise ValueError("Advanced candidate has no ensemble for its multi-agent roster")
            return FusionLayer(ensemble=None, layer=layer, roster=candidate.roster)
        ensemble = Ensemble.load(candidate.ensemble)
        if set(ensemble.agent_names) != set(candidate.roster):
            raise ValueError("Advanced candidate roster disagrees with its ensemble")
        if ensemble.calibration_hash and ensemble.calibration_hash != layer.calibration_hash:
            raise ValueError(
                f"fusion ensemble claims calibration {ensemble.calibration_hash} but its layer "
                f"claims {layer.calibration_hash} -- these were not fitted together"
            )
        return FusionLayer(ensemble=ensemble, layer=layer, roster=candidate.roster)

    fusion_dir = Path(cfg.inference.get("fusion_dir", "data/processed/fusion"))
    ensemble_path = fusion_dir / "ensemble.json"
    layer_path = (
        Path(cfg.inference.get("calibration_dir", "data/processed/calibration"))
        / f"{FUSION_DETECTOR}.layer.json"
    )
    if not (ensemble_path.is_file() and layer_path.is_file()):
        return None
    ensemble = Ensemble.load(ensemble_path)
    layer = ProbabilityLayer.load(layer_path)
    if ensemble.calibration_hash and ensemble.calibration_hash != layer.calibration_hash:
        # 8.9's refusal, applied to the pair that has to agree: weights fitted on one split
        # and calibrated on another describe different populations.
        raise ValueError(
            f"fusion ensemble claims calibration {ensemble.calibration_hash} but its layer "
            f"claims {layer.calibration_hash} -- these were not fitted together"
        )
    return FusionLayer(ensemble=ensemble, layer=layer, roster=ensemble.agent_names)
