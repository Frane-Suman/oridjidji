"""Unpromoted E12 Advanced candidate binding A1–A3, A4 evidence and uncertainty."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _locally_built(path: Path) -> bool:
    """True for quantised weights the installer built on this machine, when the operator allows it."""
    import os

    accepted = os.environ.get("ORIDJIDJI_SKIP_BUNDLE_VERIFY", "").strip().lower() in {"1", "true", "yes"}
    return accepted and "statistical" in path.parts and path.suffix in {".gguf", ".json"}


class AdvancedCandidateManifest(BaseModel):
    """Exact files and data identities for a candidate that M9 has not promoted."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[
        "advanced-candidate-v2",
        "advanced-candidate-v3",
        "advanced-candidate-v4",
        "advanced-candidate-v5",
    ] = "advanced-candidate-v4"
    path_mode: Literal["absolute", "candidate_relative"] = "absolute"
    candidate: str = Field(min_length=1)
    basic_candidate_manifest: str
    gguf_pair_manifest: str
    statistical_tokenizer: str | None = None
    statistical_pair_name: str = Field(min_length=1)
    statistical_backbone_hash: str = Field(min_length=64, max_length=64)
    statistical_quantization: Literal["Q4_K_M"] = "Q4_K_M"
    binoculars_formula_hash: str = Field(min_length=64, max_length=64)
    fast_detect_gpt_formula_hash: str = Field(min_length=64, max_length=64)
    aggregation: Literal["V0", "V4"]
    window_policy: Literal["prefix512", "coverage1024"]
    linguistic_model: str | None
    linguistic_role: Literal["explanation_only", "fused"] = "explanation_only"
    ensemble: str | None
    roster: tuple[str, ...]
    advanced_probability_layer: str
    fusion_artifact_manifest: str | None = None
    segment_reliability: str | None
    repository_hash: str = Field(min_length=64, max_length=64)
    role_hashes: dict[str, str]
    selection_hash: str = Field(min_length=64, max_length=64)
    preprocessing_hash: str = Field(min_length=64, max_length=64)
    basic_selection_hash: str | None = Field(default=None, min_length=64, max_length=64)
    basic_preprocessing_hash: str | None = Field(default=None, min_length=64, max_length=64)
    basic_probability_identity: str = Field(min_length=64, max_length=64)
    advanced_probability_identity: str = Field(min_length=64, max_length=64)
    execution_policy: Literal["sequential", "concurrent"]
    files: dict[str, str]
    runtime_status: Literal["validated_candidate", "research_only"] = "validated_candidate"
    device_transfer_status: Literal["passed", "failed", "not_checked"] = "passed"
    probability_device: Literal["cpu", "cuda"] = "cpu"
    limitations: tuple[str, ...] = ()
    limitation_codes: tuple[
        Literal[
            "cpu_cuda_transfer_failed",
            "probability_calibrated_on_cuda",
            "segment_reliability_missing",
            "e11_selected_a1",
        ],
        ...,
    ] = ()
    promotion_status: Literal["unpromoted"] = "unpromoted"

    @staticmethod
    def _portable_path(root: Path, value: str | Path) -> str:
        path = Path(value).resolve()
        try:
            relative = path.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Advanced candidate file is outside its portable root: {path}"
            ) from exc
        return relative.as_posix()

    @staticmethod
    def _resolved_path(root: Path, value: str) -> str:
        relative = Path(value)
        if relative.is_absolute():
            raise ValueError(f"portable Advanced path must be relative: {value}")
        resolved = (root / relative).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"portable Advanced path escapes its root: {value}") from exc
        return str(resolved)

    @model_validator(mode="after")
    def probability_layers_are_separate(self) -> AdvancedCandidateManifest:
        if self.version in {
            "advanced-candidate-v3",
            "advanced-candidate-v4",
            "advanced-candidate-v5",
        } and (self.statistical_tokenizer is None):
            raise ValueError("portable Advanced candidate must bind its statistical tokenizer")
        if self.version in {"advanced-candidate-v4", "advanced-candidate-v5"} and (
            self.basic_selection_hash is None or self.basic_preprocessing_hash is None
        ):
            raise ValueError("Advanced candidate must bind the Basic stage identities explicitly")
        if self.version == "advanced-candidate-v5":
            if self.runtime_status == "validated_candidate" and (
                self.segment_reliability is None
                or self.device_transfer_status != "passed"
                or self.probability_device != "cpu"
            ):
                raise ValueError(
                    "validated Advanced candidate requires CPU transfer, CPU calibration, "
                    "and segment reliability"
                )
            if self.runtime_status == "research_only" and not self.limitations:
                raise ValueError("research-only Advanced runtime must publish its limitations")
            codes = set(self.limitation_codes)
            if self.runtime_status == "research_only" and not codes:
                raise ValueError("research-only Advanced runtime must publish limitation codes")
            if self.device_transfer_status == "failed" and "cpu_cuda_transfer_failed" not in codes:
                raise ValueError("failed device transfer must be declared as a limitation")
            if self.probability_device == "cuda" and "probability_calibrated_on_cuda" not in codes:
                raise ValueError("CUDA calibration must be declared as a limitation")
            if self.segment_reliability is None and "segment_reliability_missing" not in codes:
                raise ValueError("missing segment reliability must be declared as a limitation")
            required_codes: set[str] = set()
            if self.device_transfer_status == "failed":
                required_codes.add("cpu_cuda_transfer_failed")
            if self.probability_device == "cuda":
                required_codes.add("probability_calibrated_on_cuda")
            if self.segment_reliability is None:
                required_codes.add("segment_reliability_missing")
            if not required_codes <= codes:
                raise ValueError("Advanced runtime limitation codes do not match its state")
            if self.device_transfer_status == "failed" and "cpu_cuda_transfer_failed" not in codes:
                raise ValueError("failed CPU/CUDA transfer needs an explicit limitation code")
            if self.probability_device == "cuda" and "probability_calibrated_on_cuda" not in codes:
                raise ValueError("CUDA calibration needs an explicit limitation code")
            if self.segment_reliability is None and "segment_reliability_missing" not in codes:
                raise ValueError("missing segment reliability needs an explicit limitation code")
        if self.basic_probability_identity == self.advanced_probability_identity:
            raise ValueError("Advanced fusion cannot reuse the Basic probability layer")
        if self.linguistic_role == "fused":
            raise ValueError("A4 remains explanation-only under E5, E5b and E11-R")
        if not self.roster or self.roster[0] != "encoder" or "linguistic" in self.roster:
            raise ValueError("Advanced roster must start with A1 and exclude explanation-only A4")
        if len(set(self.roster)) != len(self.roster):
            raise ValueError("Advanced roster contains duplicate agents")
        if len(self.roster) > 1 and self.ensemble is None:
            raise ValueError("A multi-agent Advanced roster requires a fitted ensemble")
        statistical_names = {"binoculars", "binoculars_v4"} & set(self.roster)
        expected = "binoculars" if self.aggregation == "V0" else "binoculars_v4"
        if statistical_names and statistical_names != {expected}:
            raise ValueError("Binoculars aggregation and roster disagree")
        if self.aggregation == "V4" and self.window_policy != "coverage1024":
            raise ValueError("Binoculars V4 requires the registered coverage1024 policy")
        if self.aggregation == "V0" and self.window_policy != "prefix512":
            raise ValueError("Binoculars V0 requires the registered prefix512 policy")
        required = {"calibration_fit", "calibration_conformal", "dev", "development_challenge"}
        if not required <= set(self.role_hashes) or any(
            len(value) != 64 for value in self.role_hashes.values()
        ):
            raise ValueError("Advanced candidate needs every consumed role hash")
        return self

    def verify(self) -> None:
        for name, expected in self.files.items():
            path = Path(name)
            if not path.is_file():
                raise ValueError(f"Advanced candidate file mismatch: {path}")
            if _sha(path) != expected and not _locally_built(path):
                raise ValueError(f"Advanced candidate file mismatch: {path}")
        gguf_path = Path(self.gguf_pair_manifest)
        gguf = json.loads(gguf_path.read_text(encoding="utf-8"))
        if gguf.get("name") != self.statistical_pair_name:
            raise ValueError("Advanced candidate opened a different statistical pair")
        if gguf.get("quantization") != self.statistical_quantization:
            raise ValueError("Advanced candidate opened a different quantization")
        for role in ("observer", "performer"):
            member = gguf_path.parent / str(gguf[role])
            expected = str(gguf[f"{role}_sha256"])
            if not member.is_file() or _sha(member) != expected:
                raise ValueError(f"Advanced candidate {role} weight hash mismatch")

        # The Basic manifest is a transitive trust boundary. Verifying only its JSON hash
        # would allow the encoder files it names to drift after packaging.
        from oridjidji.artefacts import EncoderCandidateManifest

        basic = EncoderCandidateManifest.load(Path(self.basic_candidate_manifest))
        if basic.repository_hash != self.repository_hash:
            raise ValueError("Basic and Advanced candidates use different repositories")
        if self.version in {"advanced-candidate-v4", "advanced-candidate-v5"} and (
            basic.selection_hash != self.basic_selection_hash
            or basic.preprocessing_hash != self.basic_preprocessing_hash
        ):
            raise ValueError("Basic candidate identity disagrees with the Advanced manifest")
        if self.version == "advanced-candidate-v5":
            from oridjidji.artefacts import ProbabilityLayer

            basic_layer = ProbabilityLayer.load(Path(basic.probability_layer))
            if basic_layer.calibration_hash != self.basic_probability_identity:
                raise ValueError("Basic probability identity disagrees with the Advanced manifest")
            layer = ProbabilityLayer.load(Path(self.advanced_probability_layer))
            if layer.calibration_hash != self.advanced_probability_identity:
                raise ValueError("Advanced probability identity disagrees with its layer")
            if self.fusion_artifact_manifest is not None:
                artifact = json.loads(
                    Path(self.fusion_artifact_manifest).read_text(encoding="utf-8")
                )
                semantic_expected: dict[str, object] = {
                    "backbone_hash": self.statistical_backbone_hash,
                    "pair_name": self.statistical_pair_name,
                    "probability_identity": self.advanced_probability_identity,
                    "repository_hash": self.repository_hash,
                    "selection_hash": self.selection_hash,
                    "preprocessing_hash": self.preprocessing_hash,
                    "role_hashes": self.role_hashes,
                }
                if any(artifact.get(name) != value for name, value in semantic_expected.items()):
                    raise ValueError("Advanced fusion artifact identities disagree with manifest")

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        root = path.parent.resolve()
        payload = self.model_dump(mode="json")
        payload["version"] = "advanced-candidate-v5"
        payload["path_mode"] = "candidate_relative"
        names = (
            "basic_candidate_manifest",
            "gguf_pair_manifest",
            "statistical_tokenizer",
            "linguistic_model",
            "ensemble",
            "advanced_probability_layer",
            "fusion_artifact_manifest",
            "segment_reliability",
        )
        for name in names:
            if payload[name] is not None:
                payload[name] = self._portable_path(root, str(payload[name]))
        files = payload["files"]
        if not isinstance(files, dict):  # pragma: no cover - protected by Pydantic
            raise TypeError("Advanced candidate files must be a mapping")
        payload["files"] = {
            self._portable_path(root, str(name)): str(expected) for name, expected in files.items()
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def save_absolute(self, path: Path) -> Path:
        """Save a local research manifest whose verified files span storage roots."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(mode="json")
        payload["version"] = "advanced-candidate-v5"
        payload["path_mode"] = "absolute"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> AdvancedCandidateManifest:
        manifest_path = path.resolve()
        payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("path_mode", "absolute") == "candidate_relative":
            root = manifest_path.parent
            names = (
                "basic_candidate_manifest",
                "gguf_pair_manifest",
                "statistical_tokenizer",
                "linguistic_model",
                "ensemble",
                "advanced_probability_layer",
                "fusion_artifact_manifest",
                "segment_reliability",
            )
            for name in names:
                if payload.get(name) is not None:
                    payload[name] = cls._resolved_path(root, str(payload[name]))
            files = payload.get("files")
            if not isinstance(files, dict):
                raise ValueError("Advanced candidate files must be a mapping")
            payload["files"] = {
                cls._resolved_path(root, str(name)): str(expected)
                for name, expected in files.items()
            }
            payload["path_mode"] = "absolute"
        result = cls.model_validate(payload)
        result.verify()
        return result
