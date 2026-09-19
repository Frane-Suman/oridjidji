"""Support objects the served detector needs from the fitted artefacts.

Consolidated here so the distribution carries no training, evaluation or data-generation
package. Contains the calibration layer, the candidate manifests, the few metrics the
detectors compute internally, and three constants. Nothing here fits a model or generates
data - it only loads and validates what a fitted model shipped with.

Deliberately one file: it is the single boundary between the served detector and the
artefacts it consumes.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from oridjidji.fusion.calibration import Calibrator
from oridjidji.fusion.conformal import ConformalQuantiles

# --- from evaluation/metrics.py -----------------------------------------


# --- from data/writer.py ------------------------------------------------


def content_hash(text: str) -> str:
    """SHA-256 of the exact text — the dedup key and the record's identity."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- from training/encoder_candidate.py ---------------------------------


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class EncoderCandidateManifest(BaseModel):
    """Exact Basic candidate inputs; final multi-agent bundles compose this later."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal["encoder-candidate-v1", "encoder-candidate-v2"] = "encoder-candidate-v2"
    # v1 manifests were written with host-absolute paths. They remain readable so old
    # experiments reproduce, while every newly saved v2 manifest is relocatable as one
    # candidate directory.
    path_mode: Literal["absolute", "candidate_relative"] = "absolute"
    candidate: str
    backend: Literal["onnx", "int8", "pytorch"]
    model: str
    tokenizer: str
    probability_layer: str
    weights_hash: str
    repository_hash: str
    selection_hash: str
    preprocessing_hash: str
    calibration_fit_hash: str
    calibration_conformal_hash: str
    runtime_profile: str
    runtime_profile_hash: str
    git_sha: str
    git_dirty: bool
    reproduction_command: str
    files: dict[str, str]
    promotion_status: Literal["unpromoted"] = "unpromoted"

    @staticmethod
    def _portable_path(root: Path, value: str | Path) -> str:
        path = Path(value).resolve()
        try:
            relative = path.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"candidate file is outside its portable root: {path}") from exc
        return relative.as_posix()

    @staticmethod
    def _resolved_path(root: Path, value: str) -> str:
        relative = Path(value)
        if relative.is_absolute():
            raise ValueError(f"portable candidate path must be relative: {value}")
        resolved = (root / relative).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError(f"portable candidate path escapes its root: {value}") from exc
        return str(resolved)

    @classmethod
    def create(
        cls,
        *,
        candidate: str,
        backend: Literal["onnx", "int8", "pytorch"],
        model: str | Path,
        tokenizer: str | Path,
        probability_layer: str | Path,
        weights_hash: str,
        repository_hash: str,
        selection_hash: str,
        preprocessing_hash: str,
        calibration_fit_hash: str,
        calibration_conformal_hash: str,
        runtime_profile: str | Path,
        git_sha: str,
        git_dirty: bool,
        reproduction_command: str,
    ) -> EncoderCandidateManifest:
        model_path = Path(model).resolve()
        tokenizer_path = Path(tokenizer).resolve()
        layer_path = Path(probability_layer).resolve()
        profile_path = Path(runtime_profile).resolve()
        required = [model_path, tokenizer_path, layer_path, profile_path]
        for path in required:
            if not path.is_file():
                raise ValueError(f"candidate file is missing: {path}")
        layer_index = json.loads(layer_path.read_text(encoding="utf-8"))
        component_names = [layer_index["calibrator"], *layer_index["conformal"].values()]
        paths = {
            *[path.resolve() for path in model_path.parent.iterdir() if path.is_file()],
            layer_path,
            *[(layer_path.parent / str(name)).resolve() for name in component_names],
            profile_path,
        }
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise ValueError(f"candidate probability-layer component is missing: {missing[0]}")
        return cls(
            candidate=candidate,
            backend=backend,
            model=str(model_path),
            tokenizer=str(tokenizer_path),
            probability_layer=str(layer_path),
            weights_hash=weights_hash,
            repository_hash=repository_hash,
            selection_hash=selection_hash,
            preprocessing_hash=preprocessing_hash,
            calibration_fit_hash=calibration_fit_hash,
            calibration_conformal_hash=calibration_conformal_hash,
            runtime_profile=str(profile_path),
            runtime_profile_hash=file_sha256(profile_path),
            git_sha=git_sha,
            git_dirty=git_dirty,
            reproduction_command=reproduction_command,
            files={str(path): file_sha256(path) for path in sorted(paths)},
        )

    def verify(self) -> None:
        for name, expected in self.files.items():
            path = Path(name)
            if not path.is_file():
                raise ValueError(f"candidate file is missing: {path}")
            actual = file_sha256(path)
            if actual != expected:
                raise ValueError(f"candidate file hash mismatch for {path}: {actual} != {expected}")
        layer = ProbabilityLayer.load(self.probability_layer)
        if file_sha256(self.runtime_profile) != self.runtime_profile_hash:
            raise ValueError("candidate runtime-profile identity disagrees with its file")
        if layer.calibration_fit_hash != self.calibration_fit_hash:
            raise ValueError("candidate calibration-fit identity disagrees with its layer")
        if layer.calibration_conformal_hash != self.calibration_conformal_hash:
            raise ValueError("candidate conformal identity disagrees with its layer")

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        root = target.parent.resolve()
        payload = self.model_dump(mode="json")
        payload["version"] = "encoder-candidate-v2"
        payload["path_mode"] = "candidate_relative"
        for name in ("model", "tokenizer", "probability_layer", "runtime_profile"):
            payload[name] = self._portable_path(root, str(payload[name]))
        files = payload["files"]
        if not isinstance(files, dict):  # pragma: no cover - protected by Pydantic
            raise TypeError("candidate files must be a mapping")
        payload["files"] = {
            self._portable_path(root, str(name)): str(expected) for name, expected in files.items()
        }
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> EncoderCandidateManifest:
        manifest_path = Path(path).resolve()
        payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("path_mode", "absolute") == "candidate_relative":
            root = manifest_path.parent
            for name in ("model", "tokenizer", "probability_layer", "runtime_profile"):
                payload[name] = cls._resolved_path(root, str(payload[name]))
            files = payload.get("files")
            if not isinstance(files, dict):
                raise ValueError("candidate files must be a mapping")
            payload["files"] = {
                cls._resolved_path(root, str(name)): str(expected)
                for name, expected in files.items()
            }
            # The returned object deliberately exposes resolved paths. Existing serving
            # code therefore needs no path-mode branches and cannot open a cwd-relative
            # lookalike. The serialized manifest remains relative.
            payload["path_mode"] = "absolute"
        result = cls.model_validate(payload)
        result.verify()
        return result


# --- from training/export.py --------------------------------------------


def session_options(intra_op_threads: int | None) -> Any:
    """onnxruntime `SessionOptions` pinned to a thread count, or None to keep its default.

    Kept here rather than inline because both the serving path and the sweep have to build
    the session the same way — a sweep that measures a session the server does not use is
    measuring nothing.
    """
    if intra_op_threads is None:
        return None
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = int(intra_op_threads)
    return options


def onnx_load_kwargs(model_dir: str | Path, intra_op_threads: int | None = None) -> dict[str, Any]:
    """Arguments for `ORTModelForSequenceClassification.from_pretrained`, in one place.

        **There were two implementations of this and they drifted.** The quantiser writes
        `model_quantized.onnx`, not the `model.onnx` optimum looks for by default. The serving
        path in `detectors/encoder.py` knew that; `score_texts` did not, so `train export` ran
        the whole pipeline and then died measuring the INT8 variant it had just written itself
    . A rule applied in one of two places is a rule that will be wrong in one of
        them, so both now call this.
    """
    path = Path(model_dir)
    graphs = sorted(p.name for p in path.glob("*.onnx"))
    kwargs: dict[str, Any] = {}
    if graphs and "model.onnx" not in graphs:
        kwargs["file_name"] = graphs[0]
    options = session_options(intra_op_threads)
    if options is not None:
        kwargs["session_options"] = options
    return kwargs


# --- from training/backbone_precheck.py ---------------------------------

BACKBONE_DIR = pathlib.Path("data/processed/backbones")


@dataclass(frozen=True)
class BackboneCandidate:
    """One E4 pair: what it is called, what it is, and where its weights live."""

    name: str
    observer_id: str
    performer_id: str
    observer_dir: str
    performer_dir: str
    observer_revision: str | None = None
    performer_revision: str | None = None

    def resolve(self) -> tuple[str, str]:
        """Local paths when the weights are downloaded, Hub ids otherwise.

        Falling back rather than failing keeps a fresh clone able to run E4 by fetching —
        but a machine that has already paid for the download never pays twice.
        """
        return (
            _local_or_hub(self.observer_dir, self.observer_id),
            _local_or_hub(self.performer_dir, self.performer_id),
        )


def _local_or_hub(local_name: str, model_id: str) -> str:
    candidate = BACKBONE_DIR / local_name
    return str(candidate) if (candidate / "config.json").is_file() else model_id


E4_BACKBONE_PAIRS: tuple[BackboneCandidate, ...] = (
    BackboneCandidate(
        name="eurollm_1_7b",
        observer_id="utter-project/EuroLLM-1.7B",
        performer_id="utter-project/EuroLLM-1.7B-Instruct",
        observer_dir="eurollm_base",
        performer_dir="eurollm_instruct",
    ),
    BackboneCandidate(
        name="qwen3_1_7b",
        observer_id="Qwen/Qwen3-1.7B-Base",
        performer_id="Qwen/Qwen3-1.7B",
        observer_dir="qwen3_base",
        performer_dir="qwen3_instruct",
        observer_revision="ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
        performer_revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    ),
    BackboneCandidate(
        name="gemma2_2b",
        observer_id="google/gemma-2-2b",
        performer_id="google/gemma-2-2b-it",
        observer_dir="gemma2_base",
        performer_dir="gemma2_instruct",
        observer_revision="c5ebcd40d208330abc697524c919956e692655cf",
        performer_revision="299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
    ),
)

# --- from training/fusion_fit.py ----------------------------------------

FUSION_DETECTOR = "fusion"

# --- from training/calibrate.py -----------------------------------------


@dataclass(frozen=True)
class ProbabilityLayer:
    """Everything needed to turn one detector's score into a verdict.

    One calibrator, and one set of Mondrian quantiles per registered alpha. Alpha is a
    serving choice — how much abstention the deployment wants — so all three are fitted and
    stored rather than one being picked here on the author's behalf.
    """

    detector: str
    calibrator: Calibrator
    quantiles: dict[float, ConformalQuantiles]
    calibration_hash: str
    created_at: str
    version: str = "probability-layer-v1"
    calibration_fit_hash: str = ""
    calibration_conformal_hash: str = ""
    operating_threshold: float | None = None

    def save(self, directory: str | Path) -> Path:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        self.calibrator.save(target / f"{self.detector}.calibrator.json")
        for alpha, quantiles in self.quantiles.items():
            quantiles.save(target / f"{self.detector}.conformal.a{alpha:g}.json")
        index = {
            "version": self.version,
            "detector": self.detector,
            "calibration_hash": self.calibration_hash,
            "calibration_fit_hash": self.calibration_fit_hash or self.calibration_hash,
            "calibration_conformal_hash": self.calibration_conformal_hash or self.calibration_hash,
            "operating_threshold": self.operating_threshold,
            "created_at": self.created_at,
            "calibrator": f"{self.detector}.calibrator.json",
            "conformal": {
                f"{alpha:g}": f"{self.detector}.conformal.a{alpha:g}.json"
                for alpha in self.quantiles
            },
        }
        path = target / f"{self.detector}.layer.json"
        path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> ProbabilityLayer:
        """Rebuild from disk, refusing a layer whose parts disagree about their identity.

        A calibrator fitted on one split beside quantiles fitted on another would serve a
        probability and an abstention zone that were never measured together. The hash is
        what makes that checkable rather than assumed.
        """
        index_path = Path(path)
        index: dict[str, Any] = json.loads(index_path.read_text(encoding="utf-8"))
        root = index_path.parent
        calibrator = Calibrator.load(root / index["calibrator"])
        quantiles = {
            float(alpha): ConformalQuantiles.load(root / name)
            for alpha, name in index["conformal"].items()
        }
        expected = str(index["calibration_hash"])
        mismatched = [
            f"conformal a={alpha:g} ({q.calibration_hash!r})"
            for alpha, q in quantiles.items()
            if q.calibration_hash != expected
        ]
        if calibrator.calibration_hash != expected:
            mismatched.append(f"calibrator ({calibrator.calibration_hash!r})")
        if mismatched:
            raise ValueError(
                f"probability layer {index_path} claims calibration {expected!r} but "
                f"{', '.join(mismatched)} disagree — these parts were not fitted together"
            )
        return cls(
            detector=str(index["detector"]),
            calibrator=calibrator,
            quantiles=quantiles,
            calibration_hash=expected,
            created_at=str(index.get("created_at", "")),
            version=str(index.get("version", "probability-layer-v1")),
            calibration_fit_hash=str(index.get("calibration_fit_hash", expected)),
            calibration_conformal_hash=str(index.get("calibration_conformal_hash", expected)),
            operating_threshold=(
                None
                if index.get("operating_threshold") is None
                else float(index["operating_threshold"])
            ),
        )


# --- from evaluation/statistical_thresholds.py --------------------------

DetectorName = Literal["binoculars", "fast_detect_gpt"]


ScoreDirection = Literal["lower_is_ai", "higher_is_ai"]


class DomainThresholdMetrics(BaseModel):
    """Observed rates at one frozen threshold in one Croatian domain."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    domain: Literal["academic", "news", "all"]
    humans: int
    ai: int
    achieved_fpr: float = Field(ge=0.0, le=1.0)
    tpr: float = Field(ge=0.0, le=1.0)


class StatisticalThreshold(BaseModel):
    """One conservative empirical human-FPR operating point."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_fpr: float = Field(gt=0.0, lt=1.0)
    raw_threshold: float
    oriented_threshold: float
    metrics: tuple[DomainThresholdMetrics, ...]


class StatisticalThresholdReport(BaseModel):
    """Standalone A2/A3 report bound to the exact Q4 Croatian calibration evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal["statistical-threshold-report-v1"] = "statistical-threshold-report-v1"
    detector: DetectorName
    direction: ScoreDirection
    rows: int
    independent_groups: int
    auroc: float = Field(ge=0.0, le=1.0)
    auprc: float = Field(ge=0.0, le=1.0)
    thresholds: tuple[StatisticalThreshold, ...]
    evidence_sha256: str = Field(min_length=64, max_length=64)
    repository_hash: str = Field(min_length=64, max_length=64)
    selection_hash: str = Field(min_length=64, max_length=64)
    preprocessing_hash: str = Field(min_length=64, max_length=64)
    calibration_role_hash: str = Field(min_length=64, max_length=64)
    backbone_hash: str = Field(min_length=64, max_length=64)
    pair_name: str
    quantization: str
    window_policy: str
    tokenizer_mapping_hash: str = Field(min_length=64, max_length=64)
    observer_weights_hash: str = Field(min_length=64, max_length=64)
    performer_weights_hash: str = Field(min_length=64, max_length=64)
    formula_hash: str = Field(min_length=64, max_length=64)
    report_identity: str = Field(min_length=64, max_length=64)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path
