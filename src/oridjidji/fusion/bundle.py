"""The versioned `DetectorBundle`.

The specification in one sentence: *"All fitted artefacts serialise into one versioned DetectorBundle with
git SHA, config hash, dataset hash, dates, metrics. **The API refuses to start if bundle
metadata doesn't match loaded code** — no silent staleness."*

**The refusal is the feature, and this project has earned it.** `find_exported_model` sorts by
recency because alphabetical order once ranked an obsolete `bcms_bertic_e1_winner_onnx` above
the checkpoint that replaced it, and it would have been served without a word. Six exports sit
side by side in `data/processed/models` right now. A bundle that merely *records* which
artefacts were fitted together adds nothing; one that refuses to serve when they no longer
belong together is the guarantee.

**What "matching" means here, deliberately narrowly.** Not the git SHA — a docstring edit
changes that and must not stop the API. What has to match is the identity of the data the
artefacts were fitted on: a calibrator and a conformal quantile fitted on different splits are
individually valid and jointly meaningless, and nothing downstream can tell. The git SHA and
dates are recorded for citation; the **calibration hash** is what is enforced.

That distinction is the whole design. A bundle that refused on every code change would be
turned off within a week, and a guarantee people disable is worse than none.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Where a built bundle lives. Not in git — it names fitted artefacts, which are not either.
BUNDLE_PATH = Path("data/processed/detector_bundle.json")


class BundleMismatch(RuntimeError):
    """Raised when a bundle's parts were not fitted together.

    An exception rather than a warning because the specification says the API *refuses to start*. A
    warning at startup is read once and then filtered out of the logs; the whole point is
    that stale artefacts cannot be served by accident.
    """


def git_sha() -> str:
    """The commit the bundle was built at, or `unknown` outside a checkout.

    Recorded for citation, never enforced — see the module docstring.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def file_digest(path: str | Path) -> str:
    """A short content hash, so a dataset can be named by what it contains.

    Reads in blocks: `dataset.parquet` is 3.9 MB today and the holdout will grow, and a
    bundle build that loads a whole corpus into memory to hash it is a bundle build that
    stops being run.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class ArtefactRef:
    """One fitted artefact: where it is, what it is, and what it was fitted on."""

    kind: str
    path: str
    #: The identity of the data it was fitted on. Empty for artefacts that carry none —
    #: recorded as empty rather than as a placeholder, because a fabricated hash would match.
    calibration_hash: str = ""
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectorBundle:
    """Every fitted artefact the system serves, and the identity that binds them."""

    artefacts: tuple[ArtefactRef, ...]
    git_sha: str
    config_hash: str
    dataset_hash: str
    created_at: str

    @property
    def calibration_hashes(self) -> set[str]:
        """The distinct calibration identities across artefacts that carry one."""
        return {a.calibration_hash for a in self.artefacts if a.calibration_hash}

    def verify(self) -> None:
        """Refuse a bundle whose parts were not fitted on the same data.

        Artefacts without a calibration hash are exempt rather than assumed to match: the
        encoder's provenance names a training run, not a calibration split, and demanding
        one would either block startup or invite a placeholder that matches everything.
        """
        hashes = self.calibration_hashes
        if len(hashes) > 1:
            detail = ", ".join(
                f"{a.kind}={a.calibration_hash[:8] or '(none)'}"
                for a in self.artefacts
                if a.calibration_hash
            )
            raise BundleMismatch(
                "fitted artefacts disagree about their calibration set — a probability and "
                "an abstention zone measured on different data are individually valid and "
                f"jointly meaningless: {detail}"
            )

    def to_json(self) -> str:
        return json.dumps(
            {
                "artefacts": [
                    {
                        "kind": a.kind,
                        "path": a.path,
                        "calibration_hash": a.calibration_hash,
                        "metrics": a.metrics,
                    }
                    for a in self.artefacts
                ],
                "git_sha": self.git_sha,
                "config_hash": self.config_hash,
                "dataset_hash": self.dataset_hash,
                "created_at": self.created_at,
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str) -> DetectorBundle:
        data: dict[str, Any] = json.loads(payload)
        required = ("artefacts", "git_sha", "config_hash", "dataset_hash")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"bundle is missing {', '.join(missing)}")
        return cls(
            artefacts=tuple(
                ArtefactRef(
                    kind=str(a["kind"]),
                    path=str(a["path"]),
                    calibration_hash=str(a.get("calibration_hash", "")),
                    metrics=dict(a.get("metrics", {})),
                )
                for a in data["artefacts"]
            ),
            git_sha=str(data["git_sha"]),
            config_hash=str(data["config_hash"]),
            dataset_hash=str(data["dataset_hash"]),
            created_at=str(data.get("created_at", "")),
        )

    def save(self, path: str | Path = BUNDLE_PATH) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path = BUNDLE_PATH) -> DetectorBundle:
        """Read a bundle and verify it. A bundle that fails verification never loads."""
        bundle = cls.from_json(Path(path).read_text(encoding="utf-8"))
        bundle.verify()
        return bundle


def collect(
    *,
    calibration_dir: str | Path = "data/processed/calibration",
    linguistic_dir: str | Path = "data/processed/linguistic",
    fusion_dir: str | Path = "data/processed/fusion",
    dataset: str | Path = "data/processed/dataset.parquet",
    config: str | Path = "configs/default.yaml",
) -> DetectorBundle:
    """Build a bundle from whatever is fitted on this machine.

    Absent artefacts are simply absent — a fresh clone has none, and the bundle should
    describe what exists rather than assert what ought to.
    """
    from oridjidji.detectors.linguistic import LinguisticModel
    from oridjidji.fusion.calibration import Calibrator
    from oridjidji.fusion.conformal import ConformalQuantiles
    from oridjidji.fusion.ensemble import Ensemble

    artefacts: list[ArtefactRef] = []

    for path in sorted(Path(calibration_dir).glob("*.calibrator.json")):
        calibrator = Calibrator.load(path)
        artefacts.append(
            ArtefactRef(
                kind="calibrator",
                path=str(path),
                calibration_hash=calibrator.calibration_hash,
                metrics={"ece": calibrator.ece, "brier": calibrator.brier},
            )
        )
    for path in sorted(Path(calibration_dir).glob("*.conformal.*.json")):
        quantiles = ConformalQuantiles.load(path)
        artefacts.append(
            ArtefactRef(
                kind=f"conformal@{quantiles.alpha:g}",
                path=str(path),
                calibration_hash=quantiles.calibration_hash,
                metrics={"n_human": quantiles.n_human, "n_machine": quantiles.n_machine},
            )
        )
    for path in sorted(Path(linguistic_dir).glob("*.linguistic.json")):
        model = LinguisticModel.load(path)
        artefacts.append(
            ArtefactRef(
                kind="linguistic",
                path=str(path),
                metrics={"dev_auroc": model.dev_auroc, "dev_tpr_at_1pct": model.dev_tpr_at_1pct},
            )
        )

    # The fused ensemble is a served artefact and carries the identity of the split its
    # weights were fitted on, so it belongs under the same agreement check as the rest: an
    # ensemble fitted on one calibration set beside a calibrator fitted on another is the
    # exact pairing the specification's refusal exists for.
    ensemble_path = Path(fusion_dir) / "ensemble.json"
    if ensemble_path.is_file():
        ensemble = Ensemble.load(ensemble_path)
        artefacts.append(
            ArtefactRef(
                kind="fusion",
                path=str(ensemble_path),
                calibration_hash=ensemble.calibration_hash,
                metrics={"n_calibration": float(ensemble.n_calibration)},
            )
        )

    return DetectorBundle(
        artefacts=tuple(artefacts),
        git_sha=git_sha(),
        config_hash=file_digest(config) if Path(config).is_file() else "",
        dataset_hash=file_digest(dataset) if Path(dataset).is_file() else "",
        created_at=datetime.now(UTC).isoformat(),
    )
