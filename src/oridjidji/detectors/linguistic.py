"""A4: the Croatian linguistic agent.

M1 built 37 feature extractors — orthographic, stylometric, human-error and AI-ism families —
and until now **nothing scored with them**. They fed the artefact probe and the data-quality
gates, which is to say they described the corpus rather than judging a document. This module
is what turns them into a detector.

**Why it is worth having even if it adds little AUROC.** The encoder sits at 0.9993, so a
37-feature model is not going to beat it, and E5 is registered precisely to ask whether this
agent earns its place at all — *"explainability-only"* being a legitimate answer. What it
offers regardless is a score a person can interrogate: which **Croatian** features carry
signal, which is the specification's question and something a transformer's logit cannot answer. For the
linear model that reading is exact and per-document; for the tree model it is global feature
importance, and the two are not interchangeable.

**The artefact must round-trip the model it names, and once it did not.** Gradient boosting
was selected on dev at 0.9623, and what got persisted were its feature *importances* in the
field `probability()` reads as logistic weights. The served agent scored **0.1679** on the
same split — a plausible-looking number in [0, 1], from a model that was never fitted. Nothing
raised. `_evaluable` is the guard that now makes that combination unsaveable rather than
merely wrong.

**Morphology is deliberately absent.** The CLASSLA family needs `classla`, which pins
`protobuf==4.21.2` against the backbone extra, so A4 ships on the 37 features that
need no parser. That is a stated limitation, not an oversight: the morphosyntactic family is
the one most likely to carry Croatian-specific signal, and its absence is the first thing to
revisit if A4 underperforms.

**Direction matters here more than it looks.** A4 is higher-is-machine, and it feeds the same
fusion input as Binoculars, which is lower-is-machine. The specification calls a flipped sign the classic
bug in this domain, so the direction is carried on the result rather than assumed downstream.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from oridjidji.schemas import AgentResult, AgentStatus, ScoreDirection

#: Where `train linguistic` writes the fitted model. Not in git — fitted artefacts never are.
LINGUISTIC_DIR = Path("data/processed/linguistic")

RATIONALE_HR = (
    "Hrvatske jezične značajke: pravopis, stilometrija i tragovi tipični za strojni tekst."
)


@dataclass(frozen=True)
class LinguisticModel:
    """A fitted classifier plus the exact feature list it was fitted on.

    **The feature list is not decoration.** The registry can grow — the specification says the AI-ism
    lexicon is discovered empirically and may be extended — and a model fed a different
    vector than it was trained on produces a number rather than an error. Carrying the list
    makes that detectable instead of silent.
    """

    method: str
    features: tuple[str, ...]
    #: Logistic weights. Empty for a tree model, which has none — see `importances`.
    coefficients: dict[str, float]
    intercept: float
    #: Mean and standard deviation per feature, so serving standardises the way training did.
    means: dict[str, float]
    scales: dict[str, float]
    dev_auroc: float
    dev_tpr_at_1pct: float
    n_train: int
    created_at: str
    #: The fitted trees, for a gradient-boosting model. Each is a flat node table:
    #: `left`/`right` children (-1 at a leaf), the `feature` index and `threshold` tested at
    #: an internal node, and the `value` a leaf contributes to the raw score.
    trees: tuple[dict[str, list[float]], ...] = ()
    learning_rate: float = 0.0
    #: The ensemble's starting raw score, before any tree — the prior's log-odds.
    init_score: float = 0.0
    #: Global feature importance for a tree model. NOT weights: unsigned, sums to one, and
    #: about the fit as a whole rather than about any one document.
    importances: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse a model this class cannot actually evaluate.

        The failure being prevented is not a crash. A tree model whose trees were dropped
        still loads, still returns numbers in [0, 1], and is silently a different model —
        which is what shipped, and what nothing downstream could detect.
        """
        if not self._evaluable():
            raise ValueError(
                f"a {self.method!r} model cannot be evaluated from what is stored: "
                + (
                    "no trees were serialised, so probability() would fall back to a linear "
                    "formula over parameters that are not weights"
                    if self.method == "gradient_boosting"
                    else "no coefficients were serialised"
                )
            )

    def _evaluable(self) -> bool:
        """Whether `probability` can reproduce this method from the stored parameters."""
        if self.method == "gradient_boosting":
            return bool(self.trees) and self.learning_rate > 0.0
        if self.method == "logistic_regression":
            return bool(self.coefficients)
        return False

    @property
    def attribution(self) -> dict[str, float]:
        """Whatever this model can say about which features matter, by its own method.

        Signed standardised weights for the regression, unsigned global importances for the
        tree ensemble. Same question, genuinely different answers — the report labels which
        it is rather than letting a shared column name imply they are comparable.
        """
        return self.importances if self.method == "gradient_boosting" else self.coefficients

    def to_json(self) -> str:
        return json.dumps(
            {
                "method": self.method,
                "features": list(self.features),
                "coefficients": self.coefficients,
                "intercept": self.intercept,
                "means": self.means,
                "scales": self.scales,
                "dev_auroc": self.dev_auroc,
                "dev_tpr_at_1pct": self.dev_tpr_at_1pct,
                "n_train": self.n_train,
                "created_at": self.created_at,
                "trees": list(self.trees),
                "learning_rate": self.learning_rate,
                "init_score": self.init_score,
                "importances": self.importances,
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str) -> LinguisticModel:
        data: dict[str, Any] = json.loads(payload)
        required = ("method", "features", "coefficients", "intercept", "means", "scales")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"linguistic model is missing {', '.join(missing)}")
        return cls(
            method=str(data["method"]),
            features=tuple(data["features"]),
            coefficients=dict(data["coefficients"]),
            intercept=float(data["intercept"]),
            means=dict(data["means"]),
            scales=dict(data["scales"]),
            dev_auroc=float(data.get("dev_auroc", 0.0)),
            dev_tpr_at_1pct=float(data.get("dev_tpr_at_1pct", 0.0)),
            n_train=int(data.get("n_train", 0)),
            created_at=str(data.get("created_at", "")),
            trees=tuple(data.get("trees", ())),
            learning_rate=float(data.get("learning_rate", 0.0)),
            init_score=float(data.get("init_score", 0.0)),
            importances=dict(data.get("importances", {})),
        )

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> LinguisticModel:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def _standardise(self, features: dict[str, float]) -> list[float]:
        """The model's feature vector, in the order it was fitted on.

        A feature the model was not fitted on is ignored; one it expects and does not get
        raises. Those are different failures: the registry growing is routine, and a vector
        arriving short means the extractor changed under a model that still claims to
        consume it. **Order is load-bearing for the tree model**, whose nodes test feature
        *indices* — a permuted vector there is a different model, silently.
        """
        missing = [name for name in self.features if name not in features]
        if missing:
            raise ValueError(
                f"feature vector is missing {len(missing)} of the model's features: "
                f"{', '.join(missing[:5])}"
            )
        return [
            (features[name] - self.means.get(name, 0.0)) / (self.scales.get(name, 1.0) or 1.0)
            for name in self.features
        ]

    def _raw_score(self, vector: list[float]) -> float:
        """The score before the logistic squash, by whichever method was fitted.

        The tree branch is three sentences: start at the prior's log-odds; walk each tree
        left or right on one feature's threshold until a leaf; add the leaves, scaled by the
        learning rate. That is the whole of gradient-boosting *inference* — the fitting is
        sklearn's, the prediction is here, so the served model is one this file can explain.
        """
        if self.method == "gradient_boosting":
            total = self.init_score
            for tree in self.trees:
                left, right = tree["left"], tree["right"]
                feature, threshold, value = tree["feature"], tree["threshold"], tree["value"]
                node = 0
                while int(left[node]) != -1:
                    index = int(feature[node])
                    node = int(left[node]) if vector[index] <= threshold[node] else int(right[node])
                total += self.learning_rate * value[node]
            return total
        total = self.intercept
        for name, standardised in zip(self.features, vector, strict=True):
            total += self.coefficients.get(name, 0.0) * standardised
        return total

    def probability(self, features: dict[str, float]) -> float:
        """P(machine) for one feature vector, standardised the way training was."""
        import math

        raw = self._raw_score(self._standardise(features))
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, raw))))

    def top_contributors(self, features: dict[str, float], n: int = 5) -> list[tuple[str, float]]:
        """The features pushing this document hardest, signed.

        The specification wants per-feature attribution, and for a standardised linear model it is exact
        rather than approximated — coefficient times standardised value is the contribution.
        """
        if self.method == "gradient_boosting":
            raise ValueError(
                "per-document attribution is exact only for the linear model; a tree "
                "ensemble needs SHAP, which the specification defers. Use `attribution` for the global "
                "importances, and do not present them as being about this document."
            )
        contributions = []
        for name in self.features:
            if name not in features:
                continue
            scale = self.scales.get(name, 1.0) or 1.0
            standardised = (features[name] - self.means.get(name, 0.0)) / scale
            contributions.append((name, self.coefficients.get(name, 0.0) * standardised))
        return sorted(contributions, key=lambda pair: -abs(pair[1]))[:n]


@dataclass(frozen=True)
class LinguisticProvenance:
    """What `/api/v1/health` needs to name this agent.

    Every served detector must answer `provenance.version` — the API reports which artefacts
    a running instance holds, and one that cannot name its own model is not reproducible. A4
    missed this on its first pass because the serving tests use a stub subclassing
    `EncoderDetector`, which already had it, so nothing exercised the real class through
    `create_app`.
    """

    method: str
    n_features: int
    dev_auroc: float

    @property
    def version(self) -> str:
        return f"linguistic-{self.method}@{self.n_features}f"


class LinguisticDetector:
    """A4 as a served agent, satisfying the same protocol as A1 and the statistical pair."""

    def __init__(self, model: LinguisticModel | None = None) -> None:
        self.model = model
        self.provenance = (
            LinguisticProvenance(
                method=model.method,
                n_features=len(model.features),
                dev_auroc=model.dev_auroc,
            )
            if model is not None
            else LinguisticProvenance(method="unfitted", n_features=0, dev_auroc=0.0)
        )

    @property
    def name(self) -> str:
        return "linguistic"

    @property
    def version(self) -> str:
        """Method, feature count, and whether it is fitted at all.

        The feature count is in the version because a model fitted on 37 features and a model
        fitted on 45 are different detectors, and the specification wants that visible rather than inferred.
        """
        if self.model is None:
            return "linguistic@unfitted"
        return self.provenance.version

    def load(self) -> None:
        """Nothing to load: the model is a handful of coefficients, already in memory."""

    def score(self, text: str) -> AgentResult:
        """Score one document. Never raises — a dead agent is a reportable state."""
        started = time.perf_counter()
        try:
            if self.model is None:
                raise RuntimeError("no fitted linguistic model; run `train linguistic`")
            from oridjidji.features.base import extract_features
            from oridjidji.preprocessing import preprocess

            document = preprocess(text)
            features = extract_features(document)
            value = self.model.probability(features)
            contributions: list[dict[str, Any]] = []
            for name, importance in self.model.attribution.items():
                if name not in features:
                    continue
                if self.model.method == "logistic_regression":
                    standardised = (features[name] - self.model.means[name]) / (
                        self.model.scales[name] or 1.0
                    )
                    contribution = importance * standardised
                    kind = "signed_local"
                else:
                    contribution = importance
                    kind = "global_importance"
                contributions.append(
                    {
                        "name": name,
                        "value": float(features[name]),
                        "contribution": float(contribution),
                        "kind": kind,
                    }
                )
            contributions.sort(key=lambda item: -abs(float(item["contribution"])))
        except Exception as exc:  # noqa: BLE001 — see docstring: this is the degrade path
            return AgentResult(
                name=self.name,
                version=self.version,
                status=AgentStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        return AgentResult(
            name=self.name,
            version=self.version,
            status=AgentStatus.OK,
            score=value,
            direction=ScoreDirection.HIGHER_IS_AI,
            latency_ms=int((time.perf_counter() - started) * 1000),
            rationale_hr=RATIONALE_HR,
            token_data={"top_features": contributions[:8]},
        )

    def predicts_machine(self, score: float) -> bool | None:
        """None until M6 calibrates it.

        A4's output is already in [0, 1] and looks like a probability, which makes this the
        easiest agent to over-trust: a logistic output is not a calibrated probability, and
        the specification forbids importing a threshold rather than fitting one on Croatian. `None` is the
        honest answer and the one M6's conformal layer needs — "cannot say" has to be
        distinguishable from "human".
        """
        return None


def find_linguistic_model(root: str | Path = LINGUISTIC_DIR) -> LinguisticModel | None:
    """The fitted model when it exists, None otherwise.

    Absent is a correct state, the same rule the encoder follows: a fresh clone has no fitted
    artefacts and the API must still start and refuse honestly.
    """
    base = Path(root)
    if not base.is_dir():
        return None
    candidates = sorted(base.glob("*.linguistic.json"), key=lambda p: (-p.stat().st_mtime, p.name))
    return LinguisticModel.load(candidates[0]) if candidates else None
