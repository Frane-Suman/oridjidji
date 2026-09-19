"""Corpus-relative norms, so a rate can be read as "unusual" rather than just large.

Appendix A asks every rate feature to be reported as a **raw rate and a z-score**.
An earlier decision deferred the second half to "M2 corpus statistics" because there was no corpus to
take statistics from; M2 shipped, so the deferral has expired.

Why the pair matters: `nebi_rate = 0.4` means nothing on its own. Against a news corpus
where the mean is 0.05, it is eight standard deviations out and close to proof a human
typed it; against a corpus where the mean is 0.5, it is unremarkable. The raw rate is
what the feature measured, the z-score is what it is worth — and the fusion layer and
the SHAP panel both need the second.

The norms are a **versioned artefact**, the same discipline the specification applies to the
Binoculars thresholds: a z-score is uninterpretable without knowing which corpus
produced it, so the file records the corpus hash, the record count and the date beside
the statistics. Norms are computed **per domain**, because a rate that is ordinary in
news can be unusual in academic prose and averaging the two would hide both.

Two deliberate refusals:

* A constant feature yields **0.0**, never `inf` or `NaN`. Several rate features are
  legitimately zero across a whole corpus (`zero_width_per_kchar` on clean text), and
  `extract_features` rejects non-finite values because one NaN silently poisons fusion.
* A missing norms file **raises**. Returning 0.0 would report "exactly average" for a
  feature nobody measured, which is worse than failing.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# Below this many non-zero observations, a rate feature's mean and sigma are dominated by
# whether the sample happened to catch an instance at all. Measured case: the ije/je error
# fires on 14 of 19,611 news records (0.071%), so a 3,000-record sample expects 2.1 hits
# and draws zero about 12% of the time — at which point the feature reports as "constant"
# and every z-score against it collapses to 0.0. That is sampling noise wearing the costume
# of a measurement, and it lands on exactly the human-error features that matter most.
MIN_NONZERO_OBSERVATIONS = 30


@dataclass(frozen=True)
class FeatureNorms:
    """Mean and sample standard deviation of one feature over one corpus."""

    mean: float
    stdev: float
    n: int
    n_nonzero: int = -1
    """How many observations were non-zero. ``-1`` means an older artefact did not record
    it — treated as unknown rather than as zero, so a stale file cannot masquerade as a
    well-sampled one."""

    @property
    def well_sampled(self) -> bool:
        """False when too few non-zero observations back this estimate.

        A constant feature is only *evidence* of invariance when the sample was large
        enough to have caught variation. Distinguishing the two is the whole point:
        `zero_width_per_kchar` is genuinely zero on clean text, while
        `ije_je_error_rate` is merely rarer than this sample can see.
        """
        if self.n_nonzero < 0:
            return False
        return self.n_nonzero >= MIN_NONZERO_OBSERVATIONS


@dataclass(frozen=True)
class NormsArtefact:
    """A norms file: the statistics plus what they were measured on."""

    corpus_hash: str
    n_records: int
    created_at: str
    features: dict[str, FeatureNorms]


def zscore(value: float, norms: FeatureNorms) -> float:
    """Standard scores away from the corpus mean; 0.0 when the feature never varies."""
    if norms.stdev <= 0.0:
        return 0.0
    return (value - norms.mean) / norms.stdev


def corpus_norms(values: Mapping[str, Sequence[float]]) -> dict[str, FeatureNorms]:
    """Mean and *sample* standard deviation per feature.

    Sample rather than population (n-1): these corpora are samples drawn from Croatian
    web and academic writing, not the populations themselves, and the whole project is
    built on that distinction.
    """
    out: dict[str, FeatureNorms] = {}
    for name, series in values.items():
        if not series:
            continue
        n = len(series)
        mean = statistics.fmean(series)
        stdev = statistics.stdev(series) if n > 1 else 0.0
        out[name] = FeatureNorms(
            mean=mean,
            stdev=stdev,
            n=n,
            n_nonzero=sum(1 for value in series if value != 0.0),
        )
    return out


def load_norms(path: str | Path) -> NormsArtefact:
    """Read a norms artefact. Raises if absent — see the module docstring."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(
            f"No feature norms at {target}. Build them with "
            "`oridjidji data build-norms`; z-scores cannot be faked with zeros."
        )
    payload = json.loads(target.read_text(encoding="utf-8"))
    features = {
        name: FeatureNorms(
            mean=float(v["mean"]),
            stdev=float(v["stdev"]),
            n=int(v["n"]),
            n_nonzero=int(v.get("n_nonzero", -1)),
        )
        for name, v in payload.get("features", {}).items()
    }
    return NormsArtefact(
        corpus_hash=str(payload.get("corpus_hash", "")),
        n_records=int(payload.get("n_records", 0)),
        created_at=str(payload.get("created_at", "")),
        features=features,
    )


def write_norms(artefact: NormsArtefact, path: str | Path) -> None:
    """Write a norms artefact as JSON, sorted so two builds diff cleanly."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "corpus_hash": artefact.corpus_hash,
        "n_records": artefact.n_records,
        "created_at": artefact.created_at,
        "features": {
            name: {"mean": n.mean, "stdev": n.stdev, "n": n.n, "n_nonzero": n.n_nonzero}
            for name, n in sorted(artefact.features.items())
        },
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def undersampled_features(artefact: NormsArtefact) -> list[str]:
    """Features whose norms rest on too few non-zero observations to be trusted.

    Reported rather than silently used: a z-score of 0.0 against such a feature says
    "the sample never saw one", not "this text is average".
    """
    return sorted(name for name, norm in artefact.features.items() if not norm.well_sampled)


def zscore_vector(
    vector: Mapping[str, float], artefact: NormsArtefact
) -> tuple[dict[str, float], list[str]]:
    """Z-score a feature vector, returning the scores and the features with no norm.

    Deliberately a **separate** vector rather than a transformation of the input:
    Appendix A asks for the raw rate *and* the z-score, and the raw rate is what the
    feature measured. Overwriting it would lose the half a reader can check by hand.

    A feature with no entry in the artefact is **omitted and named**, never scored as
    0.0. Silently returning "exactly average" for a feature added after the norms were
    built is the failure this whole module exists to prevent, and it would be invisible
    at exactly the moment the feature set grows.
    """
    scored: dict[str, float] = {}
    missing: list[str] = []
    for name, value in vector.items():
        norm = artefact.features.get(name)
        if norm is None:
            missing.append(name)
            continue
        scored[name] = zscore(value, norm)
    return scored, sorted(missing)


def norms_path_for(domain: str, directory: str | Path = "data/processed") -> Path:
    """Where the norms for one domain live.

    Norms are per-domain because a rate that is ordinary in news can be unusual in
    academic prose; averaging the two would hide both.
    """
    return Path(directory) / f"feature_norms_{domain}.json"


def build_norms_artefact(
    texts: Sequence[str],
    *,
    lexicon: object,
    content_hashes: Sequence[str],
    sample: int = 3000,
    seed: int = 42,
) -> tuple[NormsArtefact, int]:
    """Measure ``texts`` and return the artefact plus how many records were measured.

    Lives here rather than in the CLI so it can be tested: the sampling rule and the
    corpus hash are the two things that decide whether a norms file is reproducible, and
    neither was reachable from a test while they sat inside a typer command.

    The sample is drawn from a **sorted** order, never Parquet row order — row order is
    an artefact of how the corpus was built, so norms that depended on it would change
    when the build changed without any of the statistics really moving.
    """
    import random

    from oridjidji.features.base import extract_features
    from oridjidji.preprocessing import preprocess

    if len(texts) != len(content_hashes):
        raise ValueError(
            f"texts and content_hashes must align: {len(texts)} != {len(content_hashes)}"
        )

    pairs = sorted(zip(content_hashes, texts, strict=True))
    if sample and len(pairs) > sample:
        pairs = random.Random(seed).sample(pairs, sample)

    series: dict[str, list[float]] = {}
    for _, text in pairs:
        document = preprocess(text, lexicon=lexicon, check_language=False)  # type: ignore[arg-type]
        for name, value in extract_features(document, lexicon=lexicon).items():  # type: ignore[arg-type]
            series.setdefault(name, []).append(value)

    # Hash the measured records, not the source file: two builds over the same records
    # must give the same hash so an artefact can be traced to what produced it.
    from oridjidji.artefacts import content_hash as hash_text

    digest = hash_text("".join(sorted(h for h, _ in pairs)))
    artefact = NormsArtefact(
        corpus_hash=digest,
        n_records=len(pairs),
        created_at=datetime.now(UTC).isoformat(),
        features=corpus_norms(series),
    )
    return artefact, len(pairs)
