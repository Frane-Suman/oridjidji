"""Feature registry and extraction entry point.

Every feature registers through the ``@feature`` decorator, which *enforces* the
documentation contract: the docstring must state what is measured (first line),
why it should differ between human and AI Croatian (``Why:``), and its known
failure mode (``Failure mode:``) — that text doubles as thesis material and feeds
the UI's explanations.

Fast path = every family except ``morphology``; the morphology family needs a
CLASSLA parse and is opt-in.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from oridjidji.preprocessing.lexicon import Lexicon
from oridjidji.schemas import Document, FeatureInfo

if TYPE_CHECKING:
    from oridjidji.features.morphology import ParsedDocument

FAMILIES = ("orthographic", "human_error", "ai_ism", "lexical", "readability", "morphology")


@dataclass(frozen=True)
class FeatureContext:
    """Shared inputs a feature may need beyond the Document itself."""

    lexicon: Lexicon | None = None
    parsed: ParsedDocument | None = None


FeatureFunc = Callable[[Document, FeatureContext], float]


@dataclass(frozen=True)
class RegisteredFeature:
    info: FeatureInfo
    fn: FeatureFunc


_REGISTRY: dict[str, RegisteredFeature] = {}


def feature(family: str) -> Callable[[FeatureFunc], FeatureFunc]:
    """Register a feature function under its own name, enforcing the specification docs."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown feature family '{family}' (known: {FAMILIES})")

    def decorator(fn: FeatureFunc) -> FeatureFunc:
        name = fn.__name__
        doc = (fn.__doc__ or "").strip()
        if not doc or "Why:" not in doc or "Failure mode:" not in doc:
            raise ValueError(
                f"Feature '{name}' must document what it measures, 'Why:' it separates "
                "human from AI Croatian, and its 'Failure mode:'."
            )
        if name in _REGISTRY:
            raise ValueError(f"Duplicate feature name '{name}'")
        _REGISTRY[name] = RegisteredFeature(
            info=FeatureInfo(name=name, family=family, description=doc.splitlines()[0]),
            fn=fn,
        )
        return fn

    return decorator


def _load_modules(*, morphology: bool = False) -> None:
    # Imports run the @feature decorators; sys.modules makes repeats free.
    from oridjidji.features import aiism, croatian, readability, stylometry  # noqa: F401

    if morphology:
        from oridjidji.features import morphology as _morphology  # noqa: F401


def all_features(*, morphology: bool = False) -> dict[str, FeatureInfo]:
    """Registry metadata for the UI and the meta-tests."""
    _load_modules(morphology=morphology)
    return {
        name: reg.info
        for name, reg in _REGISTRY.items()
        if morphology or reg.info.family != "morphology"
    }


def extract_features(
    doc: Document,
    *,
    lexicon: Lexicon | None = None,
    morphology: bool = False,
    parsed: ParsedDocument | None = None,
    families: frozenset[str] | None = None,
) -> dict[str, float]:
    """Compute the feature vector for ``doc``.

    ``morphology=True`` adds the CLASSLA-backed family; a pre-parsed document may
    be injected (tests, cache replay), otherwise ``parse_document`` runs lazily.
    ``families`` restricts the output to named families, which the artefact probe
    uses to look at surface form alone. Every value is checked finite — a NaN here
    would silently poison the fusion layer downstream.
    """
    _load_modules(morphology=morphology)
    if morphology and parsed is None:
        from oridjidji.features.morphology import parse_document

        parsed = parse_document(doc.normalized)
    ctx = FeatureContext(lexicon=lexicon, parsed=parsed)

    out: dict[str, float] = {}
    for name, reg in _REGISTRY.items():
        if reg.info.family == "morphology" and not morphology:
            continue
        if families is not None and reg.info.family not in families:
            continue
        value = float(reg.fn(doc, ctx))
        if not math.isfinite(value):
            raise ValueError(f"Feature '{name}' produced a non-finite value: {value!r}")
        out[name] = value
    return out
