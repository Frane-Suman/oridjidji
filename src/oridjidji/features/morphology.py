"""Morphosyntactic features on a CLASSLA-Stanza parse (family ``morphology``, the specification).

Requires the optional extra: ``uv sync --extra morphology`` (classla pulls in
torch, ~2.5 GB). Excluded from the <500 ms fast path by design; parses are
disk-cached by sha256(text)+model version. Unit tests build ``ParsedDocument``
fixtures directly and never import classla.

Complexity-budget note: this file exceeds 300 lines because each of
the eight features carries its mandated what/Why/Failure-mode docstring —
the bulk is that prose plus the parse dataclasses, not logic. Splitting the family
would scatter one coherent fixture contract across files.

COMPLEXITY BUDGET (426 lines). The CLASSLA boundary. Length is the price of
keeping a heavy, optional dependency behind one typed adapter: parse caching, graceful
absence, and one function per morphosyntactic feature. Splitting it would put half the
adapter on each side of an import that may not resolve.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import fmean
from typing import Any

from oridjidji.features.base import FeatureContext, feature
from oridjidji.schemas import Document
from oridjidji.storage.cache import DiskCache

_MODEL_VERSION = "classla-2.2.1-hr-standard"
_SUBORDINATE_DEPRELS = frozenset({"acl", "advcl", "ccomp", "xcomp", "csubj"})
# Conservative enclitic list (unambiguous clitic forms only; stressed-homograph
# pronouns like *mi/ti* are deliberately excluded).
_ENCLITICS = frozenset(
    {
        "je",
        "sam",
        "si",
        "smo",
        "ste",
        "su",
        "ću",
        "ćeš",
        "će",
        "ćemo",
        "ćete",
        "bih",
        "bi",
        "bismo",
        "biste",
        "ga",
        "mu",
        "ih",
        "im",
        "se",
        "li",
    }
)


@dataclass(frozen=True)
class ParsedToken:
    form: str
    lemma: str
    upos: str
    feats: dict[str, str]
    head: int  # 1-based index within the sentence; 0 = root
    deprel: str


@dataclass(frozen=True)
class ParsedSentence:
    tokens: list[ParsedToken]


@dataclass(frozen=True)
class ParsedDocument:
    sentences: list[ParsedSentence]
    model_version: str = _MODEL_VERSION


def parsed_to_json(parsed: ParsedDocument) -> str:
    return json.dumps(
        {
            "model_version": parsed.model_version,
            "sentences": [
                [[t.form, t.lemma, t.upos, t.feats, t.head, t.deprel] for t in sentence.tokens]
                for sentence in parsed.sentences
            ],
        },
        ensure_ascii=False,
    )


def parsed_from_json(payload: str) -> ParsedDocument:
    data = json.loads(payload)
    return ParsedDocument(
        sentences=[
            ParsedSentence(
                tokens=[
                    ParsedToken(form=f, lemma=lem, upos=u, feats=dict(fe), head=int(h), deprel=d)
                    for f, lem, u, fe, h, d in sentence
                ]
            )
            for sentence in data["sentences"]
        ],
        model_version=str(data["model_version"]),
    )


def cache_key(text: str) -> str:
    return hashlib.sha256((_MODEL_VERSION + "\x00" + text).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def _default_cache() -> DiskCache:
    return DiskCache(Path("data") / "cache" / "classla")


@lru_cache(maxsize=1)
def _pipeline() -> Any:
    try:
        import classla
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "Morphology features need CLASSLA: install with `uv sync --extra morphology` "
            "and run `uv run python -c \"import classla; classla.download('hr')\"`."
        ) from exc
    return classla.Pipeline("hr", processors="tokenize,pos,lemma,depparse")


def _parse_feats(feats: str | None) -> dict[str, str]:
    if not feats:
        return {}
    out: dict[str, str] = {}
    for item in feats.split("|"):
        if "=" in item:
            key, value = item.split("=", 1)
            out[key] = value
    return out


def parse_document(text: str, *, cache: DiskCache | None = None) -> ParsedDocument:
    """CLASSLA parse with a disk cache; the cache path never imports classla."""
    store = cache if cache is not None else _default_cache()
    key = cache_key(text)
    hit = store.get(key)
    if hit is not None:
        return parsed_from_json(hit)
    doc = _pipeline()(text)  # pragma: no cover - exercised only with the extra
    sentences = [
        ParsedSentence(
            tokens=[
                ParsedToken(
                    form=word.text,
                    lemma=word.lemma or word.text,
                    upos=word.upos or "X",
                    feats=_parse_feats(word.feats),
                    head=int(word.head or 0),
                    deprel=(word.deprel or "dep"),
                )
                for word in sentence.words
            ]
        )
        for sentence in doc.sentences
    ]
    parsed = ParsedDocument(sentences=sentences)
    store.set(key, parsed_to_json(parsed))
    return parsed


# --- helpers --------------------------------------------------------------


def _require_parse(ctx: FeatureContext) -> ParsedDocument:
    if ctx.parsed is None:
        raise RuntimeError(
            "Morphology feature called without a parse (extract with morphology=True)."
        )
    return ctx.parsed


def _entropy_bits(counts: Counter[Any]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def _base_deprel(deprel: str) -> str:
    return deprel.split(":", 1)[0]


def _non_punct(sentence: ParsedSentence) -> list[ParsedToken]:
    return [t for t in sentence.tokens if t.upos != "PUNCT"]


# --- features -------------------------------------------------------------


@feature("morphology")
def pos_trigram_entropy(doc: Document, ctx: FeatureContext) -> float:
    """Shannon entropy (bits) of the UPOS-trigram distribution.

    Why: syntactic-pattern variety — LLM output repeats a narrower set of POS
    patterns than human prose of the same register.
    Failure mode: short texts yield few trigrams and unstable entropy.
    """
    parsed = _require_parse(ctx)
    trigrams: Counter[tuple[str, str, str]] = Counter()
    for sentence in parsed.sentences:
        upos = [t.upos for t in sentence.tokens]
        for i in range(len(upos) - 2):
            trigrams[(upos[i], upos[i + 1], upos[i + 2])] += 1
    return _entropy_bits(trigrams)


@feature("morphology")
def case_entropy(doc: Document, ctx: FeatureContext) -> float:
    """Shannon entropy (bits) of the grammatical-case distribution.

    Why: Croatian's seven cases spread differently in translated/generated text
    (nominative-heavy) than in fluent native prose.
    Failure mode: depends on tagger feats quality; headline-style text is
    legitimately nominative-heavy.
    """
    parsed = _require_parse(ctx)
    cases: Counter[str] = Counter(
        t.feats["Case"] for s in parsed.sentences for t in s.tokens if "Case" in t.feats
    )
    return _entropy_bits(cases)


@feature("morphology")
def mean_dependency_distance(doc: Document, ctx: FeatureContext) -> float:
    """Mean linear distance |i − head(i)| over non-punctuation, non-root tokens.

    Why: a robust syntactic-complexity measure; generated Croatian tends toward
    shorter, flatter dependencies than edited human prose.
    Failure mode: parser errors propagate directly; list-like text skews low.
    """
    parsed = _require_parse(ctx)
    distances = [
        abs(i - t.head)
        for s in parsed.sentences
        for i, t in enumerate(s.tokens, start=1)
        if t.head > 0 and t.upos != "PUNCT"
    ]
    return fmean(distances) if distances else 0.0


@feature("morphology")
def tree_depth_mean(doc: Document, ctx: FeatureContext) -> float:
    """Mean maximum dependency-tree depth per sentence (root = depth 1).

    Why: embedding depth tracks clause complexity; models under-embed relative
    to matched human writing.
    Failure mode: correlates with sentence length — read together with it.
    """
    parsed = _require_parse(ctx)
    depths: list[int] = []
    for sentence in parsed.sentences:
        if not sentence.tokens:
            continue
        max_depth = 1
        for index in range(1, len(sentence.tokens) + 1):
            depth, current, seen = 1, index, 0
            while sentence.tokens[current - 1].head > 0 and seen <= len(sentence.tokens):
                current = sentence.tokens[current - 1].head
                depth += 1
                seen += 1
            max_depth = max(max_depth, depth)
        depths.append(max_depth)
    return fmean(depths) if depths else 0.0


@feature("morphology")
def subordination_rate(doc: Document, ctx: FeatureContext) -> float:
    """Share of non-punctuation tokens heading subordinate clauses (acl/advcl/ccomp/xcomp/csubj).

    Why: subordination density separates registers and authors; generator prose
    leans on coordination instead.
    Failure mode: parser deprel quality; low absolute counts in short texts.
    """
    parsed = _require_parse(ctx)
    tokens = [t for s in parsed.sentences for t in _non_punct(s)]
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if _base_deprel(t.deprel) in _SUBORDINATE_DEPRELS) / len(tokens)


@feature("morphology")
def wackernagel_conformity(doc: Document, ctx: FeatureContext) -> float:
    """Share of enclitics sitting in second (Wackernagel) position — approximated
    as second non-punctuation token of the sentence.

    Why: models conform near-perfectly to the schoolbook rule; human informal
    writing dislocates clitics (which is grammatical but colloquial).
    Failure mode: the approximation ignores legitimate first-constituent phrases
    longer than one word, undercounting conformity; returns 1.0 (vacuous) when no
    enclitic occurs.
    """
    parsed = _require_parse(ctx)
    total = 0
    conforming = 0
    for sentence in parsed.sentences:
        tokens = _non_punct(sentence)
        for position, token in enumerate(tokens, start=1):
            if token.form.lower() in _ENCLITICS and token.upos in {"AUX", "PRON", "PART"}:
                total += 1
                if position == 2:
                    conforming += 1
    return conforming / total if total else 1.0


@feature("morphology")
def passive_rate(doc: Document, ctx: FeatureContext) -> float:
    """Share of sentences containing a passive marker (aux:pass / Voice=Pass).

    Why: passive/impersonal constructions mark formal register where generated
    Croatian clusters.
    Failure mode: Croatian often prefers the *se*-impersonal, which taggers label
    inconsistently — treat as a coarse signal.
    """
    parsed = _require_parse(ctx)
    if not parsed.sentences:
        return 0.0
    passive_sentences = 0
    for sentence in parsed.sentences:
        for token in sentence.tokens:
            if "pass" in token.deprel or token.feats.get("Voice") == "Pass":
                passive_sentences += 1
                break
    return passive_sentences / len(parsed.sentences)


@feature("morphology")
def tense_entropy(doc: Document, ctx: FeatureContext) -> float:
    """Shannon entropy (bits) of the verb-tense distribution.

    Why: humans shift tenses mid-narrative more than generators, which lock into
    one narrative tense.
    Failure mode: short or purely descriptive texts legitimately use one tense.
    """
    parsed = _require_parse(ctx)
    tenses: Counter[str] = Counter(
        t.feats["Tense"]
        for s in parsed.sentences
        for t in s.tokens
        if t.upos in {"VERB", "AUX"} and "Tense" in t.feats
    )
    return _entropy_bits(tenses)


@feature("morphology")
def aspect_ratio(doc: Document, ctx: FeatureContext) -> float:
    """Perfective share of aspect-marked verbs (Appendix A).

    Why: Croatian encodes aspect lexically, in perfective/imperfective verb pairs a
    learner or a translation model chooses badly. English-centric generators tend
    toward the perfective because English past tense maps onto it most readily, so
    the balance shifts away from native narrative prose.
    Failure mode: depends entirely on tagger `Aspect` feats, which CLASSLA leaves off
    many biaspectual and borrowed verbs; returns 0.0 when nothing is marked, which is
    an absence of measurement rather than an all-imperfective text.
    """
    parsed = _require_parse(ctx)
    aspects: Counter[str] = Counter(
        t.feats["Aspect"] for s in parsed.sentences for t in s.tokens if "Aspect" in t.feats
    )
    total = sum(aspects.values())
    if total == 0:
        return 0.0
    return aspects["Perf"] / total


@feature("morphology")
def word_order_entropy(doc: Document, ctx: FeatureContext) -> float:
    """Shannon entropy (bits) over subject/verb/object orderings (Appendix A).

    Why: Croatian word order is grammatically free and used pragmatically — native
    prose moves the topic around, while generated text tends to settle into a narrow,
    English-like SVO groove. Entropy captures that flattening without asserting any
    single order is correct.
    Failure mode: only sentences with an explicit subject *and* object contribute, so
    short or verb-only texts yield few observations and unstable entropy; 0.0 means
    either one order throughout or too little evidence, and the two are not
    distinguishable from this number alone.
    """
    parsed = _require_parse(ctx)
    orders: Counter[str] = Counter()
    for sentence in parsed.sentences:
        positions: dict[str, int] = {}
        for index, tok in enumerate(sentence.tokens):
            role = {"nsubj": "S", "obj": "O"}.get(tok.deprel)
            if role is None and tok.deprel == "root" and tok.upos in {"VERB", "AUX"}:
                role = "V"
            if role is not None and role not in positions:
                positions[role] = index
        if len(positions) == 3:
            orders["".join(sorted(positions, key=lambda r: positions[r]))] += 1
    return _entropy_bits(orders)


@feature("morphology")
def lemma_repetition_rate(doc: Document, ctx: FeatureContext) -> float:
    """Share of content-word tokens whose lemma already appeared (Appendix A).

    Why: this is repetition measured *under* inflection, which a surface type-token
    ratio cannot see — in a richly inflected language the same lemma in three cases
    looks like three types. Models reuse concepts more than they reuse word forms,
    so the gap between this and `mtld` is itself the signal.
    Failure mode: legitimately high in a text about one narrow subject, so it is a
    register-sensitive feature and must never be read alone; 0.0 when the text has no
    content words at all.
    """
    parsed = _require_parse(ctx)
    lemmas = [
        t.lemma.lower()
        for s in parsed.sentences
        for t in s.tokens
        if t.upos in {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
    ]
    if not lemmas:
        return 0.0
    return (len(lemmas) - len(set(lemmas))) / len(lemmas)
