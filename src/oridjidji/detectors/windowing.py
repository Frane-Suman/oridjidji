"""Deterministic token coverage for the Advanced statistical backbone.

Windowing chooses which causal-LM target positions are measured.  The detectors still
aggregate the resulting per-token sufficient statistics themselves; averaging completed
window scores would give short windows the same weight as long ones and is therefore never
done here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from oridjidji.detectors.backbone import ReducedScores

WindowPolicy = Literal["prefix160", "prefix512", "coverage1024"]


def word_starts_from_offsets(text: str, offsets: list[tuple[int, int]]) -> NDArray[np.bool_]:
    """Mark lexical word starts from fast-tokenizer character offsets.

    A new word starts at the first alphanumeric character covered by a token when the
    preceding character is not alphanumeric. Byte-level BPE tokens often include their
    leading space in the offset, while SentencePiece offsets often begin on the letter;
    searching inside the span handles both without model-specific marker characters.
    """
    flags: list[bool] = []
    for start, end in offsets:
        if not 0 <= start < end <= len(text):
            flags.append(False)
            continue
        lexical_start = next(
            (position for position in range(start, end) if text[position].isalnum()),
            None,
        )
        flags.append(
            lexical_start is not None
            and (lexical_start == 0 or not text[lexical_start - 1].isalnum())
        )
    return np.asarray(flags, dtype=bool)


@dataclass(frozen=True)
class TokenWindow:
    """Half-open target span plus preceding context used only to predict it."""

    context_start: int
    target_start: int
    target_stop: int

    def __post_init__(self) -> None:
        if not 0 <= self.context_start <= self.target_start < self.target_stop:
            raise ValueError("invalid token window")

    @property
    def context_tokens(self) -> int:
        return self.target_start - self.context_start


def plan_windows(
    total_tokens: int,
    policy: WindowPolicy,
    *,
    window_tokens: int = 512,
    left_context: int = 32,
) -> tuple[TokenWindow, ...]:
    """Select unique target positions under one registered coverage policy.

    ``total_tokens`` counts scoreable targets, after any BOS token.  For documents beyond
    1,024 targets the bounded policy represents the opening and ending equally; it never
    silently presents that sample as full coverage.
    """
    if total_tokens < 1:
        return ()
    if window_tokens < 1 or left_context < 0:
        raise ValueError("window_tokens must be positive and left_context non-negative")
    if policy.startswith("prefix"):
        limit = int(policy.removeprefix("prefix"))
        return (TokenWindow(0, 0, min(total_tokens, limit)),)
    if policy != "coverage1024":
        raise ValueError(f"unknown statistical window policy: {policy}")

    budget = 2 * window_tokens
    if total_tokens <= window_tokens:
        return (TokenWindow(0, 0, total_tokens),)
    if total_tokens <= budget:
        second_start = window_tokens
        return (
            TokenWindow(0, 0, window_tokens),
            TokenWindow(max(0, second_start - left_context), second_start, total_tokens),
        )
    final_start = total_tokens - window_tokens
    return (
        TokenWindow(0, 0, window_tokens),
        TokenWindow(max(0, final_start - left_context), final_start, total_tokens),
    )


def plan_word_aligned_windows(
    total_tokens: int,
    word_starts: NDArray[np.bool_],
    policy: WindowPolicy,
    *,
    window_tokens: int = 512,
    left_context: int = 32,
) -> tuple[TokenWindow, ...]:
    """Align coverage targets to word starts while context remains uncounted."""
    if policy != "coverage1024":
        return plan_windows(
            total_tokens,
            policy,
            window_tokens=window_tokens,
            left_context=left_context,
        )
    if len(word_starts) != total_tokens:
        raise ValueError("word-start mask must cover every scoreable document token")
    if total_tokens <= window_tokens:
        return (TokenWindow(0, 0, total_tokens),)
    boundaries = [int(index) for index in np.flatnonzero(word_starts)] + [total_tokens]
    first_stop = max((value for value in boundaries if value <= window_tokens), default=0)
    if first_stop <= 0:
        first_stop = min(window_tokens, total_tokens)
    if total_tokens <= 2 * window_tokens:
        second_start = first_stop
    else:
        floor = total_tokens - window_tokens
        second_start = next((value for value in boundaries if value >= floor), total_tokens)
        second_start = max(first_stop, second_start)
    if second_start >= total_tokens:
        return (TokenWindow(0, 0, first_stop),)
    return (
        TokenWindow(0, 0, first_stop),
        TokenWindow(max(0, second_start - left_context), second_start, total_tokens),
    )


def _concat(parts: list[np.ndarray], dtype: np.dtype[np.generic]) -> np.ndarray:
    return np.asarray(np.concatenate(parts), dtype=dtype)


def combine_reduced(
    parts: list[ReducedScores], *, total_tokens: int, policy: WindowPolicy
) -> ReducedScores:
    """Concatenate unique target reductions before either detector computes its scalar."""
    if not parts:
        raise ValueError("at least one reduced-score window is required")
    optional = (
        "cross_entropies",
        "word_starts",
        "max_probs",
        "pit",
        "observer_unmapped_mass",
        "performer_unmapped_mass",
    )
    for name in optional:
        present = [getattr(part, name) is not None for part in parts]
        if any(present) and not all(present):
            raise ValueError(f"window field {name} is present only in part of the document")

    def optional_concat(name: str, dtype: np.dtype[np.generic]) -> np.ndarray | None:
        values = [getattr(part, name) for part in parts]
        if values[0] is None:
            return None
        return _concat([np.asarray(value) for value in values if value is not None], dtype)

    token_ids = _concat([part.token_ids for part in parts], np.dtype(np.int64))
    return ReducedScores(
        token_ids=token_ids,
        realised_logprobs=_concat([part.realised_logprobs for part in parts], np.dtype(np.float32)),
        expected_logprobs=_concat([part.expected_logprobs for part in parts], np.dtype(np.float32)),
        logprob_variances=_concat([part.logprob_variances for part in parts], np.dtype(np.float32)),
        cross_entropies=optional_concat("cross_entropies", np.dtype(np.float32)),
        word_starts=optional_concat("word_starts", np.dtype(np.bool_)),
        max_probs=optional_concat("max_probs", np.dtype(np.float32)),
        pit=optional_concat("pit", np.dtype(np.float32)),
        observer_unmapped_mass=optional_concat("observer_unmapped_mass", np.dtype(np.float32)),
        performer_unmapped_mass=optional_concat("performer_unmapped_mass", np.dtype(np.float32)),
        total_tokens=total_tokens,
        scored_tokens=len(token_ids),
        window_policy=policy,
    )


def slice_reduced_words(scores: ReducedScores, *, start_word: int, end_word: int) -> ReducedScores:
    """Select one word interval from request-scoped prefix evidence.

    V0/prefix512 is contiguous from the document opening, so cumulative word-start flags
    recover the original word index without retaining raw text in the evidence cache.  A
    segment outside that prefix returns no rows and is therefore rendered as insufficient
    signal.  The discontinuous coverage1024 policy is refused because its second window
    would need explicit original token positions to avoid relabelling the skipped middle.
    """
    if not 0 <= start_word < end_word:
        raise ValueError("invalid word interval")
    if scores.window_policy not in {"prefix160", "prefix512"}:
        raise ValueError("word slicing currently requires contiguous prefix evidence")
    if scores.word_starts is None:
        raise ValueError("word slicing needs tokenizer-derived word boundaries")
    word_indices = np.cumsum(scores.word_starts.astype(np.int64)) - 1
    keep = (word_indices >= start_word) & (word_indices < end_word)
    indices = np.flatnonzero(keep)
    if not len(indices):
        raise ValueError("segment lies outside statistical evidence coverage")

    def optional(value: np.ndarray | None) -> np.ndarray | None:
        return None if value is None else value[indices]

    return ReducedScores(
        token_ids=scores.token_ids[indices],
        realised_logprobs=scores.realised_logprobs[indices],
        expected_logprobs=scores.expected_logprobs[indices],
        logprob_variances=scores.logprob_variances[indices],
        cross_entropies=optional(scores.cross_entropies),
        word_starts=optional(scores.word_starts),
        max_probs=optional(scores.max_probs),
        pit=optional(scores.pit),
        observer_unmapped_mass=optional(scores.observer_unmapped_mass),
        performer_unmapped_mass=optional(scores.performer_unmapped_mass),
        total_tokens=len(indices),
        scored_tokens=len(indices),
        window_policy="prefix512",
    )
