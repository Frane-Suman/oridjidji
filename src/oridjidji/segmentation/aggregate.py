"""Folding overlapping window scores into a per-span map.

The specification names **segment aggregation** as thesis-core: implemented explicitly, with the choice
argued rather than imported. This is that module.

**Median, not mean, and the reason is the failure mode.** Windows overlap ~50%, so every
interior word is covered by at least two. A single window can be anomalous — it may straddle
a quotation, a heading, or the authorship boundary itself, which is precisely the region the
map is meant to describe. A mean lets one such window drag its whole span; a median needs
half the covering windows to agree before the span moves. The specification offers "mean or median;
choose and justify", and the boundary case is the justification: at a genuine change point,
half the covering windows are mostly-human and half mostly-machine, and the median resolves
to whichever side actually dominates that word rather than to their average — which would be
a value neither author produced.

**The reliability rules are enforced here, not in the UI.** the specification calls them mandatory:

1. A word covered by **no** scoreable window has **no score** — it renders as *insufficient
   signal*, never as a low one. `SpanScore.score is None` is the only representation of that,
   so a template cannot accidentally print 0.0.
2. **A segment never produces a verdict.** There is no `verdict` on `SpanScore` and no
   function here returns one. The document verdict is the headline and the map is supporting
   evidence — so the type system, not a convention, is what prevents a caller from reading a
   span as an accusation.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from oridjidji.segmentation.windows import Window


@dataclass(frozen=True)
class SpanScore:
    """One contiguous run of words that share a score, or share having none.

    ``score is None`` means *insufficient signal* — no scoreable window covered these words.
    It is not a low score, and nothing downstream may treat it as one.
    """

    start_word: int
    end_word: int
    start_char: int
    end_char: int
    score: float | None
    n_windows: int

    @property
    def n_words(self) -> int:
        return self.end_word - self.start_word

    @property
    def insufficient(self) -> bool:
        return self.score is None


def aggregate_windows(
    windows: list[Window],
    scores: list[float | None],
    *,
    total_words: int,
) -> list[SpanScore]:
    """Per-word medians folded into contiguous spans.

    ``scores`` runs parallel to ``windows``; ``None`` marks a window the detector could not
    score, which is distinct from a window that scored low. Both leave the words they cover
    to whatever *other* windows say, and to *insufficient signal* if there are none.

    Spans are merged only where the score **and** the covering-window count agree, so a
    reader can see where the evidence thins out rather than seeing one flat block whose edges
    rest on a single window.
    """
    if len(windows) != len(scores):
        raise ValueError(f"{len(windows)} windows against {len(scores)} scores")

    per_word: list[list[float]] = [[] for _ in range(total_words)]
    for window, score in zip(windows, scores, strict=True):
        if score is None:
            continue
        for index in range(window.start_word, min(window.end_word, total_words)):
            per_word[index].append(float(score))

    # Character offsets per word, so a span can be highlighted. Words with no covering window
    # still need offsets: "insufficient signal" is rendered, not skipped.
    bounds = _word_bounds(windows, total_words)

    spans: list[SpanScore] = []
    for index in range(total_words):
        values = per_word[index]
        score = statistics.median(values) if values else None
        count = len(values)
        if spans and _mergeable(spans[-1], score, count):
            previous = spans[-1]
            spans[-1] = SpanScore(
                start_word=previous.start_word,
                end_word=index + 1,
                start_char=previous.start_char,
                end_char=bounds[index][1],
                score=previous.score,
                n_windows=previous.n_windows,
            )
            continue
        start_char, end_char = bounds[index]
        spans.append(
            SpanScore(
                start_word=index,
                end_word=index + 1,
                start_char=start_char,
                end_char=end_char,
                score=score,
                n_windows=count,
            )
        )
    return spans


def _mergeable(previous: SpanScore, score: float | None, count: int) -> bool:
    if previous.n_windows != count:
        return False
    if previous.score is None or score is None:
        return previous.score is None and score is None
    return abs(previous.score - score) < 1e-12


def _word_bounds(windows: list[Window], total_words: int) -> list[tuple[int, int]]:
    """Character span per word index, taken from whichever window covers it.

    Words outside every window still get a placeholder span so the map is total: the UI has
    to render *insufficient signal* somewhere, and a missing entry would render as nothing at
    all — which reads as "we checked and found nothing here" rather than "we could not look".
    """
    bounds: list[tuple[int, int]] = [(0, 0)] * total_words
    seen = [False] * total_words
    for window in windows:
        words = window.text.split()
        cursor = window.start_char
        for offset, word in enumerate(words):
            index = window.start_word + offset
            if index >= total_words or seen[index]:
                continue
            start = window.text.find(word, cursor - window.start_char)
            if start < 0:  # pragma: no cover - the word came from this text
                continue
            absolute = window.start_char + start
            bounds[index] = (absolute, absolute + len(word))
            seen[index] = True
            cursor = absolute + len(word)
    return bounds
