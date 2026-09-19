"""Sliding windows over a document, for the segment-level map.

**Why the map exists.** The thesis centres on *lightly post-edited* text, and for a half-edited
essay a single "63%" is close to meaningless. The honest answer is *"these three paragraphs
look machine-generated, the rest doesn't"* — the shape SemEval-2024 Task 8 Subtask C
established for change-point detection.

**Deliberately cheap: no new model and no extra training.** A window is scored by the same
document-level pipeline that scores a whole text. Everything here is bookkeeping — where the
windows fall, which are long enough to be scored at all, and which character span each one
speaks for.

**Windows end on sentence boundaries.** A window that ends mid-sentence is asking the detector
about a truncation artefact rather than about authorship, and the specification places the ground-truth
boundary at a sentence end for the same reason.

The one rule that is not bookkeeping: a window shorter than the length gate is **not scored
at all**. The specification requires those to render as *"insufficient signal"*, never as a low or high
score, and the cheapest way to guarantee that downstream is never to produce a number.
"""

from __future__ import annotations

from dataclasses import dataclass

from oridjidji.preprocessing.segment import split_sentences

#: Fraction of a window that the next one repeats. The specification says "~50% overlap": every interior
#: sentence is then covered by at least two windows, so one bad window cannot own a span.
DEFAULT_OVERLAP = 0.5


@dataclass(frozen=True)
class Window:
    """One scoreable span of a document, in characters and in words.

    Character offsets are what the UI highlights; word counts are what the length gate and
    SemEval's boundary metric are expressed in. Both are carried because converting between
    them after the fact is where off-by-one errors live.
    """

    start_char: int
    end_char: int
    start_word: int
    end_word: int
    text: str

    @property
    def n_words(self) -> int:
        return self.end_word - self.start_word

    def covers_word(self, index: int) -> bool:
        """Whether this window speaks for the word at ``index`` (half-open, like a slice)."""
        return self.start_word <= index < self.end_word


def build_windows(text: str, *, min_words: int, overlap: float = DEFAULT_OVERLAP) -> list[Window]:
    """Sentence-aligned sliding windows of at least ``min_words`` words.

    ``min_words`` is the document length gate (the specification, currently 100 and derived from the
    length-ablation curve, not asserted). A window below it is never produced, so no caller
    can accidentally score one.

    **A document shorter than the gate yields no windows at all**, which is the correct
    answer rather than an empty-list bug: there is no span in it the system may speak about.

    The final window is extended backwards rather than emitted short. Emitting a short tail
    would either break the gate or silently drop the end of the document, and dropping the
    end is the worse failure — a machine-written conclusion is exactly where a reader looks.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    sentences = split_sentences(text)
    if not sentences:
        return []

    # Locate every sentence in the original string, so highlighting lands on the real text
    # rather than on a normalised copy of it.
    spans: list[tuple[int, int, int, int]] = []  # start_char, end_char, start_word, end_word
    cursor = 0
    words_so_far = 0
    for sentence in sentences:
        start = text.find(sentence, cursor)
        if start < 0:  # pragma: no cover - split_sentences returns substrings of `text`
            continue
        end = start + len(sentence)
        n_words = len(sentence.split())
        spans.append((start, end, words_so_far, words_so_far + n_words))
        words_so_far += n_words
        cursor = end

    total_words = words_so_far
    if total_words < min_words:
        return []

    windows: list[Window] = []
    first = 0
    while first < len(spans):
        last = first
        while last < len(spans) and spans[last][3] - spans[first][2] < min_words:
            last += 1
        if last >= len(spans):
            # The tail is too short to reach the gate going forwards. Extend the FINAL window
            # backwards from the end instead of dropping the tail: a machine-written
            # conclusion is exactly where a reader looks, and an earlier version broke here
            # and silently ended the map 12 words short of the document.
            last = len(spans) - 1
            start = last
            while start > 0 and spans[last][3] - spans[start][2] < min_words:
                start -= 1
            final = _window(text, spans, start, last)
            if final.n_words >= min_words and (not windows or final != windows[-1]):
                windows.append(final)
            break
        window = _window(text, spans, first, last)
        if window.n_words >= min_words and (not windows or window != windows[-1]):
            windows.append(window)
        if last >= len(spans) - 1:
            break
        first = _advance(spans, first, last, overlap)
    return windows


def _window(text: str, spans: list[tuple[int, int, int, int]], first: int, last: int) -> Window:
    start_char, _, start_word, _ = spans[first]
    _, end_char, _, end_word = spans[last]
    return Window(
        start_char=start_char,
        end_char=end_char,
        start_word=start_word,
        end_word=end_word,
        text=text[start_char:end_char],
    )


def _advance(spans: list[tuple[int, int, int, int]], first: int, last: int, overlap: float) -> int:
    """The next window's first sentence, stepping by ``1 - overlap`` of the current one.

    Always advances by at least one sentence. A step computed as zero — which a high overlap
    on a two-sentence window produces — would loop forever emitting the same span.
    """
    span_words = spans[last][3] - spans[first][2]
    target = spans[first][2] + max(1, int(round(span_words * (1.0 - overlap))))
    index = first + 1
    while index < len(spans) and spans[index][2] < target:
        index += 1
    return max(index, first + 1)
