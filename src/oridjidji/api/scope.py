"""Which part of a text an estimate is about.

The application accepts up to 5,000 words, but the encoder reads only its opening
`max_tokens` and the statistical detectors only their prefix window. The estimate is
therefore about the beginning of a long text, and the UI says so twice: before analysis,
with approximate word counts derived from the configured windows, and with the result,
with exact counts from the tokens the detectors actually read.
"""

from __future__ import annotations

from collections.abc import Iterable

from oridjidji.api.schemas import AnalysedScope
from oridjidji.i18n import get_message
from oridjidji.preprocessing.segment import tokenize_words
from oridjidji.schemas import AgentResult, AgentStatus

ENCODER_NAME = "encoder"

# Tokens per editor word, measured on the 2,328 texts of the final set with the
# shipped tokenizers and the editor's own word counter: BERTić 1.55, Gemma-2 2.35. They
# only turn a token window into the approximate word count promised before analysis; the
# result reports exact counts.
ENCODER_TOKENS_PER_WORD = 1.55
STATISTICAL_TOKENS_PER_WORD = 2.35
_SPECIAL_TOKENS = 2  # [CLS] and [SEP] take two places in the encoder window


def _approx_words(tokens: int, tokens_per_word: float) -> int:
    """Round to the nearest 50 words, floor 50: a promise, not a measurement."""
    return max(50, int(round(tokens / tokens_per_word / 50.0)) * 50)


def editor_scope_messages(encoder_tokens: int, statistical_policy: str) -> dict[str, str]:
    """The notice shown under the editor, per mode, before any text is analysed."""
    encoder_words = _approx_words(encoder_tokens - _SPECIAL_TOKENS, ENCODER_TOKENS_PER_WORD)
    basic = get_message(
        "ui_editor_scope_basic", encoder_tokens=encoder_tokens, encoder_words=encoder_words
    )
    if statistical_policy.startswith("prefix"):
        statistical_tokens = int(statistical_policy.removeprefix("prefix"))
        advanced = get_message(
            "ui_editor_scope_advanced",
            encoder_words=encoder_words,
            statistical_tokens=statistical_tokens,
            statistical_words=_approx_words(statistical_tokens, STATISTICAL_TOKENS_PER_WORD),
        )
    else:
        advanced = get_message("ui_editor_scope_advanced_sampled", encoder_words=encoder_words)
    return {"osnovni": basic, "napredni": advanced, "encoderWords": str(encoder_words)}


def _words_read(agent: AgentResult, text: str, n_words: int) -> int | None:
    data = agent.token_data or {}
    total, scored = data.get("total_tokens"), data.get("scored_tokens")
    if isinstance(total, int) and isinstance(scored, int) and scored >= total:
        return n_words
    chars = data.get("scored_chars")
    if not isinstance(chars, int):
        return None
    return min(len(tokenize_words(text[:chars])), n_words)


def analysed_scope(
    text: str, n_words: int, agents: Iterable[AgentResult], used_names: set[str]
) -> AnalysedScope:
    """Exact words read by the detectors that produced the estimate, from their coverage."""
    encoder_words: int | None = None
    statistical: list[int | None] = []
    for agent in agents:
        if agent.name not in used_names or agent.status is not AgentStatus.OK:
            continue
        read = _words_read(agent, text, n_words)
        if agent.name == ENCODER_NAME:
            encoder_words = read
        elif agent.token_data and "window_policy" in agent.token_data:
            statistical.append(read)
    known = [words for words in statistical if words is not None]
    statistical_words = max(known) if statistical and len(known) == len(statistical) else None
    every = [encoder_words, *statistical]
    whole = all(value == n_words for value in every)
    return AnalysedScope(
        n_words=n_words,
        encoder_words=encoder_words,
        statistical_words=statistical_words,
        statistical_used=bool(statistical),
        whole_text=whole,
    )


def scope_message(scope: AnalysedScope) -> str:
    """One Croatian sentence for the result screen, the copied report and the history."""
    if scope.whole_text:
        return get_message("ui_scope_whole", n_words=scope.n_words)
    if scope.encoder_words is None:
        return get_message("ui_scope_unknown", n_words=scope.n_words)
    if not scope.statistical_used:
        return get_message(
            "ui_scope_basic", encoder_words=scope.encoder_words, n_words=scope.n_words
        )
    if scope.statistical_words is None:
        key = "ui_scope_advanced_sampled"
    elif scope.statistical_words >= scope.n_words:
        key = "ui_scope_advanced_statistical_whole"
    else:
        key = "ui_scope_advanced"
    return get_message(
        key,
        encoder_words=scope.encoder_words,
        statistical_words=scope.statistical_words,
        n_words=scope.n_words,
    )
