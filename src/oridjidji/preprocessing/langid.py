"""Language-identification gate — lingua behind a thin typed adapter.

lingua is pinned <2.2 (2.2.0 requires Python ≥3.12; DECISIONS an earlier decision), works fully
offline, and is restricted to the languages that realistically reach this tool:
hr/sr/bs (the continuum we honestly cannot separate), plus sl/en/de as the likely
"wrong paste" languages. Unit tests inject a fake detector; the real one loads
lazily and is process-cached.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Protocol

from oridjidji.i18n import get_message
from oridjidji.schemas import LanguageCheck

_BCS = frozenset({"hr", "sr", "bs"})
# Below this hr-vs-runner-up confidence gap the BCS caveat is always shown.
_BCS_GAP = 0.25


class DetectorLike(Protocol):
    """The single lingua method this module depends on."""

    def compute_language_confidence_values(self, text: str) -> Any: ...


@lru_cache(maxsize=1)
def _default_detector() -> Any:
    from lingua import Language, LanguageDetectorBuilder

    languages = [
        Language.CROATIAN,
        Language.SERBIAN,
        Language.BOSNIAN,
        Language.SLOVENE,
        Language.ENGLISH,
        Language.GERMAN,
    ]
    return LanguageDetectorBuilder.from_languages(*languages).build()


def _iso(language: Any) -> str:
    return str(language.iso_code_639_1.name).lower()


def detect_language(text: str, detector: DetectorLike | None = None) -> LanguageCheck:
    """Classify ``text``; accept the hr/sr/bs continuum, reject everything else.

    Acceptance policy: hr/sr/bs separation is genuinely hard, so any BCS
    top result is accepted and the uncertainty is *said* (``message_hr``) rather
    than hidden behind fake precision. Non-BCS text is rejected with a Croatian
    message.
    """
    det: DetectorLike = detector if detector is not None else _default_detector()
    values = det.compute_language_confidence_values(text)
    candidates = sorted(
        ((_iso(v.language), float(v.value)) for v in values),
        key=lambda pair: pair[1],
        reverse=True,
    )
    if not candidates:
        return LanguageCheck(
            language="und",
            confidence=0.0,
            accepted=False,
            is_bcs=False,
            message_hr=get_message("lang_rejected", lang="nepoznato"),
        )

    top_lang, top_conf = candidates[0]
    is_bcs = top_lang in _BCS
    message: str | None = None
    if not is_bcs:
        message = get_message("lang_rejected", lang=top_lang)
    else:
        runner_up_bcs = any(lang in _BCS for lang, _ in candidates[1:])
        close_race = len(candidates) > 1 and top_conf - candidates[1][1] < _BCS_GAP
        if top_lang != "hr" or (runner_up_bcs and close_race):
            message = get_message("lang_uncertain_bcs")

    return LanguageCheck(
        language=top_lang,
        confidence=min(top_conf, 1.0),
        candidates=candidates[:3],
        accepted=is_bcs,
        is_bcs=is_bcs,
        message_hr=message,
    )
