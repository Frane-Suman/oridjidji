"""Binoculars and Fast-DetectGPT: implemented explicitly rather than imported.

Every test here runs on **constructed** log-probability arrays rather than a loaded model.
That is deliberate: the arithmetic is what the author has to defend at an oral exam, and a
test that needs 3.4 GB of weights and a GPU is a test nobody runs. The model layer is
mocked exactly as `tests/unit` requires.

The direction assertions are the important ones. The specification names a flipped sign as "the classic
bug here", and it is: nothing crashes, the AUROC simply comes back as 1 - AUROC and reads
as a mediocre detector rather than an inverted one.
"""

from __future__ import annotations

import numpy as np
import pytest

from oridjidji.detectors import binoculars, fast_detect_gpt
from oridjidji.detectors.backbone import TokenScores, assert_same_tokenizer

pytestmark = pytest.mark.unit

VOCAB = 8


def distribution(peak: int, confidence: float) -> np.ndarray:
    """A log-probability row peaked on ``peak``; ``confidence`` in (0, 1) sets how sharply."""
    probs = np.full(VOCAB, (1.0 - confidence) / (VOCAB - 1))
    probs[peak] = confidence
    return np.log(probs)


def scores_for(
    tokens: list[int], observer_confidence: float, performer_confidence: float | None = None
) -> TokenScores:
    """Token scores where the observer puts ``observer_confidence`` on each realised token."""
    observer = np.stack([distribution(t, observer_confidence) for t in tokens])
    performer = (
        None
        if performer_confidence is None
        else np.stack([distribution(t, performer_confidence) for t in tokens])
    )
    return TokenScores(
        token_ids=np.array(tokens, dtype=np.int64),
        observer_logprobs=observer,
        performer_logprobs=performer,
    )


# --------------------------------------------------------------------- backbone contract


def test_token_scores_reject_a_length_mismatch() -> None:
    """Position t must predict token_ids[t]; an off-by-one silently scores the wrong token."""
    with pytest.raises(ValueError, match="must predict"):
        TokenScores(
            token_ids=np.array([1, 2, 3], dtype=np.int64),
            observer_logprobs=np.log(np.full((2, VOCAB), 1 / VOCAB)),
        )


def test_token_scores_reject_mismatched_model_shapes() -> None:
    """Different vocabularies make cross-perplexity meaningless, not merely noisy."""
    with pytest.raises(ValueError, match="same shape"):
        TokenScores(
            token_ids=np.array([1, 2], dtype=np.int64),
            observer_logprobs=np.log(np.full((2, VOCAB), 1 / VOCAB)),
            performer_logprobs=np.log(np.full((2, VOCAB + 1), 1 / (VOCAB + 1))),
        )


def test_a_mismatched_tokenizer_pair_is_refused() -> None:
    """The specification: refuse the pair, loudly. The comparison is positional, so different
    tokenizers mean position t denotes different things to the two models."""
    with pytest.raises(ValueError, match="mismatched vocabularies"):
        assert_same_tokenizer(32_000, 32_001, "some/base+some/instruct")


def test_realised_logprobs_pick_the_token_that_actually_appeared() -> None:
    scores = scores_for([3, 1, 4], observer_confidence=0.5)
    assert np.allclose(scores.realised_logprobs, np.log(0.5))


# --------------------------------------------------------------------- Binoculars


def test_log_perplexity_is_mean_negative_log_probability() -> None:
    """The numerator: how surprised the observer is by the text."""
    scores = scores_for([0, 1, 2, 3], observer_confidence=0.5)
    assert binoculars.log_perplexity(scores) == pytest.approx(-np.log(0.5))


def test_cross_perplexity_is_zero_when_the_models_agree_exactly() -> None:
    """Cross-entropy of a distribution with itself is its entropy, so a uniform pair gives
    log(VOCAB) — not zero. Zero only happens for a one-hot distribution."""
    tokens = [0, 1, 2]
    row = np.log(np.full(VOCAB, 1 / VOCAB))
    same = np.stack([row] * len(tokens))
    scores = TokenScores(
        token_ids=np.array(tokens, dtype=np.int64),
        observer_logprobs=same,
        performer_logprobs=same,
    )
    assert binoculars.cross_perplexity(scores) == pytest.approx(np.log(VOCAB))


def test_cross_perplexity_grows_as_the_models_disagree() -> None:
    """The denominator measures disagreement, so it must respond to disagreement."""
    tokens = [0, 1, 2, 3]
    agree = scores_for(tokens, observer_confidence=0.7, performer_confidence=0.7)
    disagree = np.stack([distribution((t + 4) % VOCAB, 0.7) for t in tokens])
    conflicting = TokenScores(
        token_ids=np.array(tokens, dtype=np.int64),
        observer_logprobs=agree.observer_logprobs,
        performer_logprobs=disagree,
    )
    assert binoculars.cross_perplexity(conflicting) > binoculars.cross_perplexity(agree)


def test_cross_perplexity_needs_both_models() -> None:
    with pytest.raises(ValueError, match="both models"):
        binoculars.cross_perplexity(scores_for([1, 2], observer_confidence=0.5))


def test_binoculars_scores_machine_text_LOWER_than_human_text() -> None:
    """The sign convention, asserted (the specification calls a flip "the classic bug here").

    Machine-like text: both models are confident AND agree — small numerator, and the
    denominator stays meaningful. Human-like text: the observer is surprised by the token
    that appeared, while the two models still broadly agree — larger numerator over a
    similar denominator, so a higher ratio.
    """
    tokens = [0, 1, 2, 3, 4, 5]

    machine = scores_for(tokens, observer_confidence=0.9, performer_confidence=0.85)

    # Human: the realised token is one the observer thought unlikely.
    surprising = np.stack([distribution((t + 3) % VOCAB, 0.9) for t in tokens])
    human = TokenScores(
        token_ids=np.array(tokens, dtype=np.int64),
        observer_logprobs=surprising,
        performer_logprobs=np.stack([distribution((t + 3) % VOCAB, 0.85) for t in tokens]),
    )

    assert binoculars.binoculars_score(machine) < binoculars.binoculars_score(human)
    assert binoculars.score_direction() == "lower_is_machine"


def test_binoculars_refuses_identical_models() -> None:
    """Same model loaded twice makes the denominator collapse; the ratio is undefined."""
    tokens = [0, 1, 2]
    one_hot = np.full((len(tokens), VOCAB), -np.inf)
    for i, t in enumerate(tokens):
        one_hot[i, t] = 0.0
    scores = TokenScores(
        token_ids=np.array(tokens, dtype=np.int64),
        observer_logprobs=one_hot,
        performer_logprobs=one_hot,
    )
    with pytest.raises(ValueError, match="same model was loaded twice|degenerate"):
        binoculars.binoculars_score(scores)


# --------------------------------------------------------------------- Fast-DetectGPT


def test_expected_logprob_is_the_negative_entropy() -> None:
    """mu_t = sum_v p log p, which is exactly -H(p). Checked against a uniform row."""
    uniform = np.log(np.full((1, VOCAB), 1 / VOCAB))
    assert fast_detect_gpt.expected_logprob(uniform)[0] == pytest.approx(-np.log(VOCAB))


def test_logprob_variance_is_zero_for_a_uniform_distribution() -> None:
    """Every candidate has the same log-probability, so it cannot vary."""
    uniform = np.log(np.full((1, VOCAB), 1 / VOCAB))
    assert fast_detect_gpt.logprob_variance(uniform)[0] == pytest.approx(0.0, abs=1e-12)


def test_logprob_variance_is_never_negative() -> None:
    """The identity guarantees it; floating point does not, and a -1e-16 becomes NaN
    under the square root in `curvature`."""
    rows = np.stack([distribution(i % VOCAB, 0.999999) for i in range(6)])
    assert np.all(fast_detect_gpt.logprob_variance(rows) >= 0.0)


def test_curvature_scores_machine_text_HIGHER_than_human_text() -> None:
    """The opposite convention from Binoculars, asserted.

    Machine text sits where the model expected it to (observed ~ expected, d near 0).
    Human text falls below the model's own expectation, so d is negative.
    """
    tokens = [0, 1, 2, 3, 4, 5, 6, 7]
    machine = scores_for(tokens, observer_confidence=0.6)

    unexpected = np.stack([distribution((t + 4) % VOCAB, 0.6) for t in tokens])
    human = TokenScores(token_ids=np.array(tokens, dtype=np.int64), observer_logprobs=unexpected)

    assert fast_detect_gpt.curvature(machine) > fast_detect_gpt.curvature(human)
    assert fast_detect_gpt.score_direction() == "higher_is_machine"


def test_the_two_detectors_declare_opposite_directions() -> None:
    """Two co-primary detectors with opposite sign conventions in one codebase is exactly
    where a flipped sign hides. Pin the difference so a careless edit has to break a test."""
    assert binoculars.score_direction() == "lower_is_machine"
    assert fast_detect_gpt.score_direction() == "higher_is_machine"
    assert binoculars.score_direction() != fast_detect_gpt.score_direction()


def test_curvature_uses_only_the_observer() -> None:
    """The analytic single-model variant reuses Binoculars' observer at zero extra RAM."""
    tokens = [0, 1, 2, 3]
    with_performer = scores_for(tokens, observer_confidence=0.7, performer_confidence=0.2)
    without = scores_for(tokens, observer_confidence=0.7)
    assert fast_detect_gpt.curvature(with_performer) == pytest.approx(
        fast_detect_gpt.curvature(without)
    )


# --------------------------------------------------------------------- calibration


def test_calibration_never_hardcodes_a_paper_threshold() -> None:
    """The specification: every threshold is derived from Croatian data, and the artefact records what
    produced it — a Binoculars threshold is meaningless without its model pair."""
    machine = np.array([0.5, 0.6, 0.55, 0.7])
    human = np.array([1.2, 1.4, 1.1, 1.3])

    thresholds = binoculars.calibrate(
        machine, human, model_pair="EuroLLM-1.7B/Instruct", calibration_hash="abc123"
    )
    assert 0.7 <= thresholds.threshold < 1.1
    assert thresholds.model_pair == "EuroLLM-1.7B/Instruct"
    assert thresholds.n_calibration == 8
    assert thresholds.achieved_auroc == pytest.approx(1.0)
    assert thresholds.predict_machine(0.6)
    assert not thresholds.predict_machine(1.3)


def test_calibration_reports_a_poor_separation_honestly() -> None:
    """Overlapping classes must yield an AUROC near chance, not a flattering number."""
    rng = np.random.default_rng(0)
    machine = rng.normal(1.0, 1.0, 200)
    human = rng.normal(1.0, 1.0, 200)
    thresholds = binoculars.calibrate(machine, human, model_pair="p", calibration_hash="h")
    assert 0.4 < thresholds.achieved_auroc < 0.6


def test_calibration_needs_both_classes() -> None:
    with pytest.raises(ValueError, match="both classes"):
        binoculars.calibrate(np.array([0.5]), np.array([]), model_pair="p", calibration_hash="h")
