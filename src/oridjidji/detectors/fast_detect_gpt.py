"""Fast-DetectGPT — conditional probability curvature (the specification, Bao et al. 2023,
arXiv:2310.05130).

**Co-primary, not an afterthought.** CEAID measured it at **0.8228** AUROC on Croatian
against Binoculars' **0.8000** — on this language it is the stronger of the two.
Binoculars keeps the product name ("dvostruki obzervator") because it is more explainable,
but the thesis must not imply it is the accuracy leader.

**The idea.** DetectGPT asked: is this text at a local maximum of the model's likelihood?
It answered by perturbing the text a hundred times and re-scoring — accurate and far too
slow. Fast-DetectGPT observes that the same question has a closed form. At each position
the model already tells you what it expected; you can compare what actually appeared
against that expectation analytically, in **one forward pass**, with no perturbations at
all:

    observed  = sum_t  log p(x_t | x_<t)              what the text actually did
    mu_t      = sum_v  p_t(v) log p_t(v)              what the model expected of itself
    var_t     = sum_v  p_t(v) (log p_t(v))^2 - mu_t^2
    d         = (observed - sum_t mu_t) / sqrt(sum_t var_t)

**HIGHER d means MORE LIKELY MACHINE — the opposite of Binoculars.** Text sampled from
the model sits, by construction, right where the model expected (d near 0). Human text
falls below the model's own expectation (d negative). Two co-primary detectors with
opposite sign conventions in one codebase is precisely how a sign bug survives review, so
both modules expose `score_direction()` and both directions are asserted in tests.

Uses the **analytic single-model variant** deliberately: it needs only the observer, so it
runs on log-probs the backbone has already computed for Binoculars, at zero additional
RAM. On a 16 GB laptop with no GPU that is the whole reason it is affordable.
"""

from __future__ import annotations

import numpy as np

from oridjidji.detectors.backbone import Array, PerTokenScores

# Same floor as Binoculars, for the same reason: below this the statistic is dominated by
# whichever handful of tokens occurred.
MIN_TOKENS = 32


def expected_logprob(logprobs: Array) -> Array:
    """Per-position mean log-probability under the model's OWN distribution.

    ``mu_t = sum_v p(v) log p(v)`` — the negative entropy of the position. This is what
    the model would score, on average, if it generated the next token itself, and it is
    the baseline the observed token is compared against.
    """
    probs = np.exp(logprobs)
    return np.asarray(np.sum(probs * logprobs, axis=1), dtype=np.float64)


def logprob_variance(logprobs: Array) -> Array:
    """Per-position variance of the log-probability under the model's own distribution.

    ``var_t = E[(log p)^2] - (E[log p])^2``, both expectations taken under ``p`` itself.

    This is the term that makes the statistic scale-free. Without it, a confident position
    (where every candidate is either near-certain or near-impossible) would dominate the
    sum over an uncertain one purely because its log-probabilities are larger in
    magnitude, and the score would measure the model's confidence rather than the text's
    conformity.
    """
    probs = np.exp(logprobs)
    mean = np.sum(probs * logprobs, axis=1)
    mean_square = np.sum(probs * np.square(logprobs), axis=1)
    # Clipped at zero: the identity guarantees non-negativity, floating point does not,
    # and a variance of -1e-16 becomes NaN under the square root two lines later.
    return np.asarray(np.maximum(mean_square - np.square(mean), 0.0), dtype=np.float64)


def curvature(scores: PerTokenScores) -> float:
    """Conditional probability curvature. **Higher means more likely machine.**

    Only the observer is used; the performer, if present, is ignored. That is the point of
    the analytic variant.
    """
    if scores.n_tokens == 0:
        raise ValueError("Cannot compute curvature of an empty token sequence.")

    # Read as reductions, so this works unchanged on a `ReducedScores` that never held the
    # vocabulary axis. `expected_logprob` and `logprob_variance` remain the tested
    # definitions and are what `TokenScores` computes these properties with.
    observed = float(np.sum(scores.realised_logprobs))
    expected = float(np.sum(scores.expected_logprobs))
    total_variance = float(np.sum(scores.logprob_variances))

    if total_variance <= 0.0:
        # Every position was a certainty, so the model had no expectation to deviate from.
        # Real text never does this; it means a degenerate input (a single repeated token)
        # and dividing would give inf or NaN rather than a score.
        raise ValueError(
            "Total log-probability variance is zero: the model was certain at every "
            "position, so there is no expectation for the text to deviate from. The "
            "curvature is undefined, not infinite."
        )
    return float((observed - expected) / np.sqrt(total_variance))


def score_direction() -> str:
    """States the sign convention in code, so a test can assert it.

    Deliberately the opposite string from `binoculars.score_direction()`, and the two are
    compared in a test — a codebase holding two co-primary detectors with opposite
    conventions is exactly where a flipped sign hides.
    """
    return "higher_is_machine"
