"""The normal inverse and the single-class formula, against closed forms.

No model, no bootstrap: these check the two pieces of `detector_power` that could
be silently wrong and would then mislabel every row of its table.
"""

from __future__ import annotations

import numpy as np
import pytest

from caustic.detect import auroc
from caustic.experiments.detector_power import (
    measure,
    norm_cdf,
    norm_ppf,
    separation,
    single_class_fraction,
    trial_scores,
)


@pytest.mark.parametrize(
    "p, want",
    [
        (0.5, 0.0),
        (0.975, 1.959963984540054),
        (0.95, 1.6448536269514722),
        (0.8413447460685429, 1.0),
        (0.001, -3.090232306167813),
    ],
)
def test_norm_ppf_matches_known_quantiles(p, want):
    assert norm_ppf(p) == pytest.approx(want, abs=1e-9)


def test_norm_cdf_matches_known_values():
    assert norm_cdf(0.0) == pytest.approx(0.5, abs=1e-15)
    assert norm_cdf(1.0) == pytest.approx(0.8413447460685429, abs=1e-12)
    assert norm_cdf(-3.0) == pytest.approx(0.001349898031630095, abs=1e-15)


@pytest.mark.parametrize("p", [1e-6, 0.01, 0.3, 0.5, 0.9, 1 - 1e-6])
def test_norm_ppf_inverts_norm_cdf(p):
    assert norm_cdf(norm_ppf(p)) == pytest.approx(p, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 2.0])
def test_norm_ppf_rejects_out_of_range(p):
    with pytest.raises(ValueError):
        norm_ppf(p)


@pytest.mark.parametrize("target", [0.6, 0.7, 0.8, 0.9])
def test_separation_realises_target_auroc(target):
    """The whole construction: d = sqrt(2) Phi^-1(A) must give AUROC A."""
    rng = np.random.default_rng(0)
    scores, labels = trial_scores(rng, 40_000, 40_000, separation(target))
    assert auroc(scores, labels) == pytest.approx(target, abs=0.01)


def test_single_class_fraction_matches_the_resamples_auroc_ci_draws():
    n, n_pos, draws = 16, 15, 40_000
    rng = np.random.default_rng(0)  # the stream auroc_ci uses at seed 0
    labels = np.concatenate([np.ones(n_pos), np.zeros(n - n_pos)]).astype(bool)
    hits = sum(
        1 for _ in range(draws)
        if (lab := labels[rng.integers(0, n, n)]).all() or not lab.any()
    )
    assert hits / draws == pytest.approx(single_class_fraction(n, n_pos), abs=0.01)


def test_single_class_fraction_is_negligible_when_balanced():
    assert single_class_fraction(20, 10) < 1e-5


def test_measure_power_separates_a_real_effect_from_the_null():
    """The `pow` column is the module's whole conclusion, so pin its two ends.

    `separation(0.5)` is exactly zero, so the null case draws both classes from
    the same distribution and a correct 95% interval excludes 0.5 about 5% of the
    time. A large, well-separated case must fire nearly always. An inverted or
    off-by-one comparison in `measure` fails one end or the other.
    """
    assert separation(0.5) == pytest.approx(0.0, abs=1e-12)
    rng = np.random.default_rng(0)
    _, hw_null, power_null = measure(rng, 64, 64, separation(0.5), n_trials=40, n_boot=200)
    _, hw_effect, power_effect = measure(rng, 64, 64, separation(0.9), n_trials=40, n_boot=200)
    assert power_null <= 0.25
    assert power_effect >= 0.9
    assert hw_effect < hw_null
