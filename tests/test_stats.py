"""Ground truth for the null model and the multiplicity correction.

Two failures are worth more than the rest here. A permutation null built wrongly
— shuffling the scores instead of the labels, or comparing against the wrong
tail — still returns plausible-looking small p-values on separable data, so the
negative control on random labels is the test that catches it. And a Holm
implementation that returns corrected values in sorted order silently reassigns
each relation's p-value to a different relation, which is caught here by
shuffling the input and unshuffling the output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.stats import holm, permutation_auroc


# --- Holm ------------------------------------------------------------------


def test_holm_matches_the_hand_worked_example():
    """p = (0.01, 0.04, 0.03, 0.005), k = 4, worked by hand.

    Sorted ascending: 0.005, 0.01, 0.03, 0.04.
    Multipliers k - j for j = 0..3:  4, 3, 2, 1.
    Products:            0.020, 0.030, 0.060, 0.040.
    Running maximum:     0.020, 0.030, 0.060, 0.060.
    Back in input order: 0.030, 0.060, 0.060, 0.020.

    Catches both a wrong multiplier sequence (plain Bonferroni would give
    0.04, 0.16, 0.12, 0.02) and a missing running maximum (which would leave
    0.040 < 0.060 out of order at the last position).
    """
    got = holm(np.array([0.01, 0.04, 0.03, 0.005]))
    assert got == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_is_order_preserving_under_a_shuffle():
    """Correcting a shuffled input and unshuffling must reproduce the original.

    This is the classic bug: returning `np.sort(p) * multipliers` assigns the
    smallest corrected value to the first slot rather than to the test that
    earned it, so every relation in a table gets another relation's p-value.
    """
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 12)
    perm = rng.permutation(12)
    direct = holm(p)
    via_shuffle = np.empty(12)
    via_shuffle[perm] = holm(p[perm])
    assert np.allclose(direct, via_shuffle)


def test_holm_is_monotone_along_the_sorted_sequence():
    """A corrected p-value may never fall below that of a smaller raw p-value."""
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 30)
    corrected = holm(p)[np.argsort(p)]
    assert np.all(np.diff(corrected) >= -1e-15)


def test_holm_of_a_single_pvalue_is_itself():
    """k = 1 means a multiplier of 1; a hardcoded `len(p)` off-by-one breaks this."""
    assert holm(np.array([0.037])) == pytest.approx([0.037])


def test_holm_clips_at_one():
    """All-ones input must stay at 1.0, not run to 5.0 through the multiplier."""
    assert np.all(holm(np.ones(5)) == 1.0)
    assert np.max(holm(np.array([0.9, 0.95, 0.99]))) == 1.0


def test_holm_never_shrinks_a_pvalue():
    """A correction that returns something below the raw value is anti-conservative."""
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 25)
    assert np.all(holm(p) >= p - 1e-15)


def test_holm_rejects_bad_input():
    with pytest.raises(ValueError):
        holm(np.array([]))
    with pytest.raises(ValueError):
        holm(np.array([0.1, 1.5]))
    with pytest.raises(ValueError):
        holm(np.array([0.1, -0.01]))
    with pytest.raises(ValueError):
        holm(np.array([0.1, np.nan]))
    with pytest.raises(ValueError):
        holm(np.ones((2, 3)))


# --- permutation AUROC -----------------------------------------------------


def _separable(n: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(bool)
    scores = labels * 10.0 + rng.normal(0, 0.01, n)
    return scores, labels


def test_permutation_is_deterministic_under_a_fixed_seed():
    """Two calls with the same seed must agree bitwise, or no reported p is reproducible."""
    s, y = _separable()
    a = permutation_auroc(s, y, n_perm=200, seed=7)
    b = permutation_auroc(s, y, n_perm=200, seed=7)
    assert a == b


def test_permutation_pvalue_is_bounded_and_never_exactly_zero():
    """The (1 + count) / (1 + n_perm) estimator has a floor; a naive count / n_perm does not."""
    rng = np.random.default_rng(3)
    for seed in range(5):
        y = np.r_[np.ones(15), np.zeros(15)].astype(bool)
        s = rng.normal(size=30) + y * 2.0
        _, p = permutation_auroc(s, y, n_perm=100, seed=seed)
        assert 0.0 < p <= 1.0


def test_perfect_separation_hits_the_floor_exactly():
    """Observed AUROC 1.0 with n_perm = 200 pins p at 1 / 201.

    C(20, 10) = 184,756 label splits, so 200 shuffles reproducing the observed
    split is a ~0.1% event and does not occur at this seed. A test that accepted
    "p is small" would pass against a wrong denominator; this pins the value.
    """
    s, y = _separable()
    observed, p = permutation_auroc(s, y, n_perm=200, seed=0)
    assert observed == 1.0
    assert p == pytest.approx(1.0 / 201.0)


def test_a_backwards_score_gets_a_large_pvalue():
    """One-sided by design: a score that anti-predicts the label is a failure, not a hit."""
    s, y = _separable()
    observed, p = permutation_auroc(-s, y, n_perm=200, seed=0)
    assert observed == 0.0
    assert p == 1.0


def test_random_labels_give_a_uniform_pvalue_on_average():
    """NEGATIVE CONTROL. 200 independent draws of random labels must average near 0.5.

    This is the test that fails if the null is constructed wrongly — shuffling
    the scores against a sorted label vector, comparing against `>` on the wrong
    tail, or forgetting to reshuffle inside the loop all leave separable-data
    tests green while driving this mean to 0 or 1. The band is [0.4, 0.6]: the
    Monte-Carlo standard error of the mean over 200 draws is about 0.02, and the
    discreteness of AUROC under ties biases the mean slightly upward of 0.5.
    """
    rng = np.random.default_rng(11)
    labels = np.r_[np.ones(15), np.zeros(15)].astype(bool)
    ps = [
        permutation_auroc(rng.normal(size=30), rng.permutation(labels), n_perm=99, seed=int(i))[1]
        for i in range(200)
    ]
    assert 0.4 <= float(np.mean(ps)) <= 0.6


def test_permutation_rejects_bad_input():
    s, y = _separable()
    with pytest.raises(ValueError):
        permutation_auroc(s[:5], y, n_perm=10)
    with pytest.raises(ValueError):
        permutation_auroc(s, np.ones(len(s)), n_perm=10)
    with pytest.raises(ValueError):
        permutation_auroc(s, y, n_perm=0)
    with pytest.raises(ValueError):
        permutation_auroc(np.full(len(s), np.nan), y, n_perm=10)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
