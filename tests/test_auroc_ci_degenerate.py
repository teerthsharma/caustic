"""What the bootstrap interval does when a resample loses a class.

`auroc` returns 0.5 when one class is empty, which is correct for a point estimate
and wrong as an input to a percentile. The relations reported in RESULTS.md run at
n = 16 to 18 with a minority class as small as one, where single-class resamples
are not a corner case but a third of all draws, so the guard's 0.5 became a spike
of mass in the reported intervals.

These tests pin the two halves of the claim: that the spike moved the bounds in the
regime the repo actually reports, and that nothing moves at large n. The second is
the control — a fix that also changed well-sampled intervals would be changing the
statistic rather than removing an artefact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.detect import auroc, auroc_ci, auroc_ci_detail


def legacy_auroc_ci(scores, labels, n_boot=2000, seed=0):
    """The pre-fix behaviour, kept here so the difference is measured, not asserted.

    Byte-for-byte the resampling loop as it stood at db689cb: same `default_rng`,
    same `rng.integers(0, n, n)` draw order, so the only difference from the current
    implementation is what happens to a single-class draw.
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n = len(scores)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = auroc(scores[idx], labels[idx])
    return auroc(scores, labels), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# --- the regime the repo reports in ----------------------------------------


def test_a_minority_class_of_one_makes_a_third_of_resamples_degenerate():
    """Closed form: with 1 negative in 16, P(no negative drawn) = (15/16)^16 = 0.356.

    This is the arithmetic that makes the artefact large rather than cosmetic, and
    it is the shape of `element_symbol` at short context in RESULTS.md: 15 wrong
    answers, 1 correct.
    """
    labels = np.array([1] * 15 + [0])
    scores = np.arange(16, dtype=float)
    ci = auroc_ci_detail(scores, labels, n_boot=4000, seed=0)
    assert ci.n_valid + ci.n_discarded == 4000
    assert ci.n_discarded / 4000 == pytest.approx((15 / 16) ** 16, abs=0.03)


def test_the_guard_pins_a_bound_to_0_5_under_perfect_separation():
    """Closed form. 15 positives above 1 negative, so every resample containing both
    classes scores exactly 1.0 and the bootstrap distribution is two atoms: 0.5 on
    the 35.6% degenerate draws, 1.0 on the rest.

    The old 2.5th percentile therefore lands in the spike and reports 0.5 for a
    detector that never once ranked a positive below the negative. The lower bound
    is the guard's value, carrying no information about the data at all.
    """
    labels = np.array([1] * 15 + [0])
    scores = np.array([1.0] * 15 + [0.0])
    _, old_lo, old_hi = legacy_auroc_ci(scores, labels, n_boot=4000, seed=0)
    new = auroc_ci_detail(scores, labels, n_boot=4000, seed=0)
    assert (old_lo, old_hi) == (0.5, 1.0)
    assert (new.lo, new.hi) == (1.0, 1.0)
    assert new.n_discarded > 1000


def test_contamination_alone_does_not_pin_a_bound():
    """The same n and class split with real spread, where the bound is not 0.5.

    Heavy contamination pins the lower bound only if under 2.5% of the usable draws
    already fall below 0.5. Here they do not, so the 2.5th percentile sits beneath
    the spike and the old bound is a real quantile — contaminated, but not the
    guard's value. A rule of the form "contamination above 2.5% implies a pinned
    lower bound" is therefore false, and this is the arm that refutes it.
    """
    labels = np.array([1] * 15 + [0])
    rng = np.random.default_rng(11)
    scores = rng.normal(size=16) + labels * 0.5
    _, old_lo, _ = legacy_auroc_ci(scores, labels, n_boot=4000, seed=0)
    new = auroc_ci_detail(scores, labels, n_boot=4000, seed=0)
    assert old_lo != pytest.approx(0.5), "not pinned, despite 36% contamination"
    assert new.lo != pytest.approx(old_lo)
    assert new.n_discarded > 1000


def test_the_point_estimate_is_untouched():
    """Only the interval was contaminated. Changing the point estimate would
    invalidate every AUROC in RESULTS.md rather than only the brackets after it."""
    labels = np.array([1] * 15 + [0])
    rng = np.random.default_rng(12)
    scores = rng.normal(size=16)
    assert auroc_ci(scores, labels, n_boot=500, seed=0)[0] == auroc(scores, labels)
    assert legacy_auroc_ci(scores, labels, n_boot=500, seed=0)[0] == auroc(scores, labels)


def test_a_warning_names_the_discarded_count():
    labels = np.array([1] * 15 + [0])
    with pytest.warns(UserWarning, match=r"single-class"):
        auroc_ci(np.arange(16, dtype=float), labels, n_boot=200, seed=0)


# --- the control: large n, where degenerate resamples cannot happen ---------


def test_at_large_n_old_and_new_agree_exactly():
    """400 points, balanced: P(a resample loses a class) is about 2^-399.

    Nothing is discarded, so the survivors are the whole set in the original draw
    order and the two implementations must return identical floats. This is the
    control that the fix touches only the small-n regime.
    """
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400).astype(bool)
    s = rng.normal(size=400) + y * 0.8
    old = legacy_auroc_ci(s, y, n_boot=600, seed=0)
    new = auroc_ci_detail(s, y, n_boot=600, seed=0)
    assert new.n_discarded == 0
    assert (new.point, new.lo, new.hi) == old


def test_at_moderate_n_the_bounds_barely_move():
    """n = 60 with a 1-in-5 minority: degenerate draws are ~1e-6, so any shift is
    resampling noise rather than the artefact."""
    rng = np.random.default_rng(3)
    y = np.array([1] * 12 + [0] * 48).astype(bool)
    s = rng.normal(size=60) + y * 0.7
    _, old_lo, old_hi = legacy_auroc_ci(s, y, n_boot=1000, seed=0)
    new = auroc_ci_detail(s, y, n_boot=1000, seed=0)
    assert new.n_discarded == 0
    assert new.lo == pytest.approx(old_lo, abs=1e-12)
    assert new.hi == pytest.approx(old_hi, abs=1e-12)


# --- the guard --------------------------------------------------------------


def test_too_few_survivors_gives_nan_rather_than_a_tight_interval():
    """The guard fires only below 10% survival, which heavy contamination does not reach.

    A minority class of one in 40 discards (39/40)^40 = 0.363 of draws and still
    leaves 64% usable, so it gets an interval — badly informed, but computed from
    real resamples. The guard is for the case where nothing survives at all, which
    a genuinely single-class input produces.
    """
    labels = np.array([1] * 39 + [0])
    scores = np.arange(40, dtype=float)
    assert auroc_ci_detail(scores, labels, n_boot=1000, seed=0).n_valid > 100

    ci = auroc_ci_detail(np.array([1.0, 2.0]), np.array([1, 1]), n_boot=100, seed=0)
    assert ci.n_valid == 0 and ci.n_discarded == 100
    assert np.isnan(ci.lo) and np.isnan(ci.hi)
    assert ci.point == 0.5  # auroc's own single-class guard, unchanged


def test_the_public_signature_is_still_three_values():
    """Fifteen call sites across caustic/experiments unpack exactly three."""
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 50).astype(bool)
    point, lo, hi = auroc_ci(rng.normal(size=50) + y, y, n_boot=200, seed=0)
    assert lo <= point <= hi
    assert isinstance(point, float) and isinstance(lo, float) and isinstance(hi, float)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
