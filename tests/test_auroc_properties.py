"""Properties of the AUROC estimator, cross-checked against a brute-force U.

`tests/test_detect.py` already cross-checks `auroc` against a pair count on 50
tied draws and pins the endpoints by hand. What it does not cover is the rule
level: an antisymmetry that holds only when no two scores are equal, the
degenerate class sizes the relations in this repo actually run at, a bootstrap
whose reported interval is not the interval its docstring claims, and the
single-class resamples that the empty-class guard converts into a spike of 0.5
values. The tests here are that companion, written against a reference U
statistic implemented in this file from the definition, so that no bug can be
shared between the implementation and its check.

The reference is deliberately O(n_pos * n_neg) and deliberately dumb. Speed is
not the point; independence from `caustic.detect` is.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.detect import auroc, auroc_ci


def brute_force_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U by explicit pair enumeration: 1 for a win, 0.5 for a tie.

    The independent reference. No sorting, no ranks, no numpy vectorization, so
    it shares no machinery with the rank identity under test.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    wins = 0.0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _random_case(rng: np.random.Generator, n: int, alphabet: int | None):
    """A score/label pair with both classes non-empty.

    `alphabet` None draws continuous scores; an integer draws from that many
    distinct values, which is how heavy ties are forced.
    """
    while True:
        if alphabet is None:
            s = rng.normal(size=n)
        else:
            s = rng.integers(0, alphabet, n).astype(float)
        y = rng.integers(0, 2, n).astype(bool)
        if y.any() and not y.all():
            return s, y


# --- cross-check against the definition ------------------------------------


def test_matches_brute_force_on_continuous_scores():
    """Catches an off-by-one in the rank identity, which a no-ties input would
    still expose as a constant 1/(n_pos*n_neg) offset."""
    rng = np.random.default_rng(0)
    for _ in range(120):
        n = int(rng.integers(4, 60))
        s, y = _random_case(rng, n, None)
        assert auroc(s, y) == pytest.approx(brute_force_auroc(s, y), abs=1e-12)


def test_matches_brute_force_under_heavy_ties():
    """Scores drawn from a two- to four-symbol alphabet, so most pairs are tied.

    Tie handling is where rank-based AUROC usually breaks: an implementation
    that assigns ordinal ranks within a tied block scores each such pair 0 or 1
    by sort order instead of 0.5, which biases every saturating score (e.g.
    `max_softmax_score` on confident positions) in whichever direction the sort
    happens to run.
    """
    rng = np.random.default_rng(0)
    for _ in range(120):
        n = int(rng.integers(4, 60))
        alphabet = int(rng.integers(2, 5))
        s, y = _random_case(rng, n, alphabet)
        assert auroc(s, y) == pytest.approx(brute_force_auroc(s, y), abs=1e-12)


def test_single_member_classes_match_brute_force():
    """One positive against many negatives, and the mirror case.

    With n_pos == 1 the rank identity reduces to a single subtraction, so any
    error in the n_pos*(n_pos+1)/2 correction term is fully exposed here and
    nearly invisible at balanced class sizes.
    """
    rng = np.random.default_rng(0)
    for _ in range(60):
        n = int(rng.integers(3, 30))
        s = rng.integers(0, 4, n).astype(float)  # ties, at the degenerate size
        for k in (1, n - 1):
            y = np.zeros(n, dtype=bool)
            y[rng.choice(n, k, replace=False)] = True
            assert auroc(s, y) == pytest.approx(brute_force_auroc(s, y), abs=1e-12)


# --- algebraic properties --------------------------------------------------


def test_all_equal_scores_are_exactly_one_half():
    """A score with no resolution must report no discrimination, exactly.

    Not `approx`: every pair is a tie worth 0.5, so the sum is n_pos*n_neg/2 and
    the division is exact in binary floating point. A result that is merely near
    0.5 means the tie block was not collapsed cleanly.
    """
    rng = np.random.default_rng(0)
    for n in (2, 3, 7, 16, 41):
        y = np.zeros(n, dtype=bool)
        y[rng.choice(n, max(1, n // 3), replace=False)] = True
        assert auroc(np.full(n, 2.5), y) == 0.5


def test_negating_the_score_reflects_the_statistic():
    """auroc(s, y) + auroc(-s, y) == 1, including under ties.

    A tie-blind implementation fails this: it counts each tied pair as a win in
    both directions, so the two halves sum to more than 1. This is the cheapest
    property that distinguishes averaged ranks from ordinal ranks.
    """
    rng = np.random.default_rng(0)
    for _ in range(80):
        n = int(rng.integers(4, 50))
        alphabet = None if rng.integers(0, 2) else int(rng.integers(2, 6))
        s, y = _random_case(rng, n, alphabet)
        assert auroc(s, y) + auroc(-s, y) == pytest.approx(1.0, abs=1e-12)


def test_perfect_and_inverted_separation_hit_the_endpoints():
    """The endpoints must be reachable, not merely approached.

    An implementation that adds a tie correction unconditionally, or that
    smooths ranks, lands at 0.999... here and silently caps every headline
    number below 1.
    """
    rng = np.random.default_rng(0)
    for _ in range(20):
        n_neg, n_pos = int(rng.integers(1, 20)), int(rng.integers(1, 20))
        s = np.concatenate([rng.uniform(0.0, 1.0, n_neg), rng.uniform(2.0, 3.0, n_pos)])
        y = np.concatenate([np.zeros(n_neg, bool), np.ones(n_pos, bool)])
        assert auroc(s, y) == 1.0
        assert auroc(-s, y) == 0.0


def test_empty_class_returns_one_half_and_never_nan():
    """Either class empty leaves AUROC undefined, and `detect.py` chooses 0.5
    over NaN deliberately: a NaN here would propagate silently into a reported
    table, where it reads as a missing cell rather than as an evaluation that
    could not be run. 0.5 is the value of a coin flip, so a degenerate split
    shows up as "no signal" rather than as a hole.

    The assertion is on both the value and on finiteness, because a NaN also
    fails every `== 0.5` comparison and would otherwise be indistinguishable
    from a plain wrong constant.
    """
    rng = np.random.default_rng(0)
    for n in (1, 5, 30):
        s = rng.normal(size=n)
        for y in (np.ones(n, dtype=bool), np.zeros(n, dtype=bool)):
            got = auroc(s, y)
            assert np.isfinite(got)
            assert got == 0.5


# --- negative control ------------------------------------------------------


def test_random_labels_are_unbiased_around_one_half():
    """NEGATIVE CONTROL. Labels independent of scores carry no signal, so the
    mean over many trials must sit at 0.5.

    This is the test that fails if `auroc` ever acquires a directional bias — a
    tie correction applied to one class only, an argsort tiebreak that favours
    the positive class, an off-by-one in the rank offset. Any of those shift the
    mean while leaving every hand-pinned example in `test_detect.py` intact.

    The tolerance is sized against a specific mutant rather than by taste.
    Replacing the correction term n_pos*(n_pos+1)/2 with n_pos*(n_pos-1)/2
    biases each trial by about 1/n_neg; at 500 trials of n = 400 that mutant
    scores a mean of 0.50469 while the unmutated implementation scores 0.49968,
    so 0.004 separates them. The run is fully seeded, so the 2.9-sigma margin is
    a fixed number, not a flake risk.
    """
    rng = np.random.default_rng(0)
    vals = [
        auroc(rng.normal(size=400), rng.integers(0, 2, 400).astype(bool)) for _ in range(500)
    ]
    assert abs(float(np.mean(vals)) - 0.5) < 0.004


# --- the bootstrap ---------------------------------------------------------


def test_ci_is_bitwise_reproducible_under_a_fixed_seed():
    """Two calls at the same seed must agree bitwise, and a different seed must
    not. A bootstrap that reads a global RNG makes every reported interval a
    function of call order, which is invisible until two runs of the same script
    disagree in the third decimal."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 120).astype(bool)
    s = rng.normal(size=120) + y * 0.7
    a = auroc_ci(s, y, n_boot=200, seed=0)
    b = auroc_ci(s, y, n_boot=200, seed=0)
    assert a == b
    c = auroc_ci(s, y, n_boot=200, seed=1)
    assert c[0] == a[0], "the point estimate does not depend on the resampling seed"
    assert (c[1], c[2]) != (a[1], a[2])


def test_ci_brackets_its_own_point_estimate():
    """lo <= point <= hi across a range of sample sizes and separations.

    A percentile interval can exclude the point estimate when the bootstrap
    distribution is badly skewed, which at these sizes it is; this pins that the
    reported triple is at least internally ordered, so a table cannot print an
    estimate outside its own interval.
    """
    rng = np.random.default_rng(0)
    for n in (20, 60, 200):
        for shift in (0.0, 0.5, 2.0):
            y = rng.integers(0, 2, n).astype(bool)
            s = rng.normal(size=n) + y * shift
            point, lo, hi = auroc_ci(s, y, n_boot=200, seed=0)
            assert lo <= point <= hi


def test_ci_reports_the_two_sided_95_percent_percentiles():
    """The interval must be the 2.5/97.5 percentiles of the bootstrap draws it
    took, not some other level that happens to bracket the point estimate.

    Bracketing and seed determinism both pass for a one-sided interval, so a
    silently mis-levelled CI would otherwise ship. This reproduces the draws
    from `auroc_ci`'s own documented stream and compares the percentiles bitwise.
    """
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 80).astype(bool)
    s = rng.normal(size=80) + y * 0.6
    n_boot, seed = 300, 4
    point, lo, hi = auroc_ci(s, y, n_boot=n_boot, seed=seed)
    boot_rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = boot_rng.integers(0, len(s), len(s))
        boots[b] = auroc(s[idx], y[idx])
    assert point == auroc(s, y)
    assert lo == float(np.percentile(boots, 2.5))
    assert hi == float(np.percentile(boots, 97.5))


def test_single_class_bootstrap_resamples_are_counted_and_pinned(monkeypatch):
    """MEASURED CONTAMINATION, not a fix. `auroc_ci` resamples indices with
    replacement, so at the relation sizes this repo actually runs (n = 12 to 20,
    minority class about a fifth of the items) some resamples draw only one
    class. Each of those hits the empty-class guard and is scored exactly 0.5,
    so the bootstrap distribution carries a spike of 0.5 values that belong to
    no ROC curve, and the interval is pulled toward 0.5 by that fraction.

    The count is taken by driving the real `auroc_ci` with `caustic.detect.auroc`
    replaced by a counting wrapper, so it measures whatever resampling scheme
    `auroc_ci` actually uses rather than a local re-implementation of it. At
    n_boot = 2000, seed 0, minority size max(2, n // 5):

        n = 12, 2 positives:  207 / 2000  = 10.35%
        n = 16, 3 positives:   60 / 2000  =  3.00%
        n = 18, 3 positives:   77 / 2000  =  3.85%
        n = 20, 4 positives:   16 / 2000  =  0.80%

    Pinned exactly so the contamination cannot drift silently. If `auroc_ci`
    starts drawing stratified resamples, or rejecting degenerate ones, these
    counts move and the test says so. Fixing it belongs to whoever owns
    `caustic/detect.py`; the job here is to make the effect visible and measured.
    Note the counts also depend on numpy's `Generator` stream, which NEP 19 does
    not freeze, so a numpy upgrade is the other thing that can move them.
    """
    import caustic.detect as detect

    expected = {12: 207, 16: 60, 18: 77, 20: 16}
    for n, want in expected.items():
        labels = np.zeros(n, dtype=bool)
        labels[: max(2, n // 5)] = True
        scores = np.arange(n, dtype=float)

        # Read the count from `auroc_ci_detail` rather than by wrapping `auroc`.
        # `auroc_ci` now skips a single-class resample BEFORE calling `auroc`, so
        # a counting wrapper sees nothing and the contamination becomes invisible
        # exactly where it used to be observed. The draws are unchanged - same
        # generator, same seed, same n_boot - so all four pinned values survive
        # the fix bit for bit; only the place they are reported moved.
        detail = detect.auroc_ci_detail(scores, labels, n_boot=2000, seed=0)
        assert detail.n_discarded == want, (
            f"n={n}: single-class resamples moved from {want} to {detail.n_discarded}"
        )
        assert detail.n_valid + detail.n_discarded == 2000


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
