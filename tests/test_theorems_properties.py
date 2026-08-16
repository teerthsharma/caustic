"""The counting theorems as properties, not as pinned examples.

`tests/test_theorems.py` pins each closed form on instances whose value was
written down by hand. That register catches a wrong constant on the rows it
pins but leaves the shape unconstrained off them: `certified_reduction`
rewritten as `(m_after - m_before) / n` reads the same in prose, reproduces all
four pinned deltas, and still differs bitwise from the difference-of-floors
definition on 947 of the 2,000 triples sampled below. Everything here is driven
by randomized or exhaustive input instead, and every closed form is pinned with
`==` rather than a tolerance wherever it is exact in binary floating point on
the range used.

Two witnesses carry the argument that the bound is a *bound*:
`test_t1_slack_witness_shows_the_bound_is_not_an_estimate` (actual errors above
it) and `test_non_injective_relation_falls_below_the_bound` (actual errors
below it, which is why `RelationSpec.injective` exists at all).
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.regime import RelationSpec, orbit_partition
from caustic.theorems import (
    certified_error_floor,
    certified_reduction,
    orbit_error_bound,
    path_integral_change,
    pooling_recovery_bound,
    recovery_ceiling_gain,
    volume_decay_rate,
)

CONTINENT_TPLS = ("The continent {e} is in is", "{e} is on the continent of")


def make_fn(mapping: dict[str, int], default: int = 0):
    """answer_fn keyed on whichever entity name appears in the prompt.

    Local copy of the helper in `tests/test_regime.py`, deliberately not imported
    across test modules. Keys are tried longest-first: naive insertion-order
    substring matching makes "C1" match the prompt for "C15", which silently
    mislabels entities and shows up only as a partition that disagrees with a
    hand count.
    """

    def fn(prompt: str) -> int:
        for k in sorted(mapping, key=len, reverse=True):
            if k in prompt:
                return mapping[k]
        return default

    return fn


def random_block_sizes(rng, n: int, m: int) -> list[int]:
    """A uniform-ish composition of `n` into exactly `m` positive parts."""
    cuts = rng.choice(np.arange(1, n), size=m - 1, replace=False)
    edges = np.concatenate([[0], np.sort(cuts), [n]])
    return [int(b - a) for a, b in zip(edges, edges[1:])]


# --- Theorem 1 as a property ------------------------------------------------


def test_t1_equals_the_sum_of_block_sizes_minus_one():
    """`n - m` is a closed form for `sum(s_i - 1)`; check it over 10,000 partitions.

    Catches any rewrite that gets the two endpoints right (`m = 1` and `m = n`)
    but the interior wrong, which pinning only the measured `n = 20, m = 15` row
    would let through.
    """
    rng = np.random.default_rng(0)
    for _ in range(10_000):
        n = int(rng.integers(1, 60))
        m = int(rng.integers(1, n + 1))
        sizes = random_block_sizes(rng, n, m)
        assert sum(sizes) == n and len(sizes) == m, "fixture built a bad partition"
        assert orbit_error_bound(n, m) == sum(s - 1 for s in sizes)


def test_t1_tightness_witness_attains_the_bound_exactly():
    """Every orbit holding exactly one correct answer: bound 6, actual 6.

    Catches a bound that is merely valid but loose. Nine entities in orbits of
    sizes 4, 3 and 2; the first member of each orbit is given the true answer,
    so the errors are exactly `sum(s_i - 1) = 3 + 2 + 1 = 6`.
    """
    truth = np.arange(9)
    pred = np.array([0, 0, 0, 0, 4, 4, 4, 7, 7])
    m = len(np.unique(pred))
    assert m == 3
    assert orbit_error_bound(9, m) == 6
    assert int((pred != truth).sum()) == 6, "the bound is attained, not merely respected"


def test_t1_slack_witness_shows_the_bound_is_not_an_estimate():
    """Theorem 1's measured instance: n=20, m=15, bound 5, actual 9.

    Catches any reading of `n - m` as an estimate of the error count. Five
    entities are merged onto entity 0's answer (giving the bound of 5) and four
    further entities are given wrong answers that are still unique, so `m` stays
    at 15 while the true error count rises to 9. A test suite that only ever
    exercises the equality case would not distinguish a bound from an estimator.
    """
    truth = np.arange(20)
    pred = truth.copy()
    pred[15:] = 0  # orbit {0, 15, 16, 17, 18, 19}
    pred[1:5] = [101, 102, 103, 104]  # wrong, but still singletons
    m = len(np.unique(pred))
    assert m == 15
    assert orbit_error_bound(20, m) == 5
    assert int((pred != truth).sum()) == 9
    assert orbit_error_bound(20, m) < int((pred != truth).sum())


# --- the certified rates ----------------------------------------------------


def test_certified_floor_is_exactly_the_rational_quotient():
    """`(n - m) / n` bit-for-bit over every partition with n <= 120.

    Pinned with `==`, not `approx`: a single IEEE division of two exact integers
    is correctly rounded, so any deviation means the implementation is not
    computing that quotient. Catches a floor divided by `m`, or one that scales
    by the orbit count instead of the entity count.
    """
    for n in range(1, 121):
        for m in range(1, n + 1):
            assert certified_error_floor(n, m) == (n - m) / n


def test_certified_reduction_is_the_difference_of_two_floors():
    """The delta an intervention buys is exactly floor(before) - floor(after).

    Catches a reduction computed from raw orbit counts (`(m_after - m_before) / n`
    reads the same in the docstring but rounds differently) and any accidental
    sign flip, which would report a merging intervention as an improvement.
    """
    rng = np.random.default_rng(0)
    for _ in range(2_000):
        n = int(rng.integers(1, 200))
        a = int(rng.integers(1, n + 1))
        b = int(rng.integers(1, n + 1))
        assert certified_reduction(n, a, b) == certified_error_floor(
            n, a
        ) - certified_error_floor(n, b)
        assert certified_reduction(n, a, b) == -certified_reduction(n, b, a)


def test_certified_reduction_is_zero_on_the_null_intervention():
    """An intervention that does not move the partition proves nothing new."""
    for n in range(1, 60):
        for m in range(1, n + 1):
            assert certified_reduction(n, m, m) == 0.0


# --- Theorem 2 as a property ------------------------------------------------


def test_t2_is_exactly_the_reciprocal():
    """`1/k` bit-for-bit for every block size up to 4096."""
    for k in range(1, 4097):
        assert pooling_recovery_bound(k) == 1 / k


def test_recovery_ceiling_gain_is_the_block_size_ratio():
    """Shrinking the largest block from k0 to k1 multiplies the ceiling by k0/k1.

    Pinned exactly on dyadic block sizes, where `1/k` is representable and the
    double reciprocal cannot round. Off the dyadic grid it can: `(1/5)/(1/3)`
    differs from `3/5` in the last bit, and 363 of the 1,600 ordered pairs drawn
    from 1..40 miss bit-equality for that reason. Those are held to 2 ulp of the
    exact quotient, which is the worst a double rounding can cost -- not to
    `pytest.approx`, whose default `abs=1e-12` floor would dominate every ratio
    below ~1e3 and admit an error nine orders of magnitude larger. Catches an
    inverted ratio, which would report every widening intervention as a gain.
    """
    dyadic = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for k0, k1 in itertools.product(dyadic, repeat=2):
        assert recovery_ceiling_gain(k0, k1) == k0 / k1

    for k0, k1 in itertools.product(range(1, 41), repeat=2):
        exact = k0 / k1
        assert abs(recovery_ceiling_gain(k0, k1) - exact) <= 2 * math.ulp(exact)


def test_t2_no_deterministic_decoder_beats_the_bound_on_a_pooled_block():
    """Exhaustive over every deterministic `h` on a pooled block, k = 1..10.

    `f` is constant on the block, so `h . f` takes one value there and the whole
    space of deterministic decoders is enumerated by that value. The empirical
    recovery rate is then `1/k` at best. Catches any claim that a cleverer
    downstream head recovers a pooled entity.
    """
    for k in range(1, 11):
        entities = np.arange(k)
        rates = [float(np.mean(entities == guess)) for guess in range(k)]
        assert max(rates) == pooling_recovery_bound(k)
        assert all(r <= pooling_recovery_bound(k) for r in rates)


def test_t2_partial_pooling_caps_recovery_within_every_orbit():
    """Exhaustive over all k^k answer maps and all k^k decoders, k = 1..4.

    Theorem 2 is a statement about one block, so it is checked per orbit: within
    an orbit of size `s`, no `h` exceeds `1/s`. Globally the best decoder recovers
    exactly one entity per orbit, hence `m` of `k`. Catches a bound stated over
    the whole entity set -- `m/k` routinely exceeds `1/largest_orbit`, so the two
    readings are not interchangeable.
    """
    for k in range(1, 5):
        for f in itertools.product(range(k), repeat=k):
            orbits: dict[int, list[int]] = {}
            for e, a in enumerate(f):
                orbits.setdefault(a, []).append(e)
            best = 0
            for h in itertools.product(range(k), repeat=k):
                best = max(best, sum(h[f[e]] == e for e in range(k)))
                for block in orbits.values():
                    rate = sum(h[f[e]] == e for e in block) / len(block)
                    assert rate <= pooling_recovery_bound(len(block))
            assert best == len(orbits), "one recovery per orbit, no more"


# --- every rejection branch -------------------------------------------------


def test_orbit_error_bound_rejects_an_empty_entity_set():
    with pytest.raises(ValueError):
        orbit_error_bound(0, 1)


def test_orbit_error_bound_rejects_orbit_counts_outside_one_to_n():
    """`m` counts nonempty blocks of `n` entities, so it lives in [1, n]."""
    with pytest.raises(ValueError):
        orbit_error_bound(5, 0)
    with pytest.raises(ValueError):
        orbit_error_bound(5, 6)
    with pytest.raises(ValueError):
        orbit_error_bound(5, -1)


def test_the_rate_helpers_inherit_the_same_rejections():
    """`certified_error_floor` and `certified_reduction` must not launder a bad m.

    Catches a refactor that computes the rate arithmetically instead of routing
    through `orbit_error_bound`, which would silently return a negative floor for
    `m > n`.
    """
    with pytest.raises(ValueError):
        certified_error_floor(5, 6)
    with pytest.raises(ValueError):
        certified_error_floor(0, 1)
    with pytest.raises(ValueError):
        certified_reduction(5, 6, 5)
    with pytest.raises(ValueError):
        certified_reduction(5, 5, 0)


def test_pooling_recovery_bound_rejects_empty_blocks():
    with pytest.raises(ValueError):
        pooling_recovery_bound(0)
    with pytest.raises(ValueError):
        pooling_recovery_bound(-3)


def test_recovery_ceiling_gain_rejects_empty_blocks_on_either_side():
    with pytest.raises(ValueError):
        recovery_ceiling_gain(0, 4)
    with pytest.raises(ValueError):
        recovery_ceiling_gain(4, 0)


def test_path_integral_change_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        path_integral_change(lambda h: h, np.zeros(3), np.zeros(4))


def test_volume_decay_rate_rejects_negative_step_counts():
    with pytest.raises(ValueError):
        volume_decay_rate(np.array([-1.0]), -1)


# --- negative control -------------------------------------------------------


def test_non_injective_relation_falls_below_the_bound():
    """Six countries, one continent each: `n - m` counts 4 errors where there are 0.

    This is precisely why the `injective` precondition exists. Theorem 1 derives
    `sum(s_i - 1)` from the fact that an injective `R` gives distinct entities
    distinct correct answers, so a shared orbit convicts all but one member.
    Drop injectivity and the premise is gone: "which continent is {e} in" gives
    Japan and India the same correct answer, so an orbit of size 3 can be
    entirely correct. Here `n = 6`, `m = 2`, so the quantity reads 4 while the
    true error count is 0 -- it is not a lower bound on this relation, it is
    simply the wrong quantity.

    `OrbitReport.certified_errors` is therefore required to return 0 whenever
    `RelationSpec.injective` is False, rather than reporting a number that
    happens to be arithmetically available.
    """
    ents = ("Japan", "India", "Korea", "France", "Spain", "Italy")
    ASIA, EUROPE = 1, 2
    gold = {"Japan": ASIA, "India": ASIA, "Korea": ASIA,
            "France": EUROPE, "Spain": EUROPE, "Italy": EUROPE}
    answers = {"Japan": ASIA, "India": ASIA, "Korea": ASIA,
               "France": EUROPE, "Spain": EUROPE, "Italy": EUROPE}
    r = orbit_partition(RelationSpec(CONTINENT_TPLS, ents, injective=False), make_fn(answers))

    assert r.n_distinct == 2
    assert r.largest_orbit == 3
    actual = sum(answers[e] != gold[e] for e in ents)
    assert actual == 0, "every answer here is correct"
    assert orbit_error_bound(len(ents), r.n_distinct) == 4
    assert actual < orbit_error_bound(len(ents), r.n_distinct)

    assert r.certified_errors == 0
    assert r.certified_error_rate == 0.0
    assert not r.collapsed


def test_the_same_partition_under_injectivity_does_certify():
    """Control for the case above: identical partition, injective flag flipped.

    Isolates the precondition as the only variable. The partition is byte-for-byte
    the same, so any difference in `certified_errors` comes from `injective` and
    nothing else.
    """
    ents = ("Japan", "India", "Korea", "France", "Spain", "Italy")
    answers = {"Japan": 1, "India": 1, "Korea": 1, "France": 2, "Spain": 2, "Italy": 2}
    inj = orbit_partition(RelationSpec(CONTINENT_TPLS, ents, injective=True), make_fn(answers))
    non = orbit_partition(RelationSpec(CONTINENT_TPLS, ents, injective=False), make_fn(answers))

    assert inj.orbits == non.orbits
    assert inj.n_distinct == non.n_distinct == 2
    assert inj.certified_errors == 4
    assert non.certified_errors == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
