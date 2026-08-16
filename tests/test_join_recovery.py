"""Theorem 2*: what a receiver can recover when it sees every paraphrase.

Theorem 2 caps recovery of the entity from ONE answer at `1/k`, and reads that
as a pooling equilibrium: the block is unrecoverable and no downstream capability
gets it back. That is true of one answer and false of the answer vector.

Give the receiver `(f_1(e), ..., f_T(e))` and it can distinguish exactly the
blocks of the JOIN of the `T` partitions, so recovery is capped at `m_join / n`
instead. The two bounds differ by exactly the information a single template
throws away, and the gap is the reason repair is possible at all: if the join
were as coarse as the worst template, no prefix could recover anything, because
the entity would not be present in the model's outputs to begin with.

Measured across the repository's four relations and five templates, the join is
discrete in 8 of 8 conditions - `m_join = n` - so the ceiling is 1.0 even where
a single template pools 16 entities into 3 orbits.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.regime import RelationSpec, join_partition
from caustic.theorems import join_recovery_bound, pooling_recovery_bound

TPLS = ("A {e} x", "B {e} y", "C {e} z")


def make_fn(table):
    """Answer function reading a per-(entity, template) table, keyed by first letter."""

    def fn(prompt):
        for e in sorted(table, key=len, reverse=True):
            if e in prompt:
                return table[e][prompt[0]]
        raise KeyError(prompt)

    return fn


def test_the_join_recovers_what_a_single_template_pools():
    """Template A pools everything; B and C separate. The join is discrete.

    Under Theorem 2 the receiver is capped at 1/3. Under Theorem 2* it is capped
    at 1.0, and the difference is not slack - the entity really is present in
    the answer vector and really is absent from A's answer alone.
    """
    table = {"e1": {"A": 9, "B": 1, "C": 1}, "e2": {"A": 9, "B": 2, "C": 2},
             "e3": {"A": 9, "B": 3, "C": 3}}
    spec = RelationSpec(TPLS, tuple(table))
    rep = join_partition(spec, make_fn(table))
    assert rep.n_distinct == 3 and rep.largest_orbit == 1
    assert pooling_recovery_bound(3) == pytest.approx(1 / 3)
    assert join_recovery_bound(3, rep.n_distinct) == 1.0


def test_the_join_is_never_coarser_than_any_single_template():
    """Domination: refining by more coordinates can only split blocks.

    Checked over 5,000 random tables. A violation would mean the join lost
    information that a component partition had, which is impossible.
    """
    rng = np.random.default_rng(0)
    for _ in range(5_000):
        n = int(rng.integers(2, 12))
        T = int(rng.integers(2, 5))
        ents = tuple(f"e{i}" for i in range(n))
        keys = list("ABC")[:T]
        table = {e: {k: int(rng.integers(0, 4)) for k in keys} for e in ents}
        spec = RelationSpec(tuple(f"{k} {{e}} z" for k in keys), ents)
        m_join = join_partition(spec, make_fn(table)).n_distinct
        for k in keys:
            m_single = len({table[e][k] for e in ents})
            assert m_join >= m_single


def test_identical_templates_leave_the_join_equal_to_the_single_partition():
    """Negative control: no free lunch when the paraphrases agree.

    If every template returns the same answer, the join adds nothing and the two
    bounds coincide. This is what would fail if `join_partition` were splitting
    on template index rather than on the answer tuple.
    """
    table = {"e1": {"A": 7, "B": 7, "C": 7}, "e2": {"A": 7, "B": 7, "C": 7},
             "e3": {"A": 3, "B": 3, "C": 3}}
    spec = RelationSpec(TPLS, tuple(table))
    rep = join_partition(spec, make_fn(table))
    assert rep.n_distinct == 2 and rep.largest_orbit == 2
    assert join_recovery_bound(3, rep.n_distinct) == pytest.approx(2 / 3)


def test_the_bound_is_a_ceiling_on_the_fraction_recovered():
    """Enumerate every receiver on a small case rather than argue.

    A receiver is any map from the observed answer tuple to an entity. The best
    possible is one correct guess per join block, so the fraction recovered can
    never exceed m_join / n.
    """
    table = {"e1": {"A": 1, "B": 1, "C": 1}, "e2": {"A": 1, "B": 1, "C": 1},
             "e3": {"A": 2, "B": 2, "C": 2}, "e4": {"A": 2, "B": 2, "C": 2}}
    ents = tuple(table)
    spec = RelationSpec(TPLS, ents)
    rep = join_partition(spec, make_fn(table))
    best = max(
        sum(1 for e, a in zip(ents, rep.answers) if h.get(a) == e)
        for h in [{a: e for a, e in zip(set(rep.answers), pick)}
                  for pick in [(x, y) for x in ents for y in ents]]
    )
    assert best / len(ents) <= join_recovery_bound(len(ents), rep.n_distinct)
    assert join_recovery_bound(4, rep.n_distinct) == pytest.approx(0.5)


def test_it_refuses_a_join_coarser_than_one_block_or_finer_than_the_entities():
    """The count has to come from a real partition of these entities."""
    with pytest.raises(ValueError, match="n_join"):
        join_recovery_bound(4, 5)
    with pytest.raises(ValueError, match="n_join"):
        join_recovery_bound(4, 0)


def test_join_partition_needs_at_least_two_templates():
    """With one template the join is the partition, and the caller wants that one."""
    with pytest.raises(ValueError, match="two templates"):
        join_partition(RelationSpec(("A {e} x",), ("e1", "e2")), make_fn({}))
