"""Kernel-grade contracts: determinism, ties, and edge shapes.

Modelled on the discipline that a topology-derived attention schedule needs
before it can be trusted. A parity claim verified only on well-separated
continuous inputs failed 123 of 200 tie-heavy cases in prior work, so ties get a
dedicated section here rather than a passing mention.

Everything is exact. No tolerance appears in this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.regime import RelationSpec, orbit_partition, symmetry_scores
from caustic.repair import repair_by_context

TPLS = ("R {e} ?", "S {e} !")


def const(v):
    return lambda p: v


# --- determinism -----------------------------------------------------------


def test_partition_is_bitwise_identical_across_repeated_calls():
    """Non-determinism here invalidates every before/after comparison."""
    ents = tuple(f"E{i:03d}" for i in range(50))
    fn = lambda p: hash(p) % 7  # deterministic within a process
    spec = RelationSpec(TPLS, ents)
    a, b = orbit_partition(spec, fn), orbit_partition(spec, fn)
    assert a.answers == b.answers
    assert a.n_distinct == b.n_distinct and a.largest_orbit == b.largest_orbit
    assert a.orbits == b.orbits


def test_answer_fn_is_called_exactly_once_per_entity():
    """A duplicated call would silently double-count an orbit."""
    calls = []
    ents = tuple(f"E{i}" for i in range(30))
    orbit_partition(RelationSpec(TPLS, ents), lambda p: calls.append(p) or 1)
    assert len(calls) == 30
    assert len(set(calls)) == 30


def test_symmetry_scores_call_count_is_templates_times_entities():
    calls = []
    ents = tuple(f"E{i}" for i in range(8))
    symmetry_scores(RelationSpec(TPLS, ents), lambda p: calls.append(p) or 1)
    assert len(calls) == 8 * len(TPLS)


# --- ties ------------------------------------------------------------------


def test_every_entity_tied_gives_one_orbit_not_n():
    """The degenerate tie. A partition keyed on identity rather than value
    would report n singletons here and hide a total collapse."""
    ents = tuple(f"E{i}" for i in range(40))
    r = orbit_partition(RelationSpec(TPLS, ents), const(3))
    assert r.n_distinct == 1 and r.largest_orbit == 40
    assert r.certified_errors == 39


def test_answers_that_compare_equal_but_are_distinct_objects_share_an_orbit():
    """Orbits are keyed by value. Two ints that are equal must not split."""
    ents = ("A", "B")
    fn = lambda p: 10**18 + (0 if "A" in p else 0)  # equal, separately constructed
    r = orbit_partition(RelationSpec(TPLS, ents), fn)
    assert r.n_distinct == 1 and r.largest_orbit == 2


def test_exactly_half_tied_is_counted_correctly():
    ents = tuple(f"E{i:02d}" for i in range(20))
    fn = lambda p: 0 if int(p.split()[1][1:]) < 10 else int(p.split()[1][1:])
    r = orbit_partition(RelationSpec(TPLS, ents), fn)
    assert r.largest_orbit == 10
    assert r.n_distinct == 11  # one orbit of 10, plus 10 singletons
    assert r.certified_errors == 9


# --- edge shapes -----------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 100, 257])
def test_partition_totals_always_reconcile(n):
    ents = tuple(f"E{i}" for i in range(n))
    r = orbit_partition(RelationSpec(TPLS, ents), lambda p: len(p) % 5)
    assert sum(len(v) for v in r.orbits.values()) == n
    assert r.largest_orbit <= n and r.n_distinct <= n
    assert r.certified_errors == n - r.n_distinct


def test_minimum_viable_relation_of_two_entities():
    r = orbit_partition(RelationSpec(TPLS, ("A", "B")), const(1))
    assert r.largest_orbit == 2 and r.certified_errors == 1


def test_unicode_and_punctuation_entities_are_handled():
    ents = ("Ãland", "Cote d'Ivoire", "Sao Tome & Principe", "Timor-Leste")
    r = orbit_partition(RelationSpec(TPLS, ents), lambda p: len(p))
    assert sum(len(v) for v in r.orbits.values()) == 4


def test_duplicate_entities_are_rejected_rather_than_partitioned():
    """Two copies of one entity used to give one answer, hence an orbit of two,
    hence a certified error the model did not make. Not deduplicating was the
    lesser of two wrong behaviours; the spec now refuses to be built at all."""
    with pytest.raises(ValueError, match="duplicate entity"):
        RelationSpec(TPLS, ("A", "A", "B"))


def test_repair_of_an_already_separated_partition_is_not_a_repair():
    spec = RelationSpec(TPLS, tuple(f"E{i}" for i in range(6)))
    r = repair_by_context(spec, lambda p: hash(p), prefix="P ")
    assert not r.repaired


def test_every_public_name_resolves():
    """`__all__` integrity, which no test checked and a merge quietly broke.

    Three keep-both copies of the same `from .regime import` line each carried
    one branch's new name, and the union lost `injective_subset`: it stayed in
    `__all__`, was never bound, and `from caustic import *` raised on a package
    whose 355 tests were green.
    """
    import caustic

    missing = [n for n in caustic.__all__ if not hasattr(caustic, n)]
    assert missing == [], f"listed in __all__ but not importable: {missing}"


def test_no_public_name_is_listed_twice():
    """The same union-of-lists artifact that cost the export above."""
    import caustic

    dupes = sorted({n for n in caustic.__all__ if caustic.__all__.count(n) > 1})
    assert dupes == [], f"duplicated in __all__: {dupes}"
