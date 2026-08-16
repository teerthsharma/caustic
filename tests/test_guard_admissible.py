"""The guard reports the floor it is entitled to, not the weaker one.

`guard` computes the withheld set from the partition and, when `gold_keys` is
supplied, from the answer set as well. Its precision floor should use the same
resource. Theorem 6 proves `1 - b/|S|`; Theorem 6* proves `1 - b_adm/|S|`, where
`b_adm` counts only the flagged orbits whose shared answer could be somebody's
correct answer. Reporting the weaker number when the stronger one is available
understates a guarantee a caller may be relying on.

The counting is at the ORBIT level, not the entity level, which is the mistake
this file exists to catch: three entities pooled on an inadmissible token are
one orbit contributing zero, not three entities contributing one each.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.guard import guard
from caustic.regime import RelationSpec

TPLS = ("The capital of {e} is", "{e}'s capital is")
ENTS = ("France", "Japan", "Peru", "Chile")


def make_fn(base):
    def fn(prompt):
        for k in sorted(base, key=len, reverse=True):
            if k in prompt:
                return base[k]
        return 0

    return fn


def test_a_wholly_inadmissible_collapse_certifies_precision_one():
    """The `" the" x 128` shape: one orbit, an answer nobody's truth could be.

    Theorem 6 proves 3/4 here. Theorem 6* proves 4/4, because making any member
    correct would require the truth to assign token 99 to somebody, and 99 is
    not in G.
    """
    rep = guard(
        RelationSpec(TPLS, ENTS),
        make_fn(dict.fromkeys(ENTS, 99)),
        candidates={},
        gold_keys={1, 2, 3, 4},
    )
    assert set(rep.abstain) == set(ENTS)
    assert rep.certified_precision == 1.0


def test_an_admissible_collapse_still_leaves_one_entity_uncertified():
    """Negative control: token 1 IS somebody's capital, so one member may be right."""
    rep = guard(
        RelationSpec(TPLS, ENTS),
        make_fn(dict.fromkeys(ENTS, 1)),
        candidates={},
        gold_keys={1, 2, 3, 4},
    )
    assert set(rep.abstain) == set(ENTS)
    assert rep.certified_precision == pytest.approx(3 / 4)


def test_admissible_orbits_are_counted_once_not_once_per_entity():
    """The orbit-level count, which an entity-level one would get wrong.

    Three entities share admissible token 1 and one entity holds inadmissible
    token 99. Flagged = 4. One admissible orbit, so the floor is 3/4. Counting
    admissible ENTITIES instead would give 1/4 and understate it threefold.
    """
    rep = guard(
        RelationSpec(TPLS, ENTS),
        make_fn({"France": 1, "Japan": 1, "Peru": 1, "Chile": 99}),
        candidates={},
        gold_keys={1, 2, 3, 4},
    )
    assert set(rep.abstain) == set(ENTS)
    assert rep.certified_precision == pytest.approx(3 / 4)


def test_an_inadmissible_singleton_is_certified_outright():
    """A lone wrong answer is fully certified: it forms no orbit to hide in."""
    rep = guard(
        RelationSpec(TPLS, ENTS),
        make_fn({"France": 1, "Japan": 2, "Peru": 3, "Chile": 99}),
        candidates={},
        gold_keys={1, 2, 3, 4},
    )
    assert rep.abstain == ["Chile"]
    assert rep.certified_precision == 1.0


def test_without_gold_keys_the_guard_falls_back_to_theorem_6():
    """No G means every shared answer must be assumed admissible."""
    rep = guard(RelationSpec(TPLS, ENTS), make_fn(dict.fromkeys(ENTS, 99)), candidates={})
    assert set(rep.abstain) == set(ENTS)
    assert rep.certified_precision == pytest.approx(3 / 4)


def test_the_floor_never_falls_below_the_theorem_6_value():
    """Domination, checked through the guard rather than the bare function."""
    base = {"France": 1, "Japan": 1, "Peru": 99, "Chile": 99}
    spec = RelationSpec(TPLS, ENTS)
    without = guard(spec, make_fn(base), candidates={})
    withg = guard(spec, make_fn(base), candidates={}, gold_keys={1, 2, 3, 4})
    assert withg.certified_precision >= without.certified_precision
    assert withg.certified_precision == pytest.approx(3 / 4)


def test_the_two_routes_to_the_floor_are_the_same_number():
    """Lemma: `n - m*` equals `|S*| - b_adm`, so the guard needs only one of them.

    Theorem 1* counts over ALL entities: `n - |f(E) and G|`. Theorem 6* counts
    over the flagged ones: `|S*| - b_adm`. They agree, and not by accident.

    Proof. An unflagged entity is a singleton whose answer is admissible. Two
    unflagged entities cannot share an answer, or they would form an orbit of
    size two and be flagged. So the unflagged entities contribute exactly one
    distinct admissible answer each, giving `m* = b_adm + u` for `u` unflagged.
    Then `n - m* = (|S*| + u) - (b_adm + u) = |S*| - b_adm`. []

    Checked over 20,000 random configurations. Without this lemma the guard
    would have to count orbits separately, and an entity-level count is the
    natural mistake: three entities pooled on one admissible token are one
    admissible orbit, not three.
    """
    rng = np.random.default_rng(0)
    for _ in range(20_000):
        n = int(rng.integers(2, 25))
        answers = rng.integers(0, n, n).tolist()
        gold = set(rng.choice(np.arange(2 * n), size=n, replace=False).tolist())

        orbits: dict[int, int] = {}
        for a in answers:
            orbits[a] = orbits.get(a, 0) + 1
        flagged = {a for a, k in orbits.items() if k > 1 or a not in gold}
        n_flagged = sum(orbits[a] for a in flagged)
        b_adm = sum(1 for a in flagged if a in gold and orbits[a] > 1)

        m_star = len(set(answers) & gold)
        assert n - m_star == n_flagged - b_adm


def test_colliding_gold_keys_are_refused_rather_than_certified():
    """The guard must not certify errors on a relation it cannot measure.

    `guard` reimplemented Theorem 1* inline and skipped the precondition check
    that `admissible_distinct` performs. On the repository's own motivating
    collision — Asmara/Asuncion sharing token 1634, Lusaka/Ljubljana sharing
    444 — it reported 2 certified errors and a 0.5 precision floor over four
    answers that were all correct.
    """
    gold = {"Eritrea": 1634, "Paraguay": 1634, "Zambia": 444, "Slovenia": 444}
    ents = tuple(gold)
    spec = RelationSpec(TPLS, ents)

    def fn(prompt):
        for k in sorted(gold, key=len, reverse=True):
            if k in prompt:
                return gold[k]
        return 0

    with pytest.raises(ValueError, match="not injective at this encoding"):
        guard(spec, fn, candidates={}, gold_keys=set(gold.values()))
