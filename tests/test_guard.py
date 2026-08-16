"""The inference-time path: measure, decide, act, and say what to withhold.

Every piece of this exists already — `orbit_partition` measures, `select_prefix`
decides using Theorem 1 as a ground-truth-free objective, `repair_by_context`
acts, and Theorem 6 bounds the precision of the withheld set. Nothing composed
them, so a caller wanting a guard had to know all four and get the order right.

The tests below pin the properties that make the composition safe rather than
merely convenient: it never ships answers worse than doing nothing, it withholds
a set whose precision it can prove, and it refuses relations where the
underlying theorem does not hold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.guard import GuardReport, guard
from caustic.regime import RelationSpec

TPLS = ("The capital of {e} is", "{e}'s capital is")
ENTS = ("France", "Japan", "Peru", "Chile")


def make_fn(base, prefixed=None):
    """Answer function that can respond differently once a prefix is present.

    `base` is consulted when the prompt starts with a template; `prefixed` when
    a prefix has been prepended. Entity keys are matched longest-first, since a
    naive substring match makes "C1" match "C15".
    """

    def fn(prompt):
        table = base
        if prefixed is not None and not prompt.startswith(("The capital", "France", "Japan")):
            if not prompt.startswith(TPLS[0][:6]) and not prompt.startswith(TPLS[1][:3]):
                table = prefixed
        for k in sorted(table, key=len, reverse=True):
            if k in prompt:
                return table[k]
        return 0

    return fn


def test_a_collapsed_relation_is_repaired_and_the_floor_falls():
    """The whole point: collapse in, a better partition out, chosen blind."""
    collapsed = dict.fromkeys(ENTS, 7)
    fixed = {"France": 1, "Japan": 2, "Peru": 3, "Chile": 4}
    rep = guard(
        RelationSpec(TPLS, ENTS),
        make_fn(collapsed, fixed),
        candidates={"prose": "Mechanical calculators. "},
    )
    assert rep.intervened
    assert rep.baseline_floor == pytest.approx(0.75)
    assert rep.floor == 0.0
    assert rep.abstain == []


def test_the_guard_declines_when_no_candidate_beats_doing_nothing():
    """Declining is a real outcome, and the abstain set must survive it.

    A prefix can make things worse — an incoherent one drove the largest orbit
    from 4 to 20 in this repository's own measurements. The guard therefore
    always enters the empty prefix and reports the withheld set either way.
    """
    base = {"France": 7, "Japan": 7, "Peru": 3, "Chile": 4}
    rep = guard(RelationSpec(TPLS, ENTS), make_fn(base), candidates={"noop": ""})
    assert not rep.intervened
    assert set(rep.abstain) == {"France", "Japan"}
    assert rep.certified_precision == pytest.approx(0.5)


def test_the_withheld_set_carries_its_proved_precision():
    """Theorem 6, surfaced where a caller acts on it rather than in a docstring."""
    base = {"France": 7, "Japan": 7, "Peru": 7, "Chile": 4}
    rep = guard(RelationSpec(TPLS, ENTS), make_fn(base), candidates={})
    assert set(rep.abstain) == {"France", "Japan", "Peru"}
    assert rep.certified_errors == 2
    assert rep.certified_precision == pytest.approx(2 / 3)


def test_kept_answers_exclude_exactly_the_withheld_entities():
    """`kept` is what a caller ships, so it must be the complement, not a copy."""
    base = {"France": 7, "Japan": 7, "Peru": 3, "Chile": 4}
    rep = guard(RelationSpec(TPLS, ENTS), make_fn(base), candidates={})
    assert set(rep.kept) == {"Peru", "Chile"}
    assert set(rep.kept) & set(rep.abstain) == set()
    assert set(rep.kept) | set(rep.abstain) == set(ENTS)


def test_an_inadmissible_answer_is_withheld_even_from_a_singleton():
    """With the answer set, a lone wrong answer becomes certifiable.

    Theorem 1 is silent on singletons — no partition-only argument can certify
    one wrong. Supplying `gold_keys` adds the entities whose answer is not a
    correct answer for anybody, and those are wrong outright.
    """
    base = {"France": 1, "Japan": 2, "Peru": 3, "Chile": 99}
    rep = guard(
        RelationSpec(TPLS, ENTS),
        make_fn(base),
        candidates={},
        gold_keys={1, 2, 3, 4},
    )
    assert rep.abstain == ["Chile"]
    assert rep.certified_errors == 1
    assert rep.certified_precision == 1.0


def test_a_non_injective_relation_is_refused_rather_than_scored():
    """The objective is Theorem 1; on a many-to-one relation the sign inverts."""
    spec = RelationSpec(TPLS, ENTS, injective=False)
    with pytest.raises(ValueError, match="injective"):
        guard(spec, make_fn(dict.fromkeys(ENTS, 7)), candidates={})


def test_the_guard_costs_one_forward_pass_per_entity_per_candidate():
    """Cost is counted, not asserted: this has to be cheap enough to deploy."""
    calls = []
    base = {"France": 7, "Japan": 7, "Peru": 3, "Chile": 4}

    def counting_fn(prompt):
        calls.append(prompt)
        for k in sorted(base, key=len, reverse=True):
            if k in prompt:
                return base[k]
        return 0

    guard(RelationSpec(TPLS, ENTS), counting_fn, candidates={"a": "x ", "b": "y "})
    # 3 candidates counting the empty prefix, 4 entities, one pass each.
    assert len(calls) == 3 * len(ENTS)


def test_the_report_is_readable_without_reading_the_source():
    """A guard nobody can interpret at 2am is not a guard."""
    base = {"France": 7, "Japan": 7, "Peru": 3, "Chile": 4}
    rep = guard(RelationSpec(TPLS, ENTS), make_fn(base), candidates={})
    text = str(rep)
    assert "withhold 2 of 4" in text
    assert "precision >= 0.500" in text
    assert isinstance(rep, GuardReport)
