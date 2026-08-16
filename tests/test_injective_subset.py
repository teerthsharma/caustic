"""Restricting a relation to the entities the observable can actually separate.

`verify_injective` reports that a relation's gold answers collide at the
encoding `answer_fn` compares, and every experiment then skips the whole
relation. That throws away most of a usable relation: `small_capital` has 18
entities and 16 distinct gold first tokens under Qwen, so 16 of them satisfy the
precondition and only the two colliding pairs do not.

`injective_subset` keeps one entity per colliding group, so the certificate
applies to the restriction rather than to nothing. The choice of which member to
keep is arbitrary and therefore made deterministic — first in the caller's order
— so two runs restrict identically.

The restriction is a scope reduction, not a repair. It says the bound holds on
these 16 entities, and says nothing about the 2 removed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.regime import RelationSpec, injective_subset, verify_injective

TPLS = ("The capital of {e} is", "{e}'s capital is")


def first_letter(answer: str) -> str:
    """Stand-in for a tokenizer: collides exactly when two answers share a letter."""
    return answer[0]


def test_a_colliding_relation_is_restricted_rather_than_refused():
    """Eritrea/Asmara and Paraguay/Asuncion collide on ' As'; keep one."""
    # Asmara/Asuncion share 'A'; Lusaka/Lima share 'L'. The stand-in collides
    # more readily than a tokenizer would, which is what makes it a test.
    gold = {"Eritrea": "Asmara", "Paraguay": "Asuncion", "Zambia": "Lusaka", "Peru": "Lima"}
    spec = RelationSpec(TPLS, tuple(gold))
    sub, dropped = injective_subset(spec, gold, first_letter)
    assert set(sub.entities) == {"Eritrea", "Zambia"}
    assert dropped == ["Paraguay", "Peru"]
    assert verify_injective(sub, {e: gold[e] for e in sub.entities}, first_letter) == []


def test_an_already_injective_relation_is_returned_unchanged():
    """Negative control: no entity is dropped when the precondition holds."""
    gold = {"France": "Paris", "Japan": "Tokyo", "Peru": "Lima"}
    spec = RelationSpec(TPLS, tuple(gold))
    sub, dropped = injective_subset(spec, gold, first_letter)
    assert sub.entities == spec.entities
    assert dropped == []


def test_the_kept_member_is_the_first_in_caller_order():
    """Which member survives is arbitrary, so it is pinned rather than left to a set."""
    gold = {"Zambia": "Lusaka", "Slovenia": "Ljubljana", "Peru": "Madrid"}
    spec = RelationSpec(TPLS, tuple(gold))
    sub, dropped = injective_subset(spec, gold, first_letter)
    assert sub.entities == ("Zambia", "Peru")
    assert dropped == ["Slovenia"]


def test_the_restriction_is_deterministic_across_runs():
    """Two calls on the same input must restrict identically."""
    gold = {"A1": "Xa", "B1": "Xb", "C1": "Yc", "D1": "Yd", "E1": "Ze"}
    spec = RelationSpec(TPLS, tuple(gold))
    assert injective_subset(spec, gold, first_letter) == injective_subset(spec, gold, first_letter)


def test_other_spec_fields_survive_the_restriction():
    """Templates and the injective flag belong to the relation, not the entities."""
    gold = {"A1": "Xa", "B1": "Xb", "C1": "Yc"}
    spec = RelationSpec(TPLS, tuple(gold), injective=True)
    sub, _ = injective_subset(spec, gold, first_letter)
    assert sub.templates == TPLS
    assert sub.injective is True


def test_a_restriction_below_two_entities_is_rejected():
    """A partition needs two entities; one is not a scope reduction but a failure."""
    gold = {"A1": "Xa", "B1": "Xb"}
    with pytest.raises(ValueError, match="partition needs two"):
        injective_subset(RelationSpec(TPLS, tuple(gold)), gold, first_letter)


def test_gold_must_cover_every_entity():
    """A missing gold would silently drop an entity as if it had collided."""
    gold = {"A1": "Xa"}
    with pytest.raises(ValueError, match="every entity"):
        injective_subset(RelationSpec(TPLS, ("A1", "B1")), gold, first_letter)
