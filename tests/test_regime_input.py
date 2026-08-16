"""Preconditions of the orbit bound, checked rather than assumed.

Two things the certificate depends on were previously unenforced: that the
entity list is a genuine set, and that `injective=True` is true of the map the
bound is actually applied to. Both are checked here with stub functions — no
tokenizer, no model, no GPU — so every expected value is a closed form written
out by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.regime import RelationSpec, orbit_partition, verify_injective

TPLS = ("The capital of {e} is", "{e}'s capital is")
ENTS = ("France", "Italy", "Japan", "Spain")

# Stub first_token: the answer's first character. "Asmara"/"Asuncion" collide on
# "A" exactly as they collide on token 1634 under the Qwen2.5 tokenizer.
FIRST_CHAR = lambda s: ord(s[0])


# --- entity validation -----------------------------------------------------


def test_duplicate_entity_is_rejected_by_name():
    """A repeated entity used to inflate certified_errors; the message must name it."""
    with pytest.raises(ValueError, match="Italy"):
        RelationSpec(TPLS, ("France", "Italy", "Italy", "Spain"))


def test_duplicate_entity_would_have_certified_an_error_that_does_not_exist():
    """The regression this guard exists for, shown as the number it corrupts.

    Three entities, one duplicated, a perfect model answering each entity with
    its own distinct answer: 4 slots, 3 distinct answers, so the bound reports
    1 certified error while the model made none. Constructing the spec is now
    the failure point, so the wrong number can never be produced again.
    """
    ents = ("France", "Italy", "Italy", "Spain")
    answers = {"France": 1, "Italy": 2, "Spain": 3}
    fn = lambda p: next(v for k, v in answers.items() if k in p)

    with pytest.raises(ValueError):
        RelationSpec(TPLS, ents)

    # What the pre-fix path computed, reproduced by bypassing the constructor.
    bad = RelationSpec(TPLS, ("France", "Italy", "Spain"))
    object.__setattr__(bad, "entities", ents)
    r = orbit_partition(bad, fn)
    assert r.n_distinct == 3
    assert r.certified_errors == 1, "the pre-fix bug: a certified error with zero errors"
    assert sum(fn(TPLS[0].format(e=e)) != answers[e] for e in set(ents)) == 0


@pytest.mark.parametrize("blank", ["", " ", "\t", "\n  "])
def test_blank_entity_is_rejected(blank):
    """A blank entity formats into a prompt the relation does not describe."""
    with pytest.raises(ValueError):
        RelationSpec(TPLS, ("France", blank))


def test_distinct_non_blank_entities_are_accepted():
    """Negative control: the validation must not reject a legitimate spec."""
    assert RelationSpec(TPLS, ENTS).entities == ENTS


# --- verify_injective ------------------------------------------------------


def test_no_collision_returns_empty_list():
    """Negative control: distinct first tokens, so the precondition holds."""
    gold = {"France": "Paris", "Italy": "Rome", "Japan": "Tokyo", "Spain": "Madrid"}
    assert verify_injective(RelationSpec(TPLS, ENTS), gold, FIRST_CHAR) == []


def test_the_small_capital_collision_is_found():
    """Asmara/Asuncion and Lusaka/Ljubljana, the two pairs measured on Qwen2.5-0.5B."""
    ents = ("Eritrea", "Paraguay", "Zambia", "Slovenia", "Latvia")
    gold = {"Eritrea": "Asmara", "Paraguay": "Asuncion", "Zambia": "Lusaka",
            "Slovenia": "Ljubljana", "Latvia": "Riga"}
    assert verify_injective(RelationSpec(TPLS, ents), gold, FIRST_CHAR) == [
        ("Eritrea", "Paraguay"),
        ("Zambia", "Slovenia"),
    ]


def test_pairs_come_back_in_combinations_order_over_spec_entities():
    """Three entities on one token give all 3 pairs, ordered by the entity tuple."""
    ents = ("C", "B", "A")
    gold = {"C": "same", "B": "so", "A": "s"}
    assert verify_injective(RelationSpec(TPLS, ents), gold, FIRST_CHAR) == [
        ("C", "B"),
        ("C", "A"),
        ("B", "A"),
    ]


def test_a_collision_forces_an_orbit_the_bound_reads_as_an_error():
    """Why the check matters: a perfect model still loses on a collided pair.

    `answer_fn` returns the gold first token, so the model is right about every
    entity. The two colliding entities still land in one orbit and the bound
    certifies 1 error against 0 real ones.
    """
    ents = ("Eritrea", "Paraguay", "Latvia")
    gold = {"Eritrea": "Asmara", "Paraguay": "Asuncion", "Latvia": "Riga"}
    spec = RelationSpec(TPLS, ents)
    perfect = lambda p: FIRST_CHAR(gold[next(e for e in ents if e in p)])
    r = orbit_partition(spec, perfect)
    assert r.certified_errors == 1
    assert len(verify_injective(spec, gold, FIRST_CHAR)) == 1


def test_gold_missing_an_entity_is_rejected():
    """Silently skipping an uncovered entity would report a clean audit falsely."""
    gold = {"France": "Paris", "Italy": "Rome", "Japan": "Tokyo"}
    with pytest.raises(ValueError, match="Spain"):
        verify_injective(RelationSpec(TPLS, ENTS), gold, FIRST_CHAR)


def test_gold_may_carry_entities_outside_the_spec():
    """Relation dicts are shared across experiments; extra keys are not an error."""
    gold = {e: a for e, a in zip(ENTS, "PRTM")}
    gold["Kenya"] = "Nairobi"
    assert verify_injective(RelationSpec(TPLS, ENTS), gold, FIRST_CHAR) == []


def test_non_injective_declaration_does_not_suppress_the_report():
    """The function reports collisions as measured; acting on them is the caller's."""
    ents = ("Japan", "India", "France")
    gold = {"Japan": "Asia", "India": "Asia", "France": "Europe"}
    spec = RelationSpec(TPLS, ents, injective=False)
    assert verify_injective(spec, gold, FIRST_CHAR) == [("Japan", "India")]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
