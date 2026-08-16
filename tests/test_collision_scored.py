"""The collision score borrows from templates it is not scoring.

`symmetry_scores` averages collision over every template, while a caller acts on
the answer from ONE template — `orbit_partition` uses `templates[0]`. When the
scored template's partition differs from the others, the averaged score is
measuring a different function than the label, and it can invert.

Measured on Qwen2.5-0.5B, seed 0, `currency`, 12 entities, accuracy 0.500:

    template  m_i  collisions   AUROC(collision -> wrong)
        0      12       0           0.5000
        1      12       0           0.5000
        2       5       9           0.4028
        3      12       0           0.5000
        4       4       9           0.4167
    AVERAGED            12          0.3056   <- shipped

The averaged score is below every one of its five components. Template 0 has no
collisions at all, so the whole signal is borrowed, and against template 0's
label the borrowed templates sit under chance. `capital` is the opposite case:
template 0 collapses there, the other templates agree, and averaging helps
(0.8889 -> 0.9949). Averaging is therefore not safe in general; it is safe when
the partitions agree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.regime import RelationSpec, symmetry_scores

TPLS = ("A {e} x", "B {e} y", "C {e} z")


def make_fn(table):
    """Answer function reading a per-(entity, template) table, longest key first."""

    def fn(prompt):
        for e in sorted(table, key=len, reverse=True):
            if e in prompt:
                return table[e][prompt[0]]
        raise KeyError(prompt)

    return fn


def test_scored_collision_ignores_templates_that_are_not_scored():
    """Template A separates all three entities; B and C collapse them.

    The averaged collision is high for every entity because of B and C, but the
    partition a caller acts on is A's, where nothing collides. The scored
    collision must read 0.0 for all three; the averaged one must not.
    """
    table = {"e1": {"A": 1, "B": 9, "C": 9}, "e2": {"A": 2, "B": 9, "C": 9},
             "e3": {"A": 3, "B": 9, "C": 9}}
    out = symmetry_scores(RelationSpec(TPLS, tuple(table)), make_fn(table))
    for e in table:
        assert out[e]["collision"] == pytest.approx(2.0 / 3.0)
        assert out[e]["collision_scored"] == 0.0


def test_scored_collision_equals_averaged_when_partitions_agree():
    """Negative control: identical partitions across templates give one number.

    This is the `capital` regime, where averaging is safe and helps.
    """
    table = {"e1": {"A": 1, "B": 1, "C": 1}, "e2": {"A": 1, "B": 1, "C": 1},
             "e3": {"A": 3, "B": 3, "C": 3}}
    out = symmetry_scores(RelationSpec(TPLS, tuple(table)), make_fn(table))
    for e in table:
        assert out[e]["collision_scored"] == pytest.approx(out[e]["collision"])


def test_scored_template_is_selectable_and_matches_the_partition_used():
    """A caller scoring template 1 must be able to score collision on template 1."""
    table = {"e1": {"A": 1, "B": 7, "C": 3}, "e2": {"A": 2, "B": 7, "C": 4}}
    out = symmetry_scores(RelationSpec(TPLS, tuple(table)), make_fn(table), scored_template=1)
    for e in table:
        assert out[e]["collision_scored"] == 1.0  # both entities answer 7 under B


def test_an_out_of_range_scored_template_is_rejected():
    """Silently clamping to template 0 would reintroduce the borrowing bug."""
    table = {"e1": {"A": 1, "B": 1, "C": 1}, "e2": {"A": 2, "B": 2, "C": 2}}
    with pytest.raises(ValueError, match="scored_template"):
        symmetry_scores(RelationSpec(TPLS, tuple(table)), make_fn(table), scored_template=3)
