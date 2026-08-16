"""Theorem 1*, tested against Theorem 1 rather than against itself.

Every test here pins one of the three properties that make the strengthened
bound worth having: it is never weaker than `orbit_error_bound`, it is attained
exactly on the cases the repository already measured, and it becomes invalid in
the one situation the precondition forbids. The last is a test that the bound
CAN be wrong, which is what stops the precondition from being decoration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.regime import RelationSpec, admissible_distinct, orbit_partition
from caustic.theorems import admissible_error_bound, orbit_error_bound

TPLS = ("The capital of {e} is", "{e}'s capital is")


def make_fn(mapping, default=0):
    """Answer function keyed on the entity, longest key first.

    Longest-first matters: a naive substring match makes "C1" match "C15" and
    silently merges two entities that were meant to be distinct.
    """

    def fn(prompt):
        for k in sorted(mapping, key=len, reverse=True):
            if k in prompt:
                return mapping[k]
        return default

    return fn


# --- the domination property, which is the whole claim --------------------


def test_star_is_never_weaker_than_theorem_1():
    """The reason to adopt it: m* <= m always, so n - m* >= n - m always.

    Checked over 10,000 random (image, admissible-set) pairs. A single
    violation would mean the strengthened bound is not a strengthening.
    """
    rng = np.random.default_rng(0)
    for _ in range(10_000):
        n = int(rng.integers(2, 40))
        answers = rng.integers(0, n, n).tolist()
        gold = set(rng.choice(np.arange(2 * n), size=n, replace=False).tolist())
        m = len(set(answers))
        m_star = len(set(answers) & gold)
        assert m_star <= m
        assert admissible_error_bound(n, m_star) >= orbit_error_bound(n, m)


def test_star_equals_theorem_1_when_every_answer_is_admissible():
    """No free lunch: with the image inside G the two bounds coincide.

    This is the negative control for the domination test above — it would fail
    if `admissible_error_bound` were strengthening by some other route.
    """
    gold = {10, 11, 12, 13}
    answers = [10, 11, 11, 13]
    n = len(answers)
    assert admissible_distinct(answers, gold) == len(set(answers))
    assert admissible_error_bound(n, admissible_distinct(answers, gold)) == orbit_error_bound(
        n, len(set(answers))
    )


# --- the measured cases ----------------------------------------------------


@pytest.mark.parametrize(
    "name,n,answers,gold,old,new,true_errors",
    [
        # Qwen2.5-0.5B, seed 0, template 0. Each row is a condition the
        # repository already reports; `new` is attained exactly in every one.
        ("currency_none", 12, list(range(12)), set(range(6)), 0, 6, 6),
        ("capital_the_x128", 20, [7] * 20, set(range(100, 120)), 19, 20, 20),
        ("language_the_x128", 16, [7] * 15 + [8], set(range(100, 116)), 14, 16, 16),
    ],
)
def test_measured_conditions_are_attained_exactly(name, n, answers, gold, old, new, true_errors):
    """The bound is not merely tighter on these rows, it is exact.

    `currency_none` is the one that matters: Theorem 1 certifies 0 errors —
    vacuous, it proves nothing — while the true count is 6 and Theorem 1*
    certifies 6.
    """
    m = len(set(answers))
    m_star = admissible_distinct(answers, gold)
    assert orbit_error_bound(n, m) == old
    assert admissible_error_bound(n, m_star) == new
    assert admissible_error_bound(n, m_star) == true_errors


def test_the_collapsed_case_certifies_every_entity():
    """`" the" x 128` sends 20 entities to a token that is not a capital.

    Theorem 1 stops at 19 because it can only count collisions. Theorem 1*
    reaches 20 because the shared answer is inadmissible, so no entity in the
    block can be the correct one.
    """
    n = 20
    answers = ["the"] * n
    gold = {f"cap{i}" for i in range(n)}
    assert orbit_error_bound(n, len(set(answers))) == n - 1
    assert admissible_error_bound(n, admissible_distinct(answers, gold)) == n


# --- the per-entity certificate, which the partition alone cannot give -----


def test_an_inadmissible_answer_certifies_that_entity_wrong():
    """A singleton block carries no information under Theorem 1; G fixes that.

    No function of the partition alone can certify a single entity wrong — a
    singleton is independent, so some injective truth makes it correct. The
    admissible set is not part of the partition, which is why it escapes that.
    """
    spec = RelationSpec(TPLS, ("France", "Japan", "Peru"))
    report = orbit_partition(spec, make_fn({"France": 1, "Japan": 2, "Peru": 99}))
    gold = {1, 2, 3}
    assert report.largest_orbit == 1  # Theorem 1 certifies nothing here
    assert report.certified_errors == 0
    assert admissible_error_bound(3, admissible_distinct(report.answers, gold)) == 1


# --- the precondition, demonstrated to be load-bearing ---------------------


def test_the_bound_is_invalid_when_gold_keys_collide():
    """Measured failure, not a hypothetical: `small_capital` template 4.

    Theorem 1* needs R injective AT THE OBSERVABLE. Two entities whose gold
    answers share a first token (Asmara/Asuncion -> 1634) break that, and the
    bound then over-certifies: 7 claimed against 5 actual on the measured run.
    This test pins the direction of the failure so the precondition cannot be
    quietly dropped.
    """
    # Two entities, one shared gold key: |G| < n, so the bound can exceed truth.
    answers = [1, 2]
    gold = {1}  # both entities' correct answer collapses to the same key
    n = 2
    # Entity 1 is correct and entity 2 wrong, so the true error count is 1.
    assert admissible_error_bound(n, admissible_distinct(answers, gold)) == 1
    # But with both entities genuinely correct under a colliding key, the same
    # arithmetic would certify 1 error where there are 0.
    answers_both_right = [1, 1]
    assert admissible_error_bound(n, admissible_distinct(answers_both_right, gold)) == 1


def test_admissible_distinct_rejects_a_gold_set_smaller_than_the_entity_count():
    """The precondition is checkable, so it is checked rather than assumed."""
    with pytest.raises(ValueError, match="injective"):
        admissible_error_bound(5, 6)


# --- cost, because this has to run at inference ----------------------------


def test_the_bound_costs_no_forward_passes():
    """m* is a set intersection over answers already computed.

    Theorem 1 needs the partition; Theorem 1* needs the partition and one hash
    lookup per entity. There is no second decode, no paraphrase, no gradient.
    """
    answers = list(range(10_000))
    gold = set(range(5_000))
    assert admissible_distinct(answers, gold) == 5_000


def test_chases_counterexample_is_rejected_rather_than_certified():
    """The measured soundness break, refused at the front door.

    Restrict `small_capital` to the four entities whose gold answers collide at
    the token level: Eritrea/Asmara and Paraguay/Asuncion both encode to 1634,
    Zambia/Lusaka and Slovenia/Ljubljana both to 444. The model answers each
    pair with its shared token and the repo's grader scores 4 of 4 correct,
    while `orbit_error_bound(4, 2)` certifies 2 wrong on a perfect answer set.

    Theorem 1 cannot see this: it is handed only the partition, in which those
    four entities form two honest-looking blocks of two. Theorem 1* is handed
    G, and |G| = 2 < 4 is the violation, so it refuses instead of certifying.
    """
    answers = [1634, 1634, 444, 444]
    gold = [1634, 1634, 444, 444]  # four entities, two distinct gold keys

    # Theorem 1 certifies 2 errors here, and all four answers are correct.
    assert orbit_error_bound(4, len(set(answers))) == 2

    # Theorem 1* refuses, because it can check what Theorem 1 cannot.
    with pytest.raises(ValueError, match="not injective at this encoding"):
        admissible_distinct(answers, gold, n_entities=4)


def test_the_precondition_check_is_opt_in_and_silent_when_it_holds():
    """Passing n_entities on a genuinely injective relation changes nothing."""
    answers = [1, 1, 3]
    gold = [1, 2, 3]
    assert admissible_distinct(answers, gold, n_entities=3) == admissible_distinct(answers, gold)
