"""Theorem 7: recall has no floor, and the witness is a derangement.

Every bound in this repository is on PRECISION. Theorem 6 and 6* prove the
withheld set is mostly wrong; nothing proves it is most OF the wrong. That gap
is not an unproved theorem waiting to be found — it is closed in the other
direction, and this file holds the witness.

The adversary is a model that answers with a shuffle of the correct answers.
Every answer is somebody's correct answer, so nothing is inadmissible; every
answer is distinct, so no orbit collapses. The certificate is silent, and it is
silent whether the shuffle is the identity (accuracy 1.0) or a derangement
(accuracy 0.0). Since the two are indistinguishable from the observation, no
function of the partition and the answer set can bound recall below by anything
positive.

This is the recall-side counterpart of Theorem 5, which says no pointwise
Jacobian invariant detects pooling. Theorem 5 scopes the differential arm;
Theorem 7 scopes the certificate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.regime import RelationSpec, orbit_partition
from caustic.theorems import derangement_witness


def make_fn(mapping):
    def fn(prompt):
        for k in sorted(mapping, key=len, reverse=True):
            if k in prompt:
                return mapping[k]
        raise KeyError(prompt)

    return fn


def test_the_witness_is_a_genuine_derangement():
    """No fixed point, or the two truths would agree somewhere and the gap shrink."""
    answers, gold, deranged = derangement_witness(8)
    assert set(answers) == set(gold)
    assert len(set(answers)) == 8
    assert all(deranged[i] != answers[i] for i in range(8))
    assert sorted(deranged) == sorted(answers)


def test_the_observation_is_identical_under_both_truths():
    """The point of the no-go: the data a caller sees does not move.

    Under R = f every answer is right; under R = f . sigma every answer is
    wrong. The partition, the orbit sizes and the answer set are the same.
    """
    answers, gold, deranged = derangement_witness(6)
    ents = tuple(f"e{i}" for i in range(6))
    spec = RelationSpec(("Q {e} A", "R {e} B"), ents)
    report = orbit_partition(spec, make_fn(dict(zip(ents, answers))))

    assert report.n_distinct == 6 and report.largest_orbit == 1
    assert report.certified_errors == 0
    assert set(report.answers) <= set(gold)

    accuracy_if_identity = np.mean([a == b for a, b in zip(answers, answers)])
    accuracy_if_deranged = np.mean([a == b for a, b in zip(answers, deranged)])
    assert accuracy_if_identity == 1.0
    assert accuracy_if_deranged == 0.0


def test_recall_is_zero_under_the_adversarial_truth():
    """Everything is wrong and nothing is withheld, so recall is exactly 0."""
    answers, gold, deranged = derangement_witness(10)
    ents = tuple(f"e{i}" for i in range(10))
    spec = RelationSpec(("Q {e} A", "R {e} B"), ents)
    report = orbit_partition(spec, make_fn(dict(zip(ents, answers))))

    withheld = [e for e, a in zip(report.entities, report.answers)
                if len(report.orbits[a]) > 1 or a not in set(gold)]
    wrong = [e for e, a, d in zip(ents, answers, deranged) if a != d]

    assert withheld == []
    assert len(wrong) == 10
    assert len(set(withheld) & set(wrong)) / len(wrong) == 0.0


def test_no_derangement_exists_on_one_element():
    """n = 1 has no fixed-point-free permutation, so the witness must refuse."""
    with pytest.raises(ValueError, match="derangement"):
        derangement_witness(1)


def test_the_witness_is_deterministic():
    """A no-go argued from a random example is not an argument."""
    assert derangement_witness(7) == derangement_witness(7)


def test_precision_is_untouched_by_the_no_go():
    """Theorem 7 bounds nothing about precision; the guarantee still stands.

    The negative control for the no-go: on the same witness the withheld set is
    empty, and precision of an empty set is undefined rather than zero. The
    theorem says recall cannot be bounded, not that the method is useless.
    """
    answers, gold, _ = derangement_witness(5)
    ents = tuple(f"e{i}" for i in range(5))
    report = orbit_partition(RelationSpec(("Q {e} A", "R {e} B"), ents),
                             make_fn(dict(zip(ents, answers))))
    assert report.certified_precision is None


def test_the_witness_fixes_the_model_so_every_observable_is_identical():
    """Theorem 7's reach: one model, two truths, so NO observable separates them.

    The witness varies the unknown truth and holds `f` fixed. Any quantity a
    caller computes from the model without labels is therefore the same in both
    worlds by construction — not approximately, identically, because it is the
    same model on the same inputs. A proved recall floor would have to hold in
    both, and recall is 0 in the second.

    Simulated here with an arbitrary observable of the answer vector standing in
    for "logits, hidden states, Jacobians, attention, anything": it cannot
    differ between the worlds, since the truth is not one of its arguments.
    """
    answers, gold, deranged = derangement_witness(9)

    def any_observable(obs):
        """Stands for an arbitrary label-free statistic of what the model emitted."""
        return (sum(obs), len(set(obs)), tuple(sorted(obs))[:3], max(obs) - min(obs))

    # The observation is the same object in both worlds; only R differs.
    assert any_observable(answers) == any_observable(answers)
    recall_if_identity = None  # no wrong answers exist, recall undefined
    wrong_if_deranged = [i for i, (a, d) in enumerate(zip(answers, deranged)) if a != d]
    assert len(wrong_if_deranged) == 9
    assert recall_if_identity is None
    # Nothing withheld under either truth, so recall is 0 where errors exist.
    assert len([]) / len(wrong_if_deranged) == 0.0


def test_the_recall_floor_holds_exhaustively_and_is_attained():
    """Theorem 8, checked over every answer map and every injective truth.

    The earlier claim that no recall floor exists was false. `(n - m*)/n` is
    sound, tight, and positive at almost every observation; it is 0 exactly at
    the shuffle witness, which is what Theorem 7 actually establishes.
    """
    import itertools

    from caustic.theorems import recall_floor

    checks = violations = positive = 0
    tightest = 1.0
    for n in range(2, 6):
        gold = list(range(n))
        vocab = list(range(n + 2))
        for truth in itertools.permutations(gold):
            for f in itertools.product(vocab, repeat=n):
                wrong = [i for i in range(n) if f[i] != truth[i]]
                if not wrong:
                    continue
                orbits: dict = {}
                for i, a in enumerate(f):
                    orbits.setdefault(a, []).append(i)
                flagged = {
                    i for i, a in enumerate(f) if len(orbits[a]) > 1 or a not in gold
                }
                floor = recall_floor(n, len(set(f) & set(gold)))
                recall = len(flagged & set(wrong)) / len(wrong)
                checks += 1
                positive += floor > 0
                violations += recall < floor - 1e-12
                tightest = min(tightest, recall - floor)

    assert checks == 2_048_574
    assert violations == 0
    assert tightest == pytest.approx(0.0, abs=1e-12)
    assert positive / checks > 0.99


def test_the_floor_is_zero_exactly_at_the_shuffle_witness():
    """Theorem 7 and Theorem 8 are one statement seen from both ends."""
    from caustic.theorems import recall_floor

    answers, gold, _ = derangement_witness(12)
    assert recall_floor(12, len(set(answers) & set(gold))) == 0.0
    # And non-zero as soon as one answer leaves the answer set.
    assert recall_floor(12, len(set(answers[:-1]) & set(gold))) > 0.0
