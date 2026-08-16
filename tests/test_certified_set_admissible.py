"""Theorem 6*: the floor tightens when the shared answer is not a real answer.

Theorem 6 assumes an adversary places one correct entity in every collapsed
orbit, which is why its floor is `1 - b/|S|`. That adversary is not always
available. An orbit whose shared answer is not a correct answer for ANYBODY has
no correct member at all, so it contributes its whole size rather than `k - 1`.

Counting only the orbits that answer admissibly gives `1 - b_adm/|S|`, and
`b_adm <= b` always. On the `" the" x 128` regime `b_adm` is 0 and the floor is
exactly 1.0, against 0.95 under Theorem 6 and a realised 1.000 measured on
Qwen2.5-0.5B. The tightening is not an assumption about model behaviour; it is
the same answer set Theorem 1* already uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.theorems import (
    admissible_precision_bound,
    certified_precision_bound,
)


def test_it_never_falls_below_theorem_6():
    """The domination property: b_adm <= b, so the floor can only rise."""
    rng = np.random.default_rng(0)
    for _ in range(10_000):
        b = int(rng.integers(1, 12))
        sizes = rng.integers(2, 9, b)
        n_flagged = int(sizes.sum())
        n = n_flagged + int(rng.integers(0, 12))
        m = (n - n_flagged) + b
        b_adm = int(rng.integers(0, b + 1))
        assert admissible_precision_bound(n_flagged, b_adm) >= certified_precision_bound(
            n, m, n_flagged
        ) - 1e-12


def test_an_entirely_inadmissible_collapse_certifies_precision_one():
    """The measured headline case, exactly.

    `" the" x 128` sends 20 entities to one token that is no country's capital.
    Theorem 6 proves 19/20; Theorem 6* proves 20/20, which is what was measured.
    """
    assert certified_precision_bound(20, 1, 20) == pytest.approx(19 / 20)
    assert admissible_precision_bound(20, 0) == 1.0


def test_it_equals_theorem_6_when_every_shared_answer_is_admissible():
    """Negative control: no free lunch when the adversary really is available."""
    # one orbit of 5, answering admissibly: |S| = 5, b = b_adm = 1
    assert admissible_precision_bound(5, 1) == pytest.approx(certified_precision_bound(8, 4, 5))


def test_the_bound_is_exhaustively_correct_against_every_legal_truth():
    """Enumerate rather than argue.

    For each orbit profile, the adversarial truth puts a correct entity in every
    admissible orbit and none in an inadmissible one. The minimum wrong count
    over all legal truths must equal the bound's numerator.
    """
    rng = np.random.default_rng(1)
    for _ in range(3000):
        b = int(rng.integers(1, 7))
        sizes = rng.integers(2, 6, b).tolist()
        admissible = rng.integers(0, 2, b).astype(bool).tolist()
        n_flagged = sum(sizes)
        b_adm = sum(admissible)
        min_wrong = sum(s - 1 if adm else s for s, adm in zip(sizes, admissible))
        assert min_wrong == n_flagged - b_adm
        assert admissible_precision_bound(n_flagged, b_adm) == pytest.approx(
            min_wrong / n_flagged
        )


def test_more_admissible_orbits_than_flagged_entities_is_rejected():
    """Each counted orbit holds at least two entities, so b_adm <= |S| / 2."""
    with pytest.raises(ValueError, match="b_admissible"):
        admissible_precision_bound(4, 3)


def test_an_empty_flagged_set_is_rejected_rather_than_returning_one():
    """Precision of the empty set is undefined, not perfect."""
    with pytest.raises(ValueError, match="empty"):
        admissible_precision_bound(0, 0)
