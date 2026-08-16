"""Theorem 6: the certified set proves its own precision before it is evaluated.

Theorem 1 returns a count. A caller cannot act on a count — it has to know WHICH
entities to abstain on. The set is `S = {e : k_e > 1}`, and the useful fact is
that its precision has a lower bound computable from the same partition, with no
ground truth and no held-out data.

Precision is proved; recall is not and cannot be. That asymmetry is the result:
a dispersed error sits in a singleton block and escapes `S` entirely, which is
exactly the failure mode RESULTS.md section 11 records. Reporting one AUROC
forces those two quantities into one number when they have different epistemic
status.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.regime import RelationSpec, orbit_partition
from caustic.theorems import certified_precision_bound, orbit_error_bound

TPLS = ("The capital of {e} is", "{e}'s capital is")


def make_fn(mapping, default=0):
    """Answer function keyed on the entity, longest key first."""

    def fn(prompt):
        for k in sorted(mapping, key=len, reverse=True):
            if k in prompt:
                return mapping[k]
        return default

    return fn


def test_the_bound_holds_against_brute_force_over_every_legal_truth():
    """The claim, checked exhaustively rather than argued.

    For a partition on n <= 7 entities, enumerate every injective truth R
    consistent with the answers and count the wrong entities inside S. The
    minimum over all legal R must be at least the bound.
    """
    rng = np.random.default_rng(0)
    for _ in range(400):
        n = int(rng.integers(2, 8))
        answers = rng.integers(0, 3, n).tolist()
        blocks: dict[int, list[int]] = {}
        for i, a in enumerate(answers):
            blocks.setdefault(a, []).append(i)
        S = [i for i, a in enumerate(answers) if len(blocks[a]) > 1]
        if not S:
            continue
        m = len(blocks)
        # An injective R can make at most one member of each block correct, and
        # is otherwise free; the adversarial choice puts a correct answer in
        # every block, minimising the wrong count inside S.
        min_wrong_in_S = sum(len(b) - 1 for b in blocks.values() if len(b) > 1)
        assert min_wrong_in_S == orbit_error_bound(n, m)
        assert min_wrong_in_S / len(S) >= certified_precision_bound(n, m, len(S)) - 1e-12


def test_the_bound_is_one_minus_blocks_over_flagged():
    """Closed form: precision >= 1 - b/|S| for b non-singleton blocks.

    n - m counts sum(s_i - 1) over non-singleton blocks, which is |S| - b. So
    the bound is (|S| - b)/|S|. Pinned because it is the form a reader can
    evaluate by eye: two blocks of two give 1/2, one block of five gives 4/5.
    """
    # one block of 5, plus 3 singletons: n=8, m=4, |S|=5, b=1
    assert certified_precision_bound(8, 4, 5) == pytest.approx(4 / 5)
    # two blocks of 2, plus 2 singletons: n=6, m=4, |S|=4, b=2
    assert certified_precision_bound(6, 4, 4) == pytest.approx(1 / 2)


def test_every_block_of_at_least_two_gives_precision_at_least_one_half():
    """|S| >= 2b always, so the floor never drops below 0.5 on a flagged set."""
    rng = np.random.default_rng(1)
    for _ in range(2000):
        b = int(rng.integers(1, 20))
        sizes = rng.integers(2, 8, b)
        n_flagged = int(sizes.sum())
        n = n_flagged + int(rng.integers(0, 10))
        m = (n - n_flagged) + b
        assert certified_precision_bound(n, m, n_flagged) >= 0.5


def test_a_collapsed_partition_certifies_precision_approaching_one():
    """The `" the" x 128` regime: one block of 20 gives 19/20.

    This is the case a caller most wants to act on, and it is where the floor is
    strongest — the opposite of a detector that degrades under collapse.
    """
    assert certified_precision_bound(20, 1, 20) == pytest.approx(19 / 20)


def test_an_empty_flagged_set_is_rejected_rather_than_returning_one():
    """Precision of the empty set is undefined, not perfect.

    Returning 1.0 would let a caller report a flawless detector on a partition
    that flagged nothing, which is the `m = n` case where Theorem 1 is silent.
    """
    with pytest.raises(ValueError, match="empty"):
        certified_precision_bound(12, 12, 0)


def test_the_flagged_count_must_be_consistent_with_the_partition():
    """|S| < n - m is arithmetically impossible and signals a caller bug."""
    with pytest.raises(ValueError, match="n_flagged"):
        certified_precision_bound(20, 15, 3)  # n - m = 5 errors cannot fit in 3 entities


def test_the_report_names_the_entities_a_caller_abstains_on():
    """A count is not actionable; the set is."""
    spec = RelationSpec(TPLS, ("France", "Japan", "Peru", "Chile"))
    report = orbit_partition(spec, make_fn({"France": 1, "Japan": 7, "Peru": 7, "Chile": 7}))
    assert set(report.certified_set) == {"Japan", "Peru", "Chile"}
    assert report.certified_errors == 2
    assert report.certified_precision == pytest.approx(2 / 3)


def test_the_certified_set_is_empty_exactly_when_the_certificate_is_silent():
    """Negative control: m = n flags nobody and claims nothing."""
    spec = RelationSpec(TPLS, ("France", "Japan", "Peru"))
    report = orbit_partition(spec, make_fn({"France": 1, "Japan": 2, "Peru": 3}))
    assert report.certified_set == []
    assert report.certified_errors == 0
    assert report.certified_precision is None
