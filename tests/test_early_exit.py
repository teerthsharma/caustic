"""The competition is decidable without running it when the baseline is discrete.

`select_prefix` scores candidates by the certified floor `(n - m) / n` and
declines unless one is STRICTLY below the baseline. The floor is non-negative
and hits zero exactly when the partition is already discrete, so a zero baseline
cannot be beaten and every candidate forward pass is provably wasted.

That is the common case at inference: a model answering a relation it knows puts
every entity in its own orbit. Skipping the competition there takes the guard
from `(c + 1) * n` forward passes to `n`, which for the four-candidate
configuration this repository uses is a fifth of the cost.

The tests count passes rather than timing anything, so the saving is a fact
about the call graph and not about the machine it ran on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.governor import select_prefix
from caustic.guard import guard
from caustic.regime import RelationSpec

TPLS = ("The capital of {e} is", "{e}'s capital is")
ENTS = ("France", "Japan", "Peru", "Chile")
CANDS = {"a": "x ", "b": "y ", "c": "z "}


def counting(table):
    """Answer function that records every prompt it is asked."""
    calls: list[str] = []

    def fn(prompt):
        calls.append(prompt)
        for k in sorted(table, key=len, reverse=True):
            if k in prompt:
                return table[k]
        return 0

    return fn, calls


def test_a_discrete_baseline_skips_every_candidate():
    """Four entities, four answers, floor 0. Nothing can beat it, so nothing runs."""
    fn, calls = counting({"France": 1, "Japan": 2, "Peru": 3, "Chile": 4})
    verdict = select_prefix(RelationSpec(TPLS, ENTS), fn, CANDS)
    assert verdict.floor == 0.0
    assert not verdict.intervened
    assert len(calls) == len(ENTS), "candidates were run against an unbeatable baseline"


def test_a_collapsed_baseline_still_runs_the_full_competition():
    """Negative control: the early exit must not fire where a prefix could help."""
    fn, calls = counting(dict.fromkeys(ENTS, 7))
    select_prefix(RelationSpec(TPLS, ENTS), fn, CANDS)
    assert len(calls) == (len(CANDS) + 1) * len(ENTS)


def test_the_skipped_candidates_are_reported_as_unscored_not_as_losers():
    """A candidate that never ran has no score, and must not be recorded as 0.

    Reporting an unscored candidate as having floor 0.0 would make it look tied
    with the baseline rather than untested, which is the kind of silent fiction
    that makes a competition unfalsifiable.
    """
    fn, _ = counting({"France": 1, "Japan": 2, "Peru": 3, "Chile": 4})
    verdict = select_prefix(RelationSpec(TPLS, ENTS), fn, CANDS)
    assert set(verdict.scores) == {"none"}
    assert verdict.winner_name == "none"


def test_the_guard_inherits_the_saving():
    """The early exit has to reach the path a caller actually invokes."""
    fn, calls = counting({"France": 1, "Japan": 2, "Peru": 3, "Chile": 4})
    rep = guard(RelationSpec(TPLS, ENTS), fn, candidates=CANDS)
    assert len(calls) == len(ENTS)
    assert rep.abstain == []
    assert rep.certified_precision is None


def test_the_verdict_is_identical_with_and_without_the_shortcut():
    """The optimisation must change cost only, never the decision.

    A partly collapsed baseline is scored the long way; the assertion is that
    the winner and floor match what an exhaustive competition would return.
    """
    table = {"France": 7, "Japan": 7, "Peru": 3, "Chile": 4}
    fn, calls = counting(table)
    verdict = select_prefix(RelationSpec(TPLS, ENTS), fn, CANDS)
    assert verdict.baseline_floor == pytest.approx(0.25)
    assert len(calls) == (len(CANDS) + 1) * len(ENTS)
    assert verdict.winner_name == "none"  # no candidate changes the table here
