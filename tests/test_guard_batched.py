"""The whole guard on one batched forward pass, where the baseline is discrete.

Two savings compose. The early exit skips every candidate when the baseline
partition is already discrete, since the certified floor is zero and cannot be
beaten. Batching collapses the remaining n sequential prompts into one call.
Together, the common case at inference — a model answering a relation it knows —
costs ONE batched forward pass rather than (c + 1) * n sequential ones.

For the four-candidate configuration this repository uses on 20 entities that is
100 sequential passes reduced to 1 batched call, measured at 16.61x on the
per-batch step alone.

The tests count calls rather than timing anything, so they are facts about the
call graph and hold on any machine.
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
DISCRETE = {"France": 1, "Japan": 2, "Peru": 3, "Chile": 4}
COLLAPSED = dict.fromkeys(ENTS, 7)


def counting(table):
    """Batched answer function that records each batch it is handed."""
    batches: list[list[str]] = []

    def fn(prompts):
        batches.append(list(prompts))
        out = []
        for p in prompts:
            for k in sorted(table, key=len, reverse=True):
                if k in p:
                    out.append(table[k])
                    break
            else:
                out.append(0)
        return out

    return fn, batches


def test_a_healthy_relation_costs_one_batched_call():
    """Early exit plus batching: the common case is a single forward pass."""
    fn, batches = counting(DISCRETE)
    rep = guard(RelationSpec(TPLS, ENTS), None, candidates=CANDS, batch_fn=fn)
    assert len(batches) == 1
    assert len(batches[0]) == len(ENTS)
    assert rep.abstain == []


def test_a_collapsed_relation_costs_one_batched_call_per_candidate():
    """Negative control: the competition still runs, but batched.

    Four candidates counting the empty prefix means four calls, not sixteen.
    """
    fn, batches = counting(COLLAPSED)
    guard(RelationSpec(TPLS, ENTS), None, candidates=CANDS, batch_fn=fn)
    assert len(batches) == len(CANDS) + 1
    assert all(len(b) == len(ENTS) for b in batches)


def test_each_batch_carries_its_candidate_prefix():
    """The prefix must reach the prompts, or every candidate scores the baseline."""
    fn, batches = counting(COLLAPSED)
    select_prefix(RelationSpec(TPLS, ENTS), None, CANDS, batch_fn=fn)
    prefixes = {b[0].split("The capital")[0] for b in batches}
    assert prefixes == {"", "x ", "y ", "z "}


def test_the_batched_and_sequential_guards_agree():
    """A cost change must not be a decision change."""
    table = {"France": 7, "Japan": 7, "Peru": 3, "Chile": 4}
    fn, _ = counting(table)

    def seq(prompt):
        return fn([prompt])[0]

    a = guard(RelationSpec(TPLS, ENTS), seq, candidates=CANDS)
    b = guard(RelationSpec(TPLS, ENTS), None, candidates=CANDS, batch_fn=fn)
    assert a.abstain == b.abstain
    assert a.answers == b.answers
    assert a.floor == b.floor
    assert a.winner_name == b.winner_name


def test_supplying_neither_function_is_rejected_by_the_guard():
    """An empty partition would certify zero errors over nothing."""
    with pytest.raises(ValueError, match="answer_fn or batch_fn"):
        guard(RelationSpec(TPLS, ENTS), None, candidates=CANDS)
