"""Wrapping a logits function into one that cannot answer outside the answer set.

Theorem 1* already requires the correct answers as a set. `constrain_to` uses
the same set at decode time: argmax over its tokens rather than over the whole
vocabulary. Measured on Qwen2.5-0.5B this takes accuracy from 0.550 to 1.000 on
`capital` under coherent context, against 0.750 for the shipped prefix repair,
on one forward pass with a cheaper argmax.

Two properties matter enough to pin. The output is always admissible, so the
injectivity precondition holds by construction and no tokenizer collision can
manufacture a false certificate. And where the free argmax is already
admissible, constraining must not change it — otherwise the repair would be
altering answers it had no reason to touch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.repair import constrain_to

VOCAB = 10
GOLD = (2, 5, 7)


def logits_for(peak):
    """A logits function whose argmax is `peak`, ties broken away from it."""

    def fn(prompt):
        v = np.full(VOCAB, -1.0)
        v[peak] = 5.0
        # Give one admissible token a middling score so the constrained pick is
        # a real choice rather than the only finite value.
        v[GOLD[0]] = max(v[GOLD[0]], 1.0)
        return v

    return fn


def test_the_answer_is_always_inside_the_answer_set():
    """The precondition holds by construction, not by luck."""
    for peak in range(VOCAB):
        fn = constrain_to(logits_for(peak), GOLD)
        assert fn("any prompt") in GOLD


def test_an_already_admissible_argmax_is_left_alone():
    """Negative control: constraining must not move an answer that was fine."""
    for peak in GOLD:
        assert constrain_to(logits_for(peak), GOLD)("p") == peak


def test_an_inadmissible_argmax_is_replaced_by_the_best_admissible_one():
    """Token 9 is not an answer, so the best of {2, 5, 7} wins instead."""
    assert constrain_to(logits_for(9), GOLD)("p") == GOLD[0]


def test_it_costs_exactly_one_call_to_the_logits_function():
    """The repair has to run at inference, so the call count is asserted."""
    calls = []

    def counting(prompt):
        calls.append(prompt)
        return np.arange(VOCAB, dtype=float)

    constrain_to(counting, GOLD)("p")
    assert len(calls) == 1


def test_ties_resolve_deterministically_to_the_lowest_key():
    """A sampled or order-dependent answer would make the partition noise."""

    def flat(prompt):
        return np.zeros(VOCAB)

    fn = constrain_to(flat, (7, 2, 5))
    assert fn("p") == 2
    assert fn("p") == 2


def test_an_empty_answer_set_is_rejected():
    """Returning None would put an unhashable non-answer into the partition."""
    with pytest.raises(ValueError, match="at least one"):
        constrain_to(logits_for(1), ())


def test_a_key_outside_the_logits_vector_is_rejected():
    """Indexing past the vocabulary would raise deep inside numpy instead."""
    with pytest.raises(ValueError, match="outside the logits"):
        constrain_to(logits_for(1), (2, 999))("p")
