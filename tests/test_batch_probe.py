"""Catching the two batched failures that produce a well-formed wrong partition.

A batched caller reads the answer at `logits[:, -1, :]`, so the tokenizer must
pad on the LEFT. Right-padded, the final position is a pad token, every answer
is the model's continuation of padding, and the resulting partition is
internally consistent and nobody's. Because the certificate is derived purely
from the partition, no self-check downstream can fail.

A single-element batch has nothing to pad to, so `batch_fn([prompts[0]])`
returns prompt 0's true answer under any padding side. Comparing it against
`batch_fn(prompts)[0]` costs one extra prompt out of `n` and separates the two
cases. It also catches a `batch_fn` that reorders its input, which the length
check cannot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.regime import RelationSpec, orbit_partition

TPLS = ("The capital of {e} is", "{e}'s capital is")
ENTS = ("France", "Japan", "Peru", "Chile")
TABLE = {"France": 1, "Japan": 7, "Peru": 7, "Chile": 4}


def _answer(prompt):
    for k in sorted(TABLE, key=len, reverse=True):
        if k in prompt:
            return TABLE[k]
    return 0


def left_padded(prompts):
    """Correct behaviour: the answer does not depend on batch composition."""
    return [_answer(p) for p in prompts]


def right_padded(prompts):
    """Reads a pad position for every prompt shorter than the longest.

    Simulated by returning a fixed junk answer for short prompts when batched,
    and the true answer when the batch has one element and so needs no padding.
    """
    if len(prompts) == 1:
        return [_answer(prompts[0])]
    longest = max(len(p) for p in prompts)
    return [_answer(p) if len(p) == longest else 999 for p in prompts]


def reordering(prompts):
    """Sorts by length internally and forgets to restore the caller's order."""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    return [_answer(prompts[i]) for i in order]


def test_a_left_padded_batch_passes_the_probe():
    """Negative control: the correct path must not be penalised."""
    rep = orbit_partition(RelationSpec(TPLS, ENTS), None, batch_fn=left_padded)
    assert rep.answers == tuple(_answer(TPLS[0].format(e=e)) for e in ENTS)


def test_a_right_padded_batch_is_caught():
    """The silent catastrophe, made loud."""
    with pytest.raises(ValueError, match="left"):
        orbit_partition(RelationSpec(TPLS, ENTS), None, batch_fn=right_padded)


def test_a_reordering_batch_is_caught():
    """The length check cannot see a permutation; the probe can."""
    with pytest.raises(ValueError, match="left|order"):
        orbit_partition(RelationSpec(TPLS, ENTS), None, batch_fn=reordering)


def test_the_probe_costs_one_extra_prompt_not_one_extra_batch_per_entity():
    """It has to stay cheap enough to leave on by default."""
    sizes = []

    def counting(prompts):
        sizes.append(len(prompts))
        return left_padded(prompts)

    orbit_partition(RelationSpec(TPLS, ENTS), None, batch_fn=counting)
    assert sorted(sizes) == [1, len(ENTS)]


def test_the_probe_uses_the_shortest_prompt_not_the_first():
    """Right padding corrupts in proportion to shortfall, so probe the shortest.

    On this repository's own `capital` relation the first prompt is also the
    longest, so a probe of `prompts[0]` passes a right-padded batch and the
    check is useless exactly where it is needed.
    """
    probed = []

    def recording(prompts):
        if len(prompts) == 1:
            probed.append(prompts[0])
        return left_padded(prompts)

    orbit_partition(RelationSpec(TPLS, ENTS), None, batch_fn=recording)
    built = [TPLS[0].format(e=e) for e in ENTS]
    assert probed == [min(built, key=len)]
    assert probed[0] != built[0]  # the first is not the shortest here


def test_the_probe_can_be_declined_by_a_caller_who_has_verified_once():
    """Opting out is allowed; opting out silently is not, so it is explicit."""
    sizes = []

    def counting(prompts):
        sizes.append(len(prompts))
        return right_padded(prompts)

    rep = orbit_partition(
        RelationSpec(TPLS, ENTS), None, batch_fn=counting, verify_batch=False
    )
    assert sizes == [len(ENTS)]
    assert 999 in rep.answers  # the junk gets through, as the caller asked
