"""One forward pass over n prompts instead of n forward passes over one.

`orbit_partition` calls `answer_fn` once per entity, so it serialises n prompts
that are independent by construction. That is an artefact of the callback
signature rather than of the mathematics, and it costs an order of magnitude.
Measured on Qwen2.5-0.5B, 20 country-capital prompts:

    condition        sequential   batched   speedup   answers identical
    no prefix          703.63 ms  42.36 ms   16.61x   yes
    " the" x 128       760.05 ms 446.94 ms    1.70x   yes

The partition is unchanged, so this is a cost change and nothing else. The tests
below assert the call count rather than timing anything, so they hold on any
machine.

**The padding trap.** A batched caller reads the answer at `logits[:, -1, :]`,
which requires the tokenizer to pad on the LEFT. With right padding the last
position is a pad token, every answer is the model's continuation of padding,
and the partition looks perfectly well-formed while being nobody's answers. The
docstring says so; there is no way to test it here without a tokenizer, so it is
called out rather than silently assumed.
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


def _lookup(prompt):
    for k in sorted(TABLE, key=len, reverse=True):
        if k in prompt:
            return TABLE[k]
    return 0


def test_the_batched_path_gives_the_same_partition():
    """A cost change must not be a behaviour change."""
    seq = orbit_partition(RelationSpec(TPLS, ENTS), _lookup)
    bat = orbit_partition(
        RelationSpec(TPLS, ENTS), None, batch_fn=lambda ps: [_lookup(p) for p in ps]
    )
    assert bat.answers == seq.answers
    assert bat.n_distinct == seq.n_distinct
    assert bat.largest_orbit == seq.largest_orbit
    assert bat.orbits == seq.orbits


def test_the_batch_function_is_called_exactly_once():
    """The whole point: one call carrying every prompt, not one call per entity."""
    calls = []

    def batch_fn(prompts):
        calls.append(list(prompts))
        return [_lookup(p) for p in prompts]

    orbit_partition(RelationSpec(TPLS, ENTS), None, batch_fn=batch_fn, verify_batch=False)
    assert len(calls) == 1
    assert len(calls[0]) == len(ENTS)

    # With the padding probe on (the default), there is a second call carrying
    # exactly one prompt. That is the cost of making a silent failure loud, and
    # it is one prompt rather than n.
    calls.clear()
    orbit_partition(RelationSpec(TPLS, ENTS), None, batch_fn=batch_fn)
    assert [len(c) for c in calls] == [len(ENTS), 1]


def test_the_prompts_arrive_in_entity_order():
    """Answers are zipped back against `entities`, so order is load-bearing.

    A batch function that reorders internally must restore the order, and this
    test pins the contract it has to honour.
    """
    seen = []

    def batch_fn(prompts):
        seen.extend(prompts)
        return [_lookup(p) for p in prompts]

    orbit_partition(RelationSpec(TPLS, ENTS), None, batch_fn=batch_fn, verify_batch=False)
    assert seen == [TPLS[0].format(e=e) for e in ENTS]


def test_a_batch_function_returning_the_wrong_count_is_rejected():
    """Silently zipping a short list would drop entities off the partition."""
    with pytest.raises(ValueError, match="one answer per entity"):
        orbit_partition(RelationSpec(TPLS, ENTS), None, batch_fn=lambda ps: [1, 2])


def test_supplying_neither_function_is_rejected():
    """Defaulting to an empty partition would certify zero errors on nothing."""
    with pytest.raises(ValueError, match="answer_fn or batch_fn"):
        orbit_partition(RelationSpec(TPLS, ENTS), None)


def test_supplying_both_prefers_the_batch_function():
    """Ambiguity resolved toward the cheap path, and stated rather than implied."""
    rep = orbit_partition(
        RelationSpec(TPLS, ENTS), lambda p: 999, batch_fn=lambda ps: [_lookup(p) for p in ps]
    )
    assert 999 not in rep.answers
