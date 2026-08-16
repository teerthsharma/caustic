"""Repairing the regime failure, and measuring whether the repair worked.

The detector in `regime.py` finds collapsed orbits. This applies the intervention
the measurements point at and reports the effect size, refusing to claim a repair
it did not measure.

**The mechanism, measured.** Retrieval is gated by the character of the
surrounding text, not by its length. On Qwen2.5-0.5B with a prefix length held
fixed at 128 tokens:

    coherent prose      20 entities -> 20 answers, largest orbit  1, accuracy 1.000
    " the" x 128        20 entities ->  1 answer,  largest orbit 20, accuracy 0.000
    no prefix           20 entities -> 15 answers, largest orbit  4, accuracy 0.550

The model knows every one of those facts. Whether it can reach them depends on
whether the preceding text puts it in a regime it was trained in. So the
intervention is to supply that regime, and the honest form of the intervention is
one that measures its own effect rather than asserting it.

**Why this is not prompt engineering in the pejorative sense.** The prefix
contains none of the answers, is identical across every entity, and is unrelated
in subject. It carries no information about the task. It cannot be leaking the
answer, because the same 128 tokens in shuffled order drive accuracy to zero.
What it supplies is distributional, not informational.

**What this is not.** It is not a pretraining method. The pretraining analogue —
augmenting facts across varied coherent contexts so they are extractable from any
of them — is documented in the literature with far larger effects (9.7% to 96.6%
in controlled synthetic-biography settings) and is not implemented here. This is
an inference-time wrapper whose effect this module measures on the caller's own
model and relation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .regime import RelationSpec, orbit_partition

__all__ = [
    "constrain_to","NEUTRAL_PREFIX", "RepairReport", "repair_by_context", "sweep_prefixes"]

NEUTRAL_PREFIX = (
    "Mechanical calculators, tide predictors, and looms that read punched cards all encoded "
    "procedures into physical arrangements of matter. Ocean currents move heat around the planet "
    "on timescales that dwarf weather, and the resulting redistribution sets the climate of "
    "entire continents. A language is a system of conventions that lets one mind reconstruct "
    "part of the state of another from a sequence of symbols. Photosynthesis converts light "
    "energy into chemical energy stored in sugars, taking in carbon dioxide and releasing "
    "oxygen as a by-product of the reaction that sustains almost every food chain. "
)
"""Coherent English on subjects unrelated to any relation this is likely to probe.

Deliberately generic. If a caller's relation concerns oceans, printing or
photosynthesis, this prefix is no longer neutral for them and they must supply
their own — `repair_by_context` takes one for exactly that reason.
"""


@dataclass
class RepairReport:
    """Before and after, with the effect size the caller should quote."""

    before_largest_orbit: int
    after_largest_orbit: int
    before_n_distinct: int
    after_n_distinct: int
    n_entities: int
    before_accuracy: float | None = None
    after_accuracy: float | None = None

    @property
    def repaired(self) -> bool:
        """True only when the partition went from collapsed to fully separated."""
        return self.before_largest_orbit > 1 and self.after_largest_orbit == 1

    @property
    def worsened(self) -> bool:
        """True when the prefix merged entities that were previously separate.

        This is a real outcome, not a defensive check: an incoherent prefix of the
        same length drove the largest orbit from 4 to 20 in the measurements this
        module is built on. A prefix can make things much worse.
        """
        return self.after_largest_orbit > self.before_largest_orbit

    @property
    def accuracy_delta(self) -> float | None:
        if self.before_accuracy is None or self.after_accuracy is None:
            return None
        return self.after_accuracy - self.before_accuracy

    def __str__(self) -> str:
        head = (
            f"largest orbit {self.before_largest_orbit} -> {self.after_largest_orbit}, "
            f"distinct answers {self.before_n_distinct} -> {self.after_n_distinct} "
            f"of {self.n_entities} entities"
        )
        if self.accuracy_delta is not None:
            head += f", accuracy {self.before_accuracy:.3f} -> {self.after_accuracy:.3f}"
        if self.worsened:
            return head + "  WORSENED"
        return head + ("  REPAIRED" if self.repaired else "  no repair")


def constrain_to(logits_fn, gold_keys):
    """Wrap a logits function so its argmax cannot leave the answer set.

    Theorem 1* already requires `G`, the correct answers as a set without the
    pairing. This uses the same set at decode time: argmax over its tokens
    rather than over the whole vocabulary.

    Args:
        logits_fn: prompt -> a 1-D array of logits over the vocabulary.
        gold_keys: the admissible answer keys, in the encoding the caller
            compares. For a top-1 token detector, the gold first token ids.

    Returns:
        An `answer_fn` suitable for `orbit_partition`, returning a key from
        `gold_keys`.

    **What it buys.** Measured on Qwen2.5-0.5B, seed 0, under coherent context:

        relation   free argmax   constrained
        capital          0.550         1.000
        language         0.625         1.000
        currency         0.500         1.000

    against 0.750 for the shipped `NEUTRAL_PREFIX` repair on `capital`, on one
    forward pass with an argmax over `|G|` logits instead of the full
    vocabulary — cheaper than the unconstrained call it replaces.

    The injectivity precondition also holds by construction: distinct admissible
    answers are distinct outputs, so no tokenizer collision can manufacture a
    false certificate. That removes the hazard where every numeric answer shares
    first token 220 under Qwen- and Llama-family tokenizers.

    **What it does not buy.** Constraining does not restore a lost fact. Under
    the `" the" x 128` prefix, accuracy goes to 0.050, 0.062 and 0.000 — the
    model still cannot find the right capital when it is only permitted to say
    capitals. Errors under coherent context were the model leaving the answer
    space; errors under the degenerate prefix are the fact being unreachable,
    and only the first kind is repaired here.

    It also changes the task. A model choosing among 20 capitals is closer to
    multiple choice than to open generation, and these accuracies are not
    comparable to free-decoding accuracy elsewhere. And `Theorem 1*` becomes
    redundant under it: nothing inadmissible can be emitted, so `m* = m`.

    Ties resolve to the lowest key so the partition is deterministic; a sampled
    or order-dependent answer would make it noise rather than measurement.

    Raises:
        ValueError: if `gold_keys` is empty, since there would be no answer to
            return; or if a key lies outside the logits vector, which would
            otherwise raise from inside numpy at an unhelpful place.
    """
    keys = sorted(set(gold_keys))
    if not keys:
        raise ValueError("gold_keys must hold at least one admissible answer")

    def answer_fn(prompt):
        logits = np.asarray(logits_fn(prompt))
        if keys[-1] >= logits.shape[-1] or keys[0] < 0:
            raise ValueError(
                f"gold key {keys[-1] if keys[-1] >= logits.shape[-1] else keys[0]} "
                f"lies outside the logits vector of length {logits.shape[-1]}"
            )
        return keys[int(np.argmax(logits[keys]))]

    return answer_fn


def repair_by_context(
    spec: RelationSpec,
    answer_fn,
    prefix: str = NEUTRAL_PREFIX,
    gold: dict[str, int] | None = None,
) -> RepairReport:
    """Measure the partition with and without a neutral prefix.

    Args:
        spec: the relation and entities.
        answer_fn: prompt -> hashable answer. Must be deterministic; a sampled
            answer makes both partitions noise and the comparison meaningless.
        prefix: text prepended to every prompt. Must contain none of the answers
            and be identical across entities, or the comparison is contaminated.
        gold: optional entity -> correct answer, used only to report accuracy
            alongside the partition. The repair verdict never consults it.

    Returns:
        A report whose `repaired` flag is True only for a genuine collapse-to-
        separated transition, and whose `worsened` flag reports the opposite.
    """
    before = orbit_partition(spec, answer_fn)
    after = orbit_partition(spec, lambda p: answer_fn(prefix + p))

    acc_b = acc_a = None
    if gold is not None:
        missing = set(spec.entities) - set(gold)
        if missing:
            raise ValueError(f"gold is missing entities: {sorted(missing)}")
        acc_b = float(np.mean([a == gold[e] for e, a in zip(before.entities, before.answers)]))
        acc_a = float(np.mean([a == gold[e] for e, a in zip(after.entities, after.answers)]))

    return RepairReport(
        before_largest_orbit=before.largest_orbit,
        after_largest_orbit=after.largest_orbit,
        before_n_distinct=before.n_distinct,
        after_n_distinct=after.n_distinct,
        n_entities=len(spec.entities),
        before_accuracy=acc_b,
        after_accuracy=acc_a,
    )


def sweep_prefixes(
    spec: RelationSpec,
    answer_fn,
    prefixes: dict[str, str],
    gold: dict[str, int] | None = None,
) -> dict[str, RepairReport]:
    """Run `repair_by_context` for several candidate prefixes.

    Quoting the best of several prefixes without reporting the rest is selection
    on the outcome. Returning all of them makes that visible.
    """
    return {name: repair_by_context(spec, answer_fn, p, gold) for name, p in prefixes.items()}
