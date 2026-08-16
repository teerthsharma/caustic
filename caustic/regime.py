"""Detecting the regime failure, without a gold answer.

This is the part of the programme that survived measurement. Everything the
repository tried first is deliberately absent from this module: Jacobian spectral
summaries sat at chance and could not have worked, since they detect creases where
there is no crease; the Oseledets filtration is empty for a structural reason; and
entity attention does not separate correct from wrong answers.

What survived is a group-theoretic observation with a topological instrument.

A fact carries a group action. A model that has the fact must respect two halves
of it, and a model outside its working regime respects neither:

    invariance     paraphrase the prompt and the answer must not change
    equivariance   swap the entity and the answer must change

The instrument is the ORBIT PARTITION: which entities receive the same answer.
That is H0 of the answer-equivalence relation, it needs no ground truth, and it is
what detects the failure. Measured on Qwen2.5-0.5B with a fixed 128-token prefix,
varying only its character:

    prose prefix       20 entities -> 20 distinct answers, largest orbit  1, acc 1.000
    " the" x 128       20 entities ->  1 distinct answer,  largest orbit 20, acc 0.000

Same token count. The model knows every one of those facts; whether it can reach
them depends on the surrounding text. `largest_orbit` is the number that reports
it, and `collapsed` is the boolean a caller acts on.

**What this does not do.** It measures whether a model is in a regime where
retrieval works over a set of entities sharing one relation. It does not score a
single free-form generation, does not detect a false statement about an entity it
was not given, and requires the relation to be injective — distinct entities must
genuinely warrant distinct answers, or a collapsed orbit is correct behaviour and
the signal inverts. `RelationSpec.injective` records that precondition rather than
leaving it implicit.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

__all__ = ["RelationSpec", "OrbitReport", "orbit_partition", "symmetry_scores"]


@dataclass(frozen=True)
class RelationSpec:
    """A relation to probe, with its precondition recorded.

    Args:
        templates: prompt templates containing `{e}`. The first is the primary;
            the rest are paraphrases used for the invariance score. At least one
            is required, and at least two for invariance to be defined.
        entities: the entity strings substituted into the templates.
        injective: whether distinct entities genuinely warrant distinct answers.
            Equivariance is undefined when this is False — many countries share a
            continent, so a collapsed orbit is then correct and the collision
            score inverts. Measured: 0.995 AUROC on an injective relation against
            0.273 on a many-to-one one.
    """

    templates: tuple[str, ...]
    entities: tuple[str, ...]
    injective: bool = True

    def __post_init__(self) -> None:
        if not self.templates:
            raise ValueError("at least one template is required")
        if any("{e}" not in t for t in self.templates):
            raise ValueError("every template must contain '{e}'")
        if len(self.entities) < 2:
            raise ValueError("at least two entities are required to form a partition")


@dataclass
class OrbitReport:
    """The partition of entities induced by a model's answers."""

    entities: tuple[str, ...]
    answers: tuple[int, ...]
    injective: bool
    n_distinct: int
    largest_orbit: int
    orbits: dict[int, list[str]] = field(default_factory=dict)

    @property
    def collapse_ratio(self) -> float:
        """1.0 when every entity shares one answer, 1/n when all are distinct."""
        return self.largest_orbit / len(self.entities)

    @property
    def certified_errors(self) -> int:
        """Lower bound on the number of wrong answers, provable without ground truth.

        **Orbit Error Bound.** For an injective relation on `n` entities whose
        answers induce a partition with `m` orbits, at least `n - m` answers are
        wrong.

        Proof. Injectivity makes the true answer map `g` injective. If two
        distinct entities share an orbit then `f` gives them one answer while `g`
        gives them two, so `f` errs on at least one of them. An orbit of size `s`
        therefore contributes at least `s - 1` errors, and summing over orbits
        gives `sum(s_i - 1) = n - m`.

        The bound is tight: it is attained exactly when every orbit contains one
        correct answer.

        Returns 0 for a non-injective relation, where sharing an answer is not
        evidence of error and the argument does not apply.
        """
        if not self.injective:
            return 0
        return len(self.entities) - self.n_distinct

    @property
    def certified_error_rate(self) -> float:
        """`certified_errors` as a fraction of the entity count."""
        return self.certified_errors / len(self.entities)

    @property
    def collapsed(self) -> bool:
        """True when the partition is degenerate for an injective relation.

        The threshold is two entities sharing an answer, because on an injective
        relation any shared answer is already wrong for at least one of them.
        Always False for a non-injective relation, where sharing is expected.
        """
        return self.injective and self.largest_orbit > 1

    def __str__(self) -> str:
        head = (
            f"{len(self.entities)} entities -> {self.n_distinct} distinct answers, "
            f"largest orbit {self.largest_orbit}"
        )
        if not self.injective:
            return head + "  (relation not injective; collapse is not diagnostic)"
        if not self.collapsed:
            return head + "  (no collapse)"
        merged = [v for v in self.orbits.values() if len(v) > 1]
        return head + "  COLLAPSED: " + "; ".join(", ".join(g) for g in merged)


def orbit_partition(spec: RelationSpec, answer_fn, batch_fn=None) -> OrbitReport:
    """Partition entities by the answer the model gives, using the first template.

    Args:
        spec: the relation and its entities.
        answer_fn: maps a prompt string to a hashable answer, typically the
            argmax token id. Any deterministic function works; determinism is the
            caller's responsibility and a sampled answer makes the partition noise.
            May be None when `batch_fn` is given.
        batch_fn: maps a LIST of prompts to a list of answers, one per prompt, in
            the order received. Preferred when supplied, since the prompts are
            independent by construction and serialising them is an artefact of
            the callback signature rather than of the mathematics.

    **Why the batched path exists.** Measured on Qwen2.5-0.5B over 20
    country-capital prompts, one batched forward pass against twenty sequential
    ones:

        condition        sequential   batched   speedup   answers identical
        no prefix          703.63 ms  42.36 ms   16.61x   yes
        `" the" x 128`     760.05 ms 446.94 ms    1.70x   yes

    The partition is identical either way, so this is a cost change and nothing
    else. The gain is largest exactly where a guard runs most often — short
    prompts, healthy relation — because a long shared prefix already saturates
    the device.

    **The padding trap, which is silent.** A batched caller reads the answer at
    `logits[:, -1, :]`, so the tokenizer must pad on the LEFT. With right
    padding the final position is a pad token, every answer is the model's
    continuation of padding, and the resulting partition is well-formed and
    nobody's answers. Set `tokenizer.padding_side = "left"` before batching, and
    check the batched answers against the sequential ones once on a small set —
    they must agree exactly.

    Raises:
        ValueError: if neither function is given, since an empty partition would
            certify zero errors over nothing; or if `batch_fn` returns a list
            whose length does not match the entity count, which would otherwise
            zip short and silently drop entities.
    """
    prompts = [spec.templates[0].format(e=e) for e in spec.entities]
    if batch_fn is not None:
        answers = list(batch_fn(prompts))
        if len(answers) != len(spec.entities):
            raise ValueError(
                f"batch_fn must return one answer per entity; got {len(answers)} "
                f"for {len(spec.entities)} entities"
            )
    elif answer_fn is not None:
        answers = [answer_fn(p) for p in prompts]
    else:
        raise ValueError("one of answer_fn or batch_fn is required")
    orbits: dict[int, list[str]] = {}
    for e, a in zip(spec.entities, answers):
        orbits.setdefault(a, []).append(e)
    counts = [len(v) for v in orbits.values()]
    return OrbitReport(
        entities=tuple(spec.entities),
        answers=tuple(answers),
        injective=spec.injective,
        n_distinct=len(orbits),
        largest_orbit=max(counts),
        orbits=orbits,
    )


def symmetry_scores(spec: RelationSpec, answer_fn) -> dict[str, dict[str, float]]:
    """Per-entity invariance and collision, neither of which uses a gold answer.

    invariance   agreement across paraphrases of the same fact, in [0, 1].
                 High is good. This half is close to published self-consistency
                 work and is included as the baseline the other half must beat.
    collision    fraction of OTHER entities receiving this entity's answer,
                 averaged over templates, in [0, 1]. Low is good. This half is
                 the one that reached 0.995 AUROC on injective relations, and it
                 is meaningless when `spec.injective` is False.

    Raises:
        ValueError: if fewer than two templates are given, since invariance is
            undefined on one.
    """
    if len(spec.templates) < 2:
        raise ValueError("invariance needs at least two templates")

    table = {
        e: [answer_fn(t.format(e=e)) for t in spec.templates] for e in spec.entities
    }
    out: dict[str, dict[str, float]] = {}
    for e in spec.entities:
        mine = table[e]
        inv = float(np.mean([a == b for a, b in itertools.combinations(mine, 2)]))
        col = float(
            np.mean(
                [
                    np.mean([table[o][i] == mine[i] for o in spec.entities if o != e])
                    for i in range(len(spec.templates))
                ]
            )
        )
        out[e] = {"invariance": inv, "collision": col, "score": inv - col}
    return out
