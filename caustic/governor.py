"""Selecting a prefix by competition, and measuring what a system prompt costs.

Composed from three constructions in the author's other repositories, each doing
a job this problem needs.

    laamba-silence   a governor that runs several candidate geometries in
                     competition and lets a comparator pick the winner, rather
                     than committing to one in advance
    seal-demon-tts   a critic loop: score the output, and if the score fails the
                     gate, correct and re-run rather than shipping it
    Epsilon          a shared frame, so a score computed on one model means the
                     same thing on another

The scorer is `certified_error_floor` from Theorem 1, which needs **no ground
truth**. That is what makes competition possible at inference time: candidates can
be ranked on a live relation where no answers are known.

**Why a competition rather than a fixed prefix.** Measured over eight neutral
prefixes on `capital`, individual accuracy ranged from 0.000 to 0.800. A single
hand-picked prefix is a bet on which of those you happened to write. The governor
does not need to be lucky, only to be able to score.

**Why the critic loop.** A prefix can make things worse. On `capital`, an
incoherent prefix of matched length drove the certified floor from 0.250 to 0.950.
`select_prefix` therefore always includes the empty prefix as a candidate, so the
governor can decline to intervene, and reports the margin by which the winner beat
doing nothing.

**On system prompts.** A system prompt is a prefix, so it carries a sign. Measured
on `capital`: a prompt instructing care and discouraging guessing moved accuracy
from 0.550 to 0.300 and merged orbits from 15 to 9, while a JSON-formatting
instruction reached 0.950. No prompt among six left the partition unchanged; every
adjusted Rand index against the no-prompt condition fell between 0.00 and 0.28.
`prompt_cost` measures that for a caller's own prompt and relation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .regime import OrbitReport, RelationSpec, orbit_partition
from .theorems import certified_error_floor

__all__ = ["PrefixVerdict", "PromptCost", "select_prefix", "prompt_cost"]


@dataclass
class PrefixVerdict:
    """The outcome of a prefix competition."""

    winner: str
    winner_name: str
    floor: float
    baseline_floor: float
    largest_orbit: int
    scores: dict[str, float]
    winner_report: OrbitReport | None = None
    """The winner's partition, so a caller need not re-run it.

    Scoring already computed one `orbit_partition` per candidate, and the
    winner's is the one a caller acts on. Discarding it costs `n` forward passes
    to recompute — 4 of every 16 in a three-candidate competition — which is
    pure waste at inference. Defaults to None so existing constructions keep
    working.
    """

    @property
    def improvement(self) -> float:
        """Certified error removed relative to using no prefix. Never negative,
        because the empty prefix is always a candidate."""
        return self.baseline_floor - self.floor

    @property
    def intervened(self) -> bool:
        return self.winner_name != "none"

    def __str__(self) -> str:
        if not self.intervened:
            return (
                f"declined to intervene; floor {self.floor:.3f}, "
                f"no candidate beat doing nothing"
            )
        return (
            f"selected {self.winner_name!r}: floor {self.baseline_floor:.3f} -> "
            f"{self.floor:.3f} ({self.improvement:+.3f}), largest orbit {self.largest_orbit}"
        )


@dataclass
class PromptCost:
    """What a system prompt does to a relation, measured rather than assumed."""

    name: str
    floor_without: float
    floor_with: float
    ari: float
    largest_without: int
    largest_with: int

    @property
    def cost(self) -> float:
        """Certified error the prompt ADDS. Negative means it helps."""
        return self.floor_with - self.floor_without

    @property
    def neutral(self) -> bool:
        """Neutral means it leaves the partition alone, not merely that it
        happens not to hurt. `ari >= 0.99` is the operational threshold; none of
        six prompts measured on `capital` came close."""
        return self.ari >= 0.99

    def __str__(self) -> str:
        verdict = "neutral" if self.neutral else ("helps" if self.cost < 0 else "costs")
        return (
            f"{self.name}: floor {self.floor_without:.3f} -> {self.floor_with:.3f} "
            f"({self.cost:+.3f}), ARI {self.ari:.4f}, {verdict}"
        )


def _adjusted_rand(a, b) -> float:
    """Adjusted Rand index between two labellings of the same items."""
    a, b = np.asarray(a), np.asarray(b)
    ua, ub = np.unique(a), np.unique(b)
    c = np.zeros((len(ua), len(ub)), dtype=np.int64)
    for i, x in enumerate(ua):
        for j, y in enumerate(ub):
            c[i, j] = int(np.sum((a == x) & (b == y)))

    def comb2(x):
        return x * (x - 1) / 2.0

    sij, si, sj = comb2(c).sum(), comb2(c.sum(1)).sum(), comb2(c.sum(0)).sum()
    n = comb2(len(a))
    exp = si * sj / n if n else 0.0
    mx = 0.5 * (si + sj)
    return float((sij - exp) / (mx - exp)) if mx != exp else 1.0


def select_prefix(
    spec: RelationSpec, answer_fn, candidates: dict[str, str]
) -> PrefixVerdict:
    """Run candidate prefixes in competition; pick the one that certifies fewest errors.

    The empty prefix is always entered, so the governor can decline. Ties are
    broken toward declining, then toward the earlier candidate, so the result does
    not depend on dictionary ordering more than it must.

    Args:
        spec: the relation. Must be injective, since the scorer is Theorem 1.
        answer_fn: prompt -> hashable answer. Must be deterministic.
        candidates: name -> prefix text. An entry named "none" is added if absent.
    """
    if not spec.injective:
        raise ValueError(
            "the scorer is Theorem 1, which requires an injective relation; "
            "on a many-to-one relation a shared answer is correct and the score inverts"
        )
    pool = {"none": "", **candidates}
    n = len(spec.entities)

    scores: dict[str, float] = {}
    reports = {}
    for name, prefix in pool.items():
        rep = orbit_partition(spec, (lambda p, _pre=prefix: answer_fn(_pre + p)))
        reports[name] = rep
        scores[name] = certified_error_floor(n, rep.n_distinct)

    baseline = scores["none"]
    # Decline unless a candidate strictly beats doing nothing.
    best = min(
        (nm for nm in pool if nm != "none"),
        key=lambda nm: (scores[nm], list(pool).index(nm)),
        default=None,
    )
    if best is None or scores[best] >= baseline:
        best = "none"

    return PrefixVerdict(
        winner=pool[best],
        winner_name=best,
        floor=scores[best],
        baseline_floor=baseline,
        largest_orbit=reports[best].largest_orbit,
        scores=scores,
        winner_report=reports[best],
    )


def prompt_cost(spec: RelationSpec, answer_fn, system_prompt: str, name: str = "prompt") -> PromptCost:
    """Measure what a system prompt does to a relation's partition.

    Reports the change in certified error floor and the adjusted Rand index of the
    partition against the no-prompt condition. A prompt is neutral only if it
    leaves the partition alone; one that improves accuracy while restructuring the
    partition has intervened, and the report says so.
    """
    n = len(spec.entities)
    without = orbit_partition(spec, answer_fn)
    with_ = orbit_partition(spec, lambda p: answer_fn(system_prompt + p))
    return PromptCost(
        name=name,
        floor_without=certified_error_floor(n, without.n_distinct),
        floor_with=certified_error_floor(n, with_.n_distinct),
        ari=_adjusted_rand(without.answers, with_.answers),
        largest_without=without.largest_orbit,
        largest_with=with_.largest_orbit,
    )
