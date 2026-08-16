"""The inference-time path, composed from the pieces that already exist.

Everything here is assembly. `orbit_partition` measures the partition,
`select_prefix` decides using Theorem 1's certified floor as an objective that
needs no ground truth, `repair_by_context` supplies the intervention, and
Theorem 6 bounds the precision of the set that gets withheld. What was missing
is that a caller wanting a guard had to know all four, call them in the right
order, and derive the withheld set themselves — so the repository measured a
phenomenon it never quite shipped.

**What a caller gets.** Given a relation and a deterministic answer function:
the answers to ship, the entities to withhold, and a lower bound on the
precision of that withholding, none of which consults an answer key.

**What it costs.** One forward pass per entity per candidate prefix, and the
empty prefix is always a candidate. With `c` named candidates that is
`(c + 1) * n` passes and nothing else — no paraphrase, no gradient, no second
decode. The abstention itself is a pass over the partition already computed.
Measured on this repository's own relation sizes the whole guard is five forward
passes per entity at `c = 4`, against 53.694 ms per position for the Jacobian
route that sat at chance.

**What it does not do.** It does not score a single free-form generation, and it
does not detect a false statement about an entity it was not given. It needs a
set of entities sharing one relation, and that relation must be injective at the
encoding `answer_fn` returns — on a many-to-one relation a shared answer is
correct behaviour, the certificate is void, and `guard` raises rather than
scoring. It detects collapse. An error that leaves every entity in its own orbit
is invisible to it, which is why `certified_precision` is a floor on precision
and no floor on recall exists or can exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .governor import select_prefix
from .regime import RelationSpec, orbit_partition
from .theorems import certified_error_floor

__all__ = ["GuardReport", "guard"]


@dataclass
class GuardReport:
    """What to ship, what to withhold, and what the withholding is worth."""

    entities: tuple[str, ...]
    answers: tuple[int, ...]
    abstain: list[str]
    kept: list[str]
    certified_errors: int
    baseline_floor: float
    floor: float
    winner_name: str
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def intervened(self) -> bool:
        """Whether a prefix beat doing nothing on the certified floor."""
        return self.winner_name != "none"

    @property
    def certified_precision(self) -> float | None:
        """Theorem 6: `(n - m) / |S|`. None when nothing is withheld.

        Precision of an empty set is undefined rather than perfect; returning
        1.0 would let a caller report a flawless guard on a partition that
        flagged nobody.
        """
        if not self.abstain:
            return None
        return self.certified_errors / len(self.abstain)

    def __str__(self) -> str:
        n = len(self.entities)
        head = (
            f"withhold {len(self.abstain)} of {n}, floor "
            f"{self.baseline_floor:.3f} -> {self.floor:.3f}"
        )
        if self.intervened:
            head += f" via {self.winner_name!r}"
        p = self.certified_precision
        if p is None:
            return head + "  (nothing withheld; the certificate is silent)"
        return head + f", precision >= {p:.3f}"


def guard(
    spec: RelationSpec,
    answer_fn,
    candidates: dict[str, str] | None = None,
    gold_keys=None,
    batch_fn=None,
) -> GuardReport:
    """Measure, decide, act, and report what to withhold. No ground truth used.

    Args:
        spec: the relation and its entities. Must be injective, since every
            bound here descends from Theorem 1.
        answer_fn: prompt -> hashable answer, typically an argmax token id.
            Determinism is the caller's responsibility; a sampled answer makes
            the partition noise rather than measurement.
        candidates: name -> prefix text, run in competition. The empty prefix is
            always entered so the guard can decline, which it must be able to do
            — a prefix can make things worse, and an incoherent one of matched
            length drove the largest orbit from 4 to 20 in this repository's own
            measurements.
        batch_fn: optional prompts -> answers, answering a whole candidate in one
            call rather than one call per entity. Composed with the early exit in
            `select_prefix`, a relation whose baseline partition is already
            discrete costs a SINGLE batched forward pass — 1 call rather than
            `(c + 1) * n`, measured at 16.61x on the per-batch step for 20
            prompts on Qwen2.5-0.5B. The tokenizer must pad on the left; see
            `orbit_partition` for why that failure is silent.
        gold_keys: optionally the set of correct answers *as a set*, without the
            pairing. Supplying it adds every entity whose answer is not a
            correct answer for anybody; those are wrong outright, so they are
            withheld even from singleton orbits, where the partition alone
            certifies nothing. Knowing which answers are capitals is not knowing
            which country each belongs to, so this is not an answer key.

    Raises:
        ValueError: if the relation is not injective. The objective and both
            bounds are Theorem 1's, and on a many-to-one relation a shared
            answer is correct, so the sign of the score inverts.
    """
    if not spec.injective:
        raise ValueError(
            "guard requires an injective relation; the certificate is Theorem 1 "
            "and on a many-to-one relation a shared answer is correct behaviour"
        )

    if answer_fn is None and batch_fn is None:
        raise ValueError("one of answer_fn or batch_fn is required")
    verdict = select_prefix(spec, answer_fn, dict(candidates or {}), batch_fn=batch_fn)
    # The competition already partitioned under every candidate, so the winner's
    # partition is in hand. Recomputing it would cost n forward passes for a
    # result that is already known.
    report = verdict.winner_report
    if report is None:  # pragma: no cover - only for a hand-built verdict
        prefix = verdict.winner
        report = orbit_partition(spec, lambda p, _pre=prefix: answer_fn(_pre + p))

    shared = {a for a, v in report.orbits.items() if len(v) > 1}
    flagged = {e for e, a in zip(report.entities, report.answers) if a in shared}
    certified = report.certified_errors

    if gold_keys is not None:
        gold = set(gold_keys)
        inadmissible = {
            e for e, a in zip(report.entities, report.answers) if a not in gold
        }
        # Each inadmissible answer is wrong outright, and each shared admissible
        # answer still leaves at most one member of its orbit correct. Counting
        # distinct admissible answers gives the ceiling on the correct set.
        certified = len(report.entities) - len(set(report.answers) & gold)
        flagged |= inadmissible

    order = list(report.entities)
    abstain = [e for e in order if e in flagged]
    return GuardReport(
        entities=report.entities,
        answers=report.answers,
        abstain=abstain,
        kept=[e for e in order if e not in flagged],
        certified_errors=certified,
        baseline_floor=verdict.baseline_floor,
        floor=certified_error_floor(len(report.entities), report.n_distinct),
        winner_name=verdict.winner_name,
        scores=verdict.scores,
    )
