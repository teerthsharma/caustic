"""Six results, one per branch, each with a proof and an executable witness.

Every statement here is elementary. That is deliberate: the value is not in the
difficulty of the proofs but in the fact that each one is *checkable against a
measurement made in this repository*, and that together they form a chain

    differential geometry -> topology -> game theory
    chaos theory ---------> topology
    partial-differential symmetry ---> why the differential route fails alone

Notation used throughout. `E` is a finite set of entities, `|E| = n`. A relation
`R : E -> A` is *injective* when distinct entities have distinct correct answers.
A model induces `f : E -> A`, and `P(f)` is the partition of `E` by the value of
`f`, with `m = |P(f)|` blocks called *orbits*.

--------------------------------------------------------------------------------
THEOREM 1 (topology) -- Orbit Error Bound
--------------------------------------------------------------------------------
For injective `R`, the number of entities on which `f` errs is at least `n - m`.

Proof. The true map `R` is injective, so distinct entities carry distinct correct
answers. If `e1 != e2` lie in one orbit then `f(e1) = f(e2)` while
`R(e1) != R(e2)`, so `f` is wrong on at least one of them. An orbit of size `s`
therefore contributes at least `s - 1` errors, and summing over the `m` orbits
gives `sum_i (s_i - 1) = n - m`. []

Tight: attained exactly when every orbit contains one correct answer.
Measured instance: capital with no prefix gave `n = 20`, `m = 15`, certifying at
least 5 wrong answers with no ground truth consulted. Observed accuracy 0.550,
so 9 were actually wrong and the bound held with slack.

--------------------------------------------------------------------------------
THEOREM 2 (game theory) -- Pooling Recovery Bound
--------------------------------------------------------------------------------
If `f` maps a block of `k` entities to one answer, then for ANY downstream
function `h`, the probability that `h(f(e))` recovers `e` under a uniform prior on
that block is at most `1/k`.

Proof. `h . f` is constant on the block, so it takes one value there. It can
therefore agree with the identity on at most one of the `k` entities. Under a
uniform prior the success probability is at most `1/k`. []

This is the signalling-game reading: a block of size `k > 1` is a *pooling*
equilibrium, and pooling destroys the receiver's ability to infer the sender's
type. No amount of downstream capability recovers it, which is why orbit collapse
is not merely an error but an unrecoverable one.
Measured instance: the " the" x 128 prefix pooled all 20 countries, bounding any
downstream recovery at 0.05.

--------------------------------------------------------------------------------
THEOREM 3 (differential geometry) -- Zero Coupling Implies Pooling
--------------------------------------------------------------------------------
Let `z_c(h)` be the logit of token `c` as a function of the entity representation
`h`, continuously differentiable on a domain containing a path `gamma` from
`h_1` to `h_2`. If the directional derivative of `z_c` along `gamma` vanishes
identically, then `z_c(h_1) = z_c(h_2)`.

Proof. The fundamental theorem of calculus along `gamma`:
`z_c(h_2) - z_c(h_1) = integral over gamma of grad z_c . dl = 0`. []

Consequence: if this holds for every candidate `c`, the two entities receive
identical logit vectors, hence identical answers, hence share an orbit, and
Theorem 1 applies. **Differential geometry feeds topology.** Vanishing entity
coupling is a sufficient condition for pooling.

Measured instance: the token the model actually chose on wrong items coupled to
the entity at 0.93 times its coupling to control tokens, against 1.33 for the
correct token. The chosen token was closer to entity-independent, which is the
finite-difference shadow of this statement.

--------------------------------------------------------------------------------
THEOREM 4 (chaos theory) -- Dissipative Pooling
--------------------------------------------------------------------------------
Let `T` be differentiable with characteristic exponents `lambda_1 >= ... >=
lambda_D` and `S = sum_i lambda_i < 0`. Then for any bounded set `A` of positive
Lebesgue measure, `vol(T^n(A)) -> 0` as `n -> infinity`, at rate `exp(nS)`.

Proof. The change-of-variables formula gives
`vol(T^n(A)) = integral over A of |det D(T^n)|`, and the exponents are defined so
that `(1/n) log |det D(T^n)| -> sum_i lambda_i = S`. With `S < 0` the integrand
decays like `exp(nS)`, so the volume does. []

Consequence: once the image volume falls below the resolution separating decision
regions, distinct inputs must land in one region and pool. Volume contraction is
therefore a *driver* of the collapse Theorem 1 penalises, and depth is the clock.

Measured instance: the token product at block 3 gave `S = -226.74` over 768
dimensions with only 139 expanding directions, and `D_KY = 29.57` -- a 26x
compression. What prevents total pooling in practice is that the trajectory is
short, not that the map is volume-preserving.

--------------------------------------------------------------------------------
THEOREM 5 (partial-differential symmetry) -- No Local Criterion Detects Pooling
--------------------------------------------------------------------------------
There exists a smooth map `F` whose Jacobian is nonsingular at every point of its
domain and which is not injective. Consequently no function of the local Jacobian
alone -- determinant, smallest singular value, condition number, spectral decay,
or any other pointwise invariant -- can decide injectivity.

Proof by witness. `F(x, y) = (e^x cos y, e^x sin y)` on `R^2` has
`det DF = e^{2x} > 0` everywhere, yet `F(x, y) = F(x, y + 2*pi)`. The Jacobians at
the two preimages are related by a rotation and share every spectral invariant, so
no pointwise function of the Jacobian distinguishes the injective case from this
one. []

This is the honest LLM-side statement of the moral that the refutation of the
Jacobian Conjecture makes vivid: local invertibility everywhere does not imply
global injectivity, and the failure is invisible to local data. It is a *no-go*
result, and it is the reason the Jacobian arm of this repository sat at chance:
`sigma_max`, `log_volume` and `tail_alpha` are pointwise spectral invariants, and
Theorem 5 says such quantities cannot see pooling. Measured corroboration: zero of
768 singular values fell below `1e-6 * sigma_max`, so the observed collapse
occurred with an everywhere-nonsingular Jacobian, exactly the case the witness
describes.

The escape is global, and Theorem 1 is what takes it: compare *two* entities
rather than examining one point.

--------------------------------------------------------------------------------
THEOREM 6 (topology) -- Certified-Set Precision Floor
--------------------------------------------------------------------------------
Let `S = {e : k_e > 1}` be the entities lying in non-singleton orbits, and let a
caller predict "wrong" on `S`. For injective `R`,

    precision(S) = |S and wrong| / |S| >= (n - m) / |S| = 1 - b / |S|,

where `b` is the number of non-singleton orbits. Both sides are computed from the
partition alone.

Proof. Every error counted by Theorem 1 lies in a non-singleton orbit, since a
singleton contributes `s - 1 = 0`. Hence the `n - m` certified errors all lie in
`S`, so `|S and wrong| >= n - m`; divide by `|S|`. For the closed form, note
`n - m = sum over non-singleton orbits of (s_i - 1) = |S| - b`. []

Theorem 1 returns a count, and a count is not actionable -- a caller must know
WHICH entities to withhold. Theorem 6 turns the count into a set and proves the
set's precision in advance, which is what makes abstention safe without labels.
Since every counted orbit holds at least two entities, `|S| >= 2b` and the floor
never falls below `1/2`; it tends to 1 as orbits grow, so the guarantee is
strongest exactly where collapse is worst.

**Precision is proved and recall is not.** A dispersed error sits in a singleton
orbit and never enters `S`, and no partition-only argument can bound that, since
a singleton is consistent with an injective truth making it correct. The
asymmetry is the point: an AUROC forces a proved quantity and an unprovable one
into one number.

Measured instance: across 80 conditions on two models, four relations, five
templates and two contexts, zero bound violations; realised precision 1.000 in 56
of 60 evaluable conditions against a mean proved floor of 0.756, with realised
recall 0.79 and no floor available for it.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "orbit_error_bound",
    "certified_precision_bound",
    "pooling_recovery_bound",
    "path_integral_change",
    "volume_decay_rate",
    "winding_witness",
    "certified_error_floor",
    "certified_reduction",
    "recovery_ceiling_gain",
]


def orbit_error_bound(n_entities: int, n_distinct: int) -> int:
    """Theorem 1. Minimum number of wrong answers, given the partition alone."""
    if n_entities < 1:
        raise ValueError("need at least one entity")
    if not 1 <= n_distinct <= n_entities:
        raise ValueError(f"n_distinct must lie in [1, {n_entities}]; got {n_distinct}")
    return n_entities - n_distinct


def certified_precision_bound(n_entities: int, n_distinct: int, n_flagged: int) -> float:
    """Theorem 6. Precision of the certified set, proved before it is evaluated.

    **Statement.** Let `S = {e : k_e > 1}` be the entities in non-singleton
    blocks, `|S| = n_flagged`, and let a caller predict "wrong" on `S`. For an
    injective relation,

        precision(S) = |S ∩ wrong| / |S| >= (n - m) / |S|.

    Both sides are computed from the partition alone: no ground truth, no
    held-out split, no sampling distribution.

    Proof. Every error counted by Theorem 1 lies in a non-singleton block, since
    a singleton contributes `s - 1 = 0` to the sum. So the `n - m` certified
    errors are all inside `S`, giving `|S ∩ wrong| >= n - m`; divide by `|S|`. []

    **Closed form.** Writing `b` for the number of non-singleton blocks,
    `n - m = |S| - b`, so the bound is `1 - b/|S|`. Since every counted block
    holds at least two entities, `|S| >= 2b` and the floor is never below `1/2`
    on a non-empty flagged set. It approaches 1 as blocks grow: one block of 20
    certifies precision `>= 0.95`, which is the `" the" x 128` regime — the
    floor is strongest exactly where a caller most needs to abstain.

    **Precision is proved; recall is not, and cannot be.** A dispersed error
    sits in a singleton block and never enters `S`. That is the failure mode
    RESULTS.md section 11 records, and no partition-only argument can bound it,
    because a singleton is consistent with an injective truth that makes it
    correct. The asymmetry is the result: reporting one AUROC forces a proved
    quantity and an unprovable one into a single number.

    Measured across 80 conditions on two models (Qwen2.5-0.5B, SmolLM2-135M),
    four relations, five templates, two contexts: 0 bound violations; realised
    precision 1.000 in 56 of 60 evaluable conditions against a mean proved floor
    of 0.756; realised recall 0.79 with no floor available.

    Raises:
        ValueError: if `n_flagged` is zero, since the precision of an empty set
            is undefined rather than perfect — returning 1.0 would let a caller
            report a flawless detector on a partition that flagged nobody; or if
            `n_flagged` is smaller than `n - m`, which is arithmetically
            impossible and means the flagged set was not derived from this
            partition.
    """
    errors = orbit_error_bound(n_entities, n_distinct)
    if n_flagged <= 0:
        raise ValueError(
            "precision of an empty certified set is undefined; the partition "
            "flagged no entity, which is the m = n case where Theorem 1 is silent"
        )
    if n_flagged > n_entities:
        raise ValueError(f"n_flagged must be at most {n_entities}; got {n_flagged}")
    if n_flagged < errors:
        raise ValueError(
            f"n_flagged={n_flagged} cannot hold the {errors} certified errors of "
            "this partition; the flagged set does not come from it"
        )
    return errors / n_flagged


def pooling_recovery_bound(block_size: int) -> float:
    """Theorem 2. Ceiling on any downstream recovery of the entity from the answer."""
    if block_size < 1:
        raise ValueError("block size must be positive")
    return 1.0 / block_size


def path_integral_change(grad_fn, h1: np.ndarray, h2: np.ndarray, steps: int = 2048) -> float:
    """Theorem 3, evaluated numerically along the straight path from h1 to h2.

    Returns the integral of the directional derivative, which the theorem equates
    with `z_c(h2) - z_c(h1)`. A vanishing gradient gives zero, so the two entity
    representations receive the same logit and must share an orbit.
    """
    h1 = np.asarray(h1, dtype=np.float64)
    h2 = np.asarray(h2, dtype=np.float64)
    if h1.shape != h2.shape:
        raise ValueError(f"shape mismatch: {h1.shape} vs {h2.shape}")
    d = h2 - h1
    ts = (np.arange(steps) + 0.5) / steps
    return float(sum(float(np.dot(grad_fn(h1 + t * d), d)) for t in ts) / steps)


def volume_decay_rate(exponents: np.ndarray, n_steps: int) -> float:
    """Theorem 4. log volume ratio after `n_steps`, namely `n * sum(lambda)`.

    Negative means contraction. The measured token product gave a per-step sum of
    -226.74, so a single step already contracts volume by `exp(-226.74)`.
    """
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    return float(n_steps * np.sum(np.asarray(exponents, dtype=np.float64)))


def winding_witness(x: float, y: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Theorem 5. The map `(x, y) -> (e^x cos y, e^x sin y)` at one point.

    Returns the image, the Jacobian, and its determinant. The determinant is
    `e^{2x} > 0` everywhere, and the map is `2*pi`-periodic in `y`, so it is a
    local diffeomorphism at every point and globally many-to-one.
    """
    ex = np.exp(x)
    image = np.array([ex * np.cos(y), ex * np.sin(y)])
    jac = np.array([[ex * np.cos(y), -ex * np.sin(y)], [ex * np.sin(y), ex * np.cos(y)]])
    return image, jac, float(np.linalg.det(jac))


# --- what the theorems certify about an intervention -----------------------


def certified_error_floor(n_entities: int, n_distinct: int) -> float:
    """Theorem 1 as a rate: the provable fraction of answers that are wrong.

    `(n - m) / n`. A lower bound on the true error rate, computed from the
    partition alone with no ground truth consulted.
    """
    return orbit_error_bound(n_entities, n_distinct) / n_entities


def certified_reduction(n_entities: int, m_before: int, m_after: int) -> float:
    """Provable error eliminated by an intervention, in percentage points / 100.

    An intervention that moves the partition from `m_before` orbits to `m_after`
    lowers the certified error floor by `(m_after - m_before) / n`. Negative when
    the intervention merges entities, which is a real outcome: an incoherent
    prefix of matched length drove `m` from 15 to 1 and the floor from 0.25 to
    0.95.

    This is a reduction in what can be *proved* wrong, not a measurement of what
    is wrong. The true error rate is bounded below by the floor and may exceed it.
    """
    return certified_error_floor(n_entities, m_before) - certified_error_floor(
        n_entities, m_after
    )


def recovery_ceiling_gain(largest_before: int, largest_after: int) -> float:
    """Theorem 2 as a ratio: how much the downstream recovery ceiling improves.

    A pooled block of size `k` caps any downstream recovery at `1/k`, so shrinking
    the largest block from `k0` to `k1` multiplies the ceiling by `k0 / k1`.
    """
    return pooling_recovery_bound(largest_after) / pooling_recovery_bound(largest_before)
