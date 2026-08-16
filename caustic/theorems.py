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
THEOREM 2* (game theory) -- Join Recovery Bound
--------------------------------------------------------------------------------
Let `f_1..f_T` be the answer maps under `T` templates and `m_join` the number of
blocks of their join. For ANY downstream `h` of the full answer tuple, the
fraction of entities recovered is at most `m_join / n`.

Proof. `h` is constant on each join block, since that block is exactly the set
of entities whose observed tuple is identical. It agrees with the identity on at
most one entity per block, giving at most `m_join` successes out of `n`. []

Theorem 2 reads a pooled block as an equilibrium no downstream capability
escapes. That is true of ONE answer and false of the answer vector: the join is
at least as fine as every component, so `m_join >= max_t m_t` and this ceiling
is never lower. The two bounds differ by exactly the information a single
template discards.

The gap is the licence for repair. If the join were as coarse as the worst
template, no prefix could recover anything, because the entity would be absent
from the model's outputs rather than from one of them. Measured across four
relations and five templates, the join is DISCRETE in 8 of 8 conditions, so the
ceiling is 1.0 even where a single template pools 16 entities into 3 orbits.
Pooling under one paraphrase hides the entity; it does not destroy it.

This is the theoretical companion to the measured 96% recoverability of the
entity from the hidden state. Both say the failure is retrieval, not
representation.

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
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "orbit_error_bound",
    "pooling_recovery_bound",
    "join_recovery_bound",
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


def pooling_recovery_bound(block_size: int) -> float:
    """Theorem 2. Ceiling on any downstream recovery of the entity from the answer."""
    if block_size < 1:
        raise ValueError("block size must be positive")
    return 1.0 / block_size


def join_recovery_bound(n_entities: int, n_join: int) -> float:
    """Theorem 2*. Recovery ceiling for a receiver that sees every paraphrase.

    **Statement.** Let `f_1..f_T` be the answer maps under `T` templates and let
    `m_join` be the number of blocks of their join. Then for ANY downstream
    function `h` of the full answer tuple, the fraction of entities recovered is
    at most `m_join / n`.

    Proof. `h` is constant on each join block, since the block is by definition
    the set of entities on which the observed tuple is identical. It can
    therefore agree with the identity on at most one entity per block, giving at
    most `m_join` successes out of `n`. []

    **Relation to Theorem 2.** Theorem 2 caps recovery from ONE answer at `1/k`
    for a block of size `k`, and reads it as a pooling equilibrium that no
    downstream capability escapes. That is a statement about one answer, not
    about the model. The join is at least as fine as every component partition,
    so `m_join >= max_t m_t` and this ceiling is never lower.

    **Why the gap matters.** If the join were as coarse as the worst template,
    repair would be impossible in principle: the entity would be absent from the
    model's outputs, not merely from one of them. Measured across four relations
    and five templates, the join was DISCRETE in 8 of 8 conditions, so the
    ceiling is 1.0 even where a single template pools 16 entities into 3 orbits.
    Pooling under one paraphrase hides the entity; it does not destroy it, and
    that is the licence under which a prefix can restore anything.

    This is also the theoretical companion to the measured 96% recoverability of
    the entity from the hidden state: both say the failure is retrieval, not
    representation.

    Raises:
        ValueError: if `n_join` is outside `[1, n_entities]`, which means the
            count did not come from a partition of these entities.
    """
    if n_entities < 1:
        raise ValueError("need at least one entity")
    if not 1 <= n_join <= n_entities:
        raise ValueError(f"n_join must lie in [1, {n_entities}]; got {n_join}")
    return n_join / n_entities


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
