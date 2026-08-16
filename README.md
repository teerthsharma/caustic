<h1 align="center">Caustic</h1>

<p align="center">
  <strong>A hallucination detector, a repair, and a governor for language models, built from the orbit partition of a relation, with five proved bounds behind them.</strong><br>
  A model can know a fact and still be unable to reach it. When it cannot, distinct entities collapse onto one answer.<br>
  That collapse is a topological invariant, it is measurable with no ground truth, and it is bounded from below by a theorem.<br>
  The same bound then selects the intervention — including how much noise to add, and whether to intervene at all.
</p>

<p align="center">
  <strong>Invented by <a href="https://teerthsharma.vercel.app/">Teerth Sharma</a></strong> ·
  <a href="https://github.com/teerthsharma/caustic">github.com/teerthsharma/caustic</a> ·
  <em>teerths57@gmail.com</em>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square&color=00aaff" alt="MIT"></a>
  <a href="#4-the-five-theorems"><img src="https://img.shields.io/badge/theorems-9%20proved%2C%202%20no--go-blueviolet?style=flat-square" alt="Theorems"></a>
  <a href="#3-detection-without-ground-truth"><img src="https://img.shields.io/badge/equivariance%20AUROC-0.995%20(collapse--type%20errors)-yellow?style=flat-square" alt="AUROC"></a>
  <a href="#7-stochastic-resonance"><img src="https://img.shields.io/badge/stochastic%20resonance-%2B0.333%20from%20noise-ff6b35?style=flat-square" alt="Resonance"></a>
  <a href="#8-system-prompts-are-never-neutral"><img src="https://img.shields.io/badge/neutral%20system%20prompts-0%20of%206-critical?style=flat-square" alt="Neutrality"></a>
  <a href="#3-detection-without-ground-truth"><img src="https://img.shields.io/badge/detector-5%20forward%20passes%2C%20no%20Jacobian-00aaff?style=flat-square" alt="Detector"></a>
  <a href="#20-limits"><img src="https://img.shields.io/badge/precondition-injective%20relations%20only-orange?style=flat-square" alt="Precondition"></a>
  <a href="#16-validation"><img src="https://img.shields.io/badge/tests-163%20closed--form-brightgreen?style=flat-square" alt="Tests"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/python-%E2%89%A5%203.10-yellow?style=flat-square" alt="Python"></a>
</p>


---

## Scope: what this does not claim

Collected here rather than scattered, because several of these correct statements made
elsewhere in this document.

**The detector detects collapse, not error.** AUROC 0.995 is measured on relations whose
errors take the form of distinct entities pooling onto one answer. Where errors disperse —
each entity wrong in its own way — pooled AUROC falls to 0.67–0.71 with bootstrap intervals
spanning 0.5.

**Those intervals are underpowered, not null.** Simulated at the relation sizes used here
(n = 12–20), a 95% bootstrap interval has power 0.00–0.38 to separate a true AUROC of 0.70
from 0.50; half-width 0.05 needs n ≈ 404. The minority classes behind the reported figures
are 1, 3, 4 and 8 items. "The detector degrades on harder relations" is a claim the design
cannot support; "the design cannot resolve whether it degrades" is what was measured.

**The precondition is token-level, and one shipped relation violates it.** Theorem 1 needs the
true answer map to be injective *in the encoding the answer function compares*, which for a
top-1 token detector means distinct gold answers must have distinct first tokens.
`small_capital` does not: `Asmara`/`Asuncion` share token 1634, `Lusaka`/`Ljubljana` share 444.
`verify_injective` is the check. No published number is affected — 0 of 14 certified errors on
that relation are collision artifacts — but the precondition held by luck rather than by
construction.

**Injectivity is necessary and not sufficient.** `currency` is injective at the token level,
12 entities to 12 distinct gold tokens, and its collision AUROC is 0.306 — below chance. The
cause is that the score averages over five templates while the label comes from one: template
0 separates every entity there, so its own collision is 0 and the whole signal is borrowed
from templates whose partition differs. `collision_scored` reports the scored template alone.

**One accuracy figure is graded on first letters.** `element_symbol` scores the first token of
the gold answer, and 4 of its 16 golds are multi-token — `Sb`→`' S'`, `Hg`→`' H'`, `Zr`→`' Z'`,
`Rb`→`' R'`. For a quarter of that relation any word with the right first letter counts as
correct, so its accuracy 1.000 is not established.

**"Pooled, full context" is a single relation.** `element_symbol` has zero wrong answers at
full context and never enters the pool, so that row is `small_capital` alone. Comparing it to
the two-relation short-context pool does not support "context does not explain the gap".

**`NEUTRAL_PREFIX` is off-topic but not chemistry-free.** It names chemical energy, carbon
dioxide, oxygen and a reaction, and shares the word *chemical* with the probe template. A
matched-length chemistry-stripped passage still repairs `element_symbol` (0.875), so the
result is coherence rather than priming — but the prefix was described as containing no
chemistry, and it does. Separately, the repair figure of 1.000 on `capital` belongs to a
longer tiled passage; the exported constant gives 0.750.

**Recall has no floor, and cannot.** Theorem 7 exhibits a model that shuffles the correct
answers among entities: nothing pools, nothing is inadmissible, accuracy is zero, and the
certificate is silent. Precision is proved; recall is measured at 0.65–0.79 and is
unboundable in principle.

---

## Abstract

**Caustic** detects a specific, measurable failure of factual retrieval in a language model,
proves what the detection certifies, and then uses that certificate to choose the repair.

The failure is this. A model can hold a fact and still be unable to reach it, and whether it
can is decided by the character of the surrounding text rather than by its length. Holding a
prefix at exactly 128 tokens and varying only what those tokens are, accuracy on twenty
country–capital pairs is **1.000** when the prefix is coherent prose and **0.000** when it is
one token repeated 128 times. The knowledge does not move between those two rows. The regime
does.

The failure has a shape, and the shape is what makes it detectable. Under an unreachable fact
the model does not answer randomly — it answers *identically*, sending distinct entities to
one answer. The object that records this is the **orbit partition**: the partition of entities
by the answer they receive, which is `H0` of the answer-equivalence relation. It needs no
correct answer to compute, and once it resolves it is exactly preserved as context grows —
adjusted Rand index `1.0000` from 128 to 512 tokens, while 15% of the individual answers still
change.

Four things follow. The failure is **detectable without ground truth**: an equivariance score
computed from five forward passes, consulting no answer key and no Jacobian, reaches **AUROC
0.995** on an injective relation whose errors are collapse. It detects *collapse*, not error,
and where errors disperse instead it falls to 0.67–0.71 with intervals spanning chance —
see [Scope](#scope-what-this-does-not-claim) before relying on the figure. It is **bounded**: for an injective relation on `n` entities
producing `m` distinct answers, at least `n − m` answers are provably wrong, and a pooled block
of `k` entities caps *any* downstream recovery at `1/k`. It is **repairable**, and the repair
measures its own effect size and flags the case where a prefix makes matters worse. And,
because the bound needs no ground truth, it is **usable as an objective**: `select_prefix` runs
candidate contexts in competition scored by the certified floor, always entering the empty
prefix so it can decline to intervene.

That last property produces the strongest result here. The wrong decision is a near-tie — the
correct token trails the chosen one by **0.2526 standard deviations** — which is the textbook
condition for **stochastic resonance**. Adding noise to the input embeddings and taking a
majority vote lifts accuracy on `language` from 0.500 to **0.833** at an intermediate noise
level and destroys it entirely past that, an inverted U with a gain of **+0.333** from noise
alone. The certified error floor, which never sees an answer, is minimised at exactly the noise
level where accuracy peaks. **Theorem 1 selects the noise level with no ground truth**, which
is what separates this from a curiosity.

Two further measurements fall out of the same instrument. **No system prompt is neutral**: of
six tested on `capital`, the one instructing care and discouraging guessing halves accuracy
from 0.550 to 0.300 and merges orbits from 15 to 9, a JSON-format instruction reaches 0.950,
and not one leaves the partition intact. And **averaging logits across an ensemble collapses**
to 0.100 on `capital`, below a single pass, while majority vote never goes catastrophic — with
a positive characteristic exponent, average ranks, not magnitudes.

Five theorems stand behind the claims, one per branch of mathematics, each with a proof and an
executable witness: topology, game theory, differential geometry, chaos theory, and a
partial-differential-symmetry no-go which shows that no pointwise function of the Jacobian —
determinant, smallest singular value, condition number, spectral decay — can detect pooling at
all.

**Keywords:** orbit partition, `H0`, answer-equivalence relation, equivariance, invariance,
group action on a fact, pooling equilibrium, signalling game, orbit error bound, certified
error, no-go theorem, local diffeomorphism, global injectivity, stochastic resonance,
sub-threshold signal, inverted U, majority vote, ensemble collapse, system-prompt neutrality,
Johnson–Lindenstrauss projection, cross-width comparability, volume contraction, characteristic
exponent, Kaplan–Yorke dimension, dissipative dynamics, logit lens, linear probe, adjusted Rand
index, AUROC, hallucination detection, retrieval regime, context coherence, inference-time
repair

---

## Table of contents

| § | Section | What is in it |
|---|---|---|
| [1](#1-the-failure-measured) | The failure, measured | The same 128 tokens, accuracy 1.000 and 0.000 |
| [2](#2-the-orbit-partition) | The orbit partition | The shape of the failure, and that it is stable |
| [3](#3-detection-without-ground-truth) | Detection without ground truth | Equivariance, AUROC 0.995, and its precondition |
| [4](#4-the-five-theorems) | **The five theorems** | Full statements, proofs, and the chain between them |
| [5](#5-what-the-method-certifies) | What the method certifies | The floor, the ceiling, and what is not claimed |
| [6](#6-the-decision-is-a-near-tie) | The decision is a near-tie | 0.2526 standard deviations, and why that matters |
| [7](#7-stochastic-resonance) | **Stochastic resonance** | Noise raises accuracy by +0.333, and Theorem 1 picks the level |
| [8](#8-system-prompts-are-never-neutral) | **System prompts are never neutral** | Six prompts, zero neutral, one that halves accuracy |
| [9](#9-the-ensemble-and-where-it-fails) | The ensemble, and where it fails | Vote survives; mean-logit collapses to 0.100 |
| [10](#10-repair-and-the-governor) | Repair, and the governor | The intervention, and selection by competition |
| [11](#11-comparing-geometry-across-model-widths) | Comparing across widths | Seeded JL, and the per-item error that limits it |
| [12](#12-what-the-model-holds-when-it-is-wrong) | Inside a wrong answer | The answer is present, ranked 3rd of 151,936 |
| [13](#13-dynamics) | Dynamics | Contraction and attractor dimension — **`distilgpt2`** |
| [14](#14-cost) | Cost | Why the shipped detector is five forward passes — **`distilgpt2`** |
| [15](#15-quick-start) | Quick start | Install, test, and a runnable example |
| [16](#16-validation) | Validation | 163 tests, each against a closed form |
| [17](#17-implementation-map) | Implementation map | File-by-file responsibility |
| [18](#18-measurement-environment) | Measurement environment | Two models, and why they must not be mixed |
| [19](#19-attribution) | **Attribution** | Four constructions adapted from other repositories |
| [20](#20-limits) | **Limits** | Collected once, at the end |

Every number on this page is measured, and appears in [`RESULTS.md`](RESULTS.md) with the
control it was compared against. There is no CI badge, because there is no CI: correctness is
argued from 163 closed-form assertions ([§16](#16-validation)), which a green check cannot
supply.

> **Two models, and the distinction is load-bearing.** Sections 1–12 are measured on
> `Qwen/Qwen2.5-0.5B` (`D = 896`, 24 blocks). Sections 13 and 14 are measured on `distilgpt2`
> (`D = 768`, 6 blocks). **No figure from §13 or §14 may be combined with one from §1–§12.**
> They describe different networks. Reconciling them is outstanding work
> ([§18](#18-measurement-environment)), not a detail.

---

## 1. The failure, measured

Prefix length held fixed at **128 tokens**. Only the character of those tokens varies. The
prefix contains none of the answers and is identical across every entity, so it carries no
task information — the control is built into the design rather than argued for afterwards.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  capital, 20 entities        accuracy   distinct answers   largest orbit
    no prefix                    0.550                 15               4
    coherent prose, 128 tok      1.000                 20               1
    shuffled words, 128 tok      0.000                  1              20
    " the" x 128                 0.000                  1              20
    random token ids, 128 tok    0.100                  3              18

  language, 12 entities
    no prefix                    0.500                  8               5
    coherent prose, 128 tok      0.750                 12               1
    shuffled words, 128 tok      1.000                 12               1
    " the" x 128                 0.000                  2               7
    random token ids, 128 tok    0.750                 10               3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Control:** identical token count in every row, so length is held constant by construction and
is not available as an explanation. **The same 128 tokens produce accuracy 1.000 or 0.000
depending only on whether they form language.**

Incoherent context does not merely fail to help. It merges all twenty countries onto a single
answer — a largest orbit of 20, worse than the orbit of 4 with no prefix at all. A prompt whose
prefix is the word `" the"` one hundred and twenty-eight times is a well-formed input, a valid
tokenisation, and an unanswerable question.

The model knows every one of these facts. Reaching them is a property of the surrounding text.

**Caveat.** `capital` and `language` disagree on the shuffled condition, 0.000 against 1.000.
The boundary between *coherent* and *merely lexically diverse* is not settled by this
measurement.

---

## 2. The orbit partition

Under the failure the model does not become noisy. It becomes constant. Distinct entities
receive one answer, and the object that records exactly that is the partition of entities
induced by the answer map `f`, whose blocks are the classes of

$$e_1 \sim e_2 \iff f(e_1) = f(e_2)$$

This is `H0` of the answer-equivalence relation "these two entities receive the same answer" —
its connected components. Three properties make it the right instrument:

- **It needs no ground truth.** The partition is computed from the model's own answers over
  all entities, whether those answers are right or wrong.
- **It cannot be confounded by accuracy.** Every entity is included regardless of correctness,
  so no class can collapse and no comparison inherits a moving base rate.
- **It is what the theorems constrain.** `n − m` and `1/k` in [§4](#4-the-five-theorems) are
  both statements about blocks of this partition.

`OrbitReport` carries it: `n_distinct` is the number of blocks, `largest_orbit` the size of the
biggest one, `collapse_ratio` that size over `n`, and `collapsed` is the boolean a caller acts
on. `certified_error_rate` is the same partition read as a bound, and it is the objective every
intervention in this repository is scored against.

### 2.1 The partition survives once it resolves

Adjusted Rand index between the entity partitions at different prefix lengths, computed over
all entities regardless of correctness.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  capital        ARI      answers changed
    0 -> 32   -0.0243                0.600
    0 -> 128   0.0000                0.450
  128 -> 512   1.0000                0.150
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Below the threshold the partition is destroyed and rebuilt. Above it the partition is
**exactly** preserved while individual answers still move — structure held, labels free. The
answer-churn column is what makes an ARI of `1.0000` evidence of stability rather than evidence
that nothing happened.

---

## 3. Detection without ground truth

A fact carries a group action, and a model that holds the fact must respect two halves of it:

```
  invariance     paraphrase the prompt  ->  the answer must NOT change
  equivariance   swap the entity        ->  the answer MUST change
```

A model outside its retrieval regime fails the second while passing the first: it is invariant
where it should be equivariant, returning the same answer whichever entity is named. Both
quantities are computed from the model's own outputs under transformations of its own input.
Neither consults a correct answer, which is the only condition under which a hallucination
detector is of any use.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  invariance (self-consistency)   collision (equivariance)
    capital              AUROC 0.859                 AUROC 0.995 [0.97, 1.00]
    language             AUROC 0.942                 AUROC 0.950 [0.83, 1.00]
    pooled, n = 32       AUROC 0.920                 AUROC 0.945 [0.85, 1.00]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

`symmetry_scores` returns both per entity, plus their difference. `invariance` is agreement
across paraphrases of the same fact, and high is good; `collision` is the fraction of *other*
entities receiving this entity's answer, and low is good. Five templates per relation means
five forward passes per entity, and no derivative of anything.

**Credit where due.** The invariance half is close to published self-consistency work and is
included as the baseline the other half has to beat. The equivariance half — penalising a model
for giving the *same* answer to different entities — is what reaches 0.995.

Adding a third symmetry does not help: an inversion test scores 0.5686 pooled and exactly
0.5000 on `capital`, and the unweighted sum of all three scores 0.9451, identical to
equivariance alone.

### 3.1 The precondition, and it is not optional

Equivariance requires the relation to be **injective**: distinct entities must genuinely warrant
distinct answers. On a many-to-one relation — many countries share a continent — distinct
entities *should* collide, and the signal inverts to **0.273**. Point it at such a relation and
you do not have a broken detector; you have a working detector with the sign reversed, which is
worse.

`RelationSpec.injective` records the precondition so it cannot be forgotten.
`OrbitReport.certified_errors` returns 0 whenever it is False, `collapsed` is then always False,
and `select_prefix` raises rather than optimising a score it knows is inverted.

---

## 4. The five theorems

Five results, one per branch, each with a proof and an executable witness in
[`caustic/theorems.py`](caustic/theorems.py). Every statement is elementary, and that is
deliberate: the value is not in the difficulty of the proofs but in the fact that each one is
checkable against a measurement made in this repository, and that together they chain.

```
   differential geometry ──┐
                           ├──▶  topology  ──▶  game theory
   chaos theory ───────────┘

   partial-differential symmetry ──▶  why the differential route fails alone
```

Read: vanishing entity coupling *causes* pooling (differential geometry feeds topology); volume
contraction *drives* pooling (chaos theory feeds topology); pooling *bounds every downstream
reader* (topology feeds game theory); and no pointwise Jacobian statistic can substitute for the
topological measurement (the no-go).

**Notation.** `E` is a finite set of entities, `|E| = n`. A relation `R : E → A` is *injective*
when distinct entities have distinct correct answers. A model induces `f : E → A`, and `P(f)` is
the partition of `E` by the value of `f`, with `m = |P(f)|` blocks called *orbits*.

### Theorem 1 (topology) — Orbit Error Bound

For injective `R`, the number of entities on which `f` errs, written `err(f)`, is at least
`n − m`:

$$\mathrm{err}(f) \geq n - m$$

**Proof.** The true map `R` is injective, so distinct entities carry distinct correct answers.
If `e1 ≠ e2` lie in one orbit then `f(e1) = f(e2)` while `R(e1) ≠ R(e2)`, so `f` is wrong on at
least one of them. An orbit of size `s` therefore contributes at least `s − 1` errors, and
summing over the `m` orbits gives `sum_i (s_i − 1) = n − m`. ∎

**Tight:** attained exactly when every orbit contains one correct answer.

**Measured instance.** `capital` with no prefix gave `n = 20`, `m = 15`, certifying at least
5 wrong answers with no ground truth consulted. Observed accuracy 0.550, so 9 were actually
wrong and the bound held with slack.

**Why this theorem does more work than the others.** It is the only one that turns into an
objective. Because the bound consults no answer key, it can be evaluated on a live relation
where no answers are known — which is what makes the prefix competition of
[§10](#10-repair-and-the-governor) and the noise selection of [§7](#7-stochastic-resonance)
possible at inference time.

### Theorem 2 (game theory) — Pooling Recovery Bound

If `f` maps a block of `k` entities to one answer, then for **any** downstream function `h`, the
probability that `h(f(e))` recovers `e` under a uniform prior on that block is at most `1/k`:

$$\Pr[\, h(f(e)) = e \,] \leq \frac{1}{k}$$

**Proof.** `h ∘ f` is constant on the block, so it takes one value there. It can therefore agree
with the identity on at most one of the `k` entities. Under a uniform prior the success
probability is at most `1/k`. ∎

This is the signalling-game reading: a block of size `k > 1` is a **pooling equilibrium**, and
pooling destroys the receiver's ability to infer the sender's type. No amount of downstream
capability recovers it — not a larger model above that layer, not a longer chain of thought, not
a better decoder — which is why orbit collapse is not merely an error but an unrecoverable one.

**Measured instance.** The `" the" x 128` prefix pooled all 20 countries, bounding any
downstream recovery at **0.05**.

### Theorem 3 (differential geometry) — Zero Coupling Implies Pooling

Let `z_c(h)` be the logit of token `c` as a function of the entity representation `h`,
continuously differentiable on a domain containing a path `γ` from `h1` to `h2`. If the
directional derivative of `z_c` along `γ` vanishes identically, then `z_c(h1) = z_c(h2)`:

$$z_c(h_2) - z_c(h_1) = \int_{\gamma} \nabla z_c \cdot d\ell = 0$$

**Proof.** The fundamental theorem of calculus along `γ`, as displayed. ∎

**Consequence.** If this holds for every candidate `c`, the two entities receive identical logit
vectors, hence identical answers, hence share an orbit, and Theorem 1 applies. **Differential
geometry feeds topology.** Vanishing entity coupling is a *sufficient* condition for pooling —
which is also why the topological measurement is the one to take: the partition observes the
consequence whether or not the coupling itself is measurable.

**Measured instance.** The token the model actually chose on wrong items coupled to the entity
at 0.93 times its coupling to control tokens, against 1.33 for the correct token. The chosen
token was closer to entity-independent, which is the finite-difference shadow of this statement.

`path_integral_change` evaluates the integral numerically along the straight path, and the suite
pins it against linear and quadratic closed forms to `1e-6`.

### Theorem 4 (chaos theory) — Dissipative Pooling

Let `T` be differentiable with characteristic exponents `λ_1 ≥ … ≥ λ_D` whose sum `S = Σ_i λ_i`
is negative. Then for any bounded set `A` of positive Lebesgue measure, the volume of its image
contracts:

$$\mathrm{vol}(T^{n}(A)) \sim e^{nS} \longrightarrow 0$$

**Proof.** The change-of-variables formula gives `vol(T^n(A)) = ∫_A |det D(T^n)|`, and the
exponents are defined so that `(1/n) log |det D(T^n)| → Σ_i λ_i = S`. With `S < 0` the integrand
decays like `exp(nS)`, so the volume does. ∎

**Consequence.** Once the image volume falls below the resolution separating decision regions,
distinct inputs must land in one region and pool. Volume contraction is therefore a *driver* of
the collapse Theorem 1 penalises, and depth is the clock. **Chaos theory feeds topology.**

**Measured instance.** The token product at block 3 gave `S = −226.74` over 768 dimensions with
only 139 expanding directions, and `D_KY = 29.57` at block 1 — a 26x compression against the
width ([§13](#13-dynamics)). What prevents total pooling in practice is that the trajectory is
short, not that the map is volume-preserving.

### Theorem 5 (partial-differential symmetry) — No Local Criterion Detects Pooling

There exists a smooth map `F` whose Jacobian is nonsingular at every point of its domain and
which is not injective. Consequently **no function of the local Jacobian alone** — determinant,
smallest singular value, condition number, spectral decay, or any other pointwise invariant —
can decide injectivity.

$$F(x, y) = (e^{x}\cos y, \; e^{x}\sin y)$$

**Proof by witness.** `det DF = e^{2x} > 0` everywhere, yet `F(x, y) = F(x, y + 2π)`. The
Jacobians at the two preimages are related by a rotation and therefore share every spectral
invariant, so no pointwise function of the Jacobian distinguishes the injective case from this
one. ∎

This is the honest statement of a moral the refutation of the Jacobian Conjecture made vivid:
local invertibility everywhere does not imply global injectivity, and the failure is invisible
to local data. It is a **no-go** result, and it is the only theorem here that tells you what
*not* to build. A detector watching a determinant, a smallest singular value or a condition
number for orbit collapse is watching a quantity that provably cannot see it.

**Measured corroboration.** Zero of 768 singular values fell below `1e-6 × sigma_max`, so the
observed collapse occurred with an everywhere-nonsingular Jacobian — exactly the case the
witness describes.

**The escape is global, and Theorem 1 is what takes it:** compare *two* entities instead of
examining one point. That is why the shipped detector is five forward passes and no Jacobian at
all.

---

## 5. What the method certifies

Theorem 1 turns the partition into a **lower bound on the error rate that requires no ground
truth**. For an injective relation on `n` entities producing `m` distinct answers, the certified
error floor is `(n − m) / n`. Every row below is computed by
`caustic.theorems.certified_error_floor`, and the measured error is shown beside it. The theorem
asserts floor ≤ measured, and it holds on every row with slack.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  relation   condition       n    m   certified floor   measured error
  capital    no prefix      20   15             0.250            0.450
  capital    prose 128      20   20             0.000            0.000
  capital    " the" x128    20    1             0.950            1.000
  language   no prefix      12    8             0.333            0.500
  language   prose 128      12   12             0.000            0.250
  language   " the" x128    12    2             0.833            1.000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 5.1 Certified reduction

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  capital,  no prefix -> prose            +0.250   certified error removed
  language, no prefix -> prose            +0.333   certified error removed
  capital,  " the" x128 -> prose          +0.950   full measured span
  capital,  no prefix -> " the" x128      -0.700   certified error ADDED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

On `capital`, the choice of prefix moves the **provable** error floor across a **95-point
range** — from 0.950 under a degenerate prefix to 0.000 under coherent prose, at identical token
count. The fourth row is not a footnote: a prefix can add 0.700 to the certified floor, and
`repair_by_context` reports that case as `WORSENED` rather than quietly returning a number.

### 5.2 Downstream ceiling, from Theorem 2

A pooled block of size `k` caps **any** downstream recovery of the entity at `1/k`.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  largest orbit 20 (" the" x128)   recovery ceiling  0.05
  largest orbit  1 (prose 128)     recovery ceiling  1.00      20x gain
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

No downstream component recovers what a pooled block destroyed, however capable it is. That is
what makes collapse worth detecting rather than merely scoring.

### 5.3 What these numbers are, and are not

They are bounds on a **certificate**, measured on one model and two injective relations.
`+0.250` means a quarter of the answers were provably wrong before the intervention and none are
provably wrong after — not that the true error rate fell by exactly that much. On `capital` the
true error happened to fall further, 0.450 to 0.000; on `language` it fell from 0.500 to 0.250
while the certified floor went to zero, which is the bound behaving exactly as a bound should.

The certificate is one-sided by construction. It can prove a model wrong. It can never prove a
model right.

---

## 6. The decision is a near-tie

The decision the model gets wrong is a near-tie, and this is what makes everything downstream of
here possible.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  vocabulary                                             151,936
  wrong items                                            15 of 32

  logit gap, chosen minus gold          mean 0.8338   median 0.8085
  logit standard deviation over vocab   mean 3.2942
  gap as a fraction of one sd           mean 0.2526

  p(chosen) 0.2042      p(gold) 0.1005      ratio 2.76
  mass on those two     0.3047
  mass on the other 151,934 tokens        0.6953

  effective support exp(H)   wrong 30.0   correct 27.5   of 151,936
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The model is not confidently wrong. It is wrong by **a quarter of a standard deviation**,
choosing between roughly thirty live candidates while the remaining 151,934 tokens absorb 69.5%
of the probability mass.

That number explains the size of the coherence effect. A prefix has to move logits by about 0.83
to flip these decisions, and [§13](#13-dynamics) measured a positive leading exponent, so a
small change at the input grows along the expanding directions. The chain is:

```
prefix character changes
  -> perturbation grows            lambda_1 = +0.1653
  -> logits shift by order 0.83
  -> flips a 0.2526 sd near-tie among about 30 candidates
  -> distinct entities land on one answer, which is orbit collapse
  -> certified error floor moves from 0.000 to 0.950
```

Chaos theory earns its place here as the *explanation* rather than as a detector. The exponents
do not separate correct from wrong answers and are useless as a signal. What they explain is why
an intervention this weak — 128 tokens about photosynthesis, containing none of the answers —
produces an effect this large. The decision was never robust.

**And it names the next experiment.** A signal present but below the threshold at which the
readout commits to it is, in the exact technical sense, a **sub-threshold signal**. Systems with
sub-threshold signals have a known and falsifiable response to noise.

---

## 7. Stochastic resonance

**Stochastic resonance** is the phenomenon in which a nonlinear system's response to a
sub-threshold signal *improves* under moderate noise and degrades again under too much. It
predicts a specific shape, and only one of the three possible shapes is evidence:

```
  monotone increase   noise is smoothing something; not resonance
  monotone decrease   no resonance; noise only destroys
  inverted U          resonance, with an optimum at intermediate noise
```

The peak must also exceed the zero-noise baseline, or the curve is a decline with a flat start.

**Where the noise goes.** Into the input embeddings, scaled by their own standard deviation, so
`sigma` is dimensionless. Injecting into the logits directly would be a different and much
weaker claim — it would test whether adding noise to a ranking changes the ranking, which is
trivially true. Each level is scored by **majority vote over 16 draws**, because resonance
appears in the expectation and a single noisy pass is a worse estimator than a clean one.

### 7.1 The curve

Measured on `language`, 12 entities, `Qwen/Qwen2.5-0.5B`, seed 0.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  sigma   accuracy   vs baseline   distinct answers   certified floor
   0.00      0.500        +0.000                  8             0.333
   0.40      0.583        +0.083                  9             0.250
   0.80      0.833        +0.333                 11             0.083   <- peak
   1.20      0.667        +0.167                 11             0.083
   1.60      0.000        -0.500                  2             0.833
   3.00      0.000        -0.500                  1             0.917
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Inverted U confirmed.** Accuracy rises to 0.833 at `sigma = 0.80` and falls to 0.000 by
`sigma = 1.60`, a gain of **+0.333 over the zero-noise baseline from noise alone**. No prefix
was added, no answer was consulted, no parameter was changed. The only intervention is Gaussian
noise on the input embeddings and a vote.

**The distinct-answer column is the mechanism, not a diagnostic beside it.** It runs 8 → 9 → 11
→ 11 → 2 → 1. Moderate noise breaks orbit collapse — entities that were sharing an answer stop
sharing it — and excessive noise causes orbit collapse, driving all twelve languages onto one
answer. At `sigma = 3.00` the model gives the same answer to every question, which is at least
self-consistent, and is the sharpest available demonstration that an invariance-only detector
would score that condition as healthy.

The certified-floor column is `certified_error_floor(12, m)` applied to the column beside it, so
it is a function of the partition and of nothing else.

### 7.2 Theorem 1 selects the noise level with no ground truth

This is the result that makes the section a method rather than a curiosity.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  sigma                 0.00    0.40    0.80    1.20    1.60    3.00
  accuracy   (needs gold)
                       0.500   0.583   0.833   0.667   0.000   0.000
  certified floor (no gold)
                       0.333   0.250   0.083   0.083   0.833   0.917
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                                        ^^^^^
                       accuracy peaks and the certified floor is minimised
                       at the same sigma, and the floor never saw an answer
```

The accuracy row requires an answer key. The floor row does not — it is `(n − m) / n` computed
from the model's own answers. **The two rows agree on the argmin.** A caller with no labels can
sweep `sigma`, score each level by the certified floor, and land on the level that in fact
maximises accuracy.

That is a closed loop: a quantity a deployed system can compute selects a hyperparameter a
deployed system otherwise cannot tune. The same loop drives prefix selection in
[§10](#10-repair-and-the-governor), where the empty prefix is always a candidate so the governor
can decline.

### 7.3 The optimum is relation-dependent, and `capital` declines

Under identical noise on the identical model, `capital` does **not** show resonance. Accuracy
declines monotonically from 0.550 to 0.200.

There is no universally good `sigma`. The optimum depends on the relation, which is precisely
why it has to be selected per relation rather than fixed — and why §7.2 is the load-bearing half
of this section. Reporting a fixed `sigma = 0.8` as a recommended setting would have been wrong
on one of the two relations tested.

```bash
python -m caustic.experiments.stochastic_resonance
```

---

## 8. System prompts are never neutral

A system prompt is a prefix. [§1](#1-the-failure-measured) established that prefixes move the
certified error floor by up to 95 points at fixed length. It follows that "does this system
prompt damage factual retrieval" is an empirical question with a cheap answer, and that the
answer is not automatically no.

Six prompts, `capital`, 20 entities, `Qwen/Qwen2.5-0.5B`. `ARI` is the adjusted Rand index of
the partition against the no-prompt condition; `entropy` is mean predictive entropy.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  prompt         tok      acc   orbits   largest       ARI    entropy
  none             0    0.550       15         4    1.0000     3.5622
  helpful         14    0.750       19         2    0.1923     3.0786
  cautious        29    0.300        9         6    0.2721     3.2967
  persona         27    0.900       20         1    0.0000     4.3754
  json            22    0.950       20         1    0.0000     3.6461
  long_policy     54    0.750       18         3    0.1465     4.0753
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Three readings, in order of how uncomfortable they are.

**Not one of six is neutral.** A prompt is neutral when it leaves the partition alone, `ARI =
1.0000`. The highest achieved is 0.2721 and two are exactly 0.0000. Every one of these prompts
restructured which entities share an answer. A prompt that improves accuracy while restructuring
the partition has intervened on factual retrieval; it has not left it alone.

**The prompt instructing care is the worst one.** `cautious` — "if you are not certain of a
fact, say so rather than guessing" — **halves accuracy, 0.550 to 0.300**, and merges orbits from
15 to 9. By Theorem 1 that moves the certified error floor from 0.250 to 0.550: it more than
doubles the number of answers that are *provably* wrong. The instruction most likely to appear
verbatim in a production system prompt did the most damage of the six.

**Formatting instructions are not free either, and here the sign is positive.** `json`, 22
tokens about output format and nothing about facts, reaches **0.950** with a fully separated
partition. `persona`, which asks for confidence and wit, reaches 0.900. Neither prompt contains
information about capitals. Both moved the partition further than `helpful` did.

`prompt_cost` measures this for a caller's own prompt and relation, returning the change in
certified floor and the ARI against the no-prompt condition:

```python
from caustic.governor import prompt_cost

cost = prompt_cost(spec, top1, "You are a careful assistant. ", name="cautious")
print(cost)          # cautious: floor 0.250 -> 0.550 (+0.300), ARI 0.2721, costs
print(cost.neutral)  # False — neutrality is ARI >= 0.99, not "did not obviously hurt"
```

```bash
python -m caustic.experiments.ensemble_and_neutrality
```

---

## 9. The ensemble, and where it fails

[§6](#6-the-decision-is-a-near-tie) measured a near-tie and a positive leading exponent, which
together say a single forward pass sits on an unstable decision. The classical response to a
positive exponent is not to predict one trajectory but to average over many: individual orbits
are unpredictable while the invariant measure is stable. Here that means asking the same question
under eight different neutral prefixes and taking a consensus.

Two consensus rules, because they fail differently and reporting only one is choosing the
flattering one after the fact.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  relation     n   single   best-1   worst-1   mean-logit    vote
  capital     20    0.550    0.800     0.000        0.100   0.600
  language    12    0.500    0.917     0.417        0.917   0.917
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  single            no prefix, one forward pass
  best-1 / worst-1  the luckiest and unluckiest individual prefix
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Mean-logit averaging collapses on `capital` to 0.100** — below the 0.550 of a single pass with
no ensemble at all. The ensemble was built to reduce variance and, under mean-logit, reduced
accuracy instead. The cause is visible in the same row: individual prefixes span 0.000 to 0.800,
and a single confident outlier dominates a mean of magnitudes. Averaging logits gives the most
opinionated member of the ensemble a veto over the other seven.

**Majority vote never goes catastrophic.** It reaches 0.600 on `capital`, above the single-pass
baseline, and 0.917 on `language`, equalling best-1 without knowing in advance which prefix that
was. It does not always win — on `language` mean-logit ties it — but it does not lose by 45
points either.

**The lesson, stated as a rule.** With a positive characteristic exponent, **average ranks, not
magnitudes.** A magnitude is a trajectory-dependent quantity in a system where trajectories
diverge; a rank is not. This is why every consensus in this repository — the resonance vote of
[§7](#7-stochastic-resonance), the governor of [§10](#10-repair-and-the-governor) — is
rank-based or partition-based, and none of them averages a logit.

The failure is the more useful half of the result, and it is the half a paper reporting only the
`language` row would have omitted.

---

## 10. Repair, and the governor

### 10.1 The intervention

The detector finds collapsed orbits. `repair_by_context` applies the intervention the
measurements point at, and reports the effect size rather than asserting it: prepend coherent
prose that contains none of the answers, is identical across every entity, and is unrelated in
subject.

```python
from caustic import NEUTRAL_PREFIX, repair_by_context

report = repair_by_context(spec, answer_fn, prefix=NEUTRAL_PREFIX, gold=gold)
print(report)
# largest orbit 20 -> 1, distinct answers 1 -> 20 of 20 entities,
# accuracy 0.000 -> 1.000  REPAIRED
```

| field | meaning |
|---|---|
| `repaired` | True only for a genuine collapsed → fully separated transition |
| `worsened` | True when the prefix **merged** entities that were previously separate |
| `accuracy_delta` | reported only when `gold` is supplied; the verdict never consults it |
| `before/after_largest_orbit`, `before/after_n_distinct` | the raw partition on both sides |

**`worsened` is not a defensive check.** An incoherent prefix of the same length drove the
largest orbit from 4 to 20 in the measurements this module is built on
([§1](#1-the-failure-measured)), adding 0.700 to the certified floor. A prefix can make things
much worse, and a repair function that cannot say so is a repair function that will eventually
lie.

**Why this is not prompt engineering in the pejorative sense.** The prefix carries no
information about the task: it contains none of the answers, it is identical across entities,
and it is unrelated in subject. It cannot be leaking an answer, because the *same 128 tokens in
shuffled order* drive accuracy to zero on `capital`. What it supplies is distributional, not
informational. `NEUTRAL_PREFIX` is 128 tokens on mechanical calculators, ocean currents,
language and photosynthesis — the least interesting paragraph in the repository, and on
`capital` it is worth 0.550 → 1.000.

Two honest constraints ship with it. If a caller's relation concerns oceans, looms or
photosynthesis then `NEUTRAL_PREFIX` is no longer neutral for them, which is why the prefix is a
parameter. And `sweep_prefixes` returns a report for *every* candidate prefix, because quoting
the best of several without reporting the rest is selection on the outcome.

**What this is not.** It is not a pretraining method. The pretraining analogue — augmenting facts
across varied coherent contexts so they are extractable from any of them — is documented in the
literature with far larger effects and is not implemented here. This is an inference-time wrapper
whose effect this module measures on the caller's own model and relation.

### 10.2 The governor: selection by competition

A single hand-picked prefix is a bet on which of several you happened to write. Over the eight
neutral prefixes of [§9](#9-the-ensemble-and-where-it-fails), individual accuracy on `capital`
ranged from **0.000 to 0.800**. The governor does not need to be lucky, only to be able to
score — and Theorem 1 gives it a score that needs no ground truth.

`select_prefix` runs the candidates in competition, ranks them by certified error floor, and:

- **always enters the empty prefix**, under the name `none`, so declining to intervene is a
  first-class outcome rather than a failure to find one;
- **declines unless a candidate strictly beats doing nothing**, breaking ties toward declining
  and then toward the earlier candidate;
- **reports the margin** over doing nothing as `improvement`, which is therefore never negative;
- **raises on a non-injective relation** rather than optimising a score it knows is inverted
  ([§3.1](#31-the-precondition-and-it-is-not-optional)).

```python
from caustic import select_prefix

verdict = select_prefix(spec, top1, {"prose": NEUTRAL_PREFIX, "terse": "Answer briefly. "})
print(verdict)            # selected 'prose': floor 0.250 -> 0.000 (+0.250), largest orbit 1
print(verdict.intervened) # False when nothing beat the empty prefix
print(verdict.scores)     # every candidate's floor, including 'none'
```

The `scores` dictionary is returned in full for the same reason `sweep_prefixes` returns every
report: a governor that shows only its winner is a governor that cannot be audited.

---

## 11. Comparing geometry across model widths

Every geometric quantity in this repository lives in `R^D`. Entity coupling is a gradient norm,
the characteristic exponents are singular values of a `D × D` product, and `D_KY` is derived
from them. A coupling ratio measured at `D = 896` is not directly comparable to one from a model
of width 4096, so a result on one model says nothing about another. That is the sharpest
limitation in [`RESULTS.md`](RESULTS.md), and [`caustic/bridge.py`](caustic/bridge.py) is the
partial answer to it.

**The construction.** A seeded Johnson–Lindenstrauss projection into a fixed `k`. The
JL lemma states that a random linear map into `R^k` preserves pairwise squared distances to
within `1 ± eps` with high probability, for `k = O(log n / eps^2)` **independent of the source
dimension**. Norms therefore survive, and a *ratio* of norms survives with the distortions
partially cancelling. The seed is the load-bearing detail: two runs sharing `CANONICAL_SEED`
share a projection matrix exactly, so numbers from different models land in one frame rather
than merely in one dimension.

**The result.** A true coupling ratio of 1.33, recovered after projection from three source
widths. Measured on synthetic gradients built with the observed structure.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  target k    D = 896   D = 2048   D = 4096    spread    per-item error
        64     1.3445     1.3376     1.3444    0.0052              9.3%
       256     1.3316     1.3287     1.3288    0.0022              5.4%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**The limitation, stated as prominently as the result.** Per-item relative error is **9.3% at
`k = 64` and 5.4% at `k = 256`**. The projection makes *population* statistics comparable across
widths. It does **not** make individual measurements comparable. A claim about a distribution
survives the projection; a claim about a single entity does not, and a 5.4% per-item error is
larger than most of the effects this repository would want to attribute to a single entity.

`jl_distortion_bound` returns `inf` rather than a small number whenever `k` is too small for any
useful guarantee, because a small number would be read as a promise.

---

## 12. What the model holds when it is wrong

Measured on items the model answers **wrongly**, with grouped cross-validation so that no
template appears in both folds.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  entity still linearly recoverable from h_22        0.9624   (chance 0.0312)
  correct answer token recoverable from h_22         1.0000   (chance 0.0400)
  median rank of the correct answer            3 of 151,936
  correct answer within the top 10                 88 / 100
  correct answer within the top 1000              100 / 100
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Recoverability on **wrong** items (0.9624) exceeds that on correct ones (0.8981).

The context survives. The answer survives, near the top. Nothing was destroyed — the right
answer lost a competition. This is the state Theorem 2 addresses: the loss happens where distinct
entities are mapped onto one answer, not in the representation feeding that map. It is also why
Theorem 5 matters operationally. There is no local degeneracy at the moment of failure to go
looking for, because nothing locally degenerate has happened.

It is also the precondition for [§7](#7-stochastic-resonance). Noise can only lift a signal that
is present, and this table is where the signal is shown to be present.

### 12.1 Where the answer appears

Median rank of the correct answer, each layer read through the final norm and unembedding.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  layer      all      correct       wrong
     20    30552        29594       31626
     22        7            3          12
     24        2            1           3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The answer materialises across two layers, late and abruptly. It does not degrade afterwards.

**Caveat.** The lens applies the final norm to intermediate states that were not trained to be
read through it, so ranks before layer 24 are indicative. Layer 24 is exact — it reproduces the
model's own output.

---

## 13. Dynamics

> **Measured on `distilgpt2`, `D = 768`, 6 blocks — not on the model used in §1–§12.** No figure
> in this section may be combined with one from §1–§12.

Characteristic exponents of the token-position Jacobian product, block 3, 46 steps, against a
shuffled-token control that preserves the token multiset.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                lambda_1      sum      expanding   last-step drift
  grounded       +0.1653  -226.74     139 / 768            0.0012
  shuffled       +0.1852  -170.37     151 / 768            0.0003
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Positive leading exponent with a strongly negative sum: dissipative dynamics on a
low-dimensional attractor. That is exactly the hypothesis `S < 0` of **Theorem 4**, measured
rather than assumed, and supplying it to the chain is the whole job of this section. The drift
column is what makes the values quotable: each is small against the value it drifts on.

Kaplan–Yorke dimension per block, validated against the textbook Lorenz spectrum
`(0.906, 0, −14.572)` giving `2.0622`, reproduced to `1e-3`:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  block    D_KY     D / D_KY    expanding
      1   29.57         26.0        9 / 768
      3  298.08          2.6      139 / 768
      5  674.67          1.1      354 / 768
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Block 1 places 768 dimensions of transport on a 29.57-dimensional attractor.

**`D_KY` is not a width-invariant constant.** It varies across depth with `cv 0.6801`, and the
block-0 value saturates the formula rather than measuring a dimension. Both facts are stated
because the number is otherwise easy to over-read: contraction is the hypothesis Theorem 4 needs,
and `D_KY` describes how strong it is at a given depth, not a second invariant of the network.

---

## 14. Cost

> **Measured on `distilgpt2`, `D = 768`** — same caveat as [§13](#13-dynamics).

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  block forward                    0.588 ms
  full 768 x 768 exact Jacobian   53.694 ms      91.3x forward
  top-8 Krylov, 20 iterations   1428.938 ms    2429.5x forward
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The exact Jacobian is **26.6x cheaper** than the Krylov estimator of its own top eight singular
values, which it matches to `1.685e-04`. The estimator costs 26.6x more than the quantity it was
brought in to approximate, and it is not a broken estimator: batched reverse-mode AD vectorises
across all outputs, while `k`-column power iteration runs `k × iters` sequential passes. The
crossover width is not measured and is not assumed.

None of this is on the detector's path. **The shipped detector costs five forward passes and no
Jacobian at all**, which Theorem 5 predicts in advance: the Jacobian could not have supplied the
missing signal at any price.

---

## 15. Quick start

```bash
git clone https://github.com/teerthsharma/caustic.git
cd caustic
pip install -e ".[dev]"          # numpy, torch, pytest
python -m pytest -q              # 163 passed, no model download, no GPU
```

The detector, the repair, the governor, the bridge and the theorems need only `numpy`. A live
model needs the extra:

```bash
pip install -e ".[experiments]"  # transformers
```

### 15.1 Detect, repair, and let the governor choose

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from caustic import NEUTRAL_PREFIX, RelationSpec, orbit_partition, repair_by_context
from caustic import select_prefix

MODEL = "Qwen/Qwen2.5-0.5B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()


def top1(prompt: str) -> int:
    """Prompt -> top-1 next-token id. Must be deterministic: a sampled answer
    makes the partition noise."""
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        return int(model(ids).logits[0, -1].argmax())


facts = {"France": "Paris", "Japan": "Tokyo", "Peru": "Lima",
         "Kenya": "Nairobi", "Norway": "Oslo"}

spec = RelationSpec(
    templates=("The capital of {e} is", "{e}'s capital is"),
    entities=tuple(facts),
    injective=True,          # distinct countries have distinct capitals
)

# 1. Detect. No ground truth is consulted anywhere in this block.
report = orbit_partition(spec, top1)
print(report)                      # n entities -> m distinct answers, largest orbit s
print(report.certified_errors)     # Theorem 1: n - m, proved
print(report.certified_error_rate) # the same bound as a rate, which is the objective
print(report.collapsed)            # the boolean to act on

# 2. Repair with a fixed prefix, and measure the effect. `gold` is optional and
#    is used only to print an accuracy column beside the verdict.
gold = {e: tok(" " + a, add_special_tokens=False).input_ids[0]
        for e, a in facts.items()}

print(repair_by_context(spec, top1, prefix=NEUTRAL_PREFIX, gold=gold))
# largest orbit s -> 1, distinct answers m -> 5 of 5 entities,
# accuracy a -> b  REPAIRED

# 3. Or do not pick the prefix yourself. The governor runs candidates in
#    competition scored by the certified floor, always enters the empty prefix,
#    and declines when nothing beats doing nothing.
verdict = select_prefix(spec, top1, {
    "prose": NEUTRAL_PREFIX,
    "terse": "Answer with a single word. ",
})
print(verdict)              # selected 'prose': floor x -> y (+d), largest orbit 1
print(verdict.intervened)   # False if the empty prefix won
print(verdict.improvement)  # certified error removed relative to doing nothing
print(verdict.scores)       # every candidate, including 'none'
```

`orbit_partition` uses the first template. `symmetry_scores(spec, top1)` uses all of them and
needs at least two, since invariance is undefined on one. Set `injective=False` for a
many-to-one relation: `certified_errors` and `collapsed` then correctly report nothing, and
`select_prefix` raises ([§3.1](#31-the-precondition-and-it-is-not-optional)).

### 15.2 The theorems and the bridge, directly

```python
from caustic import (
    orbit_error_bound, pooling_recovery_bound, winding_witness,
    comparable_ratio, jl_distortion_bound,
)
from caustic.theorems import certified_error_floor

orbit_error_bound(20, 15)                      # 5     -- Theorem 1, measured instance
certified_error_floor(20, 15)                  # 0.25  -- the same bound as a rate
pooling_recovery_bound(20)                     # 0.05  -- Theorem 2, collapsed ceiling
image, jac, det = winding_witness(0.0, 0.0)    # Theorem 5, nonsingular and many-to-one

comparable_ratio(grad_entity, grad_control, target_dim=256)  # width-comparable, §11
jl_distortion_bound(n_points=32, target_dim=256)             # inf if k is too small
```

### 15.3 Reproducing the measurements

Each experiment prints its own table and its own control.

```bash
python -m caustic.experiments.coherence_vs_length       # §1
python -m caustic.experiments.orbit_invariant           # §2.1
python -m caustic.experiments.symmetry_break            # §3
python -m caustic.experiments.stochastic_resonance      # §7
python -m caustic.experiments.ensemble_and_neutrality   # §8, §9
python -m caustic.experiments.answer_presence           # §12
python -m caustic.experiments.attractor_dimension       # §13
python -m caustic.experiments.probe_cost                # §14
```

---

## 16. Validation

**163 tests, every one against a closed-form or independently computed answer.** Collected with
`python -m pytest --collect-only -q`.

| assertion | tolerance |
|---|---|
| Jacobian of a position-wise linear block equals its weight matrix | `1e-10` |
| Krylov estimate matches exact `svdvals` | `1e-6` |
| power-law exponent recovered from a synthetic spectrum | `1e-9` |
| log-volume equals `torch.linalg.slogdet` | `1e-8` |
| diagonal cocycle returns `log a`; orthogonal returns `0` | `1e-8` |
| exponents sum to the mean `log\|det\|` | `1e-8` |
| Kaplan–Yorke reproduces the Lorenz value `2.0622` | `1e-3` |
| Theorem 1 bound never exceeds true error count | 2000 random instances |
| Theorem 2 bound beaten by no constant decoder | exhaustive |
| Theorem 3 path integral matches linear and quadratic closed forms | `1e-6` |
| Theorem 4 exponent sum equals `log\|det A^n\|` | `1e-9` |
| Theorem 5 both preimages share every spectral invariant | `1e-9` |
| flat spectrum returns exponent zero (negative control) | `1e-9` |
| partition is bitwise identical across repeated calls | exact |
| `select_prefix` declines when no candidate beats the empty prefix | exact |
| JL projection is bitwise identical under a shared seed | exact |

Some of those rows are negative controls — inputs whose correct answer is "nothing here" —
because the failure mode of a spectral pipeline is not an exception, it is a plausible number
from noise. A passing suite is therefore a statement about the mathematics, not about the last
time the code changed.

---

## 17. Implementation map

The package ships flat at the repository root.

| file | responsibility |
|---|---|
| [`caustic/regime.py`](caustic/regime.py) | `RelationSpec`, `OrbitReport`, `orbit_partition`, `symmetry_scores` — the detector |
| [`caustic/repair.py`](caustic/repair.py) | `NEUTRAL_PREFIX`, `repair_by_context`, `sweep_prefixes` — the intervention and its effect size |
| [`caustic/governor.py`](caustic/governor.py) | `select_prefix`, `prompt_cost` — selection by competition against the certified floor, and what a system prompt costs |
| [`caustic/bridge.py`](caustic/bridge.py) | `jl_matrix`, `project`, `comparable_ratio`, `jl_distortion_bound` — the seeded shared frame across model widths |
| [`caustic/theorems.py`](caustic/theorems.py) | the five statements, their proofs in the module docstring, one executable witness each, plus `certified_error_floor` and `certified_reduction` |
| [`caustic/jacobian.py`](caustic/jacobian.py) | `block_map`, `exact_jacobian`, `singular_values`, `top_singular_values` |
| [`caustic/cocycle.py`](caustic/cocycle.py) | `lyapunov_spectrum`, `finite_time_spectrum` — QR characteristic exponents of a matrix product, with the running trace so convergence is visible |
| [`caustic/attractor.py`](caustic/attractor.py) | `kaplan_yorke_dimension`, `embedding_bound`, `metric_entropy`, `spectrum_report` |
| [`caustic/spectrum.py`](caustic/spectrum.py) | `sigma_max`, `log_volume`, `stable_rank`, `tail_alpha` — scalar summaries of a singular spectrum |
| [`caustic/oseledets.py`](caustic/oseledets.py) | `growth_filtration`, `filtration_entropy`, `tolerance_sweep` |
| [`caustic/detect.py`](caustic/detect.py) | `auroc`, `auroc_ci`, and the Mahalanobis and PCA baselines any score must beat |
| [`caustic/experiments/`](caustic/experiments) | one runnable file per measurement, each printing its own control |
| [`tests/`](tests) | 163 closed-form assertions |
| [`RESULTS.md`](RESULTS.md) | every number on this page, with its control |

---

## 18. Measurement environment

Every number came from one host. None is projected past the measured range.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Hardware   NVIDIA GeForce RTX 4060 Laptop, 8 GiB · Windows 11
  Software   Python 3.11.9 · PyTorch 2.5.1+cu121 · transformers 5.3.0
             float32 · seed 0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Two models, and the distinction matters.**

| used for | model | width | blocks |
|---|---|---|---|
| retrieval, partitions, detection, resonance, prompts, ensemble ([§1](#1-the-failure-measured)–[§12](#12-what-the-model-holds-when-it-is-wrong)) | `Qwen/Qwen2.5-0.5B` | `D = 896` | 24 |
| Jacobian dynamics and cost ([§13](#13-dynamics)–[§14](#14-cost)) | `distilgpt2` | `D = 768` | 6 |

The dynamics and cost sections were measured on a smaller model and were not re-run on the one
that knows facts. **No figure from §13 or §14 may be combined with one from §1–§12**: they
describe different networks. The `lambda_1 = +0.1653` that appears in the causal chain of
[§6](#6-the-decision-is-a-near-tie) is a `distilgpt2` measurement used to explain a
`Qwen2.5-0.5B` observation, and it is offered as a mechanism, not as a joint measurement.
Reconciling the two is outstanding work, not a detail.

[§11](#11-comparing-geometry-across-model-widths) is the beginning of the fix, and its per-item
error of 5.4% at `k = 256` is the beginning of why it is not yet the whole fix.

Requirements: Python `>= 3.10`, `numpy >= 1.24`, `torch >= 2.4`; `transformers >= 4.40` for the
experiments and `pytest >= 8` for the suite.

---

## 19. Attribution

Four constructions in this repository were adapted from the author's other work. Each is named
here with the file that carries it.

| source | construction | used in |
|---|---|---|
| [**Epsilon**](https://github.com/teerthsharma/Epsilon) | a seeded Johnson–Lindenstrauss map into a shared frame, so two agents can compare geometry without exchanging full representations | [`caustic/bridge.py`](caustic/bridge.py) — the same map carries geometry across model widths ([§11](#11-comparing-geometry-across-model-widths)) |
| [**epsilon-cli**](https://github.com/teerthsharma/epsilon-cli) | the stochastic-resonance framing: a sub-threshold signal, an inverted-U response, and the requirement that the peak beat the zero-noise baseline | [`caustic/experiments/stochastic_resonance.py`](caustic/experiments/stochastic_resonance.py) ([§7](#7-stochastic-resonance)) |
| [**EPSILON-PHASE**](https://github.com/teerthsharma/EPSILON-PHASE) | adaptive noise scheduling — sweeping an injected-noise level rather than fixing it, and selecting per regime | the `sigma` sweep and its Theorem-1-selected optimum ([§7.2](#72-theorem-1-selects-the-noise-level-with-no-ground-truth)) |
| [**laamba-silence**](https://github.com/teerthsharma/laamba-silence) | a governor that runs several candidates in competition and lets a comparator pick the winner, rather than committing to one in advance | [`caustic/governor.py`](caustic/governor.py) — `select_prefix` ([§10.2](#102-the-governor-selection-by-competition)) |

What is new here is what each construction is pointed at: a scorer that requires no ground truth.
The JL frame carries a certified quantity rather than an agent state; the resonance sweep is
selected by Theorem 1 rather than by held-out accuracy; the competition is scored by the orbit
partition rather than by a reward model.

---

## 20. Limits

Collected once, here, rather than scattered through the sections above.

- **One model per claim, two models in the repository.** §1–§12 are `Qwen2.5-0.5B` at `D = 896`;
  §13–§14 are `distilgpt2` at `D = 768`. They must not be combined
  ([§18](#18-measurement-environment)). Whether coherence-gated retrieval is a general property
  of language models, or a behaviour of a 0.5B model outside its training regime, is **not**
  established here.
- **Two injective relations**, of 12 and 20 entities. One distractor passage per condition, one
  seed. Every effect in §7, §8 and §9 rests on those two relations, and they disagree with each
  other more than once.
- **The resonance optimum is relation-dependent.** `language` peaks at `sigma = 0.80` with a gain
  of +0.333; `capital` declines monotonically under identical noise, 0.550 to 0.200. There is no
  recommended `sigma`, only a procedure for finding one ([§7.3](#73-the-optimum-is-relation-dependent-and-capital-declines)).
- **Mean-logit ensembling is unsafe under a positive exponent.** It scored 0.100 on `capital`,
  below a single pass, because one confident outlier dominates a mean of magnitudes
  ([§9](#9-the-ensemble-and-where-it-fails)). Average ranks, not magnitudes.
- **The system-prompt result is a measurement on one relation.** Six prompts on `capital`. That
  `cautious` halves accuracy is a fact about these 20 entities and this model; that *no prompt is
  neutral* is the claim that generalises, and it generalises because neutrality is defined as
  `ARI = 1.0` and nothing reached it.
- **The JL bridge carries populations, not individuals.** Per-item relative error is 9.3% at
  `k = 64` and 5.4% at `k = 256` ([§11](#11-comparing-geometry-across-model-widths)). Population
  statistics compare across widths; single-entity claims do not survive the projection.
- **Top-1 token comparison.** Answers are compared by their first token, so a correct answer
  phrased differently counts as disagreement, and the partition, the accuracy column and every
  certified floor inherit that convention.
- **Injectivity is a required precondition**, not a convenience. On a many-to-one relation the
  equivariance signal inverts to 0.273, `certified_errors` is meaningless, `collapsed` is
  correctly always False, and `select_prefix` raises. The requirement was diagnosed by
  measurement rather than predicted in advance.
- **The bounds are bounds on a certificate.** Theorems 1 and 2 are exact and they constrain a
  partition; they say nothing about how often such partitions arise in deployment
  ([§5.3](#53-what-these-numbers-are-and-are-not)). The certificate is one-sided: it can prove a
  model wrong, never right.
- **No pointwise Jacobian statistic can detect pooling.** Theorem 5 is a no-go, not a gap to be
  closed by a better spectral summary. Zero of 768 singular values fell below
  `1e-6 × sigma_max`, so the observed collapse happened with an everywhere-nonsingular Jacobian.
- **The logit-lens ranks before layer 24 are indicative**, since the final norm is applied to
  states not trained to be read through it. Layer 24 is exact.
- **`D_KY` is not a width-invariant constant** (`cv 0.6801` across depth), and the block-0 value
  saturates the formula rather than measuring a dimension.
- **The two relations disagree on the shuffled condition** (0.000 against 1.000), so the boundary
  between coherent text and merely lexically diverse text is not resolved by these measurements.

---

## License

MIT. See [`LICENSE`](LICENSE) and the declaration in [`pyproject.toml`](pyproject.toml).

<p align="center">
  <strong>Invented by <a href="https://teerthsharma.vercel.app/">Teerth Sharma</a></strong><br>
  <a href="https://github.com/teerthsharma/caustic">github.com/teerthsharma/caustic</a> ·
  <em>teerths57@gmail.com</em><br>
  <em>A caustic is where a map folds and distinct preimages merge.<br>
  The partition is how you see it without being told where to look.</em>
</p>
