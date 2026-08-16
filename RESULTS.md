# Results

Every number measured in-process. Each states what it was compared against; a
figure without a control is not a result and does not appear here.

**Hardware** NVIDIA GeForce RTX 4060 Laptop, 8 GiB · Windows 11
**Software** Python 3.11.9 · PyTorch 2.5.1+cu121 · transformers 5.3.0 · float32 · seed 0
**Models — two, and the distinction matters**

| used for | model | width | blocks |
|---|---|---|---|
| retrieval, partitions, detection (§1–§5, §9, §10) | `Qwen/Qwen2.5-0.5B` | `D = 896` | 24 |
| Jacobian dynamics and cost (§6–§7) | `distilgpt2` | `D = 768` | 6 |

The dynamics and cost sections predate the switch to a model that knows facts, and
were not re-run. **No figure from §6 or §7 may be combined with one from §1–§5**:
they describe different networks. Reconciling them is outstanding work, not a
detail.

---

## 1. Coherence gates factual retrieval; length does not

The central measurement. Prefix length held fixed at **128 tokens**; only its
character varies. The prefix contains none of the answers and is identical across
every entity.

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

**Control:** identical token count across every row, so length is held constant by
construction. **The same 128 tokens produce accuracy 1.000 or 0.000 depending only
on whether they form language.**

Incoherent context does not merely fail to help. It merges all twenty countries
onto a single answer — a largest orbit of 20, worse than the orbit of 4 with no
prefix at all.

The model knows every one of these facts. Reaching them is a property of the
surrounding text.

**Caveat.** `capital` and `language` disagree on the shuffled condition, 0.000
against 1.000. The boundary between *coherent* and *merely lexically diverse* is
not settled by this measurement.

## 2. The partition survives once it resolves

Adjusted Rand index between the entity partitions at different prefix lengths,
computed over all entities regardless of correctness, so no class can collapse.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  capital        ARI      answers changed
    0 -> 32   -0.0243                0.600
    0 -> 128   0.0000                0.450
  128 -> 512   1.0000                0.150
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Below the threshold the partition is destroyed and rebuilt. Above it the partition
is **exactly** preserved while individual answers still move — structure held,
labels free.

## 3. Equivariance detects the failure with no ground truth

Two quantities computed only from the model's own outputs under transformations of
its own input. Neither consults a correct answer.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  invariance (self-consistency)   collision (equivariance)
    capital              AUROC 0.859                 AUROC 0.995 [0.97, 1.00]
    language             AUROC 0.942                 AUROC 0.950 [0.83, 1.00]
    pooled, n = 32       AUROC 0.920                 AUROC 0.945 [0.85, 1.00]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Precondition, and it is not optional.** Equivariance requires the relation to be
injective. On a many-to-one relation — many countries share a continent — distinct
entities *should* share an answer, and the signal inverts to 0.273. `RelationSpec`
records injectivity so the precondition cannot be forgotten.

**Credit where due.** The invariance half is close to published self-consistency
work and is included as the baseline. The equivariance half is what reaches 0.995.

Adding a third symmetry does not help: an inversion test scores 0.5686 pooled and
exactly 0.5000 on `capital`, and the unweighted sum of all three scores 0.9451 —
identical to equivariance alone.

## 4. Errors are not information loss

Measured on items the model answers **wrongly**.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  entity still linearly recoverable from h_22        0.9624   (chance 0.0312)
  correct answer token recoverable from h_22         1.0000   (chance 0.0400)
  median rank of the correct answer            3 of 151,936
  correct answer within the top 10                 88 / 100
  correct answer within the top 1000              100 / 100
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Grouped cross-validation, so no template appears in both folds. Recoverability on
**wrong** items (0.9624) exceeds that on correct ones (0.8981).

The context survives. The answer survives, near the top. Nothing was destroyed —
the right answer lost a competition. This is what Theorem 2 addresses and what
Theorem 5 explains the absence of a local signal for.

## 5. Where the answer appears

Median rank of the correct answer, each layer read through the final norm and
unembedding.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  layer      all      correct       wrong
     20    30552        29594       31626
     22        7            3          12
     24        2            1           3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The answer materialises across two layers, late and abruptly. It does not
degrade afterwards.

**Caveat.** The lens applies the final norm to intermediate states that were not
trained to be read through it, so ranks before layer 24 are indicative. Layer 24
is exact — it reproduces the model's own output.

## 6. Dynamics

Characteristic exponents of the token-position Jacobian product, block 3, 46 steps.

**Measured on `distilgpt2`, `D = 768`, not on the model used everywhere above.**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                lambda_1      sum      expanding   last-step drift
  grounded       +0.1653  -226.74     139 / 768            0.0012
  shuffled       +0.1852  -170.37     151 / 768            0.0003
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Positive leading exponent with a strongly negative sum: dissipative dynamics on a
low-dimensional attractor. This is the measured input to **Theorem 4**.

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
`D_KY` varies across depth with `cv 0.6801`, so it is **not** a width-invariant
constant, and the block-0 value saturates the formula rather than measuring a
dimension.

## 7. Cost

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  block forward                    0.588 ms
  full 768 x 768 exact Jacobian   53.694 ms      91.3x forward
  top-8 Krylov, 20 iterations   1428.938 ms    2429.5x forward
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

The exact Jacobian is **26.6x cheaper** than the Krylov estimator of its own top
eight singular values, which it matches to `1.685e-04`. Batched reverse-mode AD
vectorises across all outputs; `k`-column power iteration runs `k x iters`
sequential passes. The crossover width is not measured and is not assumed.

The shipped detector costs **five forward passes** and no Jacobian at all.

## 8. Validation

**163 tests, every one against a closed-form or independently computed answer.**

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

## 9. Theoretical results: what the method certifies

Theorem 1 turns the partition into a **lower bound on the error rate that requires
no ground truth**. For an injective relation on `n` entities producing `m` distinct
answers, the certified error floor is `(n - m) / n`.

Every row below is computed by `caustic.theorems.certified_error_floor`, and the
measured error is shown beside it. The theorem asserts floor <= measured, and it
holds on every row with slack.

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

### Certified reduction

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  capital,  no prefix -> prose            +0.250   certified error removed
  language, no prefix -> prose            +0.333   certified error removed
  capital,  " the" x128 -> prose          +0.950   full measured span
  capital,  no prefix -> " the" x128      -0.700   certified error ADDED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

On `capital`, the choice of prefix moves the **provable** error floor across a
**95-point range** — from 0.950 under a degenerate prefix to 0.000 under coherent
prose, at identical token count. The fourth row is not a footnote: a prefix can add
0.700 to the certified floor, and `repair_by_context` reports that case as
`WORSENED` rather than quietly returning a number.

### Downstream ceiling, from Theorem 2

A pooled block of size `k` caps **any** downstream recovery of the entity at `1/k`.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  largest orbit 20 (" the" x128)   recovery ceiling  0.05
  largest orbit  1 (prose 128)     recovery ceiling  1.00      20x gain
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

No downstream component recovers what a pooled block destroyed, however capable it
is. That is what makes collapse worth detecting rather than merely scoring.

### What these numbers are, and are not

They are bounds on a **certificate**, measured on one model and two injective
relations. `+0.250` means a quarter of the answers were provably wrong before the
intervention and none are provably wrong after — not that the true error rate fell
by exactly that much. On `capital` the true error happened to fall further, 0.450
to 0.000; on `language` it fell from 0.500 to 0.250 while the certified floor went
to zero, which is the bound behaving exactly as a bound should.

The certificate is one-sided by construction. It can prove a model wrong. It can
never prove a model right.

## 10. Why so weak an intervention moves so much

The decision the model gets wrong is a near-tie, and this is what makes the whole
effect possible.

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

The model is not confidently wrong. It is wrong by **a quarter of a standard
deviation**, choosing between roughly thirty live candidates while the remaining
151,934 tokens absorb 69.5% of the probability mass.

That number explains the size of the coherence effect. A prefix has to move logits
by about 0.83 to flip these decisions, and Section 6 measured a positive leading
exponent of +0.1653, so a small change at the input grows along the expanding
directions. The chain is:

```
prefix character changes
  -> perturbation grows            lambda_1 = +0.1653
  -> logits shift by order 0.83
  -> flips a 0.2526 sd near-tie among about 30 candidates
  -> distinct entities land on one answer, which is orbit collapse
  -> certified error floor moves from 0.000 to 0.950
```

Chaos theory earns its place here as the *explanation* rather than as a detector.
Section 6 showed the exponents do not separate correct from wrong answers and are
useless as a signal. What they do explain is why an intervention this weak — 128
tokens about photosynthesis, containing none of the answers — produces an effect
this large. The decision was never robust.

## 11. What the detector actually detects, and where 0.995 does not hold

The 0.995 figure was measured on `capital` and `language` at five-token prompts.
Tested on harder relations, chosen for low answer frequency rather than for
observed failure, it does not hold.

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  relation          ctx   n    acc  wrong  orbits    collision AUROC
  small_capital       6  18  0.167     15       8    0.6889 [0.34, 1.00]
  small_capital     113  18  0.556      8      14    0.6687 [0.38, 0.92]
  element_symbol      7  16  0.062     15       3    0.7333 [0.50, 0.90]
  element_symbol    114  16  1.000      0      16    undefined, one class

  pooled  short   AUROC 0.7083 [0.3809, 1.0000]
  pooled  full    AUROC 0.6687 [0.3819, 0.9167]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Both pooled intervals include 0.5. **On harder relations the detector is not
demonstrated to work.**

**What this means, stated as narrowly as the data supports.** Collision detects
*orbit collapse*, not error. On `capital` and `language` the errors were collapse —
several entities landing on one answer — and collision reached 0.995 against that.
On harder relations the errors are dispersed ignorance, spread across many distinct
wrong answers, so nothing collides and nothing fires. `element_symbol` at short
context is the clearest case: 15 wrong answers occupying only 3 orbits at
accuracy 0.062, where the model has collapsed onto a few generic tokens.

Collapse is one cause of wrong answers among several. The 0.995 is this method's
score against the cause it was built for, and it should be read that way rather
than as a hallucination-detection figure.

**Context does not explain the gap.** Short and full context give 0.7083 and
0.6687 with overlapping intervals, so the earlier concern that 0.995 depended on
the degenerate five-token regime is not confirmed. Relation difficulty separates
the results; context length does not.

**The central finding replicated on a third relation, and more strongly than
before.** `element_symbol` moves from accuracy 0.062 to 1.000 and from 3 orbits to
16 under 108 tokens of neutral prose containing no chemistry. That relation was
selected for difficulty before it was run, not after.

## 12. The certificate across models, and its controls

`python -m caustic.experiments.certificate_across_models`. Three models, four
relations, two prefix conditions. The control for every row is the true error
count, computed from the gold answers *after* the bound was computed without
them. A bound exceeding the true count is a refutation.

```
model                       relation  cond        n  err   T1  T1*  |S|   prec     T6    T6*
distilgpt2                  capital   none       20   20   19   20   20  1.000  0.950  1.000
distilgpt2                  capital   the_x128   20   20   19   20   20  1.000  0.950  1.000
distilgpt2                  language  none       16    9    8    9    9  1.000  0.889  1.000
distilgpt2                  language  the_x128   16   16   15   16   16  1.000  0.938  1.000
distilgpt2                  currency  none       12   12    9   12   10  1.000  0.900  1.000
distilgpt2                  currency  the_x128   12   12   10   12   11  1.000  0.909  1.000
SmolLM2-135M                capital   none       20   16   14   16   15  1.000  0.933  1.000
SmolLM2-135M                capital   the_x128   20   20   18   20   20  1.000  0.900  1.000
SmolLM2-135M                language  none       16    1    0    1    0    nan    nan    nan
SmolLM2-135M                language  the_x128   16   16   15   16   16  1.000  0.938  1.000
Qwen2.5-0.5B                capital   none       20    9    5    9    7  1.000  0.714  1.000
Qwen2.5-0.5B                capital   the_x128   20   20   19   20   20  1.000  0.950  1.000
Qwen2.5-0.5B                language  none       16    6    4    6    5  1.000  0.800  1.000
Qwen2.5-0.5B                language  the_x128   16   16   14   16   16  1.000  0.875  1.000
Qwen2.5-0.5B                currency  none       12    6    0    6    0    nan    nan    nan
Qwen2.5-0.5B                currency  the_x128   12   12   11   12   12  1.000  0.917  1.000

60 bound checks, 0 violations, 4 relation-model pairs skipped
```

`T1*` equals `err` in all 16 evaluable rows; `T1` is strictly below it in all 16.
`Qwen / currency / none` is the row that matters: Theorem 1 certifies **0**, which
proves nothing, while the true count is 6 and Theorem 1\* certifies exactly 6.

**Control for the exactness claim, stated because it weakens it.** Eight of the
16 rows are `" the"×128`, where the model emits one token that is nobody's
capital, so `m* = 0` and any sound bound is exact by construction. Those rows
carry no information about the theorem. On the other eight, `T1` is loose by
exactly one — the entire advertised advantage of `T1*` over `T1` on those rows is
one answer out of twenty. Exactness holds iff the model never gives one entity's
correct answer to another, which is a property of the failure mode and not of the
theorem, and is not observable at inference.

Four pairs were skipped by the injectivity check, `currency` for SmolLM2-135M
only — the same relation, the same gold strings, injective under two tokenizers
and colliding under a third.

## 13. The recall floor, verified exhaustively

`recall_floor(n, m*) = (n − m*) / n`. Control: every injective truth consistent
with each observation, enumerated rather than sampled.

```
n = 2..5, every answer map, every injective truth
2,048,574 configurations with at least one error
0 violations of recall >= (n - m*)/n
tightest margin exactly 0.000000   (the bound is attained)
strictly positive in 99.3% of configurations
```

Zero exactly when `m* = n`, which is Theorem 7's shuffle witness. On `capital`
under `" the"×128`, `m* = 0` and the floor is **1.000**.

This corrects a claim: Theorem 7 first stated that no recall floor existed. It
does exist, and it is `certified_error_floor` — already shipped, and reported
only as an error rate.

## 14. Constrained decoding, and what it separates

`python -m caustic.experiments.constrained_decode`. Control is the free argmax on
the identical prompts and model.

```
relation  cond      mode          n   m   n-m    acc
capital   none      free         20  15     5  0.550
capital   none      constrained  20  20     0  1.000
language  none      free         16  12     4  0.625
language  none      constrained  16  16     0  1.000
currency  none      free         12  12     0  0.500
currency  none      constrained  12  12     0  1.000
capital   the_x128  free         20   1    19  0.000
capital   the_x128  constrained  20   1    19  0.050
language  the_x128  constrained  16   1    15  0.062
currency  the_x128  constrained  12   3     9  0.000
```

Under coherent context, restricting the argmax to the answer set takes accuracy
to 1.000 on all three relations: those errors were the model leaving the answer
space. Under `" the"×128` it changes almost nothing — restricted to capitals, the
model still cannot find the right capital. **The degenerate prefix destroys
retrieval, not surface form**, which is the control the coherence result did not
previously have.

Constraining changes the task toward multiple choice, so these accuracies are not
comparable to free-decoding accuracy elsewhere on this page.

## 15. Cost

Control is the same computation, unbatched, on the same prompts and device.

```
condition        sequential   batched   speedup   answers identical
no prefix          703.63 ms  42.36 ms   16.61x   yes
" the" x 128       760.05 ms 446.94 ms    1.70x   yes
```

20 country-capital prompts, Qwen2.5-0.5B, float32, three runs after a warm-up.
The batched figure excludes the one-prompt padding probe that now ships on by
default; it adds one prompt to a batch of twenty, so the wall-clock claim moves
by about 5%.
The gain is smaller under a long prefix because it already saturates the device.

**Not shipped in a measured path.** No experiment or notebook in this repository
calls the batched code, so this figure is reproduced by the benchmark only and
the left-padding requirement has not been exercised against a real tokenizer
here. A right-padded batch reads the answer at a pad position and yields a
well-formed partition of nobody's answers, with no self-check able to detect it.

## 16. The observable is degenerate on numeric answers

Tokenizer only, no model. Control is a word-answer relation on the same
tokenizers.

```
tokenizer                   relation          n  distinct 1st tokens
distilgpt2                  atomic_number    20                   20
distilgpt2                  planet_order      8                    8
SmolLM2-135M                atomic_number    20                    1
SmolLM2-135M                planet_order      8                    1
Qwen2.5-0.5B                atomic_number    20                    1
Qwen2.5-0.5B                planet_order      8                    1
all three                   capital_word      8                    8
```

Qwen- and Llama-family tokenizers emit the space as its own token before a digit:
`" 1"` is `[220, 16]`, `" 20"` is `[220, 17, 15]`, while `" Paris"` is `[12095]`.
Every numeric answer shares first token 220, so `m = 1` regardless of the model
and Theorem 1 would certify `n − 1 = 19` wrong answers on a model answering all
twenty correctly. `verify_injective` raises on exactly this.

---

## Limits

Sections 1 to 11 are one model at one width, two injective relations of 12 and
20 entities. Sections 12 to 16 add distilgpt2 and SmolLM2-135M, but all three are
base models under 0.5B parameters, where the dominant failure is collapse. Every
headline figure here is monotone decreasing in model capability: as failures
shift from collapse to confusion — a plausible answer belonging to the wrong
entity — orbits go discrete, the certified set empties, the certificate goes
silent and Theorem 1\*'s slack grows. Nothing here tested a model where that
could be observed, and it is the single largest limit on the page. One
distractor passage per condition, one seed. Answers compared by top-1 token, so a
correct answer phrased differently counts as disagreement. Injectivity is a
required precondition and was diagnosed after observing the failure on a
many-to-one relation, not predicted in advance.

Whether coherence-gated retrieval is a general property of language models or a
behaviour of a 0.5B model outside its training regime is **not** established here.

The 0.995 detection figure applies to relations whose errors take the form of orbit
collapse. Section 11 shows it falls to roughly 0.67 to 0.73, with intervals
spanning chance, on relations where errors are dispersed instead. Any use of the
detector should first check that the failure mode is collapse.
