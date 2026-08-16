"""Could this design ever have resolved AUROC 0.7 from 0.5 at the sizes it ran at?

Section 11 of RESULTS.md narrowed the headline from collision AUROC 0.995 to "the
detector detects collapse, not error", on the evidence of pooled AUROC 0.6687 and
0.7083 with bootstrap intervals spanning 0.5. Two readings survive that evidence
and the repo cannot currently tell them apart:

    (a) the effect really is near chance on harder relations, or
    (b) at n = 16 to 18 entities a 95% interval is wide enough to span 0.5 for a
        true AUROC of 0.7, so the interval was never able to say otherwise.

Reading (b) is a property of the design, not of the model, so it is decidable by
simulation with no model and no GPU. This module draws scores whose TRUE AUROC is
known and measures what `caustic.detect.auroc_ci` reports about them.

**Construction.** Positives are drawn from `N(d, 1)` and negatives from `N(0, 1)`.
For two independent normals the probability that a positive outranks a negative is
`P(X_pos > X_neg) = Phi(d / sqrt(2))`, which is exactly AUROC, so a target `A` is
hit by `d = sqrt(2) * Phi^-1(A)`. That identity is checked empirically at large n
before any power number is reported; if the realised AUROC did not match the
target, every row below would be measuring the wrong thing.

**What the two reported quantities are.** The half-width is `(hi - lo) / 2` of the
percentile bootstrap interval, averaged over independent trials. The power is the
fraction of trials whose interval excludes 0.5 — the fraction of runs in which the
experiment could have reported a detection at all.

**What this does NOT show.** It says nothing about whether collision is a good
score. It assumes normal, equal-variance, independent scores, which real collision
values are not: they are bounded in [0, 1], heavily tied, and correlated across
entities that share an answer. Ties and correlation both cost effective sample
size, so the powers here are optimistic — an upper bound on what the design could
have achieved, not an estimate of what it did achieve.

**Caveat that the numbers inherit, and it is not small.** `auroc` returns a hard
0.5 whenever a resample contains only one class, and `auroc_ci` resamples entities
with replacement, so at small n with a skewed minority class a sizeable share of
bootstrap replicates are scored 0.5 by that guard rather than by the data. That
pulls the interval toward 0.5 and inflates the apparent half-width. A sibling unit
is fixing that behaviour in `caustic/detect.py`; this module does not touch it. The
`1cls` column reports the exact fraction of single-class resamples,
`(n_pos/n)^n + (n_neg/n)^n`, so the contamination is visible per row. Every
half-width here should be recomputed once the sibling fix lands.

    python -m caustic.experiments.detector_power
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from caustic.detect import auroc, auroc_ci

SEED = 0
N_TRIALS = 120
N_BOOT = 400
VERIFY_N = 200_000
TARGET_HALF_WIDTH = 0.05

AUROC_TARGETS = (0.6, 0.7, 0.8, 0.9)
N_GRID = (12, 16, 18, 20, 24, 32, 48, 64, 96, 128, 192, 256)

# The repo's own per-relation entity counts, from caustic.experiments.coupling_gap
# RELATIONS and caustic.experiments.detector_at_context HARD.
REPO_N = {
    12: "currency",
    16: "language, element_symbol",
    18: "small_capital",
    20: "capital",
}

# The class splits the repo actually evaluated on. `n_wrong` is the positive class
# for the collision AUROC. The first three and the last are read off RESULTS.md
# section 11; the `cap+lang` row is section 3's pooled block at n = 32 with the
# wrong-count from section 10 ("wrong items 15 of 32"), which was measured when
# `language` carried 12 entities rather than the 16 in `coupling_gap.RELATIONS`
# today. It is included because it is the split the 0.995 headline was pooled at.
REPO_CASES = (
    ("small_capital short", 18, 15),
    ("small_capital full", 18, 8),
    ("element_symbol short", 16, 15),
    ("cap+lang pooled (sec 3)", 32, 15),
    ("hard pooled short", 34, 30),
)


def norm_cdf(x: float) -> float:
    """Standard normal CDF via `math.erfc`, which is accurate in the far tail."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def norm_ppf(p: float, tol: float = 1e-12) -> float:
    """Standard normal quantile by bisection on `norm_cdf`. No scipy dependency.

    Bisection rather than a rational approximation because the grid here is four
    values computed once: correctness is free and speed buys nothing.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1); got {p}")
    lo, hi = -40.0, 40.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def separation(target_auroc: float) -> float:
    """Mean separation `d` giving a true AUROC of `target_auroc`."""
    return math.sqrt(2.0) * norm_ppf(target_auroc)


def single_class_fraction(n: int, n_pos: int) -> float:
    """Exact share of bootstrap resamples containing only one class.

    `auroc_ci` draws `n` indices uniformly with replacement, so a resample is
    all-negative with probability `((n - n_pos)/n)^n` and all-positive with
    probability `(n_pos/n)^n`. Every such resample is scored a flat 0.5 by
    `auroc`'s empty-class guard.
    """
    return (n_pos / n) ** n + ((n - n_pos) / n) ** n


def trial_scores(rng: np.random.Generator, n_pos: int, n_neg: int, d: float):
    scores = np.concatenate([rng.normal(d, 1.0, n_pos), rng.normal(0.0, 1.0, n_neg)])
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    return scores, labels


def measure(rng: np.random.Generator, n_pos: int, n_neg: int, d: float,
            n_trials: int = N_TRIALS, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Mean point AUROC, mean CI half-width, and power against the 0.5 null.

    Each trial gets its own bootstrap seed. Reusing one seed would give every
    trial the identical 400 resampling index vectors, so the bootstrap's own
    Monte-Carlo error would be perfectly correlated across trials and would not
    shrink with `n_trials` — including the realised single-class share, which
    would then sit at whatever that one stream happened to contain rather than at
    the closed form reported in the `1cls` column.
    """
    pts, halves, excl = [], [], 0
    for t in range(n_trials):
        scores, labels = trial_scores(rng, n_pos, n_neg, d)
        pt, lo, hi = auroc_ci(scores, labels, n_boot=n_boot, seed=SEED + t)
        pts.append(pt)
        halves.append(0.5 * (hi - lo))
        excl += int(lo > 0.5 or hi < 0.5)
    return float(np.mean(pts)), float(np.mean(halves)), excl / n_trials


def verify_construction(rng: np.random.Generator) -> None:
    print("construction check: realised AUROC at n_pos = n_neg = "
          f"{VERIFY_N:,} against the target")
    print(f"  {'target':>7} {'d':>8} {'realised':>9} {'error':>9}")
    for a in AUROC_TARGETS:
        d = separation(a)
        scores, labels = trial_scores(rng, VERIFY_N, VERIFY_N, d)
        got = auroc(scores, labels)
        print(f"  {a:>7.2f} {d:>8.4f} {got:>9.4f} {got - a:>+9.4f}")

    # The single-class fraction is closed form; replay auroc_ci's own index stream
    # to confirm the formula describes the resamples it actually draws. Only
    # splits whose fraction is above the resolution of 20,000 draws are shown: at
    # a balanced split the probability is 1e-6 and a replay would confirm nothing.
    for n, n_pos in ((16, 15), (18, 15), (12, 11)):
        check = np.random.default_rng(SEED)
        labels = np.concatenate([np.ones(n_pos), np.zeros(n - n_pos)]).astype(bool)
        hits = sum(
            1 for _ in range(20_000)
            if (lab := labels[check.integers(0, n, n)]).all() or not lab.any()
        )
        print(f"  single-class resamples n={n:>3} n_pos={n_pos:>3}: "
              f"exact {single_class_fraction(n, n_pos):.4f}  "
              f"replayed {hits / 20_000:.4f}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"seed={SEED}  trials={N_TRIALS} per cell  n_boot={N_BOOT} per interval  "
          f"bootstrap seed={SEED}+trial")
    print("no model, no GPU: score distributions are simulated with a known true AUROC\n")

    verify_construction(rng)

    print("\npower of a 95% bootstrap interval against the null AUROC = 0.5")
    print("balanced classes, n_pos = n_neg = n/2;  hw = mean CI half-width, "
          "pow = fraction of trials whose CI excludes 0.5")
    hdr = f"{'n':>5} {'1cls':>7}" + "".join(
        f" | {'A=' + format(a, '.2f'):>7} {'hw':>6} {'pow':>5}" for a in AUROC_TARGETS
    )
    print(hdr)
    print("-" * (len(hdr) + 24))
    grid_hw = {}
    for n in N_GRID:
        n_pos = n // 2
        line = f"{n:>5} {single_class_fraction(n, n_pos):>7.4f}"
        for a in AUROC_TARGETS:
            pt, hw, power = measure(rng, n_pos, n - n_pos, separation(a))
            grid_hw[(a, n)] = hw
            line += f" | {pt:>7.3f} {hw:>6.3f} {power:>5.2f}"
        if n in REPO_N:
            line += f"   <- repo n ({REPO_N[n]})"
        print(line)

    print("\nrows marked <- are the entity counts the repo's relations actually have.")
    print("the balanced split above is the FRIENDLIEST case for those n; the repo's")
    print("real splits are skewed, and skew costs both power and interval width:\n")

    hdr2 = (f"{'repo case':<24} {'n':>4} {'n_pos':>6} {'1cls':>7}"
            + "".join(f" | {'A=' + format(a, '.2f'):>7} {'hw':>6} {'pow':>5}"
                      for a in AUROC_TARGETS))
    print(hdr2)
    print("-" * len(hdr2))
    for label, n, n_pos in REPO_CASES:
        line = f"{label:<24} {n:>4} {n_pos:>6} {single_class_fraction(n, n_pos):>7.4f}"
        for a in AUROC_TARGETS:
            pt, hw, power = measure(rng, n_pos, n - n_pos, separation(a))
            line += f" | {pt:>7.3f} {hw:>6.3f} {power:>5.2f}"
        print(line)

    # Control for the one setting this module lowered to stay fast. RESULTS.md
    # sections 3 and 11 were computed at n_boot = 4000, and a 2.5th percentile
    # from 400 replicates rests on about ten order statistics, so it is biased
    # inward. The bias direction matters: inward means the table UNDERSTATES how
    # wide the repo's intervals were, which argues against this module's own
    # reading (b) rather than for it.
    ctrl_n, ctrl_a = 18, 0.70
    _, hw_400, pow_400 = measure(rng, ctrl_n // 2, ctrl_n - ctrl_n // 2, separation(ctrl_a))
    _, hw_4k, pow_4k = measure(rng, ctrl_n // 2, ctrl_n - ctrl_n // 2, separation(ctrl_a),
                               n_boot=4000)
    print(f"\ncontrol on n_boot, at n = {ctrl_n} balanced and true AUROC {ctrl_a:.2f}")
    print(f"  n_boot={N_BOOT:>5}   hw {hw_400:.4f}   power {pow_400:.2f}   (this module)")
    print(f"  n_boot={4000:>5}   hw {hw_4k:.4f}   power {pow_4k:.2f}   "
          "(RESULTS.md sections 3 and 11)")

    # Minimum n for a half-width of 0.05 at true AUROC 0.7. The half-width falls as
    # c / sqrt(n), so c is fitted on the large-n end of the grid where the
    # single-class contamination is nil, then the prediction is measured directly.
    big = [n for n in N_GRID if n >= 64]
    c = float(np.median([grid_hw[(0.70, n)] * math.sqrt(n) for n in big]))
    n_star = int(math.ceil((c / TARGET_HALF_WIDTH) ** 2))
    n_star += n_star % 2
    pt, hw, power = measure(rng, n_star // 2, n_star - n_star // 2, separation(0.70),
                            n_trials=max(N_TRIALS // 4, 25))
    print(f"\nminimum n for a CI half-width of {TARGET_HALF_WIDTH:.2f} at true AUROC 0.70")
    print(f"  fit          hw = {c:.4f} / sqrt(n)   over n in {big}")
    print(f"  predicted n  {n_star}   (balanced, {n_star // 2} wrong / {n_star - n_star // 2} correct)")
    print(f"  measured     AUROC {pt:.3f}  hw {hw:.4f}  power {power:.2f}   "
          f"at n = {n_star}, {max(N_TRIALS // 4, 25)} trials")

    print("\nreading")
    print("-------")
    print("Stated before the numbers were seen, and nothing here was tuned toward")
    print("either outcome.")
    print()
    print("  power at the repo's n is LOW   -- the 0.6687 and 0.7083 pooled figures in")
    print("     RESULTS.md section 11 are uninformative, not null. An interval that")
    print("     spans 0.5 at a true AUROC of 0.7 is the expected result of the design,")
    print("     so it is not evidence that the effect is absent. Any later per-model or")
    print("     per-relation AUROC comparison at these sizes must be reported as")
    print("     underpowered rather than as a conclusion.")
    print()
    print("  power at the repo's n is ADEQUATE -- an effect of 0.7 would have been")
    print("     resolved and was not, so the detector really is near chance on")
    print("     dispersed-error relations and HEAD's narrowing is well supported.")
    print()
    print("The powers above are an optimistic ceiling: real collision scores are tied")
    print("and correlated, and the single-class guard in `auroc` widens intervals")
    print("further at the skewed splits in the second table. Recompute once the")
    print("`auroc_ci` fix lands.")


if __name__ == "__main__":
    main()
