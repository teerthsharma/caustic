"""How much of the "intervals span 0.5" finding was the empty-class guard.

Section 11 of RESULTS.md narrowed the headline from 0.995 to "the detector detects
collapse, not error", and the stated evidence was that the pooled intervals include
0.5. Those intervals came from a percentile bootstrap that scored every single-class
resample as exactly 0.5, because that is what `auroc` returns when a class is empty.
At n = 16 to 18 with a minority class as small as one, that is not a rare event:
with 1 correct answer out of 16 it is (15/16)^16 = 0.356 of all draws.

So the reported bounds were computed over a distribution containing a spike of
probability mass at the exact value the conclusion turns on. This rerun measures how
much of each bound was the spike.

**The prediction under test.** If the 0.5 spike sits at the bottom of the bootstrap
distribution, then any contamination above 2.5% puts the 2.5th percentile inside the
spike and pins the lower bound to exactly 0.5, independent of the data. That premise
holds only when under 2.5% of the usable draws themselves fall below 0.5, which a
detector well above chance satisfies and a detector with a two-item minority class
does not. Both quantities are printed — the `pinned` column tests the conclusion, the
`<0.5` column tests the premise — so the prediction can fail visibly rather than
being assumed.

**Provenance caveat.** The discard counts are draws from `np.random.default_rng`,
whose stream NEP 19 explicitly declines to freeze. A numpy upgrade can move every
count in the `discarded` column with no code change, so these are measurements of one
environment, not constants.

**This is a re-measurement, not a simulation.** It reruns the same forward passes as
`detector_at_context` — same model, same relations, same templates, same seed — and
recomputes the intervals from the resulting score and label vectors two ways: the
pre-fix loop that scores degenerate resamples 0.5, and the current one that discards
them. Only the interval arithmetic differs between the columns. CPU is sufficient.

    python -m caustic.experiments.ci_recompute
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from caustic.detect import auroc, auroc_ci_detail
from caustic.experiments.detector_at_context import HARD, MODEL
from caustic.repair import NEUTRAL_PREFIX

SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_BOOT = 4000


def legacy_auroc_ci(scores, labels, n_boot=2000, seed=0):
    """The `auroc_ci` body exactly as it stood at db689cb, kept for the comparison.

    Same generator, same `rng.integers(0, n, n)` draw order, so a column-to-column
    difference can only come from how a single-class draw is handled. The loop body
    is duplicated verbatim in `tests/test_auroc_ci_degenerate.py`, so that the two
    transcriptions of a frozen reference stay a literal text diff apart; the test
    module must not import this one, which pulls in torch and transformers. The one
    divergence is that this copy also hands back the raw draws, which the pinning
    diagnostic needs and which no arithmetic here depends on.
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n = len(scores)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = auroc(scores[idx], labels[idx])
    return (
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
        boots,
    )


def main() -> None:
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def top1(text: str) -> int:
        ids = tok(text, return_tensors="pt").input_ids.to(DEV)
        with torch.no_grad():
            return int(model(ids).logits[0, -1].argmax())

    # Rebuild the score and label vectors of detector_at_context, unchanged.
    cases: list[tuple[str, np.ndarray, np.ndarray]] = []
    pooled: dict[str, tuple[list[float], list[float]]] = {"short": ([], []), "full": ([], [])}
    for rel, (tpls, pairs) in HARD.items():
        for label, prefix in (("short", ""), ("full", NEUTRAL_PREFIX)):
            ents = list(pairs)
            table = {e: [top1(prefix + t.format(e=e)) for t in tpls] for e in ents}
            gold = {e: tok(" " + pairs[e], add_special_tokens=False).input_ids[0] for e in ents}
            correct = [int(table[e][0] == gold[e]) for e in ents]
            collide = [
                float(np.mean([
                    np.mean([table[o][i] == table[e][i] for o in ents if o != e])
                    for i in range(len(tpls))
                ]))
                for e in ents
            ]
            w = [c for c, ok in zip(collide, correct) if not ok]
            r = [c for c, ok in zip(collide, correct) if ok]
            if not w or not r:
                continue
            pooled[label][0].extend(w)
            pooled[label][1].extend(r)
            cases.append(
                (f"{rel}/{label}", np.array(w + r), np.array([1] * len(w) + [0] * len(r)))
            )
    for label in ("short", "full"):
        w, r = pooled[label]
        if w and r:
            cases.append(
                (f"POOLED/{label}", np.array(w + r), np.array([1] * len(w) + [0] * len(r)))
            )

    print(f"model={MODEL}  device={DEV}  seed={SEED}  n_boot={N_BOOT}")
    print("re-measurement: same forward passes as detector_at_context, two interval rules")
    print("old = single-class resamples scored 0.5   new = single-class resamples discarded\n")

    hdr = (
        f"{'case':<22} {'n':>3} {'w/c':>7} {'AUROC':>7} "
        f"{'old 95%':>16} {'new 95%':>16} {'discarded':>11} {'d lo':>7} {'d hi':>7} "
        f"{'pinned':>7} {'<0.5':>6} {'0.5?':>5}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, s, y in cases:
        old_lo, old_hi, old_boots = legacy_auroc_ci(s, y, n_boot=N_BOOT, seed=SEED)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            new = auroc_ci_detail(s, y, n_boot=N_BOOT, seed=SEED)
        n_w, n_c = int(y.sum()), int((1 - y).sum())
        # An undefined interval is not evidence that the interval contains 0.5;
        # the nan comparisons would both be False and score it "yes" silently.
        if np.isnan(new.lo) or np.isnan(new.hi):
            spans = "n/a"
        else:
            spans = "yes" if not (new.lo > 0.5 or new.hi < 0.5) else "NO"
        # A lower bound sitting exactly on 0.5 is the guard's value, not a
        # measurement. It happens when under 2.5% of the *usable* draws fall below
        # 0.5, so the 2.5th percentile lands inside the spike the guard created.
        pinned = "PINNED" if abs(old_lo - 0.5) < 1e-12 else "-"
        below = float(np.mean(old_boots < 0.5))
        print(
            f"{name:<22} {len(s):>3} {f'{n_w}/{n_c}':>7} {new.point:>7.4f} "
            f"{f'[{old_lo:.4f}, {old_hi:.4f}]':>16} {f'[{new.lo:.4f}, {new.hi:.4f}]':>16} "
            f"{f'{new.n_discarded}/{N_BOOT}':>11} "
            f"{new.lo - old_lo:>+7.4f} {new.hi - old_hi:>+7.4f} {pinned:>7} {below:>6.3f} {spans:>5}"
        )

    print()
    print("reading")
    print("-------")
    print("pinned = the old lower bound equals 0.5 to floating tolerance, which is the")
    print("guard's value rather than a measurement. It requires the <0.5 column to be")
    print("under 0.025: contamination alone does not pin a bound if the usable draws")
    print("already reach below 0.5, which is what a small minority class produces.")
    print("The last column is the only one that bears on RESULTS.md Section 11.")
    print("If the corrected intervals still span 0.5, the narrowing at db689cb stands on")
    print("firmer ground: it was not the empty-class guard that put chance inside the")
    print("interval. If any pooled interval no longer spans 0.5, the stated reason was")
    print("partly an artefact and the sentence 'Both pooled intervals include 0.5' has to")
    print("be corrected -- though a corrected interval that excludes 0.5 at these sample")
    print("sizes still rests on a minority class of a few items and does not by itself")
    print("restore the detector on hard relations.")
    print("A large discard count with a near-zero delta means the guard was harmless here,")
    print("not that it is harmless in general.")


if __name__ == "__main__":
    main()
