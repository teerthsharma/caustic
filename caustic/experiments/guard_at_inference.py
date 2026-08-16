"""Does the guard earn its forward passes, measured against the cheap baselines.

Theorem 6 bounds the precision of the withheld set from the partition alone. That
is arithmetic until someone checks it against a model, and the check has never
been run: every number in `caustic/guard.py` comes from an injected dictionary.

Two questions, and the second is the one that decides whether any of this ships.

**Is the floor honest?** Realised precision of `abstain` against the proved
`(n - m) / |S|`. A single condition where realised precision falls below the
floor refutes Theorem 6, so this is a falsification run, not a demonstration.

**Does it beat one forward pass?** Abstention is a risk-coverage trade: withhold
some answers, keep the rest, and the kept set should be more accurate than the
whole. Every baseline in `caustic/detect.py` can do that too — `max_softmax`,
`logit_margin` and `predictive_entropy` each cost ONE forward pass and produce a
score that can be thresholded to abstain on exactly as many items. The guard
costs `(c + 1) * n`. So the comparison is at MATCHED COVERAGE: given the same
number of withheld items, whose kept set is more accurate?

That comparison is the one this repository has never run, and it is the one a
reader should demand. A prior gate in this line of work scored 0.846 mean AUROC
against Mahalanobis at 0.899, both cheaper; the lesson was that a geometric score
which does not beat a cheap baseline on both quality and cost is a negative
result. The same standard applies here, and this experiment is written so it can
return one.

    python -m caustic.experiments.guard_at_inference
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from caustic.detect import logit_margin, max_softmax_score, predictive_entropy
from caustic.guard import guard
from caustic.regime import RelationSpec
from caustic.experiments.coupling_gap import RELATIONS

MODEL = "Qwen/Qwen2.5-0.5B"
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# The prefix pair the repository's headline rests on: same token budget, opposite
# character. Collapse is what the certificate is for, so it must be measured in
# both regimes rather than only the healthy one.
PROSE = (
    "Mechanical calculators were built from gears and levers. Tide predictors "
    "used pulleys and cams to sum harmonic terms. Slide rules encoded "
    "multiplication as addition of lengths. Nomograms let an engineer read off "
    "a result by laying a straightedge across three scales. "
)
CANDIDATES = {"prose": PROSE}


def _coverage_accuracy(scores: np.ndarray, correct: np.ndarray, n_abstain: int) -> float:
    """Accuracy of the kept set when the `n_abstain` highest scores are withheld.

    Ties are broken by index so the comparison is deterministic. Returns nan when
    nothing is kept, which is a real outcome and must not be silently averaged.
    """
    if n_abstain >= len(scores):
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    kept = order[n_abstain:]
    return float(np.mean(correct[kept]))


def main() -> None:
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def logits_of(text: str) -> np.ndarray:
        ids = tok(text, return_tensors="pt").input_ids.to(DEV)
        with torch.no_grad():
            return model(ids).logits[0, -1].float().cpu().numpy()

    def top1(text: str) -> int:
        return int(logits_of(text).argmax())

    def gold_tok(answer: str) -> int:
        return tok(" " + answer, add_special_tokens=False).input_ids[0]

    conditions = {"none": "", "the_x128": " the" * 128}

    print(f"model={MODEL} seed={SEED} device={DEV} candidates={len(CANDIDATES) + 1}")
    print("floor is Theorem 6: (n - m) / |S|, computed with no ground truth.")
    print("realised is the fraction of withheld entities that were actually wrong.")
    print("a realised value BELOW the floor refutes Theorem 6.\n")
    print(
        f"{'relation':<10}{'cond':<9}{'n':>3}{'|S|':>5}{'floor':>7}{'realised':>10}"
        f"{'acc_all':>9}{'acc_kept':>10}  verdict"
    )
    print("-" * 78)

    rows = []
    for rel, (tpl, facts) in RELATIONS.items():
        ents = tuple(facts)
        gold = {e: gold_tok(facts[e]) for e in ents}
        if len(set(gold.values())) < len(ents):
            print(f"{rel:<10}{'-':<9}  skipped: gold not injective at the token level")
            continue
        spec = RelationSpec((tpl,), ents)
        for cname, pre in conditions.items():
            fn = (lambda p, _pre=pre: top1(_pre + p))
            rep = guard(spec, fn, candidates=CANDIDATES)
            answered = dict(zip(rep.entities, rep.answers))
            correct = np.array([answered[e] == gold[e] for e in ents])

            withheld = set(rep.abstain)
            realised = (
                float(np.mean([not correct[i] for i, e in enumerate(ents) if e in withheld]))
                if withheld
                else float("nan")
            )
            floor = rep.certified_precision
            acc_all = float(correct.mean())
            kept_idx = [i for i, e in enumerate(ents) if e not in withheld]
            acc_kept = float(np.mean(correct[kept_idx])) if kept_idx else float("nan")

            if floor is None:
                verdict = "silent"
            elif realised + 1e-12 < floor:
                verdict = "REFUTES THM 6"
            else:
                verdict = "holds"
            print(
                f"{rel:<10}{cname:<9}{len(ents):>3}{len(withheld):>5}"
                f"{(floor if floor is not None else float('nan')):>7.3f}{realised:>10.3f}"
                f"{acc_all:>9.3f}{acc_kept:>10.3f}  {verdict}"
            )
            rows.append((rel, cname, ents, gold, pre, correct, len(withheld), acc_kept))

    # --- the comparison that decides whether the guard is worth its cost -----
    print(f"\n{'':-<78}")
    print("matched coverage: withhold the same number of items by each score.")
    print("baselines cost ONE forward pass; the guard costs (c + 1) * n.\n")
    print(
        f"{'relation':<10}{'cond':<9}{'k':>3}{'guard':>8}{'msp':>8}{'margin':>8}"
        f"{'entropy':>9}  guard beats cheapest?"
    )
    print("-" * 78)

    wins = losses = ties = 0
    for rel, cname, ents, gold, pre, correct, k, acc_kept in rows:
        if k == 0 or k >= len(ents):
            continue
        L = np.stack([logits_of(pre + RELATIONS[rel][0].format(e=e)) for e in ents])
        base = {
            "msp": max_softmax_score(L),
            "margin": logit_margin(L),
            "entropy": predictive_entropy(L),
        }
        accs = {nm: _coverage_accuracy(s, correct, k) for nm, s in base.items()}
        best_base = max(v for v in accs.values() if not np.isnan(v))
        if acc_kept > best_base + 1e-12:
            mark, wins = "yes", wins + 1
        elif acc_kept < best_base - 1e-12:
            mark, losses = "no", losses + 1
        else:
            mark, ties = "tie", ties + 1
        print(
            f"{rel:<10}{cname:<9}{k:>3}{acc_kept:>8.3f}{accs['msp']:>8.3f}"
            f"{accs['margin']:>8.3f}{accs['entropy']:>9.3f}  {mark}"
        )

    print(f"\nguard wins {wins}, loses {losses}, ties {ties} at matched coverage.\n")
    print("reading:")
    print("  any REFUTES THM 6 row  -> the precision floor is wrong and the theorem falls.")
    print("  guard loses on most rows -> the certificate does not earn (c+1)*n passes;")
    print("                              a one-pass baseline abstains better and the")
    print("                              honest conclusion is to ship the baseline.")
    print("  guard wins on collapse rows only -> it is a collapse detector, which is")
    print("                              what RESULTS.md section 11 already concluded,")
    print("                              and it should be scoped to that regime.")
    print("  'silent' rows are m = n, where Theorem 1 certifies nothing and the guard")
    print("  withholds nobody. Silence is the correct output there, not a failure.")


if __name__ == "__main__":
    main()
