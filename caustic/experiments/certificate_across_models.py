"""Do the certificate bounds hold on models other than the one they were found on.

Every AUROC in this repository sits on 12 to 20 entities, where a 95% bootstrap
interval has power 0.00-0.38 to separate a true 0.70 from 0.50. Sweeping more
models does not repair that: an underpowered comparison repeated five times is
five underpowered comparisons.

Theorems 1*, 6 and 6* are different in kind. They are deterministic inequalities
over a finite set, with no sampling distribution and no minority class, so they
either hold on a run or they do not. One violation refutes the theorem. Zero
violations across models is evidence of the sort the AUROC work cannot produce
at these sizes, and it costs one forward pass per entity per model.

Checked per (model, relation, condition):

    Theorem 1    errors >= n - m
    Theorem 1*   errors >= n - |f(E) n G|
    Theorem 6    realised precision of S >= (n - m) / |S|
    Theorem 6*   realised precision of S >= (|S| - b_adm) / |S|

Relations whose gold answers are not injective at the tokenizer's encoding are
skipped rather than scored, since that is the hypothesis of every bound above
and it is per-tokenizer: a relation injective for one model may collide for
another.

    python -m caustic.experiments.certificate_across_models
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from caustic.experiments.coupling_gap import RELATIONS
from caustic.regime import RelationSpec, orbit_partition
from caustic.theorems import (
    admissible_error_bound,
    admissible_precision_bound,
    certified_precision_bound,
    orbit_error_bound,
)

MODELS = ("distilgpt2", "HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B")
SEED = 0
DEV = "cpu"
CONDITIONS = {"none": "", "the_x128": " the" * 128}


def main() -> None:
    torch.manual_seed(SEED)
    print(f"models={len(MODELS)} seed={SEED} device={DEV}")
    print("a single `VIOLATION` row refutes the theorem named in that column.\n")
    print(
        f"{'model':<28}{'relation':<10}{'cond':<10}{'n':>3}{'err':>5}"
        f"{'T1':>5}{'T1*':>5}{'|S|':>5}{'prec':>7}{'T6':>7}{'T6*':>7}  verdict"
    )
    print("-" * 104)

    checks = violations = skipped = 0
    for name in MODELS:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(DEV).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        def top1(text: str) -> int:
            ids = tok(text, return_tensors="pt").input_ids.to(DEV)
            with torch.no_grad():
                return int(model(ids).logits[0, -1].argmax())

        for rel, (tpl, facts) in RELATIONS.items():
            ents = tuple(facts)
            gold = {e: tok(" " + facts[e], add_special_tokens=False).input_ids[0] for e in ents}
            G = set(gold.values())
            if len(G) < len(ents):
                skipped += 1
                print(f"{name:<28}{rel:<10}{'-':<10}  skipped: gold not injective under this tokenizer")
                continue
            spec = RelationSpec((tpl,), ents)
            for cname, pre in CONDITIONS.items():
                rep = orbit_partition(spec, lambda p, _pre=pre: top1(_pre + p))
                n = len(ents)
                errs = sum(1 for e, a in zip(rep.entities, rep.answers) if a != gold[e])
                t1 = orbit_error_bound(n, rep.n_distinct)
                t1s = admissible_error_bound(n, len(set(rep.answers) & G))

                flagged = [
                    e for e, a in zip(rep.entities, rep.answers) if len(rep.orbits[a]) > 1
                ]
                if flagged:
                    wrong_in_S = sum(1 for e in flagged if rep.answers[list(rep.entities).index(e)] != gold[e])
                    prec = wrong_in_S / len(flagged)
                    b_adm = sum(1 for a, v in rep.orbits.items() if len(v) > 1 and a in G)
                    t6 = certified_precision_bound(n, rep.n_distinct, len(flagged))
                    t6s = admissible_precision_bound(len(flagged), b_adm)
                else:
                    prec = t6 = t6s = float("nan")

                # bool(): an empty `flagged` makes `flagged and (...)` evaluate to
                # the empty list, not False, and the counter then rejects it.
                bad = bool(
                    errs < t1
                    or errs < t1s
                    or (flagged and (prec < t6 - 1e-12 or prec < t6s - 1e-12))
                )
                checks += 2 + (2 if flagged else 0)
                violations += bad
                print(
                    f"{name:<28}{rel:<10}{cname:<10}{n:>3}{errs:>5}{t1:>5}{t1s:>5}"
                    f"{len(flagged):>5}{prec:>7.3f}{t6:>7.3f}{t6s:>7.3f}"
                    f"  {'VIOLATION' if bad else 'holds'}"
                )
        del model

    print(f"\n{checks} bound checks, {violations} violations, {skipped} relations skipped\n")
    print("reading:")
    print("  0 violations -> the bounds are deterministic inequalities and survived every")
    print("     run; unlike an AUROC this needs no power, so the evidence is real at n=12.")
    print("  any violation -> the named theorem is false as stated, or the relation was")
    print("     not injective at that tokenizer's encoding and should have been skipped.")
    print("  skipped rows are the precondition doing its job: injectivity is per-tokenizer,")
    print("     so a relation safe for one model can collide for another.")


if __name__ == "__main__":
    main()
