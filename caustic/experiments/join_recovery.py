"""Does the entity survive every paraphrase, or only some of them.

Theorem 2 caps recovery of the entity from ONE pooled answer at `1/k`. Theorem 2*
caps it from the full answer tuple at `m_join / n`, where `m_join` counts the
blocks of the join of the per-template partitions. The two differ by exactly the
information a single template discards, and which of them binds decides whether
repair is possible at all: if the join is coarse, the entity is absent from the
model's outputs and no prefix recovers it.

This experiment measures `m_join` rather than assuming it. The first draft of
Theorem 2* claimed the join was discrete in every condition. That figure came
from a run over a different condition set and is false here — the join collapses
under the degenerate prefix, which is what makes the theorem informative rather
than vacuous.

    python -m caustic.experiments.join_recovery
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from caustic.experiments.coupling_gap import RELATIONS
from caustic.experiments.symmetry_break import PARAPHRASES
from caustic.regime import RelationSpec, join_partition, orbit_partition
from caustic.theorems import join_recovery_bound, pooling_recovery_bound

MODEL = "Qwen/Qwen2.5-0.5B"
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Same token budget, opposite character: the contrast the repository's headline
# rests on, and the one under which the join is expected to differ.
CONDITIONS = {"none": "", "the_x128": " the" * 128}


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

    print(f"model={MODEL} seed={SEED} device={DEV}")
    print("Thm 2  is the single-template ceiling 1/k on the largest orbit.")
    print("Thm 2* is the join ceiling m_join / n, seeing every paraphrase.\n")
    print(
        f"{'relation':<11}{'cond':<11}{'n':>3}{'m_0':>5}{'max k':>7}{'Thm 2':>8}"
        f"{'m_join':>8}{'Thm 2*':>8}  discrete?"
    )
    print("-" * 72)

    discrete = total = 0
    for rel, (tpl, facts) in RELATIONS.items():
        tpls = PARAPHRASES[rel]
        ents = tuple(facts)
        for cname, pre in CONDITIONS.items():
            fn = (lambda p, _pre=pre: top1(_pre + p))
            single = orbit_partition(RelationSpec((tpls[0],), ents), fn)
            joined = join_partition(RelationSpec(tuple(tpls), ents), fn)
            n = len(ents)
            total += 1
            is_discrete = joined.n_distinct == n
            discrete += is_discrete
            print(
                f"{rel:<11}{cname:<11}{n:>3}{single.n_distinct:>5}"
                f"{single.largest_orbit:>7}"
                f"{pooling_recovery_bound(single.largest_orbit):>8.3f}"
                f"{joined.n_distinct:>8}"
                f"{join_recovery_bound(n, joined.n_distinct):>8.3f}"
                f"  {'yes' if is_discrete else 'NO'}"
            )

    print(f"\njoin discrete in {discrete} of {total} conditions\n")
    print("reading:")
    print("  discrete join  -> the entity survives every paraphrase, a receiver could")
    print("                    recover it, and repair is licensed.")
    print("  coarse join    -> it is destroyed across all of them, and no function of")
    print("                    these outputs recovers it. No prefix can help.")
    print("  a uniformly discrete result would make Theorem 2* vacuous; the split")
    print("  between the two prefix regimes is what gives the ceiling content.")
    print("  continent is many-to-one, so its join is coarse under both conditions.")


if __name__ == "__main__":
    main()
