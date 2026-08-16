"""Does the orbit bound's precondition actually hold on the relations it was run on?

Theorem 1 needs the true answer map to be injective. Every experiment here reads
the model's answer as an argmax TOKEN ID and scores it against the first token of
the gold string, so the map the bound is applied to is `first_token . gold`. That
composition is where injectivity can be lost: two different capitals can begin
with the same token, and then no model, however good, can place those entities in
different orbits. Each such pair forces an orbit of two and the bound counts one
certified error that may not be an error at all.

`RelationSpec.injective` is a boolean the caller supplies and nothing checks. This
audit checks it, for every relation dict the repository actually probes, using
only the tokenizer — no model weights, no forward passes.

**What each outcome means.** A relation with 0 collisions had its precondition
hold under this tokenizer, and its published certified-error counts stand. A
relation declared injective with collisions > 0 had the precondition fail, and up
to one certified error per colliding pair is spurious. A relation declared
non-injective is listed for completeness; the bound returns 0 there regardless.

**What this does not show.** It does not show that the relation is injective as a
relation, nor that any particular reported error was spurious — a collision is an
opportunity for a false certified error, realised only when the model puts both
members of the pair on the shared token. It is tokenizer-specific: a different
tokenizer moves the whole table.

    python -m caustic.experiments.injectivity_audit
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from caustic.experiments.coupling_gap import RELATIONS
from caustic.experiments.detector_at_context import HARD
from caustic.regime import RelationSpec, verify_injective

MODEL = "Qwen/Qwen2.5-0.5B"
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)

    def first_token(answer: str) -> int:
        return int(tok(" " + answer, add_special_tokens=False).input_ids[0])

    # (name, first template, entity -> gold answer), in the order the repo defines them.
    relations = [(name, tpl, facts) for name, (tpl, facts) in RELATIONS.items()]
    relations += [(name, tpls[0], facts) for name, (tpls, facts) in HARD.items()]

    print(f"model={MODEL} seed={SEED} relations={len(relations)}")
    print("tokenizer only; no forward pass runs, so the table is a property of the")
    print("tokenizer and the relation dicts, not of the weights or the device.")
    print("injectivity is checked on first_token(gold), the map Theorem 1 is applied to\n")

    hdr = (
        f"{'relation':<16} {'n':>3} {'distinct':>9} {'declared':>9} "
        f"{'holds':>6} {'ceiling':>8}  collisions"
    )
    print(hdr)
    print("-" * (len(hdr) + 22))

    failures = 0
    for name, tpl, facts in relations:
        # Declared injectivity is read off the data, not a hardcoded name list: a
        # relation whose gold STRINGS repeat is many-to-one by construction, the
        # shape `continent` has. Anything else claims injectivity.
        declared = len(set(facts.values())) == len(facts)
        spec = RelationSpec((tpl,), tuple(facts), injective=declared)
        pairs = verify_injective(spec, facts, first_token)
        ids = [first_token(a) for a in facts.values()]
        n_distinct = len(set(ids))
        # The bound over-counts by at most (group size - 1) per collided token,
        # which is n - n_distinct, NOT the number of pairs: three entities on one
        # token give 3 pairs but can inflate `n - m` by only 2.
        ceiling = len(facts) - n_distinct
        holds = "yes" if not pairs else "no"
        if declared and pairs:
            failures += 1
            holds = "NO"
        shown = "; ".join(f"{a}/{b}" for a, b in pairs[:3])
        if len(pairs) > 3:
            shown += f"; +{len(pairs) - 3} more"
        print(
            f"{name:<16} {len(facts):>3} {n_distinct:>9} "
            f"{'injective' if declared else 'many-one':>9} {holds:>6} {ceiling:>8}  "
            f"{shown or '-'}"
        )

    print()
    print("reading")
    print("-------")
    print("holds=yes with declared=injective: the precondition is satisfied and every")
    print("certified error on that relation is a real error. holds=NO with")
    print("declared=injective: the precondition FAILS, and `ceiling` bounds how many")
    print("of that relation's certified errors could be artefacts of the tokenisation")
    print("rather than model mistakes -- an upper bound, realised only where the model")
    print("actually puts a collided pair on one answer. declared=many-one is not a")
    print("failure; the bound already returns 0 there.")
    print()
    print(f"relations declared injective whose precondition fails: {failures}")


if __name__ == "__main__":
    main()
