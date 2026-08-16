"""Is the `element_symbol` repair generic coherence, or is it domain priming?

`RESULTS.md` reports `element_symbol` moving from accuracy 0.062 to 1.000 under
`NEUTRAL_PREFIX`, and describes that prefix as "neutral prose containing no
chemistry". It is not chemistry-free. Its final sentence reads:

    Photosynthesis converts light energy into chemical energy stored in sugars,
    taking in carbon dioxide and releasing oxygen as a by-product of the reaction

That is `chemical`, `energy`, `carbon dioxide`, `oxygen`, `reaction`, `sugars`,
and `matter` earlier in the passage — and the relation is probed with the
template "The chemical symbol for {e} is", which shares the word `chemical` with
the prefix. `repair.py`'s own docstring for `NEUTRAL_PREFIX` warns that a caller
whose relation concerns photosynthesis must supply their own prefix. Nobody
checked whether `element_symbol` is such a caller.

Two hypotheses make different predictions, and only a control relation separates
them:

    generic coherence   any coherent, off-topic prose of the same length repairs
                        `element_symbol`, and stripping the chemistry vocabulary
                        costs nothing
    domain priming      the chemistry vocabulary is what repairs it, so stripping
                        hurts and enriching helps — and neither does anything
                        extra for a relation that is not about chemistry

Four prefixes, matched on token count under the model's own tokenizer:

    neutral     `NEUTRAL_PREFIX` verbatim, the published baseline
    stripped    the same passage with only the chemistry removed: the last
                sentence swapped for one about courtroom procedure, and
                "arrangements of matter" for "arrangements of gears and levers".
                Sentences two and three are byte-identical to the original, so
                the contrast is closer to isolating the chemistry than to a
                rewrite, though sentences one and four differ in wording as well
                as in vocabulary and the contrast is not clean.
    enriched    deliberately chemistry-laden: reactions, bonds, atoms, the
                periodic table, electron shells, valence. Contains none of the
                sixteen element names and none of their symbols.
    the_x_k     `" the"` repeated to the same token count, the repository's
                established negative control, documented in `repair.py` to drive
                the largest orbit from 4 to 20.

`element_symbol` is the relation under test; `small_capital` is the control. Both
come from `caustic.experiments.detector_at_context` unmodified. The control is
what separates "the enriched passage is simply better prose" from "the enriched
passage primes this one topic": a prefix effect that shows up on both relations
is not priming.

**What a result here does not show.** One model, one seed, one prefix per
condition, top-1 token comparison — and that last is lenient here, because four
of the sixteen symbols are two tokens under Qwen's BPE (` Sb`, ` Hg`, ` Zr`,
` Rb`), so their gold reduces to a first token of ` S`, ` H`, ` Z`, ` R`. Those
four remain distinct from each other and from the other twelve, so the relation
stays injective and the Orbit Error Bound holds, but the accuracy column would
score ` Sulfur` as correct for antimony. The measurement is kept exactly as
`detector_at_context` makes it so the numbers are comparable to the published
ones rather than better than them. A stripped prefix that repairs as well as the
original refutes the confound for *these* sixteen entities on *this* model; it
does not establish that `NEUTRAL_PREFIX` is neutral for an arbitrary relation,
which is a claim no single passage can support. Nor does the token-count match
control for anything but length — the four passages differ in perplexity,
sentence count and subject matter, and no attempt is made to equalise those.

All four prefixes were written in full before the model was loaded, and were
never edited afterwards; the only post-hoc change permitted to them was length,
measured with the tokenizer alone and never with a forward pass.

    python -m caustic.experiments.prefix_confound
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from caustic.regime import RelationSpec
from caustic.repair import NEUTRAL_PREFIX, RepairReport, sweep_prefixes

MODEL = "Qwen/Qwen2.5-0.5B"
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"

CHEMISTRY_STRIPPED = (
    "Mechanical calculators, tide predictors, and looms that read punched cards all encoded "
    "procedures into physical arrangements of gears and levers. Ocean currents move heat around "
    "the planet on timescales that dwarf weather, and the resulting redistribution sets the "
    "climate of entire continents. A language is a system of conventions that lets one mind "
    "reconstruct part of the state of another from a sequence of symbols. A trial settles a "
    "question of fact under rules that constrain what may be said, by whom, and in what order, "
    "so that the record survives later review. "
)

CHEMISTRY_ENRICHED = (
    "A reaction rearranges the bonds between atoms while conserving every element that entered "
    "it, so the mass on each side of the equation agrees. Chemists write each element as a one- "
    "or two-letter symbol, and arrange those symbols into a periodic table ordered by the "
    "structure of the electron shells, so that valence, oxidation state and reactivity can be "
    "read off the column an element sits in. Photosynthesis converts light energy into chemical "
    "energy stored in sugars, taking in carbon dioxide and releasing oxygen as a by-product of "
    "the reaction. "
)

RELATIONS = ("element_symbol", "small_capital")


def rows(name: str, rep: RepairReport) -> tuple[str, str]:
    """Format the before and after halves of one report as table rows.

    The certified error rate is recomputed here from `n` and `m` rather than
    taken from an `OrbitReport`, because `RepairReport` keeps the counts and not
    the partitions. It is `(n - m) / n`, the Orbit Error Bound of `regime.py`,
    valid only because both relations here are injective.
    """
    n = rep.n_entities

    def row(tag: str, m: int, largest: int, acc: float | None) -> str:
        return (
            f"  {tag:<12} {n:>3} {m:>4} {largest:>8} "
            f"{(n - m) / n:>10.3f} {acc if acc is not None else float('nan'):>9.3f}"
        )

    return (
        row("(no prefix)", rep.before_n_distinct, rep.before_largest_orbit, rep.before_accuracy),
        row(name, rep.after_n_distinct, rep.after_largest_orbit, rep.after_accuracy),
    )


def main() -> None:
    # Imported here rather than at module scope so that `rows` stays testable
    # without `transformers`, which lives in the optional `experiments` extra.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from caustic.experiments.detector_at_context import HARD

    torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def top1(text: str) -> int:
        ids = tok(text, return_tensors="pt").input_ids.to(DEV)
        with torch.no_grad():
            return int(model(ids).logits[0, -1].argmax())

    def ntok(s: str) -> int:
        return len(tok(s, add_special_tokens=False).input_ids)

    base = ntok(NEUTRAL_PREFIX)
    prefixes = {
        "neutral": NEUTRAL_PREFIX,
        "stripped": CHEMISTRY_STRIPPED,
        "enriched": CHEMISTRY_ENRICHED,
        # Trailing space so the control ends like the three prose passages do;
        # without it the last " the" glues to the prompt's first word and the
        # control differs from the others in prompt format, not only in content.
        "the_x_k": " the" * (base - 1) + " ",
    }
    lens = {k: ntok(v) for k, v in prefixes.items()}
    # Length is the one thing this experiment holds fixed. One token of slack,
    # because hand-written English does not land on an exact BPE count.
    assert all(abs(v - base) <= 1 for v in lens.values()), lens

    print(f"model={MODEL}  seed={SEED}  prefix_tokens={base} (neutral baseline)")
    print("  matched lengths: " + ", ".join(f"{k}={v}" for k, v in lens.items()))
    print("  relation under test: element_symbol;  control: small_capital (not chemistry)")
    print("  m = distinct answers;  certified = (n - m) / n, the Orbit Error Bound\n")

    hdr = f"  {'prefix':<12} {'n':>3} {'m':>4} {'largest':>8} {'certified':>10} {'acc':>9}"
    for rel in RELATIONS:
        tpls, pairs = HARD[rel]
        spec = RelationSpec(tuple(tpls), tuple(pairs))
        gold = {e: tok(" " + a, add_special_tokens=False).input_ids[0] for e, a in pairs.items()}
        reports = sweep_prefixes(spec, top1, prefixes, gold=gold)
        # sweep_prefixes recomputes the no-prefix baseline once per prefix. The
        # table prints it once, so check the four agree rather than let the
        # printing hide a nondeterministic answer_fn.
        befores = {(r.before_n_distinct, r.before_largest_orbit, r.before_accuracy)
                   for r in reports.values()}
        assert len(befores) == 1, befores

        print(rel)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        printed_before = False
        for name in prefixes:
            before, after = rows(name, reports[name])
            if not printed_before:
                print(before)
                printed_before = True
            print(after)
        print()

    print("reading")
    print("-------")
    print("Stated in advance, before the table above was produced:")
    print()
    print("If `stripped` repairs element_symbol about as well as `neutral`, the confound is")
    print("REFUTED: what the prefix supplies is generic coherence, and the published 0.062")
    print("-> 1.000 stands as reported.")
    print()
    print("If `stripped` repairs markedly WORSE and `enriched` markedly BETTER, while the")
    print("small_capital control is indifferent to which of the three coherent prefixes is")
    print("used, then the element_symbol repair is domain priming and RESULTS.md line 371 --")
    print("'16 under 108 tokens of neutral prose containing no chemistry' -- is wrong on its")
    print("face and must be corrected.")
    print()
    print("If BOTH relations move together across the three coherent prefixes, the passages")
    print("differ in general quality rather than in topic, and this experiment separates")
    print("nothing. That outcome is inconclusive, not a refutation.")
    print()
    print("`the_x_k` is the negative control and carries no verdict: it is expected to")
    print("collapse both relations, and a run in which it does not is a broken run.")


if __name__ == "__main__":
    main()
