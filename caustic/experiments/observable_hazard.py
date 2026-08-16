"""Which answer shapes make the first-token observable degenerate.

Every certificate here compares the top-1 token id, so the precondition is
injectivity of the gold answers *at that encoding*. Word answers behave: under
all three tokenizers tested, ` Paris` and ` Tokyo` are single distinct tokens.
Numeric answers do not, and the failure is total rather than marginal.

    tokenizer                   relation          n  distinct 1st tokens
    distilgpt2                  atomic_number    20                   20
    distilgpt2                  planet_order      8                    8
    SmolLM2-135M                atomic_number    20                    1
    SmolLM2-135M                planet_order      8                    1
    Qwen2.5-0.5B                atomic_number    20                    1
    Qwen2.5-0.5B                planet_order      8                    1

The cause is visible in the ids:

    distilgpt2     ' 1'  -> [352]           [' 1']
                   ' 20' -> [1160]          [' 20']
    Qwen2.5-0.5B   ' 1'  -> [220, 16]       [' ', '1']
                   ' 20' -> [220, 17, 15]   [' ', '2', '0']
                   ' Paris' -> [12095]      [' Paris']

Qwen- and Llama-family tokenizers emit the space as its own token before a
digit, so every numeric answer shares first token 220. Twenty distinct numbers
collapse to one observable value.

**What that does to the certificate.** On a numeric relation under such a
tokenizer, `m = 1` regardless of the model, so Theorem 1 certifies `n - m = 19`
wrong answers on a model that answered all twenty correctly. That is a false
certification of 19 errors — the worst failure available to a one-sided bound,
and silent, because the partition simply looks like total collapse.

`verify_injective` catches it: every gold maps to token 220, so `|G| = 1 < n`
and it raises rather than certifying. This module exists so the hazard is stated
rather than left to be rediscovered, and so the rule is usable in advance:

    safe      answers whose surface form the tokenizer keeps whole — ordinary
              words, most proper nouns
    hazardous answers the tokenizer splits from their leading space — digits,
              symbols, punctuation, and anything short and non-alphabetic

The rule is per-tokenizer, not per-relation: the same numeric relation is
perfectly injective under distilgpt2 and degenerate under Qwen2.5-0.5B.

    python -m caustic.experiments.observable_hazard
"""

from __future__ import annotations

import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TOKENIZERS = ("distilgpt2", "HuggingFaceTB/SmolLM2-135M", "Qwen/Qwen2.5-0.5B")

ELEMENTS = [
    "hydrogen", "helium", "lithium", "beryllium", "boron", "carbon", "nitrogen",
    "oxygen", "fluorine", "neon", "sodium", "magnesium", "aluminium", "silicon",
    "phosphorus", "sulfur", "chlorine", "argon", "potassium", "calcium",
]
PLANETS = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
CAPITALS = {
    "France": "Paris", "Japan": "Tokyo", "Peru": "Lima", "Egypt": "Cairo",
    "Norway": "Oslo", "Cuba": "Havana", "Ghana": "Accra", "Nepal": "Kathmandu",
}

CANDIDATES = {
    "atomic_number": {e: str(i + 1) for i, e in enumerate(ELEMENTS)},
    "planet_order": {p: str(i + 1) for i, p in enumerate(PLANETS)},
    "capital_word": CAPITALS,
}


def main() -> None:
    print(f"tokenizers={len(TOKENIZERS)}")
    print("a relation is usable only where distinct 1st tokens equals n.\n")
    print(f"{'tokenizer':<28}{'relation':<16}{'n':>3}{'1st tok':>9}{'single':>8}  usable?")
    print("-" * 74)

    for name in TOKENIZERS:
        tok = AutoTokenizer.from_pretrained(name)
        for rel, facts in CANDIDATES.items():
            answers = list(facts.values())
            ids = [tok(" " + a, add_special_tokens=False).input_ids for a in answers]
            distinct_first = len({i[0] for i in ids})
            single = sum(1 for i in ids if len(i) == 1)
            usable = distinct_first == len(answers)
            print(
                f"{name:<28}{rel:<16}{len(answers):>3}{distinct_first:>9}{single:>8}"
                f"  {'yes' if usable else 'NO - certificate would be false'}"
            )

    print("\nhow the split looks, on the two tokenizers that disagree:")
    for name in ("distilgpt2", "Qwen/Qwen2.5-0.5B"):
        tok = AutoTokenizer.from_pretrained(name)
        print(f"  {name}")
        for a in ("1", "20", "Paris"):
            ids = tok(" " + a, add_special_tokens=False).input_ids
            print(f"    {' ' + a!r:9} -> {ids}  {[tok.decode([i]) for i in ids]!r}")

    print("\nreading:")
    print("  a relation with `distinct 1st tokens` below n cannot be certified under")
    print("  that tokenizer. On the numeric relations m = 1 regardless of the model,")
    print("  so Theorem 1 would certify n - 1 errors on a model answering perfectly.")
    print("  verify_injective raises on exactly this, and is the only thing between a")
    print("  caller and that false certification.")
    print("  the rule is per-tokenizer, not per-relation: the same numeric relation is")
    print("  injective under distilgpt2 and degenerate under Qwen2.5-0.5B.")


if __name__ == "__main__":
    main()
