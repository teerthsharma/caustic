"""Two different failures hide behind one precondition check.

Theorem 1 needs the true answer map injective *in the encoding `answer_fn`
compares*. Every experiment here compares the top-1 token id, so the check runs
on first tokens, and `verify_injective` skipped 4 of 12 relation-model pairs on
that basis. Those skips are not all the same thing.

    tokenizer                   relation          n  first-tok   string
    distilgpt2                  small_capital    18         16       18   fixed
    SmolLM2-135M                currency         12         11       12   fixed
    SmolLM2-135M                small_capital    18         17       18   fixed
    Qwen2.5-0.5B                small_capital    18         16       18   fixed
    every tokenizer             continent        12          6        6   genuine

`small_capital` and `currency` collide only because of tokenization: the gold
strings are distinct, so comparing decoded strings restores injectivity under
all three tokenizers. `continent` collides at 6 under every observable, because
Europe really is the correct answer for two of those countries. The first-token
observable was conflating "this relation is many-to-one", which is a scope
limit, with "this tokenizer cannot separate these answers", which is a bug.

What the string observable costs, measured on Qwen2.5-0.5B with four greedy
decode steps:

    relation        m_firsttok  m_string   accuracy      cost
    small_capital            8        12   unchanged  646 -> 1848 ms
    element_symbol           3         5   unchanged  447 -> 1544 ms

The string partition is FINER, so `n - m` is smaller and the certificate is
weaker — but it is valid where the first-token one was not. Validity is bought
with strength and roughly threefold cost. Accuracy is unchanged on both, which
is expected here rather than reassuring: both relations sit at 0.062 and 0.167,
where first-token grading has almost no room to flatter.

    python -m caustic.experiments.observable_choice
"""

from __future__ import annotations

import sys
from pathlib import Path

import time

import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformers import AutoModelForCausalLM, AutoTokenizer
from caustic.experiments.coupling_gap import RELATIONS
import caustic.experiments.detector_at_context as dc

print("=== part 1: injectivity of the gold set, first-token vs full-string ===")
print(f"{'tokenizer':<28}{'relation':<16}{'n':>3}{'first-tok':>11}{'string':>9}")
print("-"*70)
rels=dict(RELATIONS)
H=getattr(dc,'HARD',None) or {}
for k,v in H.items():
    rels[k]=(v[0] if isinstance(v[0],str) else v[0][0], v[1])
for name in ("distilgpt2","HuggingFaceTB/SmolLM2-135M","Qwen/Qwen2.5-0.5B"):
    tk=AutoTokenizer.from_pretrained(name)
    for rel,(tpl,facts) in rels.items():
        ans=list(facts.values()); n=len(ans)
        ft=len({tk(" "+a,add_special_tokens=False).input_ids[0] for a in ans})
        st=len(set(ans))
        print(f"{name:<28}{rel:<16}{n:>3}{ft:>11}{st:>9}"+("   <- collides" if ft<n else ""))

print("\n=== part 2: cost and behaviour of decoded answers, Qwen ===")
M="Qwen/Qwen2.5-0.5B"; DEV="cuda" if torch.cuda.is_available() else "cpu"
tok=AutoTokenizer.from_pretrained(M)
mdl=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float32).to(DEV).eval()
for p in mdl.parameters(): p.requires_grad_(False)

def first_tok(t):
    ids=tok(t,return_tensors="pt").input_ids.to(DEV)
    with torch.no_grad(): return int(mdl(ids).logits[0,-1].argmax())
def decoded(t, k=4):
    ids=tok(t,return_tensors="pt").input_ids.to(DEV)
    with torch.no_grad(): out=mdl.generate(ids,max_new_tokens=k,do_sample=False,
                                           pad_token_id=tok.eos_token_id)
    return tok.decode(out[0,ids.shape[1]:]).strip()

for rel in ("small_capital","element_symbol"):
    tpl,facts = rels[rel]; ents=list(facts)
    t0=time.perf_counter(); ft=[first_tok(tpl.format(e=e)) for e in ents]; t_f=time.perf_counter()-t0
    t0=time.perf_counter(); dc_=[decoded(tpl.format(e=e)) for e in ents]; t_d=time.perf_counter()-t0
    acc_f=sum(a==tok(" "+facts[e],add_special_tokens=False).input_ids[0] for a,e in zip(ft,ents))/len(ents)
    acc_d=sum(d.split()[0].strip(".,")==facts[e] if d.split() else False for d,e in zip(dc_,ents))/len(ents)
    print(f"{rel:<16} n={len(ents)}  m_firsttok={len(set(ft))}  m_string={len(set(dc_))}"
          f"  acc_ft={acc_f:.3f} acc_str={acc_d:.3f}  cost {t_f*1000:.0f}ms -> {t_d*1000:.0f}ms")
