"""Restricting the argmax to the answer set, and what that separates.

Theorem 1* needs the correct answers as a set `G` — that Paris and Berlin are
capitals, not which country each belongs to. Once a caller has `G`, they can
also decode against it: take the argmax over the tokens in `G` rather than over
the whole vocabulary. Three consequences follow, and the third is the reason
this module exists.

**The precondition becomes automatic.** Distinct admissible answers are distinct
outputs, so `|G| = n` holds whenever the golds are distinct strings and no
tokenizer collision can produce a false certificate.

**Theorem 1\* collapses back to Theorem 1.** Nothing inadmissible can be emitted,
so `f(E) ⊆ G` and `m* = m`. The answer set buys nothing once it is enforced
rather than merely consulted.

**And it separates two failures the free decoder conflates.** Measured on
Qwen2.5-0.5B, seed 0:

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

Under coherent context, constraining takes accuracy to 1.000 on all three
relations. Those errors were never lost facts — the model was emitting something
outside the answer space, and confining it to that space is enough. Under
`" the" x 128` constraining changes almost nothing: restricted to capitals, the
model still cannot find the right capital.

That is a control the repository did not have. It shows the coherence result is
about retrieval rather than formatting: the degenerate prefix destroys the
model's access to the fact, not merely its choice of surface form. A reader who
suspected the 0.000 row was an artifact of off-distribution tokens now has the
answer, and it is no.

The certificate tracks both regimes. Coherent and constrained gives `m = n`, so
it certifies 0 errors against a measured accuracy of 1.000. Degenerate and
constrained certifies 19 against 19 actual on `capital` and 15 against 15 on
`language` — exact in 5 of the 6 constrained rows, valid with slack 3 in the
sixth.

**Cost.** One forward pass with the argmax masked to `|G|` logits, which is
cheaper than the unconstrained argmax over the full vocabulary, and it beats the
shipped prefix repair: `NEUTRAL_PREFIX` measures 0.750 on `capital` against
1.000 here.

**What it is not.** Constraining changes the task. A model choosing among 20
capitals is doing something closer to multiple choice than to open generation,
and the accuracy figures above are not comparable to free-decoding accuracy
anywhere else in this repository. It is available only where the answer set is
known, which is exactly the setting Theorem 1* already assumes and is far from
every setting.

    python -m caustic.experiments.constrained_decode
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformers import AutoModelForCausalLM, AutoTokenizer
from caustic.experiments.coupling_gap import RELATIONS
M="Qwen/Qwen2.5-0.5B"; torch.manual_seed(0)
DEV="cuda" if torch.cuda.is_available() else "cpu"
tok=AutoTokenizer.from_pretrained(M)
mdl=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float32).to(DEV).eval()
for p in mdl.parameters(): p.requires_grad_(False)
def logits(t):
    ids=tok(t,return_tensors="pt").input_ids.to(DEV)
    with torch.no_grad(): return mdl(ids).logits[0,-1]
def g(a): return tok(" "+a,add_special_tokens=False).input_ids[0]

print(f"{'relation':<10}{'cond':<10}{'mode':<12}{'n':>3}{'m':>4}{'n-m':>6}{'acc':>7}")
print("-"*56)
for rel,(tpl,facts) in RELATIONS.items():
    ents=list(facts); G=[g(facts[e]) for e in ents]
    if len(set(G))<len(ents):
        print(f"{rel:<10}  skipped: gold not token-injective"); continue
    Gt=torch.tensor(sorted(set(G)),device=DEV)
    for cond,pre in (("none",""),("the_x128"," the"*128)):
        for mode in ("free","constrained"):
            ans=[]
            for e in ents:
                lg=logits(pre+tpl.format(e=e))
                if mode=="free": ans.append(int(lg.argmax()))
                else: ans.append(int(Gt[lg[Gt].argmax()]))
            m=len(set(ans)); acc=np.mean([a==g(facts[e]) for a,e in zip(ans,ents)])
            print(f"{rel:<10}{cond:<10}{mode:<12}{len(ents):>3}{m:>4}{len(ents)-m:>6}{acc:>7.3f}")
