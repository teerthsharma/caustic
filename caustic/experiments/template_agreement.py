"""Does partition agreement predict which templates the collision score wants?

`symmetry_scores` averages collision over every template while the label is one
template's correctness, and that average inverts on `currency`: below chance at
0.3056, lower than any of its five components. The obvious fix is to average
only over templates whose partition agrees with the scored one, measuring
agreement with the adjusted Rand index the governor already computes.

**The fix does not work, and this module is the record of that.** Over 20
(relation, template) pairs the correlation between ARI against template 0 and
that template's own AUROC is -0.111 — no relationship, and if anything the wrong
sign. Filtering to ARI >= 0.5 makes matters worse on three relations of four.

What the data says instead: a template contributes signal only if it collapses.
Every template scoring exactly 0.5000 has `m = n`, so its collision column is
constant zero — it adds nothing and dilutes the mean toward zero. Meanwhile
`capital` template 1 has ARI 0.128, near-total disagreement with the scored
partition, and scores 0.7778. On the relations where the detector works, the
DISAGREEING templates carry the most signal, and the proposed filter would have
discarded them.

So averaging over templates is not principled and no simple agreement filter
repairs it. `collision` and `collision_scored` are both reported, neither
dominates, and a caller has to know which template their label comes from.

    python -m caustic.experiments.template_agreement
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from transformers import AutoModelForCausalLM, AutoTokenizer
from caustic.governor import _adjusted_rand
from caustic.detect import auroc
from caustic.experiments.coupling_gap import RELATIONS
from caustic.experiments.symmetry_break import PARAPHRASES

M="Qwen/Qwen2.5-0.5B"; torch.manual_seed(0)
DEV="cuda" if torch.cuda.is_available() else "cpu"
tok=AutoTokenizer.from_pretrained(M)
mdl=AutoModelForCausalLM.from_pretrained(M,dtype=torch.float32).to(DEV).eval()
for p in mdl.parameters(): p.requires_grad_(False)
def top1(t):
    ids=tok(t,return_tensors="pt").input_ids.to(DEV)
    with torch.no_grad(): return int(mdl(ids).logits[0,-1].argmax())
def g(a): return tok(" "+a,add_special_tokens=False).input_ids[0]

print(f"model={M} seed=0 | ARI is template i's partition against template 0's")
print(f"{'relation':<10}{'tpl':>4}{'ARI_0i':>8}{'AUROC_i':>9}   helps?")
print("-"*45)
rows=[]
for rel,(tpl,facts) in RELATIONS.items():
    tpls=PARAPHRASES[rel]; ents=list(facts)
    tab={e:[top1(t.format(e=e)) for t in tpls] for e in ents}
    correct=np.array([tab[e][0]==g(facts[e]) for e in ents])
    if correct.all() or not correct.any(): continue
    base=[tab[e][0] for e in ents]
    cols=[]
    for i in range(len(tpls)):
        lab=[tab[e][i] for e in ents]
        col=np.array([np.mean([tab[o][i]==tab[e][i] for o in ents if o!=e]) for e in ents])
        cols.append(col)
        a=auroc(col,~correct); ari=_adjusted_rand(base,lab)
        rows.append((rel,i,ari,a))
        print(f"{rel:<10}{i:>4}{ari:>8.3f}{a:>9.4f}   {'+' if a>0.5 else '-'}")
    avg=np.mean(cols,axis=0)
    keep=[c for (r,i,ari,a),c in zip([r for r in rows if r[0]==rel],cols) if ari>=0.5]
    sel=np.mean(keep,axis=0) if keep else cols[0]
    print(f"{rel:<10}{'AVG':>4}{'':>8}{auroc(avg,~correct):>9.4f}   all templates")
    print(f"{rel:<10}{'ARI>':>4}{0.5:>8.1f}{auroc(sel,~correct):>9.4f}   agreeing only ({len(keep)} of {len(tpls)})\n")
ar=np.array([r[2] for r in rows]); au=np.array([r[3] for r in rows])
print(f"corr(ARI_0i, AUROC_i) over {len(rows)} templates = {np.corrcoef(ar,au)[0,1]:.3f}")
