"""The row formatter recomputes the certified floor; that arithmetic is checked here.

`RepairReport` carries counts, not partitions, so `prefix_confound.rows` derives
`(n - m) / n` itself rather than reading `OrbitReport.certified_error_rate`. A
sign or operand slip there would misreport the Orbit Error Bound in the only
table this experiment produces, so it is pinned against a dictionary model with
no tokenizer, no weights and no download.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.experiments.prefix_confound import rows
from caustic.regime import RelationSpec
from caustic.repair import sweep_prefixes

ENTS = ("tungsten", "antimony", "tin", "potassium")
GOLD = {e: i for i, e in enumerate(ENTS, start=1)}  # 0 is the collapsed answer, never correct
SPEC = RelationSpec(("The chemical symbol for {e} is",), ENTS)


def answer_fn(prompt: str) -> int:
    """Collapsed onto one answer unless the prompt carries the GOOD prefix."""
    entity = next(e for e in ENTS if e in prompt.rsplit("for ", 1)[-1])
    return GOLD[entity] if prompt.startswith("GOOD ") else 0


def test_rows_report_the_orbit_error_bound_on_both_sides():
    out = sweep_prefixes(SPEC, answer_fn, {"good": "GOOD ", "bad": "BAD "}, gold=GOLD)

    before, after = rows("good", out["good"])
    # No prefix: 4 entities, 1 answer, so 3 of 4 are certified wrong.
    assert before.split() == ["(no", "prefix)", "4", "1", "4", "0.750", "0.000"]
    assert after.split() == ["good", "4", "4", "1", "0.000", "1.000"]

    # The before half is the same measurement whichever prefix follows it.
    assert rows("bad", out["bad"])[0] == before
    assert rows("bad", out["bad"])[1].split() == ["bad", "4", "1", "4", "0.750", "0.000"]
