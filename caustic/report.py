"""One table, printed once and written once, so a claim can be traced to a run.

Every experiment in this repo prints a fixed-width table to stdout and nothing
else. The numbers in README.md and RESULTS.md then exist only because they were
copied across by hand, which means no check can tell a transcription slip from a
result, and CI has nothing to compare a claim against. The failure mode is not
hypothetical: a table that is re-run with a different seed and re-pasted only in
part leaves the two halves of the document disagreeing, silently.

This module is the smallest thing that closes that gap. `table` reproduces the
house format the scripts already print, so nothing on stdout has to change, and
`emit` additionally writes the same rows to `results/<name>.json` together with
the git SHA and a UTC timestamp. A later reproduction matrix reads the JSON; a
reader reads the table; they are the same numbers by construction.

What this does NOT do:

    it does not retrofit the 22 existing experiment scripts, which still print
    by hand and are out of scope here

    it does not verify anything — it records. Checking a README number against
    a result file is a separate job, and this module only makes that job possible

    it does not version or namespace the output. `results/<name>.json` is
    overwritten by the next run of the same name, because a run artifact that
    accumulates silently is worse than one that is plainly the latest

Non-finite floats are rendered in the table as `nan` / `inf` / `-inf`, but JSON
has no literal for them, so they are encoded as `null`. A `null` in a result
file therefore means "the run produced a non-finite value", not "missing".
"""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

__all__ = ["table", "emit", "git_sha", "RESULTS_DIR"]

RESULTS_DIR = Path("results")

_BOOL = (bool, np.bool_)
_INT = (int, np.integer)
_FLOAT = (float, np.floating)


def _cell(v: object) -> str:
    """One value, formatted deterministically and independent of locale.

    Floats always carry four decimals: a column whose places vary between rows
    cannot be read down, and a column whose places vary between runs cannot be
    diffed.
    """
    if isinstance(v, _BOOL):
        return str(bool(v))
    if isinstance(v, _INT):
        return str(int(v))
    if isinstance(v, _FLOAT):
        return f"{float(v):.4f}"
    return "" if v is None else str(v)


def _numeric(v: object) -> bool:
    return not isinstance(v, _BOOL) and isinstance(v, _INT + _FLOAT)


def table(rows: list[dict], cols: list[str] | None = None) -> str:
    """The repo's fixed-width table: header, `-` rule, one line per row.

    Columns default to first-seen key order across `rows`. A column is
    right-aligned when every value present in it is numeric, left-aligned
    otherwise; width is the widest of the header and its cells. Missing keys
    render empty rather than raising, so a row set that grew a column mid-run is
    still printable.
    """
    if not rows:
        raise ValueError("no rows to format")
    if cols is None:
        cols = list(dict.fromkeys(k for r in rows for k in r))
    if not cols:
        raise ValueError("no columns to format")

    cells = [[_cell(r.get(c)) for c in cols] for r in rows]
    right = [all(_numeric(r[c]) for r in rows if c in r) for c in cols]
    width = [max(len(c), *(len(row[j]) for row in cells)) for j, c in enumerate(cols)]

    def line(vals: list[str]) -> str:
        return " ".join(
            v.rjust(width[j]) if right[j] else v.ljust(width[j])
            for j, v in enumerate(vals)
        ).rstrip()

    # The header keeps its trailing pad so the rule spans the full table width,
    # as the hand-written tables do; data lines are stripped.
    head = " ".join(
        c.rjust(width[j]) if right[j] else c.ljust(width[j]) for j, c in enumerate(cols)
    )
    return "\n".join([head, "-" * len(head), *(line(row) for row in cells)])


def git_sha() -> str | None:
    """HEAD of the repo containing this file, or None outside a repo.

    Never raises: a missing git, a tarball checkout and an empty repository all
    degrade to None, because a run must not fail over its own provenance stamp.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _jsonable(v: object) -> object:
    """numpy scalars down to Python, non-finite floats to None (JSON has no NaN)."""
    if isinstance(v, _BOOL):
        return bool(v)
    if isinstance(v, _INT):
        return int(v)
    if isinstance(v, _FLOAT):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, (str, type(None))):
        return v
    if isinstance(v, np.ndarray):
        # A 0-d array is a scalar wearing an array's type and is not iterable.
        return _jsonable(v.item()) if v.ndim == 0 else [_jsonable(x) for x in v]
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


def emit(name: str, rows: list[dict], meta: dict) -> str:
    """Print the table and write `results/<name>.json`; return the table string.

    `meta` is the caller's provenance — pass at least `model` and `seed` — and is
    merged with the git SHA and an ISO-8601 UTC timestamp. Those two are written
    last and so override any caller key of the same name: a run cannot claim a
    SHA it was not built from.
    """
    # Both flavours, because `Path` is only one of them and CI is not the
    # platform this repo is developed on. `PureWindowsPath("C:evil").name` is
    # "evil", so the drive prefix is caught on Windows; `PurePosixPath` leaves
    # it as "C:evil" and would let the write through on the Linux runner.
    if (
        name in ("", ".", "..")
        or PurePosixPath(name).name != name
        or PureWindowsPath(name).name != name
    ):
        raise ValueError(f"name must be a bare file stem, got {name!r}")
    text = table(rows)
    print(text)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "meta": {
            **_jsonable(dict(meta)),
            "git_sha": git_sha(),
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
        "rows": [_jsonable(dict(r)) for r in rows],
    }
    (RESULTS_DIR / f"{name}.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return text
