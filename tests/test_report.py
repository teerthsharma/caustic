"""Contracts for the result recorder, pinned to exact strings and exact JSON.

The point of this module is that a number on stdout and the same number in a
file cannot drift apart, so nothing here is checked to a tolerance: every table
is compared against the literal expected text, and every JSON payload against
the literal expected value. No model, no GPU, no network.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.report import emit, git_sha, table

ROWS = [
    {"rel": "capital", "n": 20, "m": 15, "floor": 0.25},
    {"rel": "continent", "n": 12, "m": 4, "floor": 0.6667},
]


# --- table -----------------------------------------------------------------


def test_table_is_exact_house_format():
    """Header, `-` rule of the header's width, left strings, right numerics.

    Pinned to the literal string: a change to padding or decimal places silently
    breaks every downstream diff of a pasted table, and nothing else would catch
    it.
    """
    assert table(ROWS) == (
        "rel        n  m  floor\n"
        "----------------------\n"
        "capital   20 15 0.2500\n"
        "continent 12  4 0.6667"
    )


def test_column_width_follows_the_widest_cell_not_the_header():
    """A cell longer than its header must widen the column, not overflow it."""
    out = table([{"a": "xxxxxxx"}])
    assert out.splitlines() == ["a      ", "-------", "xxxxxxx"]


def test_cols_selects_and_orders():
    """Explicit `cols` is the caller's contract with the reproduction matrix."""
    assert table(ROWS, ["m", "rel"]).splitlines() == [
        " m rel      ",
        "------------",
        "15 capital",
        " 4 continent",
    ]


def test_float_places_are_fixed_at_four():
    """0.5 and 0.5000001 must not render with different widths in one column."""
    assert table([{"x": 0.5}, {"x": 0.5000001}]).splitlines()[2:] == ["0.5000", "0.5000"]


def test_missing_key_renders_empty_rather_than_raising():
    out = table([{"a": 1, "b": 2}, {"a": 3}])
    assert out.splitlines()[3] == "3"


def test_non_finite_floats_render_but_do_not_crash():
    out = table([{"x": float("nan")}, {"x": float("-inf")}, {"x": 0.5}])
    assert out.splitlines() == ["     x", "------", "   nan", "  -inf", "0.5000"]


def test_empty_rows_is_rejected():
    """Negative control: a run that produced nothing must not write a table."""
    with pytest.raises(ValueError):
        table([])


def test_numpy_scalars_match_their_python_equivalents():
    """np.float64 must not print as `np.float64(0.25)`."""
    assert table([{"n": np.int64(20), "f": np.float64(0.25)}]) == table([{"n": 20, "f": 0.25}])


# --- emit ------------------------------------------------------------------


def _run(tmp_path, name, rows, meta):
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        text = emit(name, rows, meta)
    finally:
        os.chdir(cwd)
    return text, json.loads((tmp_path / "results" / f"{name}.json").read_text())


def test_emit_writes_the_rows_it_printed(tmp_path, capsys):
    text, payload = _run(tmp_path, "demo", ROWS, {"model": "Qwen/Qwen2.5-0.5B", "seed": 0})
    assert text == table(ROWS)
    assert capsys.readouterr().out == text + "\n"
    assert payload["name"] == "demo"
    assert payload["rows"] == ROWS
    assert payload["meta"]["model"] == "Qwen/Qwen2.5-0.5B"
    assert payload["meta"]["seed"] == 0


def test_emit_stamps_sha_and_timestamp(tmp_path):
    """Provenance is the whole point; a result file without it is untraceable."""
    _, payload = _run(tmp_path, "demo", ROWS, {"model": "m", "seed": 1})
    assert set(payload["meta"]) == {"model", "seed", "git_sha", "timestamp"}
    assert payload["meta"]["timestamp"].endswith("Z")
    sha = payload["meta"]["git_sha"]
    assert sha is None or len(sha) == 40


def test_caller_cannot_forge_the_sha(tmp_path):
    """Negative control: a meta key named git_sha is overwritten, not honoured."""
    _, payload = _run(tmp_path, "demo", ROWS, {"model": "m", "seed": 0, "git_sha": "deadbeef"})
    assert payload["meta"]["git_sha"] != "deadbeef"


def test_non_finite_floats_become_null_in_json(tmp_path):
    """JSON has no NaN literal; emitting one would produce a file json.load rejects."""
    _, payload = _run(tmp_path, "demo", [{"x": float("nan"), "y": float("-inf"), "z": 1.0}], {})
    assert payload["rows"] == [{"x": None, "y": None, "z": 1.0}]


def test_numpy_values_survive_the_round_trip(tmp_path):
    """np.bool_ must land as JSON true, not the string "True".

    A numpy comparison returns np.bool_, never bool, so a row flag written as
    `scores.mean() > thresh` is the common case, and a reproduction matrix
    comparing it against `true` would fail on a correct run.
    """
    row = {"n": np.int64(7), "f": np.float32(0.5), "ok": np.float64(2.0) > 1.0,
           "s": np.asarray(3.5)}
    _, payload = _run(tmp_path, "demo", [row], {})
    assert payload["rows"] == [{"n": 7, "f": 0.5, "ok": True, "s": 3.5}]


def test_rerunning_the_same_name_overwrites(tmp_path):
    _run(tmp_path, "demo", ROWS, {})
    _, payload = _run(tmp_path, "demo", [{"rel": "capital", "n": 1}], {})
    assert payload["rows"] == [{"rel": "capital", "n": 1}]


@pytest.mark.parametrize("name", ["../escape", "sub/demo", "sub\\demo", "C:evil", "", "."])
def test_name_that_is_not_a_bare_stem_is_rejected(tmp_path, name):
    """Negative control: nothing may write outside results/.

    `C:evil` is the one that does not look like a path: `Path("results") /
    "C:evil.json"` is `C:evil.json`, an absolute-ish write on the platform this
    repo is developed on.
    """
    with pytest.raises(ValueError):
        _run(tmp_path, name, ROWS, {})


def test_git_sha_never_raises():
    sha = git_sha()
    assert sha is None or (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
