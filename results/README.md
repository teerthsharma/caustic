# results/

Run artifacts written by `caustic.report.emit`. Everything in this directory
except this note is gitignored: a result file records one execution on one
machine, and committing it would make the repo's history a log of runs rather
than of code.

`emit("<name>", rows, meta)` prints the table to stdout and writes
`results/<name>.json`. A second run of the same name overwrites the file.

## Schema

```json
{
  "name": "coupling_gap",
  "meta": {
    "model": "Qwen/Qwen2.5-0.5B",
    "seed": 0,
    "git_sha": "0000000000000000000000000000000000000000",
    "timestamp": "2026-01-01T00:00:00Z"
  },
  "rows": [
    {"rel": "capital", "n": 20, "m": 15, "floor": 0.25}
  ]
}
```

- `name` — the stem passed to `emit`, matching the file name.
- `meta` — whatever the caller passed, plus two keys written last and therefore
  not forgeable by the caller: `git_sha` (HEAD at run time, `null` outside a git
  checkout) and `timestamp` (ISO-8601, UTC, `Z`-suffixed). Callers pass at least
  `model` and `seed`.
- `rows` — the table rows, one object per line of the printed table, with the
  same keys and values.

Non-finite floats are encoded as `null`, because JSON has no `NaN` or `Infinity`
literal. A `null` therefore means the run produced a non-finite value, not that
the field was absent — an absent field is absent.
