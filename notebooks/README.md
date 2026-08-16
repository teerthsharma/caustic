# Notebooks

Runnable front door to the repository. Every notebook here is committed **with its
executed outputs**, so the result can be read on GitHub before deciding whether to
run anything.

## `caustic_quickstart.ipynb`

The headline claim, end to end: a model that holds a fact and cannot reach it, the
orbit partition that records the failure, the Theorem 1 certificate computed with no
answer key, and `repair_by_context` reporting its own effect size — including the
case where the repair makes matters worse.

Two models are run under the same code. `HuggingFaceTB/SmolLM2-135M` does **not**
reproduce the contrast; `Qwen/Qwen2.5-0.5B`, the model `RESULTS.md` was measured on,
does. Both are shown rather than only the one that agrees.

| | |
|---|---|
| Tier | free Colab CPU, no accelerator, no manual edits |
| Runtime | 52 s measured end to end on a cold kernel with weights already cached; the two model downloads add roughly 1.2 GB on a first run |
| Installs | `caustic[experiments]` from git when no local clone is present |
| Determinism | seed 0, `.eval()`, `torch.no_grad()`, argmax next-token id — never sampled |

The first cell detects a local clone at `..` and uses it instead of installing, so the
notebook runs unchanged from a checkout as well as from Colab.
