"""Scoring a position as likely-wrong, and the baselines any such score must beat.

The atom of hallucination this module targets is narrow and measurable: at token
position t, is the model's top-1 next-token prediction wrong? Confabulation is
built out of positions where the model commits to a token that the grounded
continuation does not support, so "about to be wrong" is the smallest honest unit
of the phenomenon, and unlike a shuffled-token control it is a real label.

**The baselines are defined before any geometric score, deliberately.** A prior
gate in this line of work scored 0.846 mean AUROC against Mahalanobis at 0.899 and
PCA at 0.888, both of which are cheaper. A geometric score that does not beat
Mahalanobis on both AUROC and cost is a negative result, and writing the baselines
first is what stops them from being written to lose.

Every baseline here reads only the hidden state or the logits, costs one forward
pass, and needs no Jacobian.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np

__all__ = [
    "max_softmax_score",
    "predictive_entropy",
    "logit_margin",
    "MahalanobisScorer",
    "PCAScorer",
    "auroc",
    "auroc_ci",
    "auroc_ci_detail",
    "AurocCI",
]


# --- baselines -------------------------------------------------------------


def max_softmax_score(logits: np.ndarray) -> np.ndarray:
    """1 - max softmax probability. The standard confidence baseline.

    Higher means less confident, so higher should mean more likely wrong.
    """
    z = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    return 1.0 - p.max(axis=-1)


def predictive_entropy(logits: np.ndarray) -> np.ndarray:
    """Shannon entropy of the next-token distribution, in nats."""
    z = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    return -(p * np.log(np.maximum(p, 1e-30))).sum(axis=-1)


def logit_margin(logits: np.ndarray) -> np.ndarray:
    """Negated gap between the top two logits. Cheaper than entropy, often as good."""
    part = np.partition(logits, -2, axis=-1)
    return -(part[..., -1] - part[..., -2])


class MahalanobisScorer:
    """Whitened distance of a hidden state from the calibration mean.

    The baseline that beat a comparable geometric gate previously. Uses a
    shrinkage-regularized covariance because D is typically comparable to or
    larger than the number of calibration points, and the sample covariance is
    then singular.
    """

    def __init__(self, shrinkage: float = 0.1):
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError(f"shrinkage must be in [0, 1]; got {shrinkage}")
        self.shrinkage = shrinkage
        self.mu_: np.ndarray | None = None
        self.prec_: np.ndarray | None = None

    def fit(self, H: np.ndarray) -> MahalanobisScorer:
        H = np.asarray(H, dtype=np.float64)
        if H.ndim != 2:
            raise ValueError(f"expected (n, D); got {H.shape}")
        self.mu_ = H.mean(axis=0)
        X = H - self.mu_
        cov = (X.T @ X) / max(len(X) - 1, 1)
        trace_mean = np.trace(cov) / cov.shape[0]
        cov = (1.0 - self.shrinkage) * cov + self.shrinkage * trace_mean * np.eye(cov.shape[0])
        self.prec_ = np.linalg.pinv(cov, hermitian=True)
        return self

    def score(self, H: np.ndarray) -> np.ndarray:
        if self.mu_ is None or self.prec_ is None:
            raise RuntimeError("fit before scoring")
        X = np.asarray(H, dtype=np.float64) - self.mu_
        return np.einsum("ij,jk,ik->i", X, self.prec_, X)


class PCAScorer:
    """Reconstruction error under a rank-k PCA fitted on calibration states."""

    def __init__(self, k: int = 32):
        if k < 1:
            raise ValueError(f"k must be positive; got {k}")
        self.k = k
        self.mu_: np.ndarray | None = None
        self.V_: np.ndarray | None = None

    def fit(self, H: np.ndarray) -> PCAScorer:
        H = np.asarray(H, dtype=np.float64)
        self.mu_ = H.mean(axis=0)
        _, _, Vt = np.linalg.svd(H - self.mu_, full_matrices=False)
        self.V_ = Vt[: min(self.k, Vt.shape[0])].T
        return self

    def score(self, H: np.ndarray) -> np.ndarray:
        if self.mu_ is None or self.V_ is None:
            raise RuntimeError("fit before scoring")
        X = np.asarray(H, dtype=np.float64) - self.mu_
        return ((X - (X @ self.V_) @ self.V_.T) ** 2).sum(axis=1)


# --- evaluation ------------------------------------------------------------


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve, by the rank identity, with ties averaged.

    `labels` is 1 for the positive class (here: the prediction was wrong).
    Returns 0.5 when either class is empty, since AUROC is undefined then and a
    silent NaN would propagate into a reported table.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Average ranks so tied scores cannot inflate the statistic.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


class AurocCI(NamedTuple):
    """A bootstrap interval together with the evidence that it is trustworthy."""

    point: float
    lo: float
    hi: float
    n_valid: int
    n_discarded: int


def auroc_ci_detail(
    scores: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    _stacklevel: int = 2,
) -> AurocCI:
    """Percentile bootstrap 95% interval, with single-class resamples excluded.

    A single AUROC without an interval invites a reader to treat 0.61 and 0.59 as
    different when the sample cannot distinguish them.

    **Why single-class resamples are dropped rather than scored.** `auroc` returns
    0.5 when a class is empty, which is the right guard for a point estimate but
    the wrong input to a percentile. At the relation sizes this repo actually
    measures — n = 12 to 20, minority class sometimes 1 — resampling n indices
    with replacement leaves one class empty a large fraction of the time: with 1
    correct answer out of 16, (15/16)^16 = 0.356 of all resamples. Feeding each of
    those a hard 0.5 puts a spike of probability mass at exactly 0.5, which drags
    whichever percentile bound lies on the far side of 0.5 toward it and can pin a
    bound to 0.5 outright. The reported interval then describes the guard, not the
    data. Resamples that cannot yield an AUROC are therefore discarded, and the
    interval is taken over the survivors.

    **What that interval does NOT show.** Conditioning on both classes surviving is
    a different estimand from the unconditional sampling distribution of AUROC, and
    neither is the distribution at fixed class sizes. At a minority class of one or
    two, the surviving resamples are a handful of distinct configurations reweighted
    by multiplicity, so the interval is a coarse quantisation of very little
    evidence. Excluding the degenerate draws makes the interval honest about its
    own width; it does not make the underlying sample larger.

    In particular the survivors can collapse to one distinct value and return
    `lo == hi`. That is a property of the percentile bootstrap at any n — perfect
    separation gives a zero-width interval on 300 points too — but discarding the
    degenerate draws makes it reachable at n = 16, where it means "every resample
    that contained both classes ranked them the same way", not "certain". A
    zero-width interval here is a report that the resampling has run out of
    resolution and must be read alongside `n_valid` and the class sizes.

    **When too little survives.** If fewer than `n_boot // 10` resamples are usable,
    the interval is undefined at this sample size and `lo`/`hi` are returned as nan
    rather than raised. A raise would abort a caller mid-table and every call site
    in this repo unpacks three floats; a nan prints as `nan` and cannot be misread
    as a tight interval, whereas any numeric fallback could. `point` is still
    returned, since the point estimate does not depend on resampling.

    `n_discarded` is returned so the contamination is visible, and a warning is
    raised whenever it is non-zero.
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n = len(scores)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y = labels[idx]
        if not y.any() or y.all():
            continue
        boots.append(auroc(scores[idx], y))

    point = auroc(scores, labels)
    n_valid, n_discarded = len(boots), n_boot - len(boots)
    if n_discarded:
        warnings.warn(
            f"auroc_ci: {n_discarded}/{n_boot} bootstrap resamples were single-class "
            f"at n={n} and were discarded",
            stacklevel=_stacklevel,
        )
    if n_valid < max(1, n_boot // 10):
        return AurocCI(point, float("nan"), float("nan"), n_valid, n_discarded)
    b = np.asarray(boots)
    return AurocCI(
        point, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), n_valid, n_discarded
    )


def auroc_ci(
    scores: np.ndarray, labels: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """`(point, lo, hi)` from `auroc_ci_detail`, whose docstring carries the argument.

    Kept at three values because every call site in this repo unpacks exactly three.
    """
    ci = auroc_ci_detail(scores, labels, n_boot, seed, _stacklevel=3)
    return ci.point, ci.lo, ci.hi
