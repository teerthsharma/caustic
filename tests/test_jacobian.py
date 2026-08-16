"""Correctness contracts for the J-space estimators.

Every test here has a closed-form answer, so a wrong implementation fails rather
than merely looking different. Run with `pytest tests/` or `python tests/test_jacobian.py`.

The load-bearing ones are `test_linear_block_jacobian_is_the_weight_matrix`
(ground truth for the estimator) and `test_tail_alpha_recovers_a_known_exponent`
(ground truth for the fractal summary). If only two tests survive, keep those.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from caustic.jacobian import exact_jacobian, singular_values, top_singular_values
from caustic.spectrum import log_volume, stable_rank, summarize, tail_alpha

SEED = 0
D = 32
T = 5


class LinearBlock(torch.nn.Module):
    """A block whose Jacobian at every position is exactly W.

    Acts position-wise, so the diagonal block of the full Jacobian is W with no
    cross-position leakage. This is the ground truth the estimator must reproduce.
    """

    def __init__(self, W: torch.Tensor):
        super().__init__()
        self.W = torch.nn.Parameter(W, requires_grad=False)

    def forward(self, h):
        return h @ self.W.T


class TupleBlock(LinearBlock):
    """Same map, returning a tuple, as transformers <5 blocks do."""

    def forward(self, h):
        return (super().forward(h), None)


@pytest.fixture
def W():
    g = torch.Generator().manual_seed(SEED)
    return torch.randn(D, D, generator=g, dtype=torch.float64)


@pytest.fixture
def h_in():
    g = torch.Generator().manual_seed(SEED + 1)
    return torch.randn(1, T, D, generator=g, dtype=torch.float64)


# --- ground truth for the estimator ---------------------------------------


def test_linear_block_jacobian_is_the_weight_matrix(W, h_in):
    """d(Wh)/dh = W, exactly."""
    J = exact_jacobian(LinearBlock(W), h_in, pos=T - 1)
    assert torch.allclose(J, W, atol=1e-10), f"max dev {(J - W).abs().max():.3e}"


def test_tuple_returning_block_is_handled(W, h_in):
    """transformers <5 returns a tuple; the shim must not silently take the wrong element."""
    J = exact_jacobian(TupleBlock(W), h_in, pos=T - 1)
    assert torch.allclose(J, W, atol=1e-10)


def test_jacobian_is_deterministic(W, h_in):
    """Same input, bitwise same Jacobian. Non-determinism here invalidates every A/B."""
    block = LinearBlock(W)
    a = exact_jacobian(block, h_in, pos=2)
    b = exact_jacobian(block, h_in, pos=2)
    assert torch.equal(a, b)


def test_frozen_positions_do_not_leak(W, h_in):
    """The map isolates one position: perturbing h_in elsewhere cannot change W."""
    block = LinearBlock(W)
    J0 = exact_jacobian(block, h_in, pos=T - 1)
    perturbed = h_in.clone()
    perturbed[0, 0, :] += 10.0
    J1 = exact_jacobian(block, perturbed, pos=T - 1)
    assert torch.allclose(J0, J1, atol=1e-10)


def test_power_iteration_matches_exact_singular_values(W, h_in):
    """The Krylov estimator must agree with the exact spectrum it replaces.

    Measured at 1.685e-04 max relative error on distilgpt2 layer 3 (k=8, 20 iters);
    on this small well-conditioned case it should be far tighter.
    """
    block = LinearBlock(W)
    k = 6
    est = top_singular_values(block, h_in, pos=T - 1, k=k, iters=60).cpu().numpy()
    exact = singular_values(exact_jacobian(block, h_in, pos=T - 1)).cpu().numpy()[:k]
    rel = np.abs(est - exact) / np.maximum(exact, 1e-30)
    assert rel.max() < 1e-6, f"max rel err {rel.max():.3e}\nest   {est}\nexact {exact}"


def test_power_iteration_is_deterministic_across_calls(W, h_in):
    """Same input, bitwise same estimate.

    This test exists because the parity check above passed on five consecutive
    runs and failed on a sixth. The estimator drew its starting block from the
    global RNG, so an unlucky draw left it nearly orthogonal to a wanted singular
    direction and twenty iterations could not recover. Any A/B built on a
    non-deterministic estimator measures the estimator's own variance.
    """
    block = LinearBlock(W)
    a = top_singular_values(block, h_in, pos=T - 1, k=4, iters=30)
    torch.randn(1000)  # perturb the global RNG between calls
    b = top_singular_values(block, h_in, pos=T - 1, k=4, iters=30)
    assert torch.equal(a, b), f"drifted by {(a - b).abs().max():.3e}"


def test_power_iteration_seed_none_still_returns_the_right_answer(W, h_in):
    """Opting out of the seed must change the start, not the answer."""
    block = LinearBlock(W)
    exact = singular_values(exact_jacobian(block, h_in, pos=T - 1)).cpu().numpy()[:4]
    for _ in range(5):
        est = top_singular_values(block, h_in, pos=T - 1, k=4, iters=80, seed=None).cpu().numpy()
        assert np.abs(est - exact).max() / exact.max() < 1e-5


def test_out_of_range_position_raises(W, h_in):
    with pytest.raises(IndexError):
        exact_jacobian(LinearBlock(W), h_in, pos=T + 3)


def test_wrong_shape_raises(W):
    with pytest.raises(ValueError):
        exact_jacobian(LinearBlock(W), torch.randn(2, T, D, dtype=torch.float64), pos=0)


# --- ground truth for the spectral summaries -------------------------------


@pytest.mark.parametrize("alpha", [0.25, 0.5, 1.0, 2.0])
def test_tail_alpha_recovers_a_known_exponent(alpha):
    """On an exact power law sigma_i = (i+1)^-alpha, the fit must return alpha.

    This is the test that decides whether the reported constant 0.2374 means
    anything. A fit that cannot recover a synthetic exponent cannot be trusted on
    a measured one.
    """
    i = np.arange(768)
    sv = (i + 1.0) ** (-alpha)
    assert tail_alpha(sv) == pytest.approx(alpha, abs=1e-9)


@pytest.mark.parametrize("c", [1e-6, 1.0, 1e6])
def test_tail_alpha_is_scale_invariant(c):
    """Multiplying every singular value by c shifts the intercept, not the slope."""
    i = np.arange(768)
    sv = (i + 1.0) ** (-0.5)
    assert tail_alpha(c * sv) == pytest.approx(tail_alpha(sv), abs=1e-9)


@pytest.mark.parametrize("c", [1e-6, 1.0, 1e6])
def test_stable_rank_is_scale_invariant(c):
    rng = np.random.default_rng(SEED)
    sv = np.sort(rng.gamma(2.0, size=500))[::-1]
    assert stable_rank(c * sv) == pytest.approx(stable_rank(sv), rel=1e-12)


def test_stable_rank_bounds():
    """n for a flat spectrum, ~1 for a single dominant direction, never outside [1, n]."""
    n = 100
    assert stable_rank(np.ones(n)) == pytest.approx(n)
    spike = np.concatenate([[1.0], np.full(n - 1, 1e-8)])
    assert stable_rank(spike) == pytest.approx(1.0, abs=1e-6)
    rng = np.random.default_rng(SEED)
    for _ in range(20):
        sv = np.abs(rng.normal(size=n)) + 1e-9
        assert 1.0 - 1e-9 <= stable_rank(sv) <= n + 1e-9


def test_log_volume_equals_logdet(W, h_in):
    """sum log sigma_i of J must equal log|det J|. Ties the summary to the geometry."""
    J = exact_jacobian(LinearBlock(W), h_in, pos=0)
    sv = singular_values(J).cpu().numpy().astype(np.float64)
    expected = torch.linalg.slogdet(J.double())[1].item()
    # singular_values works in float32 while slogdet works in float64, so the
    # gap is bounded by float32 eps, not by anything about log_volume. rel=1e-8
    # held only on Windows/MKL; Linux LAPACK lands at ~1e-7.
    assert log_volume(sv) == pytest.approx(expected, rel=1e-6)


def test_tail_alpha_rejects_a_degenerate_bulk():
    with pytest.raises(ValueError):
        tail_alpha(np.ones(50), bulk=(10, 11))


def test_summarize_is_complete():
    sv = (np.arange(768) + 1.0) ** -0.3
    assert set(summarize(sv)) == {"sigma_max", "stable_rank", "tail_alpha", "log_volume"}


# --- the negative control --------------------------------------------------


def test_flat_spectrum_has_zero_tail_exponent():
    """The null case. A pipeline that reports structure in a flat spectrum reports
    it everywhere, which is the failure mode that makes fractal claims worthless."""
    assert tail_alpha(np.ones(768)) == pytest.approx(0.0, abs=1e-9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
