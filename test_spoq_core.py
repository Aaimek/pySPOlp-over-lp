"""Tests for SPOQ core definitions."""

from __future__ import annotations

import numpy as np

from spoq_core import lq_smooth, lp_smooth, spoq_grad, spoq_metric_diag, spoq_penalty


def _matlab_lp_smooth(x: np.ndarray, alpha: float, p: float) -> float:
    return float((np.sum((x**2 + alpha**2) ** (p / 2.0) - alpha**p)) ** (1.0 / p))


def _matlab_lq_smooth(x: np.ndarray, eta: float, q: float) -> float:
    return float((eta**q + np.sum(np.abs(x) ** q)) ** (1.0 / q))


def _matlab_gradlplq(x: np.ndarray, alpha: float, beta: float, eta: float, p: float, q: float) -> np.ndarray:
    lp = _matlab_lp_smooth(x, alpha, p)
    lq = _matlab_lq_smooth(x, eta, q)
    grad1 = x * ((x**2 + alpha**2) ** (p / 2.0 - 1.0)) / (lp**p + beta**p)
    grad2 = np.sign(x) * np.abs(x) ** (q - 1.0) / (lq**q)
    return grad1 - grad2


def _matlab_condlplq(
    x: np.ndarray, alpha: float, beta: float, eta: float, p: float, q: float, rho: float
) -> np.ndarray:
    lp = _matlab_lp_smooth(x, alpha, p)
    xpq = (q - 1.0) / ((eta**q + rho**q) ** (2.0 / q))
    return xpq + ((x**2 + alpha**2) ** (p / 2.0 - 1.0)) / (lp**p + beta**p)


def _finite_diff_grad(
    x: np.ndarray, alpha: float, beta: float, eta: float, p: float, q: float, eps: float = 1e-7
) -> np.ndarray:
    g = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (spoq_penalty(xp, alpha, beta, eta, p, q) - spoq_penalty(xm, alpha, beta, eta, p, q)) / (
            2.0 * eps
        )
    return g


def test_lp_lq_deterministic_values() -> None:
    x = np.array([0.0, 1.0, -2.0], dtype=float)
    alpha, eta, p, q = 0.1, 0.2, 1.0, 2.0
    lp_expected = ((0.1 - 0.1) + ((1.01) ** 0.5 - 0.1) + ((4.01) ** 0.5 - 0.1)) ** 1.0
    lq_expected = (0.2**2 + 0.0 + 1.0 + 4.0) ** 0.5
    assert np.isclose(lp_smooth(x, alpha, p), lp_expected, rtol=1e-12, atol=1e-12)
    assert np.isclose(lq_smooth(x, eta, q), lq_expected, rtol=1e-12, atol=1e-12)


def test_spoq_penalty_deterministic_value() -> None:
    x = np.array([1.0, -2.0], dtype=float)
    alpha, beta, eta, p, q = 0.5, 0.3, 0.2, 1.0, 2.0
    lp_power = np.sum((x**2 + alpha**2) ** (p / 2.0) - alpha**p)
    lq = (eta**q + np.sum(np.abs(x) ** q)) ** (1.0 / q)
    expected = np.log((lp_power + beta**p) ** (1.0 / p) / lq)
    assert np.isclose(spoq_penalty(x, alpha, beta, eta, p, q), expected, rtol=1e-12, atol=1e-12)


def test_gradient_matches_finite_difference_random_cases() -> None:
    rng = np.random.default_rng(42)
    cases = [
        (0.1, 0.05, 0.2, 1.0, 2.0),
        (1e-3, 0.1, 0.3, 0.75, 2.5),
        (0.2, 0.2, 0.1, 1.5, 3.0),
    ]
    for alpha, beta, eta, p, q in cases:
        x = rng.normal(size=8)
        g_analytic = spoq_grad(x, alpha, beta, eta, p, q)
        g_fd = _finite_diff_grad(x, alpha, beta, eta, p, q, eps=1e-7)
        assert np.allclose(g_analytic, g_fd, rtol=5e-5, atol=5e-6)


def test_python_matches_matlab_core_formulas() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=10)
    alpha, beta, eta, p, q = 7e-7, 3e-3, 1e-1, 0.75, 2.0
    rho = np.linalg.norm(x, ord=q)

    assert np.isclose(lp_smooth(x, alpha, p), _matlab_lp_smooth(x, alpha, p), rtol=1e-12, atol=1e-12)
    assert np.isclose(lq_smooth(x, eta, q), _matlab_lq_smooth(x, eta, q), rtol=1e-12, atol=1e-12)
    assert np.allclose(
        spoq_grad(x, alpha, beta, eta, p, q),
        _matlab_gradlplq(x, alpha, beta, eta, p, q),
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.allclose(
        spoq_metric_diag(x, alpha, beta, eta, p, q, rho),
        _matlab_condlplq(x, alpha, beta, eta, p, q, rho),
        rtol=1e-12,
        atol=1e-12,
    )


def test_metric_is_strictly_positive() -> None:
    x = np.array([0.0, 0.0, 1e-12, -1e-10], dtype=float)
    diag = spoq_metric_diag(x, alpha=1e-3, beta=1e-2, eta=1e-1, p=0.75, q=2.0, rho=0.0)
    assert np.all(diag > 0.0)
