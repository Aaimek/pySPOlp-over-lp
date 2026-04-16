"""Tests for stage-3 proximal machinery."""

from __future__ import annotations

import numpy as np

from spoq_prox import prox_ppxa_plus, proxB, proxl2


def test_proxl2_inside_ball_returns_input() -> None:
    x = np.array([1.0, 2.0], dtype=float)
    y = np.array([1.1, 2.1], dtype=float)
    eta = 0.5
    z = proxl2(x, y, eta)
    assert np.allclose(z, x, atol=1e-12, rtol=1e-12)


def test_proxl2_outside_ball_projects_to_boundary() -> None:
    x = np.array([3.0, 0.0], dtype=float)
    y = np.array([0.0, 0.0], dtype=float)
    eta = 1.5
    z = proxl2(x, y, eta)
    expected = np.array([1.5, 0.0], dtype=float)
    assert np.allclose(z, expected, atol=1e-12, rtol=1e-12)
    assert np.isclose(np.linalg.norm(z - y), eta, atol=1e-12, rtol=1e-12)


def test_proxB_matches_closed_form_and_nonnegativity() -> None:
    B = np.array([2.0, 0.5, 3.0], dtype=float)
    x = np.array([-1.0, 4.0, -2.0], dtype=float)
    xhat = np.array([3.0, -1.0, 1.0], dtype=float)
    teta = 2.0
    z = proxB(B, x, xhat, teta)
    expected = (x + teta * (B * xhat)) / (1.0 + teta * B)
    expected = np.maximum(expected, 0.0)
    assert np.allclose(z, expected, atol=1e-12, rtol=1e-12)
    assert np.all(z >= 0.0)


def test_prox_ppxa_plus_identity_eta_zero_returns_single_feasible_point() -> None:
    D = np.eye(2, dtype=float)
    B = np.array([1.0, 4.0], dtype=float)
    x = np.array([10.0, -3.0], dtype=float)
    y = np.array([0.25, 1.75], dtype=float)
    result = prox_ppxa_plus(D, B, x, y, eta=0.0, J=2000, prec=1e-14)
    assert result.converged
    assert np.allclose(result.z, y, atol=2e-6, rtol=1e-6)


def _bruteforce_weighted_projection_identity(
    x: np.ndarray,
    y: np.ndarray,
    eta: float,
    B: np.ndarray,
    lo: float = -0.2,
    hi: float = 1.6,
    num: int = 1201,
) -> np.ndarray:
    grid = np.linspace(lo, hi, num)
    best_z = None
    best_obj = np.inf
    for z0 in grid:
        for z1 in grid:
            z = np.array([z0, z1], dtype=float)
            if np.any(z < 0.0):
                continue
            if np.linalg.norm(z - y) > eta + 1e-12:
                continue
            obj = 0.5 * np.sum(B * (z - x) ** 2)
            if obj < best_obj:
                best_obj = obj
                best_z = z.copy()
    assert best_z is not None
    return best_z


def test_prox_ppxa_plus_matches_bruteforce_small_identity_case() -> None:
    D = np.eye(2, dtype=float)
    B = np.array([1.0, 3.0], dtype=float)
    x = np.array([1.2, -0.4], dtype=float)
    y = np.array([0.4, 0.5], dtype=float)
    eta = 0.35

    result = prox_ppxa_plus(D, B, x, y, eta=eta, J=4000, prec=1e-14)
    brute = _bruteforce_weighted_projection_identity(x=x, y=y, eta=eta, B=B)

    assert result.converged
    assert np.all(result.z >= -1e-10)
    assert np.linalg.norm(result.z - y) <= eta + 2e-6
    assert np.allclose(result.z, brute, atol=3e-3, rtol=3e-3)
