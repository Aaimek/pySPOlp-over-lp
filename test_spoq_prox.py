"""Tests for stage-3 proximal machinery."""

from __future__ import annotations

import numpy as np

from spoq_prox import prox_ppxa_plus, proxB, proxl2


def _objective(z: np.ndarray, B: np.ndarray, x: np.ndarray) -> float:
    """Objective minimized by the inner prox over the feasible set."""
    return float(0.5 * np.sum(B * (z - x) ** 2))


def _is_feasible(z: np.ndarray, D: np.ndarray, y: np.ndarray, eta: float, tol: float = 1e-6) -> bool:
    """Check nonnegativity and l2-ball feasibility."""
    return bool(np.all(z >= -tol) and np.linalg.norm(D @ z - y) <= eta + tol)


def _random_feasible_point(
    rng: np.random.Generator,
    D: np.ndarray,
    y: np.ndarray,
    eta: float,
    *,
    trials: int = 5000,
    scale: float = 2.0,
) -> np.ndarray:
    """Generate a feasible point by rejection sampling in small dimensions."""
    n = D.shape[1]
    for _ in range(trials):
        z = rng.uniform(0.0, scale, size=n)
        if np.linalg.norm(D @ z - y) <= eta:
            return z
    raise RuntimeError("Failed to sample a feasible point.")


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


def test_prox_ppxa_plus_nonidentity_operator_feasible_and_beats_random_feasible_point() -> None:
    """Validate coupling through non-identity D and compare objective values."""
    rng = np.random.default_rng(7)
    D = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float)
    B = np.array([1.5, 0.7], dtype=float)
    x = np.array([0.9, -0.2], dtype=float)
    y = np.array([0.6, 0.2], dtype=float)
    eta = 0.45

    result = prox_ppxa_plus(D, B, x, y, eta=eta, J=5000, prec=1e-14)
    random_feasible = _random_feasible_point(rng, D, y, eta, scale=1.5)

    assert result.converged
    assert _is_feasible(result.z, D, y, eta, tol=2e-6)
    assert _objective(result.z, B, x) <= _objective(random_feasible, B, x) + 1e-8


def test_prox_ppxa_plus_objective_not_worse_than_many_random_feasible_points() -> None:
    """Check the returned point beats multiple feasible comparison points."""
    rng = np.random.default_rng(21)
    D = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float)
    B = np.array([2.0, 0.8], dtype=float)
    x = np.array([1.1, -0.1], dtype=float)
    y = np.array([0.5, 0.3], dtype=float)
    eta = 0.55

    result = prox_ppxa_plus(D, B, x, y, eta=eta, J=5000, prec=1e-14)
    assert result.converged
    assert _is_feasible(result.z, D, y, eta, tol=2e-6)

    result_obj = _objective(result.z, B, x)
    for _ in range(25):
        z_feas = _random_feasible_point(rng, D, y, eta, scale=1.5)
        assert result_obj <= _objective(z_feas, B, x) + 1e-8


def test_prox_ppxa_plus_solution_stabilizes_across_iteration_budgets() -> None:
    """Verify convergence stability when J increases."""
    D = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float)
    B = np.array([1.2, 2.5], dtype=float)
    x = np.array([0.8, -0.3], dtype=float)
    y = np.array([0.4, 0.25], dtype=float)
    eta = 0.4

    r500 = prox_ppxa_plus(D, B, x, y, eta=eta, J=500, prec=1e-14)
    r2000 = prox_ppxa_plus(D, B, x, y, eta=eta, J=2000, prec=1e-14)
    r5000 = prox_ppxa_plus(D, B, x, y, eta=eta, J=5000, prec=1e-14)

    assert r500.converged and r2000.converged and r5000.converged
    assert np.allclose(r500.z, r2000.z, atol=2e-6, rtol=2e-6)
    assert np.allclose(r2000.z, r5000.z, atol=2e-6, rtol=2e-6)


def test_prox_ppxa_plus_hits_l2_and_nonnegativity_boundaries_simultaneously() -> None:
    """Construct a case whose solution lies on both active boundaries."""
    D = np.eye(2, dtype=float)
    B = np.array([1.0, 1.0], dtype=float)
    x = np.array([-1.0, -1.0], dtype=float)
    y = np.array([0.8, 0.0], dtype=float)
    eta = 0.3

    result = prox_ppxa_plus(D, B, x, y, eta=eta, J=5000, prec=1e-14)

    assert result.converged
    assert _is_feasible(result.z, D, y, eta, tol=5e-6)
    assert np.isclose(np.linalg.norm(D @ result.z - y), eta, atol=5e-6, rtol=5e-6)
    assert np.isclose(result.z[1], 0.0, atol=5e-6, rtol=0.0)


def test_prox_ppxa_plus_randomized_small_stress_cases() -> None:
    """Stress small random problems for feasibility, nonnegativity, and convergence."""
    rng = np.random.default_rng(1234)
    for _ in range(6):
        n = int(rng.integers(2, 4))
        m = n
        D = rng.normal(size=(m, n))
        B = rng.uniform(0.5, 2.0, size=n)
        x = rng.normal(size=n)
        z_anchor = rng.uniform(0.0, 1.0, size=n)
        y = D @ z_anchor + rng.normal(scale=0.05, size=m)
        eta = float(np.linalg.norm(D @ z_anchor - y) + 0.15)

        result = prox_ppxa_plus(D, B, x, y, eta=eta, J=5000, prec=1e-12)

        assert result.converged
        assert np.all(result.z >= -2e-6)
        assert np.linalg.norm(D @ result.z - y) <= eta + 5e-5
