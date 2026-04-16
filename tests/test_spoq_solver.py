"""Tests for the SPOQ solver, including MATLAB-style metric modes."""

from __future__ import annotations

import numpy as np

from spoq_solver import plot_solver_trajectory_on_spoq_contour, run_spoq_solver
from spoq_viz import SOOT_PRESET, SpoqParams


def test_run_spoq_solver_returns_expected_history_shapes() -> None:
    D = np.eye(2, dtype=float)
    y = np.array([0.5, 0.1], dtype=float)
    x0 = np.array([0.8, 0.3], dtype=float)
    sol, history = run_spoq_solver(
        x0=x0,
        D=D,
        y=y,
        eta=0.3,
        params=SOOT_PRESET,
        max_iter=8,
        gamma=1.0,
        metric_mode=1,
        prox_max_iter=2000,
        prox_prec=1e-30,
    )

    assert sol.shape == x0.shape
    assert len(history.xs) == 9
    assert len(history.psi_values) == 9
    assert len(history.feasibility_residuals) == 9
    assert len(history.step_norms) == 9
    assert len(history.relative_errors) == 9
    assert len(history.metric_diags) == 9
    assert len(history.trust_region_radii) == 9
    assert len(history.trust_region_shrinks) == 9
    assert history.metric_mode == 1


def test_run_spoq_solver_keeps_iterates_feasible() -> None:
    D = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float)
    y = np.array([0.45, 0.15], dtype=float)
    x0 = np.array([1.0, 0.2], dtype=float)
    eta = 0.35
    _, history = run_spoq_solver(
        x0=x0,
        D=D,
        y=y,
        eta=eta,
        params=SOOT_PRESET,
        max_iter=10,
        gamma=1.0,
        metric_mode=1,
        prox_max_iter=20000,
        prox_prec=1e-16,
    )

    assert all(res <= 5e-6 for res in history.feasibility_residuals[1:])
    assert all(np.all(x >= -5e-6) for x in history.xs[1:])


def test_run_spoq_solver_exhibits_near_monotonic_psi_decrease() -> None:
    params = SpoqParams(alpha=1e-3, beta=5e-2, eta=2e-1, p=1.0, q=2.0)
    D = np.eye(2, dtype=float)
    y = np.array([0.35, 0.0], dtype=float)
    x0 = np.array([0.9, 0.6], dtype=float)
    _, history = run_spoq_solver(
        x0=x0,
        D=D,
        y=y,
        eta=0.25,
        params=params,
        max_iter=12,
        gamma=0.8,
        metric_mode=1,
        prox_max_iter=3000,
        prox_prec=1e-12,
    )

    psi = np.array(history.psi_values)
    diffs = np.diff(psi)
    assert psi[-1] <= psi[0] + 1e-8
    assert np.max(diffs) <= 5e-4


def test_run_spoq_solver_reduces_step_sizes_over_iterations() -> None:
    D = np.eye(2, dtype=float)
    y = np.array([0.4, 0.2], dtype=float)
    x0 = np.array([1.2, 0.8], dtype=float)
    _, history = run_spoq_solver(
        x0=x0,
        D=D,
        y=y,
        eta=0.4,
        params=SOOT_PRESET,
        max_iter=10,
        gamma=1.0,
        metric_mode=1,
        prox_max_iter=2000,
        prox_prec=1e-12,
    )

    step_norms = np.array(history.step_norms[1:])
    assert step_norms[-1] <= step_norms[0] + 1e-8


def test_plot_solver_trajectory_on_spoq_contour_runs_for_2d_history() -> None:
    D = np.eye(2, dtype=float)
    y = np.array([0.5, 0.0], dtype=float)
    x0 = np.array([0.8, 0.4], dtype=float)
    _, history = run_spoq_solver(
        x0=x0,
        D=D,
        y=y,
        eta=0.35,
        params=SOOT_PRESET,
        max_iter=4,
        gamma=1.0,
        metric_mode=1,
        prox_max_iter=2000,
        prox_prec=1e-12,
    )

    fig, ax = plot_solver_trajectory_on_spoq_contour(history, SOOT_PRESET)
    assert fig is not None
    assert ax is not None
    fig.clf()


def test_all_metric_modes_run_and_produce_feasible_outputs() -> None:
    D = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float)
    y = np.array([0.45, 0.1], dtype=float)
    x0 = np.array([0.9, 0.6], dtype=float)
    eta = 0.35

    for metric_mode in (0, 1, 2):
        sol, history = run_spoq_solver(
            x0=x0,
            D=D,
            y=y,
            eta=eta,
            params=SOOT_PRESET,
            max_iter=6,
            gamma=1.0,
            metric_mode=metric_mode,
            prox_max_iter=15000,
            prox_prec=1e-16,
        )
        assert sol.shape == x0.shape
        assert history.metric_mode == metric_mode
        assert history.feasibility_residuals[-1] <= 5e-5
        assert np.min(sol) >= -5e-5


def test_metric_mode_one_matches_default_solver_behavior() -> None:
    D = np.eye(2, dtype=float)
    y = np.array([0.45, 0.05], dtype=float)
    x0 = np.array([0.9, 0.7], dtype=float)

    sol_default, hist_default = run_spoq_solver(
        x0=x0,
        D=D,
        y=y,
        eta=0.3,
        params=SOOT_PRESET,
        max_iter=6,
        gamma=1.0,
        prox_max_iter=15000,
        prox_prec=1e-16,
    )
    sol_mode1, hist_mode1 = run_spoq_solver(
        x0=x0,
        D=D,
        y=y,
        eta=0.3,
        params=SOOT_PRESET,
        max_iter=6,
        gamma=1.0,
        metric_mode=1,
        prox_max_iter=15000,
        prox_prec=1e-16,
    )

    assert np.allclose(sol_default, sol_mode1, atol=1e-12, rtol=1e-12)
    assert np.allclose(np.vstack(hist_default.xs), np.vstack(hist_mode1.xs), atol=1e-12, rtol=1e-12)
    assert np.allclose(hist_default.psi_values, hist_mode1.psi_values, atol=1e-12, rtol=1e-12)


def test_trust_region_mode_records_radius_and_shrink_behavior() -> None:
    params = SpoqParams(alpha=1e-3, beta=5e-2, eta=2e-1, p=1.0, q=2.0)
    D = np.eye(2, dtype=float)
    y = np.array([0.3, 0.0], dtype=float)
    x0 = np.array([1.2, 1.0], dtype=float)

    _, history = run_spoq_solver(
        x0=x0,
        D=D,
        y=y,
        eta=0.2,
        params=params,
        max_iter=6,
        gamma=1.0,
        metric_mode=2,
        prox_max_iter=15000,
        prox_prec=1e-16,
    )

    assert all(radius is None or radius >= 0.0 for radius in history.trust_region_radii)
    assert all(shrinks >= 0 for shrinks in history.trust_region_shrinks)
    assert any(radius is not None for radius in history.trust_region_radii[1:])
    for x, radius in zip(history.xs[1:], history.trust_region_radii[1:]):
        if radius is not None:
            assert np.linalg.norm(x, ord=params.q) + 5e-5 >= radius


def test_trust_region_mode_can_shrink_radius_on_challenging_case() -> None:
    params = SpoqParams(alpha=1e-3, beta=5e-2, eta=2e-1, p=1.0, q=2.0)
    D = np.eye(2, dtype=float)
    y = np.array([0.0, 0.0], dtype=float)
    x0 = np.array([2.0, 1.5], dtype=float)

    _, history = run_spoq_solver(
        x0=x0,
        D=D,
        y=y,
        eta=0.1,
        params=params,
        max_iter=4,
        gamma=1.0,
        metric_mode=2,
        prox_max_iter=15000,
        prox_prec=1e-16,
    )

    assert sum(history.trust_region_shrinks) >= 0
    assert max(history.trust_region_shrinks) >= 1
