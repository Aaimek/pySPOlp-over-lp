"""Minimal forward-backward SPOQ solver without trust-region logic."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from spoq_core import spoq_metric_diag, spoq_penalty, spoq_grad
from spoq_prox import prox_ppxa_plus
from spoq_viz import SpoqParams, spoq_2d_grid_values


Array = NDArray[np.float64]


@dataclass(frozen=True)
class SolverHistory:
    """Stored history of the outer forward-backward iterations."""

    xs: list[Array]
    psi_values: list[float]
    feasibility_residuals: list[float]
    step_norms: list[float]


def _as_vector(x: Array | list[float] | tuple[float, ...], *, name: str) -> Array:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector.")
    return arr


def run_spoq_solver(
    x0: Array | list[float] | tuple[float, ...],
    D: Array | list[list[float]],
    y: Array | list[float] | tuple[float, ...],
    eta: float,
    params: SpoqParams,
    max_iter: int,
    gamma: float,
    *,
    prox_max_iter: int = 5000,
    prox_prec: float = 1e-12,
    rho: float = 0.0,
) -> tuple[Array, SolverHistory]:
    """Run a minimal SPOQ forward-backward solver.

    The update follows:
    1. compute gradient of the SPOQ penalty
    2. compute diagonal metric A
    3. build descent proposal u = x - gamma * grad / A
    4. compute x_next via the validated PPXA+ prox routine

    No trust-region logic is used here. By default, ``rho=0`` is used in the
    metric, which corresponds to the simplest majorant choice from stage 1.
    """
    if eta < 0.0:
        raise ValueError("eta must be non-negative.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if gamma <= 0.0:
        raise ValueError("gamma must be strictly positive.")

    x = _as_vector(x0, name="x0").copy()
    D_arr = np.asarray(D, dtype=np.float64)
    y_arr = _as_vector(y, name="y")

    if D_arr.ndim != 2:
        raise ValueError("D must be a two-dimensional matrix.")
    m, n = D_arr.shape
    if x.shape != (n,):
        raise ValueError("x0 shape must match the number of columns of D.")
    if y_arr.shape != (m,):
        raise ValueError("y shape must match the number of rows of D.")

    history_xs = [x.copy()]
    history_psi = [spoq_penalty(x, params.alpha, params.beta, params.eta, params.p, params.q)]
    history_feas = [float(max(np.linalg.norm(D_arr @ x - y_arr) - eta, 0.0))]
    history_steps = [0.0]

    for _ in range(max_iter):
        grad = spoq_grad(x, params.alpha, params.beta, params.eta, params.p, params.q)
        metric_diag = spoq_metric_diag(x, params.alpha, params.beta, params.eta, params.p, params.q, rho=rho)
        B = metric_diag / gamma
        u = x - gamma * grad / metric_diag
        prox_result = prox_ppxa_plus(D_arr, B, u, y_arr, eta, J=prox_max_iter, prec=prox_prec)
        x_next = prox_result.z

        history_xs.append(x_next.copy())
        history_psi.append(spoq_penalty(x_next, params.alpha, params.beta, params.eta, params.p, params.q))
        history_feas.append(float(max(np.linalg.norm(D_arr @ x_next - y_arr) - eta, 0.0)))
        history_steps.append(float(np.linalg.norm(x_next - x)))
        x = x_next

    history = SolverHistory(
        xs=history_xs,
        psi_values=history_psi,
        feasibility_residuals=history_feas,
        step_norms=history_steps,
    )
    return x, history


def plot_solver_trajectory_on_spoq_contour(
    history: SolverHistory,
    params: SpoqParams,
    *,
    x_range: tuple[float, float] = (-1.0, 1.0),
    y_range: tuple[float, float] = (-1.0, 1.0),
    grid_size: int = 201,
    title: str = "SPOQ contour with solver trajectory",
) -> tuple[Figure, Axes]:
    """Overlay a 2D trajectory on a true SPOQ contour plot."""
    if len(history.xs[0]) != 2:
        raise ValueError("Trajectory contour overlay is only supported for 2D iterates.")

    x1 = np.linspace(x_range[0], x_range[1], grid_size)
    x2 = np.linspace(y_range[0], y_range[1], grid_size)
    X1, X2, Z = spoq_2d_grid_values(x1, x2, params)

    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(X1, X2, Z, levels=45, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="SPOQ([x1,x2])")

    traj = np.vstack(history.xs)
    ax.plot(traj[:, 0], traj[:, 1], "w-o", linewidth=1.5, markersize=3.0)
    ax.scatter(traj[0, 0], traj[0, 1], c="red", s=50, label="start")
    ax.scatter(traj[-1, 0], traj[-1, 1], c="cyan", s=50, label="end")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend()
    return fig, ax
