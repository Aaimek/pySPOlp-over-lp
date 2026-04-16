"""SPOQ outer solver logic, including MATLAB-style metric modes."""

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
    relative_errors: list[float]
    metric_mode: int
    metric_diags: list[Array]
    trust_region_radii: list[float | None]
    trust_region_shrinks: list[int]


def _as_vector(x: Array | list[float] | tuple[float, ...], *, name: str) -> Array:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector.")
    return arr


def compute_lipschitz(alpha: float, beta: float, eta: float, p: float, q: float, n: int) -> float:
    """Compute the MATLAB `ComputeLipschitz.m` constant."""
    l1 = p * alpha ** (p - 2.0) / beta**p
    l2 = p / (2.0 * alpha**2) * max(1.0, (n * alpha**p / beta**p) ** 2)
    l3 = (q - 1.0) / eta**2
    return float(l1 + l2 + l3)


def _relative_error(x_new: Array, x_old: Array) -> float:
    """Match MATLAB stopping criterion as closely as practical."""
    denom = float(np.linalg.norm(x_old) ** 2)
    numer = float(np.linalg.norm(x_new - x_old) ** 2)
    if denom > 0.0:
        return numer / denom
    if numer == 0.0:
        return 0.0
    return float(np.inf)


def run_spoq_solver(
    x0: Array | list[float] | tuple[float, ...],
    D: Array | list[list[float]],
    y: Array | list[float] | tuple[float, ...],
    eta: float,
    params: SpoqParams,
    max_iter: int,
    gamma: float,
    *,
    metric_mode: int = 1,
    prox_max_iter: int = 5000,
    prox_prec: float = 1e-12,
    rho: float = 0.0,
) -> tuple[Array, SolverHistory]:
    """Run the SPOQ outer solver with MATLAB-style metric modes.

    Metric modes follow `FB_PPXALpLq.m`:
    - `0`: global Lipschitz metric
    - `1`: variable metric without trust region
    - `2`: trust-region variable metric
    """
    if eta < 0.0:
        raise ValueError("eta must be non-negative.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if gamma <= 0.0:
        raise ValueError("gamma must be strictly positive.")
    if metric_mode not in (0, 1, 2):
        raise ValueError("metric_mode must be one of {0, 1, 2}.")

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

    lipschitz = compute_lipschitz(params.alpha, params.beta, params.eta, params.p, params.q, n)

    history_xs = [x.copy()]
    history_psi = [spoq_penalty(x, params.alpha, params.beta, params.eta, params.p, params.q)]
    history_feas = [float(max(np.linalg.norm(D_arr @ x - y_arr) - eta, 0.0))]
    history_steps = [0.0]
    history_errors = [0.0]
    history_metric_diags = [np.full(n, np.nan, dtype=np.float64)]
    history_radii: list[float | None] = [None]
    history_shrinks = [0]

    for _ in range(max_iter):
        grad = spoq_grad(x, params.alpha, params.beta, params.eta, params.p, params.q)
        bwhile = 0

        if metric_mode == 0:
            metric_diag = np.full(n, lipschitz, dtype=np.float64)
            B = metric_diag / gamma
            u = x - (1.0 / B) * grad
            x_next = prox_ppxa_plus(D_arr, B, u, y_arr, eta, J=prox_max_iter, prec=prox_prec).z
            accepted_radius: float | None = None
        elif metric_mode == 1:
            metric_diag = spoq_metric_diag(x, params.alpha, params.beta, params.eta, params.p, params.q, rho=0.0)
            B = metric_diag / gamma
            u = x - (1.0 / B) * grad
            x_next = prox_ppxa_plus(D_arr, B, u, y_arr, eta, J=prox_max_iter, prec=prox_prec).z
            accepted_radius = 0.0
        else:
            ro = float(np.sum(np.abs(x) ** params.q) ** (1.0 / params.q))
            while True:
                metric_diag = spoq_metric_diag(x, params.alpha, params.beta, params.eta, params.p, params.q, rho=ro)
                B = metric_diag / gamma
                u = x - (1.0 / B) * grad
                x_next = prox_ppxa_plus(D_arr, B, u, y_arr, eta, J=prox_max_iter, prec=prox_prec).z
                if float(np.sum(np.abs(x_next) ** params.q) ** (1.0 / params.q)) < ro:
                    ro = ro / 2.0
                    bwhile += 1
                else:
                    break
            accepted_radius = ro

        history_xs.append(x_next.copy())
        history_psi.append(spoq_penalty(x_next, params.alpha, params.beta, params.eta, params.p, params.q))
        history_feas.append(float(max(np.linalg.norm(D_arr @ x_next - y_arr) - eta, 0.0)))
        history_steps.append(float(np.linalg.norm(x_next - x)))
        history_errors.append(_relative_error(x_next, x))
        history_metric_diags.append(metric_diag.copy())
        history_radii.append(accepted_radius)
        history_shrinks.append(bwhile)

        if history_errors[-1] < prox_prec:
            x = x_next
            break
        x = x_next

    history = SolverHistory(
        xs=history_xs,
        psi_values=history_psi,
        feasibility_residuals=history_feas,
        step_norms=history_steps,
        relative_errors=history_errors,
        metric_mode=metric_mode,
        metric_diags=history_metric_diags,
        trust_region_radii=history_radii,
        trust_region_shrinks=history_shrinks,
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
