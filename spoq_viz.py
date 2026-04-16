"""Visualization utilities for the true SPOQ penalty.

All SPOQ evaluations route through the validated core implementation
in ``spoq_core.spoq_penalty``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from spoq_core import spoq_penalty


@dataclass(frozen=True)
class SpoqParams:
    """SPOQ hyperparameters and exponents."""

    alpha: float
    beta: float
    eta: float
    p: float
    q: float


SOOT_PRESET = SpoqParams(alpha=7e-7, beta=3e-3, eta=1e-1, p=1.0, q=2.0)
FOLDED_PRESET = SpoqParams(alpha=7e-7, beta=3e-3, eta=1e-1, p=0.25, q=2.0)


def spoq_1d_values(x: np.ndarray, params: SpoqParams) -> np.ndarray:
    """Evaluate true SPOQ on vectors [x_i] in 1D."""
    x = np.asarray(x, dtype=np.float64).ravel()
    vals = np.empty_like(x, dtype=np.float64)
    for i, xv in enumerate(x):
        vals[i] = spoq_penalty(
            np.array([xv], dtype=np.float64),
            alpha=params.alpha,
            beta=params.beta,
            eta=params.eta,
            p=params.p,
            q=params.q,
        )
    return vals


def spoq_2d_grid_values(x1: np.ndarray, x2: np.ndarray, params: SpoqParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate true SPOQ on 2D vectors [x1, x2] over a meshgrid."""
    x1 = np.asarray(x1, dtype=np.float64).ravel()
    x2 = np.asarray(x2, dtype=np.float64).ravel()
    X1, X2 = np.meshgrid(x1, x2, indexing="xy")
    Z = np.empty_like(X1, dtype=np.float64)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = spoq_penalty(
                np.array([X1[i, j], X2[i, j]], dtype=np.float64),
                alpha=params.alpha,
                beta=params.beta,
                eta=params.eta,
                p=params.p,
                q=params.q,
            )
    return X1, X2, Z


def check_even_symmetry_2d(X1: np.ndarray, X2: np.ndarray, Z: np.ndarray, params: SpoqParams) -> dict[str, float]:
    """Numerically check even/sign and swap symmetries on sampled points."""
    rng = np.random.default_rng(0)
    nrows, ncols = Z.shape
    errs_sign = []
    errs_swap = []
    for _ in range(min(100, nrows * ncols)):
        i = int(rng.integers(0, nrows))
        j = int(rng.integers(0, ncols))
        x = np.array([X1[i, j], X2[i, j]], dtype=np.float64)
        z0 = Z[i, j]
        z_sign = spoq_penalty(-x, params.alpha, params.beta, params.eta, params.p, params.q)
        z_swap = spoq_penalty(np.array([x[1], x[0]], dtype=np.float64), params.alpha, params.beta, params.eta, params.p, params.q)
        errs_sign.append(abs(z0 - z_sign))
        errs_swap.append(abs(z0 - z_swap))
    return {
        "max_abs_err_sign_flip": float(np.max(errs_sign) if errs_sign else 0.0),
        "max_abs_err_swap": float(np.max(errs_swap) if errs_swap else 0.0),
        "mean_abs_err_sign_flip": float(np.mean(errs_sign) if errs_sign else 0.0),
        "mean_abs_err_swap": float(np.mean(errs_swap) if errs_swap else 0.0),
    }


def plot_spoq_1d(
    x: np.ndarray,
    params_list: Iterable[SpoqParams],
    labels: Iterable[str] | None = None,
    title: str = "SPOQ 1D profile",
) -> tuple[Figure, Axes]:
    """Create a static 1D SPOQ plot."""
    x = np.asarray(x, dtype=np.float64).ravel()
    fig, ax = plt.subplots(figsize=(7, 4))
    labels_seq = list(labels) if labels is not None else None
    for idx, params in enumerate(params_list):
        y = spoq_1d_values(x, params)
        label = labels_seq[idx] if labels_seq is not None else f"p={params.p}, q={params.q}"
        ax.plot(x, y, linewidth=2.0, label=label)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("SPOQ([x])")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig, ax


def plot_spoq_2d_contour(
    x1: np.ndarray,
    x2: np.ndarray,
    params: SpoqParams,
    levels: int = 40,
    title: str | None = None,
) -> tuple[Figure, Axes, np.ndarray, np.ndarray, np.ndarray]:
    """Create static 2D contour plot of true SPOQ([x1, x2])."""
    X1, X2, Z = spoq_2d_grid_values(x1, x2, params)
    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(X1, X2, Z, levels=levels, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="SPOQ([x1,x2])")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title or f"SPOQ contour (p={params.p}, q={params.q})")
    return fig, ax, X1, X2, Z


def plot_spoq_3d_surface(
    x1: np.ndarray,
    x2: np.ndarray,
    params: SpoqParams,
    title: str | None = None,
) -> tuple[Figure, Axes, np.ndarray, np.ndarray, np.ndarray]:
    """Create static 3D surface of true SPOQ([x1, x2])."""
    X1, X2, Z = spoq_2d_grid_values(x1, x2, params)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X1, X2, Z, cmap="viridis", linewidth=0.0, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1, label="SPOQ([x1,x2])")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("SPOQ")
    ax.set_title(title or f"SPOQ surface (p={params.p}, q={params.q})")
    return fig, ax, X1, X2, Z
