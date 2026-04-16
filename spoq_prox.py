"""Inner proximal machinery ported from the MATLAB SPOQ toolbox.

This module ports:
- ``proxl2.m``
- ``proxB.m``
- ``proxPPXAplus.m``

The implementation stays close to the MATLAB algorithmic structure while
making the optimization meaning explicit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


@dataclass(frozen=True)
class PpxaResult:
    """Result bundle for the inner PPXA+ iterations."""

    z: Array
    iterations: int
    converged: bool
    final_error: float


def _as_float_vector(x: Array | list[float] | tuple[float, ...], *, name: str) -> Array:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    return arr


def _as_float_matrix(mat: Array | list[list[float]], *, name: str) -> Array:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    return arr


def proxl2(x: Array | list[float], y: Array | list[float], eta: float) -> Array:
    """Project ``x`` onto the Euclidean ball centered at ``y`` with radius ``eta``.

    MATLAB reference: ``Tools/proxl2.m``.
    """
    x_arr = _as_float_vector(x, name="x")
    y_arr = _as_float_vector(y, name="y")
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape.")
    if eta < 0.0:
        raise ValueError("eta must be non-negative.")

    t = x_arr - y_arr
    norm_t = np.linalg.norm(t, ord=2)
    if norm_t == 0.0:
        return x_arr.copy()
    s = t * min(eta / norm_t, 1.0)
    return x_arr + s - t


def proxB(B: Array | list[float], x: Array | list[float], xhat: Array | list[float], teta: float) -> Array:
    """Compute the diagonal-weighted prox with nonnegativity clipping.

    This matches MATLAB ``Tools/proxB.m``:
    ``p = (x+teta*(B.*xhat))./(1+teta*B); p(p<0)=0;``

    Interpreted optimization problem:
    minimize_u  0.5 * ||u - x||_2^2 + (teta/2) * ||u - xhat||_B^2 + I_{u >= 0}(u)
    where ``B`` is a diagonal metric encoded by its diagonal entries.
    """
    B_arr = _as_float_vector(B, name="B")
    x_arr = _as_float_vector(x, name="x")
    xhat_arr = _as_float_vector(xhat, name="xhat")
    if not (B_arr.shape == x_arr.shape == xhat_arr.shape):
        raise ValueError("B, x, and xhat must have the same shape.")
    if np.any(B_arr < 0.0):
        raise ValueError("B must contain non-negative diagonal entries.")
    if teta <= 0.0:
        raise ValueError("teta must be strictly positive.")

    p = (x_arr + teta * (B_arr * xhat_arr)) / (1.0 + teta * B_arr)
    return np.maximum(p, 0.0)


def prox_ppxa_plus(
    D: Array | list[list[float]],
    B: Array | list[float],
    x: Array | list[float],
    y: Array | list[float],
    eta: float,
    J: int,
    prec: float,
    *,
    teta: float = 1.9,
) -> PpxaResult:
    """Compute the inner prox using the PPXA+ iterations from MATLAB.

    MATLAB reference: ``Tools/proxPPXAplus.m``.

    The routine computes the proximity step associated with the sum of:
    - a nonnegative diagonal-weighted quadratic term handled by ``proxB``
    - an Euclidean-ball data-consistency term on ``D z`` handled by ``proxl2``
    """
    D_arr = _as_float_matrix(D, name="D")
    B_arr = _as_float_vector(B, name="B")
    x_arr = _as_float_vector(x, name="x")
    y_arr = _as_float_vector(y, name="y")

    m, n = D_arr.shape
    if x_arr.shape != (n,):
        raise ValueError("x must have shape (n,) matching D.shape[1].")
    if B_arr.shape != (n,):
        raise ValueError("B must have shape (n,) matching D.shape[1].")
    if y_arr.shape != (m,):
        raise ValueError("y must have shape (m,) matching D.shape[0].")
    if eta < 0.0:
        raise ValueError("eta must be non-negative.")
    if J <= 0:
        raise ValueError("J must be a positive integer.")
    if prec < 0.0:
        raise ValueError("prec must be non-negative.")
    if teta <= 0.0:
        raise ValueError("teta must be strictly positive.")

    x1k_old = x_arr.copy()
    x2k_old = D_arr @ x1k_old
    A = np.linalg.inv(np.eye(n, dtype=np.float64) + D_arr.T @ D_arr)
    zk_old = A @ (x1k_old + D_arr.T @ x2k_old)
    final_error = np.inf

    for j in range(1, J + 1):
        y1k_old = proxB(B_arr, x1k_old, x_arr, teta)
        y2k_old = proxl2(x2k_old, y_arr, eta)
        vk_old = A @ (y1k_old + D_arr.T @ y2k_old)
        x1k = x1k_old + 2.0 * vk_old - zk_old - y1k_old
        x2k = x2k_old + D_arr @ (2.0 * vk_old - zk_old) - y2k_old
        zk = vk_old
        final_error = float(np.linalg.norm(zk - zk_old, ord=2) ** 2)
        if final_error < prec:
            return PpxaResult(z=zk, iterations=j, converged=True, final_error=final_error)
        x1k_old = x1k
        x2k_old = x2k
        zk_old = zk

    return PpxaResult(z=zk_old, iterations=J, converged=False, final_error=final_error)
