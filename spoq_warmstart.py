"""Warm-start routine ported from MATLAB `Tools/pds.m`."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from spoq_prox import proxl2


Array = NDArray[np.float64]


@dataclass(frozen=True)
class WarmStartResult:
    """Result bundle for the MATLAB-style warm-start routine."""

    x: Array
    refspec: Array
    iterations: int
    converged: bool


def _as_vector(x: Array | list[float] | tuple[float, ...], *, name: str) -> Array:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector.")
    return arr


def _as_matrix(K: Array | list[list[float]], *, name: str) -> Array:
    arr = np.asarray(K, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    return arr


def proxl1(x: Array | list[float], w: float) -> Array:
    """Port of MATLAB `proxl1.m`.

    This is soft-thresholding followed by clipping to nonnegative values,
    exactly as written in the toolbox.
    """
    x_arr = _as_vector(x, name="x")
    if w < 0.0:
        raise ValueError("w must be non-negative.")
    p = np.zeros_like(x_arr)
    pos = x_arr > w
    p[pos] = x_arr[pos] - w
    neg = x_arr < -w
    p[neg] = x_arr[neg] + w
    p[p < 0.0] = 0.0
    return p


def norm2_estimate(K: Array | list[list[float]], n: int, seed: int = 0) -> float:
    """Port of MATLAB `norm2.m` using deterministic RNG."""
    K_arr = _as_matrix(K, name="K")
    if K_arr.shape[1] != n:
        raise ValueError("n must match the number of columns of K.")
    rng = np.random.default_rng(seed)
    b = rng.random(n, dtype=np.float64)
    nbiter = 50
    for _ in range(nbiter):
        tmp = K_arr.T.dot(K_arr.dot(b))
        tmpnorm = np.linalg.norm(tmp, ord=2)
        if tmpnorm == 0.0:
            return 0.0
        b = tmp / tmpnorm
    return float(np.linalg.norm(K_arr.dot(b), ord=2))


def pds_warmstart(
    K: Array | list[list[float]],
    y: Array | list[float],
    eta: float,
    nbiter: int,
    *,
    norm_seed: int = 0,
) -> WarmStartResult:
    """Port of MATLAB `pds.m` with faithful parameter/stopping logic."""
    K_arr = _as_matrix(K, name="K")
    y_arr = _as_vector(y, name="y")
    m, n = K_arr.shape
    if y_arr.shape != (m,):
        raise ValueError("y shape must match the number of rows of K.")
    if eta < 0.0:
        raise ValueError("eta must be non-negative.")
    if nbiter <= 0:
        raise ValueError("nbiter must be positive.")

    normK = norm2_estimate(K_arr, n, seed=norm_seed)
    if normK == 0.0:
        x = np.ones(n, dtype=np.float64)
        return WarmStartResult(x=x, refspec=np.zeros(nbiter, dtype=np.float64), iterations=0, converged=True)

    tau = 1.0 / normK
    sigma = 0.9 / (tau * normK**2)
    ro = 1.0
    refspec = np.zeros(nbiter, dtype=np.float64)
    xk_old = np.ones(n, dtype=np.float64)
    uk_old = K_arr.dot(xk_old)
    prec = 1e-6

    for i in range(nbiter):
        xxk = proxl1(xk_old - tau * (K_arr.T.dot(uk_old)), tau)
        zk = uk_old + sigma * (K_arr.dot(2.0 * xxk - xk_old))
        uuk = zk - sigma * proxl2(zk / sigma, y_arr, eta)
        xk = xk_old + ro * (xxk - xk_old)
        uk = uk_old + ro * (uuk - uk_old)

        norm_xk = np.linalg.norm(xk, ord=2)
        norm_uk = np.linalg.norm(uk, ord=2)
        ex = (np.linalg.norm(xk - xk_old, ord=2) ** 2) / (norm_xk**2) if norm_xk > 0.0 else np.inf
        eu = (np.linalg.norm(uk - uk_old, ord=2) ** 2) / (norm_uk**2) if norm_uk > 0.0 else np.inf
        if ex < prec and eu < prec:
            return WarmStartResult(x=xk, refspec=refspec, iterations=i + 1, converged=True)

        refspec[i] = ex
        xk_old = xk
        uk_old = uk

    return WarmStartResult(x=xk_old, refspec=refspec, iterations=nbiter, converged=False)
