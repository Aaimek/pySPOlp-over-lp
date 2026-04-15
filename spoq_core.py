"""Core SPOQ penalty definitions from Cherni et al. (2020).

This module implements the smooth lp and lq terms and the SPOQ penalty
defined around Eq. (7)-(9), together with its gradient from Eq. (15)-(16).
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import NDArray

ArrayLike = Union[NDArray[np.floating], list[float], tuple[float, ...]]


def _validate_params(alpha: float, beta: float, eta: float, p: float, q: float) -> None:
    if alpha <= 0.0:
        raise ValueError("alpha must be strictly positive.")
    if beta <= 0.0:
        raise ValueError("beta must be strictly positive.")
    if eta <= 0.0:
        raise ValueError("eta must be strictly positive.")
    if not (0.0 < p < 2.0):
        raise ValueError("p must satisfy 0 < p < 2.")
    if q < 2.0:
        raise ValueError("q must satisfy q >= 2.")


def _as_1d_array(x: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("x must be a one-dimensional vector.")
    return arr


def lp_smooth(x: ArrayLike, alpha: float, p: float) -> float:
    """Compute smoothed lp term from Eq. (7): ||x||_{p,alpha}."""
    x_arr = _as_1d_array(x)
    if alpha <= 0.0:
        raise ValueError("alpha must be strictly positive.")
    if not (0.0 < p < 2.0):
        raise ValueError("p must satisfy 0 < p < 2.")

    lp_power = np.sum((x_arr * x_arr + alpha * alpha) ** (0.5 * p) - alpha**p)
    lp_power = max(lp_power, 0.0)  # guard tiny negative from floating-point roundoff
    return float(lp_power ** (1.0 / p))


def lq_smooth(x: ArrayLike, eta: float, q: float) -> float:
    """Compute smoothed lq term from Eq. (8): ||x||_{q,eta}."""
    x_arr = _as_1d_array(x)
    if eta <= 0.0:
        raise ValueError("eta must be strictly positive.")
    if q < 2.0:
        raise ValueError("q must satisfy q >= 2.")

    lq_power = eta**q + np.sum(np.abs(x_arr) ** q)
    return float(lq_power ** (1.0 / q))


def spoq_penalty(x: ArrayLike, alpha: float, beta: float, eta: float, p: float, q: float) -> float:
    """Compute SPOQ penalty from Eq. (9)."""
    _validate_params(alpha=alpha, beta=beta, eta=eta, p=p, q=q)
    x_arr = _as_1d_array(x)

    lp_power = np.sum((x_arr * x_arr + alpha * alpha) ** (0.5 * p) - alpha**p)
    lp_power = max(lp_power, 0.0)
    lq_power = eta**q + np.sum(np.abs(x_arr) ** q)

    numerator = (lp_power + beta**p) ** (1.0 / p)
    denominator = lq_power ** (1.0 / q)
    return float(np.log(numerator / denominator))


def spoq_grad(
    x: ArrayLike, alpha: float, beta: float, eta: float, p: float, q: float
) -> NDArray[np.float64]:
    """Compute gradient of SPOQ from Eq. (15)-(16)."""
    _validate_params(alpha=alpha, beta=beta, eta=eta, p=p, q=q)
    x_arr = _as_1d_array(x)

    sq = x_arr * x_arr + alpha * alpha
    lp_power = np.sum(sq ** (0.5 * p) - alpha**p)
    lp_power = max(lp_power, 0.0)
    lq_power = eta**q + np.sum(np.abs(x_arr) ** q)

    grad_p = x_arr * (sq ** (0.5 * p - 1.0)) / (lp_power + beta**p)
    grad_q = np.sign(x_arr) * (np.abs(x_arr) ** (q - 1.0)) / lq_power
    return grad_p - grad_q


def spoq_metric_diag(
    x: ArrayLike,
    alpha: float,
    beta: float,
    eta: float,
    p: float,
    q: float,
    rho: float,
) -> NDArray[np.float64]:
    """Compute diagonal of local majorizing metric from Eq. (25)-(26)."""
    _validate_params(alpha=alpha, beta=beta, eta=eta, p=p, q=q)
    if rho < 0.0:
        raise ValueError("rho must be non-negative.")
    x_arr = _as_1d_array(x)

    lp_power = np.sum((x_arr * x_arr + alpha * alpha) ** (0.5 * p) - alpha**p)
    lp_power = max(lp_power, 0.0)

    chi = (q - 1.0) / ((eta**q + rho**q) ** (2.0 / q))
    diag_term = (x_arr * x_arr + alpha * alpha) ** (0.5 * p - 1.0) / (lp_power + beta**p)
    return chi + diag_term
