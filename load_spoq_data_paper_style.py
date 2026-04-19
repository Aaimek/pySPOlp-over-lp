"""Synthetic sparse nonnegative deconvolution data in the spirit of Cherni et al. (2020).

These are **not** exact reproductions of the paper's experimental pipeline (e.g. no full
averaging chemistry simulation). They use the same Toeplitz/Pascal peak construction as
`load_spoq_data_simulated`, Gaussian noise, and feasibility radius ``xi = sqrt(N) * sigma``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from numpy.typing import NDArray

from load_spoq_data_simulated import _pascal_matrix, _toeplitz_from_peak_shape

Array = NDArray[np.float64]


@dataclass(frozen=True)
class PaperStyleSpoqData:
    """Paper-style simulated instance (sparse nonnegative, Gaussian noise, fixed xi rule)."""

    xtrue: Array
    K: Array
    y: Array
    y0: Array
    noise: Array
    sigma: float
    xi: float
    n_sample: int
    n_peak: int
    peak_width: int
    seed: int
    label: str
    xtrue_location: Array
    xtrue_amplitude: Array
    peak_shape: Array

    def to_metadata(self) -> dict[str, object]:
        meta = asdict(self)
        for key, value in meta.items():
            if isinstance(value, np.ndarray):
                meta[key] = value.tolist()
        return meta


def load_paper_style_spoq_data(
    *,
    n_sample: int,
    n_peak: int,
    peak_width: int = 5,
    seed: int = 0,
    label: str = "paper-style",
) -> PaperStyleSpoqData:
    """Build a length-``n_sample`` sparse nonnegative signal, Toeplitz blur, and noisy data.

    - ``P`` spikes at random locations with amplitudes uniform on ``[0, 1)``.
    - Noise: i.i.d. Gaussian ``N(0, sigma^2)`` with ``sigma = 0.5 * max(|y0|) / 100`` (same relative
      level as the MATLAB simulated loader).
    - Feasibility: ``xi = sqrt(n_sample) * sigma`` (as requested; differs from the toolbox's
      ``1.1 * sqrt(n) * sigma`` rule).
    """
    if n_peak <= 0 or n_peak > n_sample:
        raise ValueError("n_peak must satisfy 0 < n_peak <= n_sample.")
    if peak_width <= 0 or peak_width > n_sample:
        raise ValueError("peak_width must be positive and not exceed n_sample.")

    rng = np.random.default_rng(seed)

    xtrue = np.zeros(n_sample, dtype=np.float64)
    xtrue_location = rng.permutation(n_sample)[:n_peak]
    xtrue_amplitude = rng.random(n_peak, dtype=np.float64)
    xtrue[xtrue_location] = xtrue_amplitude

    peak_matrix = _pascal_matrix(peak_width)
    peak_shape = np.diag(np.fliplr(peak_matrix)).astype(np.float64)
    peak_shape = peak_shape / np.sum(peak_shape)
    peak_shape_filled = np.concatenate((peak_shape, np.zeros(n_sample - peak_width, dtype=np.float64)))
    K = _toeplitz_from_peak_shape(peak_shape_filled)

    y0 = K.dot(xtrue)
    noise = rng.standard_normal(n_sample, dtype=np.float64)
    sigma = float(0.5 * np.max(np.abs(y0)) / 100.0)
    y = y0 + sigma * noise
    xi = float(np.sqrt(float(n_sample)) * sigma)

    return PaperStyleSpoqData(
        xtrue=xtrue,
        K=K,
        y=y,
        y0=y0,
        noise=noise,
        sigma=sigma,
        xi=xi,
        n_sample=n_sample,
        n_peak=n_peak,
        peak_width=peak_width,
        seed=seed,
        label=label,
        xtrue_location=xtrue_location.astype(np.int64),
        xtrue_amplitude=xtrue_amplitude,
        peak_shape=peak_shape,
    )


def load_paper_style_dataset_a(seed: int = 0) -> PaperStyleSpoqData:
    """Paper-style preset A: N=1000, P=48 (not an exact paper experimental instance)."""
    return load_paper_style_spoq_data(
        n_sample=1000,
        n_peak=48,
        seed=seed,
        label="Paper-style A (N=1000, P=48)",
    )


def load_paper_style_dataset_b(seed: int = 0) -> PaperStyleSpoqData:
    """Paper-style preset B: N=1000, P=94 (not an exact paper experimental instance)."""
    return load_paper_style_spoq_data(
        n_sample=1000,
        n_peak=94,
        seed=seed,
        label="Paper-style B (N=1000, P=94)",
    )
