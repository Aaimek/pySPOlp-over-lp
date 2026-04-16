"""Python equivalent of MATLAB `Load_SPOQ_Data_Simulated.m`."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


@dataclass(frozen=True)
class SimulatedSpoqData:
    """Container for simulated SPOQ toolbox data and parameters."""

    xtrue: Array
    K: Array
    y: Array
    y0: Array
    noise: Array
    sigma: float
    xi: float
    eta: float
    alpha: float
    beta: float
    p: float
    q: float
    nbiter: int
    n_sample: int
    n_peak: int
    peak_width: int
    seed: int
    xtrue_location: Array
    xtrue_amplitude: Array
    peak_shape: Array

    def to_metadata(self) -> dict[str, object]:
        """Return JSON-serializable metadata."""
        meta = asdict(self)
        for key, value in meta.items():
            if isinstance(value, np.ndarray):
                meta[key] = value.tolist()
        return meta


def _pascal_matrix(n: int) -> Array:
    """Construct the symmetric Pascal matrix used by MATLAB `pascal(n)`."""
    mat = np.zeros((n, n), dtype=np.float64)
    mat[:, 0] = 1.0
    mat[0, :] = 1.0
    for i in range(1, n):
        for j in range(1, n):
            mat[i, j] = mat[i - 1, j] + mat[i, j - 1]
    return mat


def _toeplitz_from_peak_shape(peak_shape_filled: Array) -> Array:
    """Build the same Toeplitz matrix structure as the MATLAB script."""
    n = peak_shape_filled.size
    col = peak_shape_filled.copy()
    row = np.concatenate(([peak_shape_filled[0]], peak_shape_filled[:0:-1]))
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    return np.where(i >= j, col[i - j], row[j - i])


def load_spoq_data_simulated(seed: int = 0) -> SimulatedSpoqData:
    """Reproduce MATLAB simulated-data workflow as closely as practical."""
    n_sample = 500
    n_peak = 20
    peak_width = 5

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
    sigma = float(0.5 * np.max(y0) / 100.0)
    y = y0 + sigma * noise

    xi = float(1.1 * np.sqrt(n_sample) * sigma)
    eta = 2e-6
    alpha = 7e-7
    beta = 3e-2
    p = 0.75
    q = 2.0
    nbiter = 5000

    return SimulatedSpoqData(
        xtrue=xtrue,
        K=K,
        y=y,
        y0=y0,
        noise=noise,
        sigma=sigma,
        xi=xi,
        eta=eta,
        alpha=alpha,
        beta=beta,
        p=p,
        q=q,
        nbiter=nbiter,
        n_sample=n_sample,
        n_peak=n_peak,
        peak_width=peak_width,
        seed=seed,
        xtrue_location=xtrue_location.astype(np.int64),
        xtrue_amplitude=xtrue_amplitude,
        peak_shape=peak_shape,
    )
