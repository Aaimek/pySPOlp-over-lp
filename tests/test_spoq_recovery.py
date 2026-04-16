"""Smoke tests for simulated-data SPOQ recovery workflow."""

from __future__ import annotations

import numpy as np

from load_spoq_data_simulated import load_spoq_data_simulated
from run_spoq_recovery import run_spoq_recovery


def _snr_db(xtrue: np.ndarray, xrec: np.ndarray) -> float:
    err = np.sum((xrec - xtrue) ** 2)
    ref = np.sum(xtrue**2)
    if err == 0.0:
        return float(np.inf)
    return float(10.0 * np.log10(ref / err))


def test_simulated_loader_shapes_and_parameters() -> None:
    data = load_spoq_data_simulated(seed=0)
    assert data.xtrue.shape == (500,)
    assert data.K.shape == (500, 500)
    assert data.y.shape == (500,)
    assert data.n_peak == 20
    assert data.peak_width == 5
    assert data.eta == 2e-6
    assert data.alpha == 7e-7
    assert data.beta == 3e-2
    assert data.p == 0.75
    assert data.q == 2.0
    assert data.nbiter == 5000


def test_run_spoq_recovery_smoke_and_snr_improves(tmp_path) -> None:
    data = load_spoq_data_simulated(seed=0)
    x0 = np.zeros_like(data.xtrue)
    snr0 = _snr_db(data.xtrue, x0)

    result = run_spoq_recovery(data, output_dir=tmp_path / "recovery", max_iter=20)

    assert result.xrec.shape == data.xtrue.shape
    assert len(result.history.xs) >= 2
    assert result.output_dir.exists()
    assert (result.output_dir / "reconstruction.png").exists()
    assert (result.output_dir / "convergence_snr.png").exists()
    assert (result.output_dir / "x_trace.csv").exists()
    assert (result.output_dir / "x_init.csv").exists()
    assert (result.output_dir / "xrec.csv").exists()
    assert result.snr_db > snr0


def test_run_spoq_recovery_can_toggle_warm_start(tmp_path) -> None:
    data = load_spoq_data_simulated(seed=0)

    warm = run_spoq_recovery(
        data,
        output_dir=tmp_path / "warm",
        max_iter=5,
        use_warm_start=True,
        warm_start_iter=10,
    )
    cold = run_spoq_recovery(
        data,
        output_dir=tmp_path / "cold",
        max_iter=5,
        use_warm_start=False,
    )

    assert warm.warmstart is not None
    assert cold.warmstart is None
    assert np.linalg.norm(warm.x_init) > 0.0
    assert np.allclose(cold.x_init, 0.0, atol=0.0, rtol=0.0)
