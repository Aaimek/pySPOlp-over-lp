"""Python equivalent of MATLAB `Run_SPOQ_Recovery.m` for simulated data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

from load_spoq_data_simulated import SimulatedSpoqData, load_spoq_data_simulated
from spoq_solver import SolverHistory, run_spoq_solver
from spoq_viz import SpoqParams
from spoq_warmstart import WarmStartResult, pds_warmstart


@dataclass(frozen=True)
class RecoveryResult:
    """Container for toolbox-style recovery outputs."""

    x_init: np.ndarray
    xrec: np.ndarray
    history: SolverHistory
    warmstart: WarmStartResult | None
    snr_db: float
    elapsed_seconds: float
    output_dir: Path


def _snr_db(xtrue: np.ndarray, xrec: np.ndarray) -> float:
    err = np.sum((xrec - xtrue) ** 2)
    ref = np.sum(xtrue**2)
    if err == 0.0:
        return float(np.inf)
    return float(10.0 * np.log10(ref / err))


def _save_trace_csv(history: SolverHistory, out_dir: Path) -> None:
    x_trace = np.vstack(history.xs)
    np.savetxt(out_dir / "x_trace.csv", x_trace, delimiter=",")
    np.savetxt(out_dir / "psi_trace.csv", np.asarray(history.psi_values)[:, None], delimiter=",")
    np.savetxt(out_dir / "step_norms.csv", np.asarray(history.step_norms)[:, None], delimiter=",")
    np.savetxt(out_dir / "feasibility_residuals.csv", np.asarray(history.feasibility_residuals)[:, None], delimiter=",")
    np.savetxt(out_dir / "relative_errors.csv", np.asarray(history.relative_errors)[:, None], delimiter=",")
    np.savetxt(
        out_dir / "trust_region_radii.csv",
        np.asarray([np.nan if r is None else r for r in history.trust_region_radii], dtype=float)[:, None],
        delimiter=",",
    )
    np.savetxt(out_dir / "trust_region_shrinks.csv", np.asarray(history.trust_region_shrinks)[:, None], delimiter=",")


def _save_vector_csv(x: np.ndarray, out_path: Path) -> None:
    np.savetxt(out_path, np.asarray(x, dtype=float)[:, None], delimiter=",")


def _plot_reconstruction(xtrue: np.ndarray, xrec: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xtrue, "r-o", linewidth=1.0, markersize=2.0, label="Original signal")
    ax.plot(xrec, "b--", linewidth=1.5, label="Estimated signal")
    ax.grid(True, alpha=0.3)
    ax.set_title("Reconstruction results")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_convergence(history: SolverHistory, xtrue: np.ndarray, out_path: Path) -> None:
    x_trace = np.vstack(history.xs)
    snr_curve = np.array([_snr_db(xtrue, x) for x in x_trace], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(snr_curve)), snr_curve, "-k", linewidth=2.0, label="TR-VMFB")
    ax.grid(True, alpha=0.3)
    ax.set_title("Algorithm convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("SNR (dB)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_spoq_recovery(
    data: SimulatedSpoqData | None = None,
    *,
    output_dir: str | Path = "spoq_simulated_recovery",
    max_iter: int | None = None,
    use_warm_start: bool = True,
    warm_start_iter: int = 10,
) -> RecoveryResult:
    """Run toolbox-style SPOQ recovery on the simulated dataset."""
    if data is None:
        data = load_spoq_data_simulated()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = SpoqParams(alpha=data.alpha, beta=data.beta, eta=data.eta, p=data.p, q=data.q)
    n_iter = data.nbiter if max_iter is None else int(max_iter)
    warmstart_result: WarmStartResult | None
    if use_warm_start:
        warmstart_result = pds_warmstart(data.K, data.y, data.xi, warm_start_iter)
        x0 = warmstart_result.x.copy()
    else:
        warmstart_result = None
        x0 = np.zeros_like(data.xtrue)

    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)
        xrec, history = run_spoq_solver(
            x0=x0,
            D=data.K,
            y=data.y,
            eta=data.xi,
            params=params,
            max_iter=n_iter,
            gamma=1.0,
            metric_mode=2,
            prox_max_iter=5000,
            prox_prec=1e-12,
        )
    elapsed = time.perf_counter() - start
    snr_db = _snr_db(data.xtrue, xrec)

    _plot_reconstruction(data.xtrue, xrec, out_dir / "reconstruction.png")
    _plot_convergence(history, data.xtrue, out_dir / "convergence_snr.png")
    _save_trace_csv(history, out_dir)
    _save_vector_csv(x0, out_dir / "x_init.csv")
    _save_vector_csv(xrec, out_dir / "xrec.csv")
    if warmstart_result is not None:
        np.savetxt(out_dir / "warmstart_refspec.csv", warmstart_result.refspec[:, None], delimiter=",")

    metadata = {
        "snr_db": snr_db,
        "elapsed_seconds": elapsed,
        "max_iter_requested": n_iter,
        "metric_mode": 2,
        "gamma": 1.0,
        "n_iter_recorded": len(history.xs) - 1,
        "use_warm_start": use_warm_start,
        "warm_start_iter": warm_start_iter,
        "warm_start_converged": None if warmstart_result is None else warmstart_result.converged,
        "warm_start_iterations_recorded": None if warmstart_result is None else warmstart_result.iterations,
        "data": data.to_metadata(),
    }
    with (out_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return RecoveryResult(
        x_init=x0,
        xrec=xrec,
        history=history,
        warmstart=warmstart_result,
        snr_db=snr_db,
        elapsed_seconds=elapsed,
        output_dir=out_dir,
    )


def main() -> None:
    data = load_spoq_data_simulated()
    result = run_spoq_recovery(data)
    print(f"Reconstruction in {len(result.history.xs) - 1} iterations")
    print(f"SNR = {result.snr_db:.6f} dB")
    print(f"Reconstruction time is {result.elapsed_seconds:.6f} seconds")
    print(f"Outputs written to: {result.output_dir}")


if __name__ == "__main__":
    main()
