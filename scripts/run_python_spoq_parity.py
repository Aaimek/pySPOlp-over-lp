"""Run small deterministic Python parity experiments for the SPOQ solver."""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spoq_core import spoq_penalty
from spoq_solver import run_spoq_solver
from spoq_viz import SpoqParams


def _problem_definitions() -> dict[str, dict[str, object]]:
    params = dict(alpha=1e-3, beta=5e-2, eta=2e-1, p=1.0, q=2.0)
    return {
        "identity_2d": {
            "D": np.eye(2, dtype=float),
            "y": np.array([0.45, 0.05], dtype=float),
            "x0": np.array([0.90, 0.70], dtype=float),
            "xi": 0.30,
            "gamma": 1.0,
            "max_iter": 12,
            "params": params,
        },
        "nonidentity_2d": {
            "D": np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float),
            "y": np.array([0.55, 0.10], dtype=float),
            "x0": np.array([0.85, 0.40], dtype=float),
            "xi": 0.28,
            "gamma": 1.0,
            "max_iter": 12,
            "params": params,
        },
    }


def _write_trace(case_dir: Path, D: np.ndarray, y: np.ndarray, xi: float, x_hist: np.ndarray, psi_hist: np.ndarray) -> None:
    step_norms = np.zeros(x_hist.shape[0], dtype=float)
    step_norms[1:] = np.linalg.norm(np.diff(x_hist, axis=0), axis=1)
    feas = np.linalg.norm((D @ x_hist.T).T - y[None, :], axis=1) - xi
    min_component = np.min(x_hist, axis=1)

    np.savetxt(case_dir / "x_trace.csv", x_hist, delimiter=",")
    np.savetxt(case_dir / "psi_trace.csv", psi_hist[:, None], delimiter=",")
    np.savetxt(case_dir / "step_norms.csv", step_norms[:, None], delimiter=",")
    np.savetxt(case_dir / "feasibility_residual.csv", feas[:, None], delimiter=",")
    np.savetxt(case_dir / "min_component.csv", min_component[:, None], delimiter=",")


def main() -> None:
    out_root = Path("spoq_parity_outputs") / "python"
    out_root.mkdir(parents=True, exist_ok=True)

    for case_name, cfg in _problem_definitions().items():
        case_dir = out_root / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        D = np.asarray(cfg["D"], dtype=float)
        y = np.asarray(cfg["y"], dtype=float)
        x0 = np.asarray(cfg["x0"], dtype=float)
        xi = float(cfg["xi"])
        gamma = float(cfg["gamma"])
        max_iter = int(cfg["max_iter"])
        params = SpoqParams(**cfg["params"])

        sol, history = run_spoq_solver(
            x0=x0,
            D=D,
            y=y,
            eta=xi,
            params=params,
            max_iter=max_iter,
            gamma=gamma,
            prox_max_iter=20000,
            prox_prec=1e-16,
        )

        x_hist = np.vstack(history.xs)
        psi_hist = np.array(
            [spoq_penalty(x, params.alpha, params.beta, params.eta, params.p, params.q) for x in history.xs],
            dtype=float,
        )
        _write_trace(case_dir, D, y, xi, x_hist, psi_hist)

        metadata = {
            "case_name": case_name,
            "D": D.tolist(),
            "y": y.tolist(),
            "x0": x0.tolist(),
            "xi": xi,
            "gamma": gamma,
            "max_iter": max_iter,
            "params": cfg["params"],
            "final_x": sol.tolist(),
            "final_psi": float(psi_hist[-1]),
        }
        with (case_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    print(f"Wrote Python parity traces to: {out_root}")


if __name__ == "__main__":
    main()
