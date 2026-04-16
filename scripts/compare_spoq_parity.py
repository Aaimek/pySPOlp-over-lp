"""Compare MATLAB and Python SPOQ parity traces and generate plots."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


def _load_case(root: Path, case_name: str) -> dict[str, np.ndarray]:
    case_dir = root / case_name
    return {
        "x_trace": np.loadtxt(case_dir / "x_trace.csv", delimiter=","),
        "psi_trace": np.loadtxt(case_dir / "psi_trace.csv", delimiter=",").reshape(-1),
        "step_norms": np.loadtxt(case_dir / "step_norms.csv", delimiter=",").reshape(-1),
        "feasibility_residual": np.loadtxt(case_dir / "feasibility_residual.csv", delimiter=",").reshape(-1),
        "min_component": np.loadtxt(case_dir / "min_component.csv", delimiter=",").reshape(-1),
    }


def _plot_metric(py: np.ndarray, matlab: np.ndarray, ylabel: str, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    its = np.arange(py.shape[0])
    ax.plot(its, py, "o-", linewidth=2.0, label="Python")
    ax.plot(its, matlab, "s--", linewidth=2.0, label="MATLAB")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_trajectory(py: np.ndarray, matlab: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(py[:, 0], py[:, 1], "o-", linewidth=2.0, label="Python")
    ax.plot(matlab[:, 0], matlab[:, 1], "s--", linewidth=2.0, label="MATLAB")
    ax.scatter(py[0, 0], py[0, 1], c="red", s=50, label="start")
    ax.scatter(py[-1, 0], py[-1, 1], c="cyan", s=50, label="python end")
    ax.scatter(matlab[-1, 0], matlab[-1, 1], c="orange", s=50, label="matlab end")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    root = Path("spoq_parity_outputs")
    py_root = root / "python"
    matlab_root = root / "matlab"
    out_root = root / "comparison"
    out_root.mkdir(parents=True, exist_ok=True)

    case_names = ["identity_2d", "nonidentity_2d"]
    missing = [case for case in case_names if not (matlab_root / case).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing MATLAB outputs for cases: "
            + ", ".join(missing)
            + ". Run `matlab_spoq_parity_run` in MATLAB first."
        )

    report_lines = [
        "# SPOQ MATLAB vs Python Parity Report",
        "",
        "This report compares the full solver traces exported by Python and MATLAB.",
        "",
    ]

    for case_name in case_names:
        py = _load_case(py_root, case_name)
        matlab = _load_case(matlab_root, case_name)
        case_dir = out_root / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        _plot_metric(py["psi_trace"], matlab["psi_trace"], "Psi(x_k)", case_dir / "psi_vs_iteration.png", f"{case_name}: Psi")
        _plot_metric(
            py["feasibility_residual"],
            matlab["feasibility_residual"],
            "||D x_k - y|| - xi",
            case_dir / "feasibility_vs_iteration.png",
            f"{case_name}: feasibility residual",
        )
        _plot_metric(py["step_norms"], matlab["step_norms"], "||x_k - x_{k-1}||", case_dir / "step_norms_vs_iteration.png", f"{case_name}: step norms")
        _plot_metric(
            py["min_component"],
            matlab["min_component"],
            "min component",
            case_dir / "min_component_vs_iteration.png",
            f"{case_name}: minimum component",
        )
        if py["x_trace"].shape[1] == 2:
            _plot_trajectory(py["x_trace"], matlab["x_trace"], case_dir / "trajectory_overlay.png", f"{case_name}: trajectory")

        final_dx = float(np.linalg.norm(py["x_trace"][-1] - matlab["x_trace"][-1]))
        final_dpsi = float(abs(py["psi_trace"][-1] - matlab["psi_trace"][-1]))
        final_dfeas = float(abs(py["feasibility_residual"][-1] - matlab["feasibility_residual"][-1]))
        max_path_dx = float(np.max(np.linalg.norm(py["x_trace"] - matlab["x_trace"], axis=1)))
        max_path_dpsi = float(np.max(np.abs(py["psi_trace"] - matlab["psi_trace"])))

        summary = {
            "case_name": case_name,
            "final_x_difference_norm": final_dx,
            "final_psi_difference_abs": final_dpsi,
            "final_feasibility_difference_abs": final_dfeas,
            "max_path_x_difference_norm": max_path_dx,
            "max_path_psi_difference_abs": max_path_dpsi,
        }
        with (case_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        report_lines.extend(
            [
                f"## {case_name}",
                "",
                f"- Final `x` difference norm: `{final_dx:.6e}`",
                f"- Final `Psi` difference: `{final_dpsi:.6e}`",
                f"- Final feasibility difference: `{final_dfeas:.6e}`",
                f"- Max path `x` difference norm: `{max_path_dx:.6e}`",
                f"- Max path `Psi` difference: `{max_path_dpsi:.6e}`",
                "",
            ]
        )

    report_path = out_root / "parity_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Wrote comparison outputs to: {out_root}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
