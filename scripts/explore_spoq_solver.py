"""Explore the minimal SPOQ solver across multiple parameter settings.

This script:
- runs the solver for several `(p, q, gamma, x0)` configurations
- saves static matplotlib contour/trajectory plots
- saves interactive Plotly contour/trajectory plots
- optionally saves interactive 3D surface/trajectory plots
- records `Psi(x_k)` and iterate trajectories for each run
- writes a short markdown summary with simple observations
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from spoq_solver import plot_solver_trajectory_on_spoq_contour, run_spoq_solver
from spoq_viz import SpoqParams, spoq_2d_grid_values


@dataclass(frozen=True)
class Config:
    p: float
    q: float
    gamma: float
    init_id: str
    x0: np.ndarray


def _safe_tag(value: float) -> str:
    return str(value).replace(".", "p")


def _config_dir(root: Path, cfg: Config) -> Path:
    return root / f"p_{_safe_tag(cfg.p)}__q_{_safe_tag(cfg.q)}__gamma_{_safe_tag(cfg.gamma)}__init_{cfg.init_id}"


def _plot_psi_curve(psi_values: list[float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(psi_values)), psi_values, "o-", linewidth=2.0, markersize=4.0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Psi(x_k)")
    ax.set_title("SPOQ penalty along iterations")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_gamma_comparison(gamma_to_psi: dict[float, list[float]], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for gamma, psi_values in sorted(gamma_to_psi.items()):
        ax.plot(np.arange(len(psi_values)), psi_values, "o-", linewidth=1.8, markersize=3.5, label=f"gamma={gamma}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Psi(x_k)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plotly_contour_with_trajectory(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    params: SpoqParams,
    trajectory: np.ndarray,
    title: str,
) -> go.Figure:
    x1 = np.linspace(x_range[0], x_range[1], 161)
    x2 = np.linspace(y_range[0], y_range[1], 161)
    X1, X2, Z = spoq_2d_grid_values(x1, x2, params)

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=X1[0, :],
            y=X2[:, 0],
            z=Z,
            colorscale="Viridis",
            contours=dict(showlabels=False),
            showscale=True,
            colorbar=dict(title="SPOQ"),
            opacity=0.9,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            mode="lines+markers",
            name="trajectory",
            line=dict(color="white", width=3),
            marker=dict(size=6, color=np.arange(len(trajectory)), colorscale="Plasma", showscale=False),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[trajectory[0, 0]],
            y=[trajectory[0, 1]],
            mode="markers",
            name="start",
            marker=dict(size=12, color="red", symbol="circle"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[trajectory[-1, 0]],
            y=[trajectory[-1, 1]],
            mode="markers",
            name="end",
            marker=dict(size=12, color="cyan", symbol="diamond"),
        )
    )
    fig.update_layout(title=title, xaxis_title="x1", yaxis_title="x2")
    return fig


def _plotly_surface_with_trajectory(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    params: SpoqParams,
    trajectory: np.ndarray,
    title: str,
) -> go.Figure:
    x1 = np.linspace(x_range[0], x_range[1], 121)
    x2 = np.linspace(y_range[0], y_range[1], 121)
    X1, X2, Z = spoq_2d_grid_values(x1, x2, params)
    traj_z = np.array(
        [spoq_2d_grid_values(np.array([pt[0]]), np.array([pt[1]]), params)[2][0, 0] for pt in trajectory],
        dtype=float,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=X1,
            y=X2,
            z=Z,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="SPOQ"),
            opacity=0.9,
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            z=traj_z,
            mode="lines+markers",
            name="trajectory",
            line=dict(color="white", width=6),
            marker=dict(size=4, color=np.arange(len(trajectory)), colorscale="Plasma"),
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="SPOQ"),
    )
    return fig


def _make_initializations() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(2026)
    return {
        "diag_pos": np.array([0.9, 0.7], dtype=float),
        "axis_x1": np.array([1.0, 0.05], dtype=float),
        "rand_mixed": rng.normal(loc=0.35, scale=0.45, size=2).astype(float),
    }


def main() -> None:
    output_root = Path("spoq_solver_exploration")
    output_root.mkdir(exist_ok=True)

    p_values = [1.0, 0.5, 0.25]
    q_values = [2.0, 3.0]
    gammas = [0.5, 1.0, 1.5]
    initializations = _make_initializations()

    # Small 2D problem so trajectories can be overlaid directly on contours.
    D = np.eye(2, dtype=float)
    y = np.array([0.45, 0.0], dtype=float)
    eta = 0.3
    max_iter = 18
    prox_max_iter = 6000
    prox_prec = 1e-14

    run_summaries: list[dict[str, object]] = []
    gamma_groups: dict[tuple[float, float, str], dict[float, list[float]]] = {}

    for p in p_values:
        for q in q_values:
            params = SpoqParams(alpha=7e-7, beta=3e-3, eta=1e-1, p=p, q=q)
            for init_id, x0 in initializations.items():
                key = (p, q, init_id)
                gamma_groups[key] = {}
                for gamma in gammas:
                    cfg = Config(p=p, q=q, gamma=gamma, init_id=init_id, x0=x0.copy())
                    cfg_dir = _config_dir(output_root, cfg)
                    cfg_dir.mkdir(parents=True, exist_ok=True)

                    sol, history = run_spoq_solver(
                        x0=cfg.x0,
                        D=D,
                        y=y,
                        eta=eta,
                        params=params,
                        max_iter=max_iter,
                        gamma=gamma,
                        prox_max_iter=prox_max_iter,
                        prox_prec=prox_prec,
                    )
                    trajectory = np.vstack(history.xs)

                    x_pad = 0.15
                    x_min = min(np.min(trajectory[:, 0]), x0[0], sol[0], y[0] - eta) - x_pad
                    x_max = max(np.max(trajectory[:, 0]), x0[0], sol[0], y[0] + eta) + x_pad
                    y_min = min(np.min(trajectory[:, 1]), x0[1], sol[1], y[1] - eta) - x_pad
                    y_max = max(np.max(trajectory[:, 1]), x0[1], sol[1], y[1] + eta) + x_pad
                    x_range = (float(x_min), float(x_max))
                    y_range = (float(y_min), float(y_max))

                    # Static contour with trajectory.
                    fig, _ = plot_solver_trajectory_on_spoq_contour(
                        history,
                        params,
                        x_range=x_range,
                        y_range=y_range,
                        grid_size=181,
                        title=f"SPOQ trajectory: p={p}, q={q}, gamma={gamma}, init={init_id}",
                    )
                    fig.tight_layout()
                    fig.savefig(cfg_dir / "trajectory_contour.png", dpi=220)
                    plt.close(fig)

                    # Psi curve.
                    _plot_psi_curve(history.psi_values, cfg_dir / "psi_vs_iteration.png")

                    # Interactive contour.
                    contour_fig = _plotly_contour_with_trajectory(
                        x_range=x_range,
                        y_range=y_range,
                        params=params,
                        trajectory=trajectory,
                        title=f"Interactive contour: p={p}, q={q}, gamma={gamma}, init={init_id}",
                    )
                    contour_fig.write_html(cfg_dir / "trajectory_contour_interactive.html", include_plotlyjs="cdn")

                    # Interactive 3D surface.
                    surface_fig = _plotly_surface_with_trajectory(
                        x_range=x_range,
                        y_range=y_range,
                        params=params,
                        trajectory=trajectory,
                        title=f"Interactive surface: p={p}, q={q}, gamma={gamma}, init={init_id}",
                    )
                    surface_fig.write_html(cfg_dir / "trajectory_surface_interactive.html", include_plotlyjs="cdn")

                    metadata = {
                        "p": p,
                        "q": q,
                        "gamma": gamma,
                        "init_id": init_id,
                        "x0": cfg.x0.tolist(),
                        "solution": sol.tolist(),
                        "psi_initial": history.psi_values[0],
                        "psi_final": history.psi_values[-1],
                        "psi_decrease": history.psi_values[0] - history.psi_values[-1],
                        "max_feasibility_residual_after_step0": float(max(history.feasibility_residuals[1:])),
                        "num_iterations": len(history.xs) - 1,
                    }
                    with (cfg_dir / "run_summary.json").open("w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2)

                    run_summaries.append(metadata)
                    gamma_groups[key][gamma] = history.psi_values

                # Bonus: compare gamma values for the same (p, q, init).
                _plot_gamma_comparison(
                    gamma_groups[key],
                    output_root / f"compare_gamma__p_{_safe_tag(p)}__q_{_safe_tag(q)}__init_{init_id}.png",
                    title=f"Gamma comparison: p={p}, q={q}, init={init_id}",
                )

    # Write a short observed-behavior summary.
    by_p = {}
    by_gamma = {}
    for item in run_summaries:
        by_p.setdefault(item["p"], []).append(item["psi_decrease"])
        by_gamma.setdefault(item["gamma"], []).append(item["psi_decrease"])

    summary_lines = [
        "# SPOQ Solver Exploration Summary",
        "",
        "This summary is based on the runs generated by `explore_spoq_solver.py`.",
        "",
        "## Observed behaviors",
        "",
    ]
    for p in sorted(by_p):
        vals = np.asarray(by_p[p], dtype=float)
        summary_lines.append(
            f"- `p={p}`: mean Psi decrease `{vals.mean():.6f}`, min `{vals.min():.6f}`, max `{vals.max():.6f}`."
        )
    for gamma in sorted(by_gamma):
        vals = np.asarray(by_gamma[gamma], dtype=float)
        summary_lines.append(
            f"- `gamma={gamma}`: mean Psi decrease `{vals.mean():.6f}`, min `{vals.min():.6f}`, max `{vals.max():.6f}`."
        )
    summary_lines.extend(
        [
            "",
            "## Qualitative notes",
            "",
            "- Lower `p` tends to produce sharper contour geometry near the axes, which often makes trajectories bend more strongly toward axis-aligned sparse regions.",
            "- Larger `gamma` typically makes early steps longer; depending on initialization, this can accelerate progress or create more oscillatory paths before settling.",
            "- Different initializations can converge along noticeably different transient paths even on the same contour landscape.",
            "- Feasibility is enforced by the prox step, so trajectories are driven into the Euclidean-ball data-consistency region after the first updates.",
        ]
    )
    (output_root / "exploration_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Wrote exploration outputs to: {output_root}")
    print(f"Generated {len(run_summaries)} configuration folders.")
    print(f"Summary: {output_root / 'exploration_summary.md'}")


if __name__ == "__main__":
    main()
