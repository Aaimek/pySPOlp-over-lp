"""Streamlit app for exploring the SPOQ penalty family and solver."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webapp.app_plotting import (  # noqa: E402
    convergence_figure,
    penalty_1d_figure,
    penalty_2d_trajectory_figure,
    penalty_3d_trajectory_figure,
    signal_recovery_figure,
)
from webapp.app_utils import (  # noqa: E402
    DATASET_DATA_DRIVEN,
    DATASET_PAPER_STYLE_A,
    DATASET_PAPER_STYLE_B,
    create_penalty_params,
    make_problem_data,
    run_full_problem,
)


st.set_page_config(page_title="SPOQ Explorer", layout="wide")


def _trajectory_bounds(trajectory: np.ndarray, y: np.ndarray | None = None, xi: float | None = None) -> tuple[tuple[float, float], tuple[float, float]]:
    pad = 0.2
    xs = trajectory[:, 0]
    ys = trajectory[:, 1]
    if y is not None and xi is not None and y.shape[0] >= 2:
        xs = np.concatenate([xs, [y[0] - xi, y[0] + xi]])
        ys = np.concatenate([ys, [y[1] - xi, y[1] + xi]])
    return ((float(xs.min() - pad), float(xs.max() + pad)), (float(ys.min() - pad), float(ys.max() + pad)))


st.title("SPOQ Explorer")
st.caption("Local didactic interface for the SPOQ penalty family and the associated recovery solver.")

with st.sidebar:
    with st.expander("Section 1 — Mode", expanded=True):
        mode = st.radio(
            "Choose what to explore",
            options=["Penalty only", "Full recovery problem"],
            help="Penalty only visualizes the SPOQ penalty itself. Full recovery problem runs the validated solver on a dataset.",
        )

    with st.expander("Section 2 — Penalty Parameters", expanded=False):
        st.caption("Penalty-family controls (SPOQ parameters).")
        p = st.select_slider("p — sparsity exponent", options=[0.25, 0.5, 0.75, 1.0, 1.5], value=0.75)
        q = st.select_slider("q — normalization exponent", options=[2.0, 3.0, 4.0], value=2.0)
        alpha = st.number_input("alpha — smoothing for p-part", min_value=1e-9, value=7e-7, format="%.8g")
        beta = st.number_input("beta — numerator stabilizer", min_value=1e-9, value=3e-3, format="%.8g")
        eta = st.number_input("eta — smoothing for q-part", min_value=1e-9, value=1e-1, format="%.8g")

    params = create_penalty_params(p=p, q=q, alpha=alpha, beta=beta, eta=eta)

    with st.expander("Section 3 — Problem / Dataset Parameters", expanded=False):
        if mode == "Full recovery problem":
            st.caption("Dataset/problem definition (from didactic to realistic).")
            dataset_options = [
                "Toy 2D",
                "Identity toy problem",
                "Diagonal operator toy problem",
                "Simulated toolbox dataset",
                DATASET_PAPER_STYLE_A,
                DATASET_PAPER_STYLE_B,
            ]
            dataset_name = st.selectbox(
                "Dataset",
                options=dataset_options,
                index=0,
                help="2D toys suit contour/surface views. Simulated / paper-style use higher-dimensional Toeplitz convolution.",
            )
            dataset_description = {
                "Toy 2D": "Simple 2D didactic problem for visualizing trajectories.",
                "Identity toy problem": "2D test case with identity forward operator.",
                "Diagonal operator toy problem": "2D test case with diagonal forward operator.",
                "Simulated toolbox dataset": "Synthetic dataset inspired by the MATLAB toolbox (N=500); ξ from the toolbox rule.",
                DATASET_PAPER_STYLE_A: (
                    "Paper-style synthetic deconvolution (not an exact paper experiment): N=1000, P=48, nonnegative spikes, "
                    "Gaussian noise, default ξ = √N σ."
                ),
                DATASET_PAPER_STYLE_B: (
                    "Paper-style synthetic deconvolution (not an exact paper experiment): N=1000, P=94, nonnegative spikes, "
                    "Gaussian noise, default ξ = √N σ."
                ),
            }
            st.info(dataset_description.get(dataset_name, "Dataset description unavailable."))

            st.markdown(
                "\n".join(
                    [
                        "**Problem variables:**",
                        "- $x^{\\mathrm{true}}$ : true signal",
                        "- $y$ : observed data",
                        "- $\\xi$ : feasibility tolerance",
                        "- $D$ (or $K$) : forward operator",
                    ]
                )
            )

            data_driven = dataset_name in DATASET_DATA_DRIVEN
            is_paper_style = dataset_name in (DATASET_PAPER_STYLE_A, DATASET_PAPER_STYLE_B)

            if data_driven:
                st.caption("Coordinates and ξ are fixed by the dataset generator (not the toy fields below).")
                if dataset_name == "Simulated toolbox dataset":
                    st.info("Toolbox-style: Gaussian noise; $\\xi = 1.1\\,\\sqrt{N}\\,\\sigma$.")
                elif is_paper_style:
                    st.info(
                        "Paper-style (simplified): Gaussian noise; **$\\xi = \\sqrt{N}\\,\\sigma$** "
                        "(σ from the noise level). Not the full paper acquisition pipeline."
                    )
                toy_x0 = toy_x1 = toy_y0 = toy_y1 = 0.0
                toy_xi = 0.3
                diag_a = 1.0
                diag_b = 0.5
                simulated_seed = st.number_input("Random seed", min_value=0, value=0, step=1)
            elif dataset_name == "Toy 2D":
                st.markdown(r"**$x^{\mathrm{true}}[0]$**")
                toy_x0 = st.number_input("xtrue0_value", value=0.9, format="%.4f", label_visibility="collapsed")
                st.markdown(r"**$x^{\mathrm{true}}[1]$**")
                toy_x1 = st.number_input("xtrue1_value", value=0.6, format="%.4f", label_visibility="collapsed")
                st.markdown(r"**$y[0]$**")
                toy_y0 = st.number_input("y0_value", value=0.45, format="%.4f", label_visibility="collapsed")
                st.markdown(r"**$y[1]$**")
                toy_y1 = st.number_input("y1_value", value=0.0, format="%.4f", label_visibility="collapsed")
                toy_xi = st.number_input("Feasibility tolerance xi (\u03be)", min_value=1e-9, value=0.3, format="%.6g")
                diag_a = 1.0
                diag_b = 0.5
                simulated_seed = 0
            elif dataset_name == "Identity toy problem":
                st.caption(r"Fixed $x^{\mathrm{true}}=(0.8,0.2)$ and $y=(0.45,0.05)$; only $\xi$ is adjustable below.")
                toy_x0 = toy_x1 = toy_y0 = toy_y1 = 0.0
                toy_xi = st.number_input("Feasibility tolerance xi (\u03be)", min_value=1e-9, value=0.3, format="%.6g")
                diag_a = 1.0
                diag_b = 0.5
                simulated_seed = 0
            else:
                # Diagonal operator toy problem
                st.caption(r"Fixed $x^{\mathrm{true}}=(0.8,0.35)$; $y=Dx^{\mathrm{true}}$; set $D$ and $\xi$ below.")
                st.markdown("**Diagonal $D$**")
                diag_a = st.number_input("Operator diagonal entry (1,1)", value=1.0, format="%.4f")
                diag_b = st.number_input("Operator diagonal entry (2,2)", value=0.5, format="%.4f")
                toy_x0 = toy_x1 = toy_y0 = toy_y1 = 0.0
                toy_xi = st.number_input("Feasibility tolerance xi (\u03be)", min_value=1e-9, value=0.3, format="%.6g")
                simulated_seed = 0
        else:
            st.caption("Only used in Full recovery problem mode.")
            dataset_name = None
            toy_x0 = toy_x1 = toy_y0 = toy_y1 = toy_xi = diag_a = diag_b = 0.0
            simulated_seed = 0

    with st.expander("Section 4 — Solver Parameters", expanded=False):
        st.caption("Outer optimization controls (backend unchanged).")
        solver_mode = st.selectbox(
            "Solver mode",
            options=[0, 1, 2],
            index=2,
            format_func=lambda x: {0: "0 — global/fixed metric", 1: "1 — variable metric", 2: "2 — trust-region variable metric"}[x],
        )
        gamma = st.number_input("gamma", min_value=1e-4, value=1.0, format="%.6g")
        max_iter = st.slider("max iterations", min_value=5, max_value=500, value=40, step=5)
        use_warm_start = st.checkbox(
            "Use warm start",
            value=True,
            help="Uses the MATLAB-style pds warm-start routine before the outer SPOQ solver.",
        )
        warm_start_iter = st.slider("Warm-start iterations", min_value=1, max_value=50, value=10, step=1, disabled=not use_warm_start)

    with st.expander("Section 5 — Visualization Mode", expanded=False):
        st.caption("Display settings only (does not change solver numerics).")
        vis_mode = st.radio("Visualization", options=["1D", "2D", "3D"], index=1)
        st.markdown(r"Use **Compute and plot** to run backend first, then display $\Psi(x_k)$ and diagnostics.")
        run_clicked = st.button("Compute and plot", type="primary")


if "app_result" not in st.session_state:
    st.session_state["app_result"] = None


if run_clicked:
    if mode == "Penalty only":
        st.session_state["app_result"] = {
            "mode": mode,
            "params": params,
        }
    else:
        problem, _ = make_problem_data(
            dataset_name,
            toy_x0=toy_x0,
            toy_x1=toy_x1,
            toy_y0=toy_y0,
            toy_y1=toy_y1,
            toy_xi=toy_xi,
            diag_a=diag_a,
            diag_b=diag_b,
            simulated_seed=int(simulated_seed),
        )
        solver_result = run_full_problem(
            problem,
            params,
            solver_mode=solver_mode,
            gamma=gamma,
            max_iter=max_iter,
            use_warm_start=use_warm_start,
            warm_start_iter=warm_start_iter,
        )
        st.session_state["app_result"] = {
            "mode": mode,
            "params": params,
            "problem": problem,
            "solver_result": solver_result,
            "solver_mode": solver_mode,
            "gamma": gamma,
            "max_iter": max_iter,
            "use_warm_start": use_warm_start,
        }


result = st.session_state["app_result"]

if result is None:
    st.info("Choose settings in the sidebar, then click **Compute and plot**.")
else:
    if result["mode"] == "Penalty only":
        st.subheader("Penalty-only exploration")
        st.markdown(
            "This mode visualizes the true SPOQ penalty defined in the paper. For 2D and 3D views, the penalty is evaluated on full vectors `[x1, x2]`."
        )
        if vis_mode == "1D":
            x = np.linspace(-1.0, 1.0, 801)
            st.plotly_chart(penalty_1d_figure(x, result["params"]), use_container_width=True)
        elif vis_mode == "2D":
            st.plotly_chart(
                penalty_2d_trajectory_figure((-1.0, 1.0), (-1.0, 1.0), result["params"], title="SPOQ contour"),
                use_container_width=True,
            )
        else:
            st.plotly_chart(
                penalty_3d_trajectory_figure((-1.0, 1.0), (-1.0, 1.0), result["params"], title="SPOQ surface"),
                use_container_width=True,
            )
        st.metric("Current p", f"{result['params'].p:g}")
        st.metric("Current q", f"{result['params'].q:g}")
    else:
        problem = result["problem"]
        solver_result = result["solver_result"]
        trajectory = np.vstack(solver_result.history.xs)

        st.subheader("Full recovery problem")
        st.markdown(problem.description)

        metric_cols = st.columns(5)
        metric_cols[0].metric("Final Psi", f"{solver_result.history.psi_values[-1]:.6g}")
        metric_cols[1].metric("Final feasibility", f"{solver_result.history.feasibility_residuals[-1]:.3e}")
        metric_cols[2].metric("Iterations", f"{len(solver_result.history.xs) - 1}")
        metric_cols[3].metric("Solver mode", str(result["solver_mode"]))
        metric_cols[4].metric("Final SNR (dB)", "n/a" if solver_result.snr_db is None else f"{solver_result.snr_db:.3f}")

        if problem.is_2d and vis_mode in {"2D", "3D"}:
            x_range, y_range = _trajectory_bounds(trajectory, problem.y, problem.xi)
            if vis_mode == "2D":
                main_fig = penalty_2d_trajectory_figure(
                    x_range,
                    y_range,
                    result["params"],
                    trajectory=trajectory,
                    title="Penalty landscape with solver trajectory",
                )
            else:
                main_fig = penalty_3d_trajectory_figure(
                    x_range,
                    y_range,
                    result["params"],
                    trajectory=trajectory,
                    title="Penalty surface with solver trajectory",
                )
            st.plotly_chart(main_fig, use_container_width=True)
        elif problem.is_2d and vis_mode == "1D":
            st.warning("1D view is less informative for a 2D recovery problem. Showing signal/coordinate traces instead.")
            st.plotly_chart(signal_recovery_figure(problem.xtrue, solver_result.x_init, solver_result.x_final), use_container_width=True)
        else:
            st.plotly_chart(signal_recovery_figure(problem.xtrue, solver_result.x_init, solver_result.x_final), use_container_width=True)

        st.plotly_chart(
            convergence_figure(
                solver_result.history.psi_values,
                solver_result.history.feasibility_residuals,
                solver_result.history.step_norms,
            ),
            use_container_width=True,
        )

        with st.expander("Run details"):
            st.write(
                {
                    "x_init_norm": float(np.linalg.norm(solver_result.x_init)),
                    "x_final_norm": float(np.linalg.norm(solver_result.x_final)),
                    "warm_start_used": result["use_warm_start"],
                    "warm_start_iterations": None if solver_result.warmstart is None else solver_result.warmstart.iterations,
                    "max_trust_region_shrinks": int(max(solver_result.history.trust_region_shrinks)),
                }
            )
