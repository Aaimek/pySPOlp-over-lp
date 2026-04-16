"""Plotly figures used by the local Streamlit SPOQ app."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from spoq_core import spoq_penalty
from spoq_viz import SpoqParams, spoq_1d_values, spoq_2d_grid_values


def _add_stem_trace(fig: go.Figure, y: np.ndarray, name: str, color: str) -> None:
    """Add a stem-style trace using vertical segments plus markers."""
    x = np.arange(y.size, dtype=float)
    xs = np.repeat(x, 3)
    ys = np.empty(3 * y.size, dtype=float)
    ys[0::3] = 0.0
    ys[1::3] = y
    ys[2::3] = np.nan
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=color, width=1.6),
            name=name,
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(color=color, size=6, symbol="diamond"),
            name=name,
        )
    )


def penalty_1d_figure(x: np.ndarray, params: SpoqParams) -> go.Figure:
    """Interactive 1D SPOQ profile."""
    y = spoq_1d_values(x, params)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(width=3), name="SPOQ([x])"))
    fig.update_layout(title="SPOQ penalty in 1D", xaxis_title="x", yaxis_title="\u03a8([x])")
    return fig


def penalty_2d_trajectory_figure(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    params: SpoqParams,
    trajectory: np.ndarray | None = None,
    title: str = "SPOQ contour",
) -> go.Figure:
    """Contour plot of SPOQ([x1, x2]) with optional trajectory overlay."""
    x1 = np.linspace(x_range[0], x_range[1], 181)
    x2 = np.linspace(y_range[0], y_range[1], 181)
    X1, X2, Z = spoq_2d_grid_values(x1, x2, params)
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=X1[0, :],
            y=X2[:, 0],
            z=Z,
            colorscale="Viridis",
            contours=dict(showlabels=True),
            colorbar=dict(title="SPOQ"),
        )
    )
    if trajectory is not None:
        fig.add_trace(
            go.Scatter(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                mode="lines+markers",
                name="trajectory",
                line=dict(color="white", width=3),
                marker=dict(size=6, color=np.arange(len(trajectory)), colorscale="Plasma"),
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


def penalty_3d_trajectory_figure(
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    params: SpoqParams,
    trajectory: np.ndarray | None = None,
    title: str = "SPOQ surface",
) -> go.Figure:
    """3D surface plot of SPOQ([x1, x2]) with optional trajectory overlay."""
    x1 = np.linspace(x_range[0], x_range[1], 121)
    x2 = np.linspace(y_range[0], y_range[1], 121)
    X1, X2, Z = spoq_2d_grid_values(x1, x2, params)
    fig = go.Figure()
    fig.add_trace(
        go.Surface(x=X1, y=X2, z=Z, colorscale="Viridis", colorbar=dict(title="SPOQ"), opacity=0.92, name="surface")
    )
    if trajectory is not None:
        traj_z = np.array(
            [spoq_penalty(pt, params.alpha, params.beta, params.eta, params.p, params.q) for pt in trajectory],
            dtype=float,
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
    fig.update_layout(title=title, scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="SPOQ"))
    return fig


def convergence_figure(psi_values: list[float], feasibility_residuals: list[float], step_norms: list[float]) -> go.Figure:
    """Three-panel convergence summary."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("\u03a8(x_k)", "Feasibility residual", "Step norm"),
    )
    it = np.arange(len(psi_values))
    fig.add_trace(go.Scatter(x=it, y=psi_values, mode="lines+markers", name="\u03a8(x_k)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=it, y=feasibility_residuals, mode="lines+markers", name="feas"), row=2, col=1)
    fig.add_trace(go.Scatter(x=it, y=step_norms, mode="lines+markers", name="step"), row=3, col=1)
    fig.update_xaxes(title_text="Iteration", row=3, col=1)
    fig.update_yaxes(title_text="\u03a8", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    fig.update_yaxes(title_text="Norm", row=3, col=1)
    fig.update_layout(height=800, title="Convergence diagnostics", showlegend=False)
    return fig


def signal_recovery_figure(xtrue: np.ndarray | None, x_init: np.ndarray, x_final: np.ndarray) -> go.Figure:
    """Signal plot for higher-dimensional recovery problems.

    The true signal is shown with a stem-style representation so it remains
    visible even when reconstruction nearly coincides with it.
    """
    fig = go.Figure()
    if xtrue is not None:
        _add_stem_trace(fig, np.asarray(xtrue, dtype=float), name="true signal", color="crimson")
    fig.add_trace(
        go.Scatter(
            y=x_init,
            mode="lines",
            name="initial point",
            line=dict(color="gray", dash="dot", width=1.5),
            opacity=0.8,
        )
    )
    fig.add_trace(
        go.Scatter(
            y=x_final,
            mode="lines",
            name="reconstruction",
            line=dict(color="royalblue", width=2.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            y=x_final,
            mode="markers",
            name="reconstruction markers",
            marker=dict(color="royalblue", size=4, symbol="circle-open"),
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Signal reconstruction",
        xaxis_title="Index",
        yaxis_title="Amplitude",
        legend_title="Curves",
    )
    return fig
