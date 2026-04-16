"""Interactive Plotly visualization for true SPOQ penalty."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from spoq_viz import SpoqParams, spoq_2d_grid_values


def plotly_spoq_2d_contour(
    x1: np.ndarray,
    x2: np.ndarray,
    params: SpoqParams,
    title: str | None = None,
) -> go.Figure:
    """Build an interactive contour plot for SPOQ([x1, x2])."""
    X1, X2, Z = spoq_2d_grid_values(x1, x2, params)
    fig = go.Figure(
        data=go.Contour(
            x=X1[0, :],
            y=X2[:, 0],
            z=Z,
            colorscale="Viridis",
            contours=dict(showlabels=True),
            colorbar=dict(title="SPOQ"),
        )
    )
    fig.update_layout(
        title=title or f"SPOQ contour (p={params.p}, q={params.q})",
        xaxis_title="x1",
        yaxis_title="x2",
    )
    return fig


def plotly_spoq_3d_surface(
    x1: np.ndarray,
    x2: np.ndarray,
    params: SpoqParams,
    title: str | None = None,
) -> go.Figure:
    """Build an interactive 3D surface plot for SPOQ([x1, x2])."""
    X1, X2, Z = spoq_2d_grid_values(x1, x2, params)
    fig = go.Figure(
        data=go.Surface(
            x=X1,
            y=X2,
            z=Z,
            colorscale="Viridis",
            colorbar=dict(title="SPOQ"),
        )
    )
    fig.update_layout(
        title=title or f"SPOQ surface (p={params.p}, q={params.q})",
        scene=dict(
            xaxis_title="x1",
            yaxis_title="x2",
            zaxis_title="SPOQ",
        ),
    )
    return fig
