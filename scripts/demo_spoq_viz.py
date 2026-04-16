"""Demo script for stage-2 SPOQ visualization.

Produces:
- 1D static matplotlib plot (SOOT vs folded case)
- 2D static contour plot
- 3D interactive Plotly surface
"""

from __future__ import annotations

import numpy as np

from plotly_spoq_viz import plotly_spoq_2d_contour, plotly_spoq_3d_surface
from spoq_viz import (
    FOLDED_PRESET,
    SOOT_PRESET,
    check_even_symmetry_2d,
    plot_spoq_1d,
    plot_spoq_2d_contour,
    plot_spoq_3d_surface,
)


def main() -> None:
    x1d = np.linspace(-1.0, 1.0, 801)
    x2d = np.linspace(-1.0, 1.0, 241)

    # 1D comparison: SOOT vs lower-p folded behavior
    fig1, _ = plot_spoq_1d(
        x=x1d,
        params_list=[SOOT_PRESET, FOLDED_PRESET],
        labels=["SOOT (p=1, q=2)", "Folded (p=0.25, q=2)"],
        title="True SPOQ in 1D",
    )
    fig1.tight_layout()
    fig1.savefig("spoq_1d_profiles.png", dpi=200)

    # 2D contour for the folded case (stronger sparsity-promoting geometry)
    fig2, _, X1, X2, Z = plot_spoq_2d_contour(
        x1=x2d,
        x2=x2d,
        params=FOLDED_PRESET,
        levels=50,
        title="True SPOQ contour: folded case (p=0.25, q=2)",
    )
    fig2.tight_layout()
    fig2.savefig("spoq_2d_contour_folded.png", dpi=220)

    # Optional static 3D matplotlib (also useful for quick export)
    fig3, _, _, _, _ = plot_spoq_3d_surface(
        x1=x2d,
        x2=x2d,
        params=FOLDED_PRESET,
        title="True SPOQ surface: folded case (p=0.25, q=2)",
    )
    fig3.tight_layout()
    fig3.savefig("spoq_3d_surface_folded_matplotlib.png", dpi=220)

    # Interactive Plotly outputs
    fig2i = plotly_spoq_2d_contour(
        x1=x2d,
        x2=x2d,
        params=FOLDED_PRESET,
        title="Interactive SPOQ contour (p=0.25, q=2)",
    )
    fig2i.write_html("spoq_2d_contour_folded_interactive.html", include_plotlyjs="cdn")

    fig3i = plotly_spoq_3d_surface(
        x1=x2d,
        x2=x2d,
        params=FOLDED_PRESET,
        title="Interactive SPOQ surface (p=0.25, q=2)",
    )
    fig3i.write_html("spoq_3d_surface_folded_interactive.html", include_plotlyjs="cdn")

    # Numerical symmetry checks on sampled grid points
    symmetry_stats = check_even_symmetry_2d(X1, X2, Z, FOLDED_PRESET)
    print("Symmetry checks for folded case:")
    for key, value in symmetry_stats.items():
        print(f"  {key}: {value:.3e}")

    print("\nGenerated files:")
    print("  - spoq_1d_profiles.png")
    print("  - spoq_2d_contour_folded.png")
    print("  - spoq_3d_surface_folded_matplotlib.png")
    print("  - spoq_2d_contour_folded_interactive.html")
    print("  - spoq_3d_surface_folded_interactive.html")


if __name__ == "__main__":
    main()
