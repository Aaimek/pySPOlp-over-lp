# Stage 2 Report: SPOQ Visualization Module

## What was implemented

- `spoq_viz.py`
  - Reusable SPOQ parameter dataclass: `SpoqParams`
  - Presets:
    - SOOT: `p=1, q=2`
    - Folded/sparser case: `p=0.25, q=2`
  - True SPOQ evaluators:
    - `spoq_1d_values(x, params)` evaluates SPOQ on vectors `[x]`
    - `spoq_2d_grid_values(x1, x2, params)` evaluates SPOQ on vectors `[x1, x2]` over a meshgrid
  - Static matplotlib visualizations:
    - `plot_spoq_1d(...)`
    - `plot_spoq_2d_contour(...)`
    - `plot_spoq_3d_surface(...)`
  - Symmetry checker:
    - `check_even_symmetry_2d(...)` for sign-flip and coordinate-swap invariances
- `plotly_spoq_viz.py`
  - Interactive Plotly contour: `plotly_spoq_2d_contour(...)`
  - Interactive Plotly surface: `plotly_spoq_3d_surface(...)`
- `demo_spoq_viz.py`
  - Produces:
    - 1D static plot
    - 2D static contour plot
    - 3D interactive surface plot (HTML)
  - Also exports optional static 3D matplotlib figure
  - Prints numerical symmetry checks

## Mapping to the paper

- All evaluations call `spoq_core.spoq_penalty(...)` from stage 1, which follows paper Eq. (9) using Eq. (7) and Eq. (8).
- 2D and 3D plots evaluate the **true vector** SPOQ on `[x1, x2]`, not a separable approximation.
- The SOOT and folded presets reproduce the qualitative comparison discussed around Figure 1:
  - SOOT (`p=1, q=2`) yields smoother valleys.
  - Lower `p` (`p=0.25`) yields sharper folds along axes.

## Qualitative behavior and checks

- Visual behavior:
  - Decreasing `p` from `1` to `0.25` sharpens axis-aligned folds and increases sparsity-promoting shape contrast near coordinate axes.
  - Level sets remain symmetric with respect to sign flips and coordinate swap.
- Numerical symmetry checks (from demo run, folded case):
  - `max_abs_err_sign_flip = 0.000e+00`
  - `max_abs_err_swap = 0.000e+00`
  - `mean_abs_err_sign_flip = 0.000e+00`
  - `mean_abs_err_swap = 0.000e+00`

## Numerical considerations near zero

- Very small `alpha`, `beta`, `eta` increase curvature near the origin; this is expected from the smoothing design.
- Evaluation remains stable because stage-1 core enforces strictly positive smoothing parameters and computes with floating-point safeguards.
- Plots use symmetric ranges around zero (`[-1, 1]`) and dense sampling to reveal geometry around the origin.

## Generated demo artifacts

- `spoq_1d_profiles.png`
- `spoq_2d_contour_folded.png`
- `spoq_3d_surface_folded_matplotlib.png`
- `spoq_2d_contour_folded_interactive.html`
- `spoq_3d_surface_folded_interactive.html`
