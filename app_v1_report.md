# SPOQ App v1

## App structure

- `app.py`
  - Streamlit user interface
  - sidebar sections for:
    - mode
    - penalty parameters
    - problem / dataset parameters
    - solver parameters
    - visualization mode
- `app_utils.py`
  - problem construction
  - solver execution
  - warm-start integration
- `app_plotting.py`
  - Plotly figures for:
    - 1D / 2D / 3D penalty views
    - trajectory overlays
    - convergence diagnostics
    - signal reconstruction views

## Supported modes

- `Penalty only`
  - 1D SPOQ profile
  - 2D contour of true SPOQ on `[x1, x2]`
  - 3D surface of true SPOQ on `[x1, x2]`
- `Full recovery problem`
  - dataset choices:
    - `Toy 2D`
    - `Simulated toolbox dataset`
    - `Identity toy problem`
    - `Diagonal operator toy problem`
  - solver choices:
    - metric mode `0`
    - metric mode `1`
    - metric mode `2`
  - optional warm start

## Included in version 1

- local Streamlit app
- Plotly-based interactive figures
- didactic parameter controls
- compute-then-display workflow
- 2D and 3D trajectory overlays for 2D problems
- signal reconstruction and convergence plots for higher-dimensional problems

## Intentionally deferred to version 2

- live iteration-by-iteration animation
- background computation / progress streaming
- richer dataset management
- side-by-side multi-run comparison dashboards
- explicit objective-landscape visualization beyond the SPOQ penalty landscape

## Launch locally

```bash
python3 -m streamlit run app.py
```
