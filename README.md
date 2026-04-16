# pySPOQ (Python SPOQ toolbox port)

Short project note:

- Core SPOQ penalty + gradient + metric implemented
- Prox / inner solver implemented
- Warm start (`pds`) implemented
- Full outer solver modes (including trust-region mode) implemented
- Simulated toolbox workflow implemented
- Local Streamlit app implemented (`webapp/`)

Validation/proof is summarized here:

- `docs/VALIDATION_SUMMARY.md`

Run the app locally:

```bash
python3 -m streamlit run webapp/app.py
```

## Quick validation (MATLAB parity)

- Python implementation reproduces MATLAB behavior on 2D identity and 2D non-identity parity cases.
- Full trajectories are compared (`x_k`, `Psi(x_k)`, feasibility, step norms).
- Detailed short note: `docs/VALIDATION_SUMMARY.md`

Identity case:

<img src="docs/validation_assets/identity_psi_vs_iteration.png" width="420" />
<img src="docs/validation_assets/identity_trajectory_overlay.png" width="420" />

Non-identity case:

<img src="docs/validation_assets/nonidentity_psi_vs_iteration.png" width="420" />
<img src="docs/validation_assets/nonidentity_trajectory_overlay.png" width="420" />
<img src="docs/validation_assets/nonidentity_feasibility_vs_iteration.png" width="420" />
