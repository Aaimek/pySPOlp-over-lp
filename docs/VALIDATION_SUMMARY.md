# Validation Summary (short)

## What is implemented (equivalents to matlab toolbox)

- SPOQ core (penalty, gradient, metric): `spoq_core.py`
- Prox/inner solver pieces: `spoq_prox.py`
- Warm start (`pds`): `spoq_warmstart.py`
- Outer solver with MATLAB modes 0/1/2: `spoq_solver.py`
- Simulated toolbox workflow: `load_spoq_data_simulated.py`, `run_spoq_recovery.py`
- Web visualization app (Streamlit + Plotly): `webapp/`

## MATLAB mapping (quick)

- `Lpsmooth.m`, `Lqsmooth.m`, `gradlplq.m`, `condlplq.m` -> Python SPOQ core
- `proxl2.m`, `proxB.m`, `proxPPXAplus.m` -> Python prox module
- `pds.m`, `proxl1.m`, `norm2.m` -> Python warm start
- `FB_PPXALpLq.m` -> Python outer solver (metric modes + trust-region logic)
- `Load_SPOQ_Data_Simulated.m`, `Run_SPOQ_Recovery.m` -> Python simulated-data + recovery scripts

## Proof it works

- Unit/integration tests pass (`tests/`).
- MATLAB parity was run on:
  - 2D identity case
  - 2D non-identity case
- Full trajectory comparison (not only final point):
  - `x_k`, `Psi(x_k)`, feasibility residuals, step norms
- Report/plots generated in:
  - `spoq_parity_outputs/comparison/`
  - `spoq_parity_outputs/comparison/parity_report.md`
- Observed differences are at numerical precision level (very small), so behavior matches MATLAB for tested cases.

### Quick visual parity (saved)

Identity case:

![Identity Psi](validation_assets/identity_psi_vs_iteration.png)
![Identity trajectory](validation_assets/identity_trajectory_overlay.png)

Non-identity case:

![Non-identity Psi](validation_assets/nonidentity_psi_vs_iteration.png)
![Non-identity trajectory](validation_assets/nonidentity_trajectory_overlay.png)
![Non-identity feasibility](validation_assets/nonidentity_feasibility_vs_iteration.png)

## Reproduce quickly

```bash
python3 -m pytest -q tests/
python3 scripts/run_python_spoq_parity.py
# MATLAB (batch example):
# /Applications/MATLAB_R2026a.app/bin/matlab -batch "cd('<repo>'); addpath('matlab_parity'); addpath('SPOQ-Sparse-Restoration-Toolbox-v1/SPOQ-Sparse-Restoration-Toolbox-v1.0/Tools'); matlab_spoq_parity_run"
python3 scripts/compare_spoq_parity.py
python3 -m streamlit run webapp/app.py
```

---

This is version 1: backend parity + didactic UI are in place. Live animation and extra polish are intentionally deferred.
