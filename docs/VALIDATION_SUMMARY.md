# Validation Summary (short)

- Implemented: SPOQ core, prox/inner solver, warm start (`pds`), full outer solver (modes 0/1/2), simulated toolbox workflow, and Streamlit app.
- MATLAB parity checked on 2D identity + 2D non-identity cases (full trajectories: `x_k`, `Psi(x_k)`, feasibility, step norms).
- Result: Python matches MATLAB at numerical precision level on tested parity cases.
- Outputs are in `spoq_parity_outputs/comparison/` (plots + `parity_report.md`).

MATLAB parity command (works on my MacBook):

```bash
/Applications/MATLAB_R2026a.app/bin/matlab -batch "cd('<repo>'); addpath('matlab_parity'); addpath('SPOQ-Sparse-Restoration-Toolbox-v1/SPOQ-Sparse-Restoration-Toolbox-v1.0/Tools'); matlab_spoq_parity_run"
```
