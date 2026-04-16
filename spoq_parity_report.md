# SPOQ MATLAB vs Python Parity Experiment

## What this parity experiment does

This experiment compares the **full trajectory** of the minimal SPOQ forward-backward solver in Python and MATLAB on two tiny deterministic 2D problems:

- `identity_2d`: `D = I`
- `nonidentity_2d`: `D = [[1, 2], [0, 1]]`

For each iteration, both sides record:

- `x_k`
- `Psi(x_k)` using the paper-consistent SPOQ formula
- step norm `||x_k - x_{k-1}||`
- feasibility residual `||D x_k - y|| - xi`
- minimum component of `x_k`

## Files

- Python trace exporter: `run_python_spoq_parity.py`
- MATLAB trace exporter: `matlab_spoq_parity_run.m`
- Python comparison/plotting script: `compare_spoq_parity.py`

## Important implementation note

The MATLAB parity script intentionally mirrors the **minimal no-trust-region outer loop** used in Python:

1. `gradlplq`
2. `condlplq(..., ro=0)`
3. `u = x - gamma * grad ./ A`
4. `proxPPXAplus(D, A/gamma, u, y, xi, J, prec)`

This is not the full trust-region solver from the toolbox. It is the closest MATLAB-side counterpart to the Python stage-4 solver.

Also, the MATLAB toolbox function `Fcost.m` is **not** used for parity because it appears inconsistent with paper Eq. (9). Both sides instead compute `Psi` with the paper-consistent formula.

## How to run

### 1. Run Python traces

```bash
python3 run_python_spoq_parity.py
```

### 2. Run MATLAB traces

From MATLAB, in this workspace directory:

```matlab
matlab_spoq_parity_run
```

This writes CSV traces into `spoq_parity_outputs/matlab/`.

### 3. Compare both sides

```bash
python3 compare_spoq_parity.py
```

This generates:

- `spoq_parity_outputs/comparison/<case>/psi_vs_iteration.png`
- `spoq_parity_outputs/comparison/<case>/feasibility_vs_iteration.png`
- `spoq_parity_outputs/comparison/<case>/step_norms_vs_iteration.png`
- `spoq_parity_outputs/comparison/<case>/min_component_vs_iteration.png`
- `spoq_parity_outputs/comparison/<case>/trajectory_overlay.png`
- `spoq_parity_outputs/comparison/parity_report.md`

## Expected outcome

If the implementations are aligned, the trajectories should be very close for both cases. Small differences may still remain because of:

- floating-point differences between Python/NumPy and MATLAB
- explicit inverse and linear algebra implementation details
- stopping behavior inside `proxPPXAplus`

The comparison script reports:

- final `x` difference
- final `Psi` difference
- final feasibility difference
- max trajectory difference over all iterations
