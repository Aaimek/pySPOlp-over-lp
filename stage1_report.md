# Stage 1 Report: SPOQ Core

## Concise implementation plan used

1. Extract the exact SPOQ definitions from the paper around Eq. (9) and gradient/metric equations nearby.
2. Map toolbox MATLAB files to paper definitions and identify potential inconsistencies.
3. Implement a minimal Python core (`spoq_core.py`) with numerical safeguards and parameter validation.
4. Add tests (`test_spoq_core.py`) for deterministic values, finite-difference gradient checks, and MATLAB-formula parity.
5. Record mappings, discrepancies, and trust warnings in this report and `compare_with_matlab.md`.

## Paper equations used as source of truth

- Smoothed p-term: Eq. (7)
  - \(\ell_{p,\alpha}(x)=\left(\sum_n ((x_n^2+\alpha^2)^{p/2}-\alpha^p)\right)^{1/p}\)
- Smoothed q-term: Eq. (8)
  - \(\ell_{q,\eta}(x)=\left(\eta^q+\sum_n |x_n|^q\right)^{1/q}\)
- SPOQ penalty: Eq. (9)
  - \(\Psi(x)=\log\left(\frac{(\ell_{p,\alpha}^p(x)+\beta^p)^{1/p}}{\ell_{q,\eta}(x)}\right)\)
- Gradient definitions:
  - Eq. (10), Eq. (11), Eq. (15), Eq. (16)
  - Final implemented gradient:
    - \(x_n(x_n^2+\alpha^2)^{p/2-1}/(\ell_{p,\alpha}^p+\beta^p)\)
    - minus \(\mathrm{sign}(x_n)|x_n|^{q-1}/\ell_{q,\eta}^q\)
- Local diagonal metric:
  - Eq. (25) and Eq. (26)
  - \(A_{q,\rho}(x)=\chi_{q,\rho}I + \frac{1}{\ell_{p,\alpha}^p(x)+\beta^p}\mathrm{Diag}((x_n^2+\alpha^2)^{p/2-1})\)
  - \(\chi_{q,\rho}=(q-1)/(\eta^q+\rho^q)^{2/q}\)

## MATLAB toolbox mapping

- SPOQ penalty components:
  - `Tools/Lpsmooth.m` -> Eq. (7)
  - `Tools/Lqsmooth.m` -> Eq. (8)
- Gradient:
  - `Tools/gradlplq.m` -> Eq. (15)-(16)
- Local metric / conditioning:
  - `Tools/condlplq.m` -> Eq. (25)-(26) diagonal form
- Visualization scripts:
  - `Display_SPOQ_Penalty_2D.m`
- Main solver:
  - `Run_SPOQ_Recovery.m` calls `Tools/FB_PPXALpLq.m`
  - trust-region loop and metric choice in `Tools/FB_PPXALpLq.m`

## Discrepancies and suspicious points

- `Tools/Fcost.m` appears inconsistent with Eq. (9):
  - it uses exponent `1/q` where Eq. (9) requires `1/p` for numerator construction.
  - this may distort reported objective values in MATLAB logging.
- `Display_SPOQ_Penalty_2D.m` computes a simplified/scalar 2D illustration; it is useful for visualization, but not a direct implementation of the full vector formula used in optimization.
- In `FB_PPXALpLq.m`, trust-region update test uses strict inequality on q-norm and halving logic; this is implementation-specific and should not be treated as a direct transcription of all details in the paper's algorithmic statements.

## Ambiguities explicitly noted

- The paper defines a full matrix majorant in Eq. (25); toolbox and Python stage 1 only use the diagonal form for vector updates.
- The meaning of `rho` in practice (trust-region radius sequence) is algorithmic-context-dependent; stage 1 exposes `rho` as an explicit parameter in `spoq_metric_diag` without solver policy.

## Stage 1 files produced

- `spoq_core.py`
- `test_spoq_core.py`
- `compare_with_matlab.md`
- `stage1_report.md`
