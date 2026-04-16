# Stage 5 Report: MATLAB-Style Outer Solver Logic

## What was added relative to the simplified solver

`spoq_solver.py` was extended from a single minimal variable-metric forward-backward loop to support the three solver modes used in MATLAB `Tools/FB_PPXALpLq.m`:

- `metric_mode = 0`: fixed/global metric using the Lipschitz constant from `ComputeLipschitz.m`
- `metric_mode = 1`: variable metric without trust region using `condlplq(..., rho=0)`
- `metric_mode = 2`: trust-region variable metric using the same radius initialization and shrink loop as MATLAB

The solver now also records MATLAB-style iteration diagnostics:

- iterates `x_k`
- `Psi(x_k)` using the paper-consistent SPOQ formula
- step norms
- feasibility residuals
- relative stopping errors
- metric diagonals used at each iteration
- accepted trust-region radius for mode `2`
- trust-region shrink counts (`Bwhile`-like bookkeeping)

## How the trust-region logic was mapped from MATLAB

The translation follows `FB_PPXALpLq.m` closely.

### MATLAB structure

For trust-region mode:

1. initialize
   - `ro = ||x_k||_q`
   - `bwhile = 0`
2. build local metric with `condlplq(x_k, ..., ro)`
3. compute proposal with that metric
4. evaluate prox step
5. if `||x_{k+1}||_q < ro`, then:
   - set `ro = ro / 2`
   - increment `bwhile`
   - retry
6. otherwise accept the step

### Python mapping

Python uses the same loop:

- radius initialization:
  - `ro = (sum(abs(x)**q))**(1/q)`
- shrink rule:
  - `ro = ro / 2`
- acceptance rule:
  - accept the candidate only when `||x_next||_q >= ro`

This is intentionally kept close to MATLAB, even though the acceptance rule is somewhat unusual from a generic trust-region perspective.

## Stopping behavior

Stopping now mirrors MATLAB `FB_PPXALpLq.m`:

\[
\text{error} = \frac{\|x_{k+1} - x_k\|_2^2}{\|x_k\|_2^2}
\]

and the loop stops when `error < prox_prec`.

### Practical note

MATLAB would divide by zero if `x_k` were exactly zero. In Python, this was made explicit:

- if both numerator and denominator are zero, error is set to `0`
- if denominator is zero and numerator is nonzero, error is set to `inf`

This preserves the intended behavior while avoiding silent `NaN` propagation.

## Ambiguities or suspicious MATLAB behavior kept explicit

- The trust-region acceptance condition depends on `||x_{k+1}||_q < ro`, not on a standard model decrease test.
  - This was preserved rather than redesigned.
- The MATLAB code uses the same symbol `prec` both for the inner PPXA stopping tolerance and the outer stopping threshold.
  - Python keeps this coupling through the same `prox_prec` argument to stay close to MATLAB.
- MATLAB stores `Bwhile` only for trust-region mode.
  - Python generalizes this into `trust_region_shrinks`, which is `0` for other modes.
- MATLAB logs `fcost` through `Fcost.m`, but that function appears inconsistent with paper Eq. (9).
  - Python records `Psi` using the validated paper-consistent formula instead of reproducing the questionable `Fcost.m` expression.

## Tests added/updated

`test_spoq_solver.py` now verifies:

- all three solver modes run
- variable-metric mode matches the prior default behavior
- trust-region mode records radii and shrink counts sensibly
- a challenging trust-region case triggers at least one radius shrink
- existing feasibility, descent, and plotting checks still pass
