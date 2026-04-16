# Stage 7 Report: MATLAB Warm Start (`pds`) Integration

## What was added

- `spoq_warmstart.py`
  - `pds_warmstart(...)`: faithful Python port of MATLAB `Tools/pds.m`
  - `proxl1(...)`: port of MATLAB `Tools/proxl1.m`
  - `norm2_estimate(...)`: deterministic port of MATLAB `Tools/norm2.m`
- `run_spoq_recovery.py`
  - now uses warm start by default to better match the MATLAB toolbox workflow
  - can disable warm start with `use_warm_start=False`
  - saves both `x_init` and `xrec`
- tests:
  - `test_spoq_warmstart.py`
  - updated `test_spoq_recovery.py`

## How `pds.m` was mapped to Python

The Python implementation follows the MATLAB structure closely:

1. compute spectral norm estimate `normK` via power iteration (`norm2.m`)
2. set:
   - `tau = 1 / normK`
   - `sigma = 0.9 / (tau * normK^2)`
   - `ro = 1`
3. initialize:
   - `xk_old = ones(N,1)`
   - `uk_old = K * xk_old`
4. iterate:
   - `xxk = proxl1(xk_old - tau * K' * uk_old, tau)`
   - `zk = uk_old + sigma * K * (2*xxk - xk_old)`
   - `uuk = zk - sigma * proxl2(zk / sigma, y, eta)`
   - relaxed updates with factor `ro`
5. stop when both primal and dual relative changes are below `1e-6`

This is a direct port of the MATLAB warm-start routine, not a redesigned initializer.

## Helper functions needed

- `proxl1.m` was not previously implemented and had to be added.
  - Important detail: the MATLAB function is **not** plain soft-thresholding.
  - It also clips negative values to zero at the end.
- `norm2.m` was also needed.
  - MATLAB uses random initialization in the power iteration.
  - Python uses a fixed RNG seed by default for determinism and reproducibility.
- `proxl2.m` was already available from stage 3 and reused directly.

## How integration changed the workflow

Before this stage:

- simulated recovery used `x0 = 0`

After this stage:

- simulated recovery uses `pds_warmstart(...)` by default
- zero initialization is still available for comparison by setting:
  - `use_warm_start=False`

The recovery outputs now save:

- `x_init.csv`
- `xrec.csv`
- `warmstart_refspec.csv` when warm start is enabled

## Ambiguities / suspicious MATLAB behavior kept explicit

- `norm2.m` uses a random initial vector without controlling the seed.
  - Python makes this deterministic for reproducibility.
- `proxl1.m` clips negatives to zero after thresholding.
  - This is unusual if read as a generic `l1` prox, but it was preserved exactly because it is part of the toolbox warm-start logic.
- `pds.m` initializes from an all-ones vector, not zeros.
  - This was preserved.

## What still remains for fuller toolbox parity

- reproduce paper-data and user-data loaders beyond the simulated dataset
- mirror MATLAB display scripts more literally if needed
- compare the Python warm-start trace directly against MATLAB `pds.m` on a shared tiny case
- optionally reproduce the toolbox’s complete end-to-end script entry structure more literally
