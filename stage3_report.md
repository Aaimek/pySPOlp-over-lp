# Stage 3 Report: Inner Proximal Machinery

## What was implemented

- `spoq_prox.py`
  - `proxl2(x, y, eta)`
  - `proxB(B, x, xhat, teta)`
  - `prox_ppxa_plus(D, B, x, y, eta, J, prec, teta=1.9)`
  - `PpxaResult` dataclass for returning the iterate, iteration count, convergence flag, and final iterate-change error
- `test_spoq_prox.py`
  - deterministic tests for `proxl2`
  - deterministic tests for `proxB`
  - small synthetic PPXA+ tests with exact/near-exact reference behavior

## MATLAB-to-Python mapping

- `Tools/proxl2.m` -> `spoq_prox.proxl2`
- `Tools/proxB.m` -> `spoq_prox.proxB`
- `Tools/proxPPXAplus.m` -> `spoq_prox.prox_ppxa_plus`

The Python implementation preserves the MATLAB iteration structure:

1. initialize `x1k_old = x`
2. initialize `x2k_old = D @ x1k_old`
3. precompute `A = inv(I + D.T @ D)`
4. iterate:
   - `y1k_old = proxB(...)`
   - `y2k_old = proxl2(...)`
   - `vk_old = A @ (y1k_old + D.T @ y2k_old)`
   - update `x1k`, `x2k`, `zk`
   - stop when `||zk - zk_old||_2^2 < prec`

## What optimization problem the inner solver is solving

From the MATLAB usage and closed forms, the inner PPXA+ machinery is solving the proximity problem associated with the constrained convex term used in the outer algorithm:

- nonnegativity constraint on the signal
- Euclidean-ball data-consistency constraint on `D z`

More explicitly, the intended proximal point appears to be the minimizer of

\[
\min_{z} \; \frac12 \|z-x\|_{B}^{2} + \iota_{\mathbb{R}_+^N}(z) + \iota_{\mathcal{B}(y,\eta)}(Dz),
\]

where:

- `B` is diagonal and encoded by its diagonal entries
- `x` is the proximal center coming from the outer step
- `\iota_{\mathbb{R}_+^N}` is the nonnegativity indicator
- `\iota_{\mathcal{B}(y,\eta)}` is the indicator of the Euclidean ball centered at `y` with radius `eta`

This interpretation is supported by:

- `proxl2`: exact projection of `Dx`-space variables onto the Euclidean ball
- `proxB`: exact closed-form prox of a diagonal weighted quadratic, followed by nonnegativity clipping
- `proxPPXAplus`: splitting between those two pieces through PPXA+

## Validation summary

- `proxl2`
  - verified on a point already inside the ball
  - verified on a point outside the ball, which is projected exactly onto the boundary
- `proxB`
  - verified against the exact MATLAB closed form on deterministic diagonal-weighted data
  - verified that negative coordinates are clipped to zero
- `prox_ppxa_plus`
  - tested with `D = I` and `eta = 0`, where the feasible set reduces to a single point `y`; the output converges to `y` within MATLAB-style stopping tolerance
  - tested on a small `2D` identity case against a brute-force weighted constrained search; the PPXA+ output matches the reference solution up to the expected numerical tolerance

## Ambiguities and suspicious behavior

- `proxB.m` comment is incomplete/misleading:
  - it says it computes the proximity operator of `f(x) = (teta/2) * ||y-x||_B^2`
  - however, the actual code uses `xhat`, not `y`, and also applies `p(p<0)=0`
  - therefore the implemented operator is really the prox of a weighted quadratic **plus nonnegativity**
- `proxPPXAplus.m` stops based only on iterate change:
  - `error = norm(zk-zk_old,2)^2`
  - there is no explicit final feasibility check
  - as a result, the returned point can be microscopically off the ball boundary due to finite precision
- `proxPPXAplus.m` uses an explicit matrix inverse:
  - `A = inv(eye(N) + D'*D)`
  - this is faithful to MATLAB but should not automatically be trusted as the best numerical strategy for larger problems
- the MATLAB function signature `function [zk,j] = proxPPXAplus(...)` is slightly awkward:
  - the caller uses only the first output
  - stage 3 returns a richer structured result instead

## Test command

```bash
python3 -m pytest -q test_spoq_prox.py
```
