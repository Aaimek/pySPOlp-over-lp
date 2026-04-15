# SPOQ Core: Python vs MATLAB Mapping

## File mapping

- `spoq_core.py:lp_smooth` maps to MATLAB `Tools/Lpsmooth.m`
- `spoq_core.py:lq_smooth` maps to MATLAB `Tools/Lqsmooth.m`
- `spoq_core.py:spoq_grad` maps to MATLAB `Tools/gradlplq.m`
- `spoq_core.py:spoq_metric_diag` maps to MATLAB `Tools/condlplq.m`
- SPOQ objective expression (paper Eq. (9)) is used for `spoq_penalty`

## Formula-level comparison

- MATLAB `Lpsmooth.m`:
  - `lpx = (sum((x.^2 + alpha.^2).^(p/2) - alpha.^p))^(1/p);`
  - Python implementation is algebraically identical.
- MATLAB `Lqsmooth.m`:
  - `lqx = (mu.^q + sum(abs(x).^q))^(1/q);`
  - Python implementation is algebraically identical.
- MATLAB `gradlplq.m`:
  - `grad1 = x.*((x.^2+alpha.^2).^(p/2-1)) / (lp^p+beta^p);`
  - `grad2 = sign(x).*abs(x).^(q-1) / (lq^q);`
  - `gradphi = grad1 - grad2;`
  - Python implementation is algebraically identical and follows paper Eq. (15)-(16).
- MATLAB `condlplq.m`:
  - `Xpq = (q-1)/((eta^q + ro^q)^(2/q));`
  - `A = Xpq + (1/(lp^p+beta^p))*(x.^2+alpha^2).^(p/2-1);`
  - Python `spoq_metric_diag` is algebraically identical to this diagonal metric.

## Notable mismatch observed in toolbox

- MATLAB `Tools/Fcost.m` appears inconsistent with paper Eq. (9):
  - It uses exponent `1/q` where `1/p` is expected in two places.
  - This likely means `Fcost.m` does **not** evaluate Eq. (9) exactly.
  - Python `spoq_penalty` intentionally follows the paper as source of truth.
