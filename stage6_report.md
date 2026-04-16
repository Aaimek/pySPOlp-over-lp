# Stage 6 Report: Simulated Toolbox Workflow

## What was implemented

- `load_spoq_data_simulated.py`
  - Python equivalent of MATLAB `Load_SPOQ_Data_Simulated.m`
- `run_spoq_recovery.py`
  - Python equivalent of MATLAB `Run_SPOQ_Recovery.m`
- `test_spoq_recovery.py`
  - small deterministic end-to-end smoke test

## Mapping to `Load_SPOQ_Data_Simulated.m`

The Python loader follows the MATLAB logic closely:

- `nSample = 500`
- `nPeak = 20`
- `peakWidth = 5`
- `xtrue` initialized to zeros
- random peak locations and amplitudes
- Pascal/binomial kernel construction
- Toeplitz convolution matrix `K`
- Gaussian noise generation
- `y0 = K @ xtrue`
- `sigma = 0.5 * max(y0) / 100`
- `y = y0 + sigma * noise`
- `xi = 1.1 * sqrt(nSample) * sigma`
- SPOQ parameters:
  - `eta = 2e-6`
  - `alpha = 7e-7`
  - `beta = 3e-2`
  - `p = 0.75`
  - `q = 2`
  - `nbiter = 5000`

## Mapping to `Run_SPOQ_Recovery.m`

The Python runner mirrors the MATLAB recovery workflow:

1. load the simulated data
2. run the solver in trust-region mode (`metric_mode = 2`)
3. compute reconstruction quality via

\[
\mathrm{SNR} = 10 \log_{10}\left(\frac{\|x_{\mathrm{true}}\|^2}{\|x_{\mathrm{rec}} - x_{\mathrm{true}}\|^2}\right)
\]

4. generate plots:
   - original vs reconstructed signal
   - convergence curve
5. save traces and metadata

## Approximations / implementation notes

- Random generation:
  - MATLAB uses `randperm`, `rand`, and `randn`
  - Python uses NumPy RNG with a fixed seed for reproducibility
- Kernel shape:
  - MATLAB uses `pascal(peakWidth)` and the anti-diagonal
  - Python reconstructs the same symmetric Pascal-style matrix explicitly and takes the anti-diagonal
- Toeplitz matrix:
  - MATLAB calls `toeplitz(...)`
  - Python builds the same Toeplitz structure directly using NumPy indexing
- Initialization:
  - user requested `x0 = zeros`; this is what the Python runner uses
  - this differs from the full MATLAB toolbox solver `FB_PPXALpLq.m`, which internally warm-starts using `pds(...)`

## What remains for fuller toolbox parity

- reproduce the MATLAB warm start from `pds(...)` exactly
- support the paper and user dataset loaders (`Load_SPOQ_Data_Paper.m`, `Load_SPOQ_Data_User.m`)
- reproduce MATLAB timing/SNR bookkeeping arrays more literally if needed
- reproduce the MATLAB data-information display script in Python if desired

## Validation performed

- loader smoke test for dimensions and parameters
- end-to-end recovery smoke test:
  - no crash
  - outputs created
  - reconstructed SNR improves over zero initialization
