"""Utility functions for the local Streamlit SPOQ app."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from load_spoq_data_paper_style import PaperStyleSpoqData, load_paper_style_dataset_a, load_paper_style_dataset_b
from load_spoq_data_simulated import SimulatedSpoqData, load_spoq_data_simulated

DATASET_PAPER_STYLE_A = "Paper-style A (N=1000, P=48)"
DATASET_PAPER_STYLE_B = "Paper-style B (N=1000, P=94)"

# Datasets where xi (and xtrue/y) come from the generator, not the toy sliders.
DATASET_DATA_DRIVEN = (
    "Simulated toolbox dataset",
    DATASET_PAPER_STYLE_A,
    DATASET_PAPER_STYLE_B,
)
from run_spoq_recovery import _snr_db
from spoq_solver import SolverHistory, run_spoq_solver
from spoq_viz import SpoqParams
from spoq_warmstart import WarmStartResult, pds_warmstart


@dataclass(frozen=True)
class ProblemData:
    """Problem instance used by the app."""

    name: str
    xtrue: np.ndarray | None
    D: np.ndarray
    y: np.ndarray
    xi: float
    is_2d: bool
    description: str


@dataclass(frozen=True)
class SolverRunResult:
    """Solver outputs displayed by the app."""

    x_init: np.ndarray
    x_final: np.ndarray
    history: SolverHistory
    warmstart: WarmStartResult | None
    snr_db: float | None


def create_penalty_params(p: float, q: float, alpha: float, beta: float, eta: float) -> SpoqParams:
    """Build SPOQ parameter container from UI values."""
    return SpoqParams(alpha=float(alpha), beta=float(beta), eta=float(eta), p=float(p), q=float(q))


def make_problem_data(
    dataset_name: str,
    *,
    toy_x0: float = 0.9,
    toy_x1: float = 0.6,
    toy_y0: float = 0.45,
    toy_y1: float = 0.0,
    toy_xi: float = 0.3,
    diag_a: float = 1.0,
    diag_b: float = 0.5,
    simulated_seed: int = 0,
) -> tuple[ProblemData, SimulatedSpoqData | PaperStyleSpoqData | None]:
    """Create supported problem instances for the app."""
    if dataset_name == "Toy 2D":
        D = np.eye(2, dtype=float)
        xtrue = np.array([toy_x0, toy_x1], dtype=float)
        y = np.array([toy_y0, toy_y1], dtype=float)
        xi = float(toy_xi)
        return (
            ProblemData(
                name=dataset_name,
                xtrue=xtrue,
                D=D,
                y=y,
                xi=xi,
                is_2d=True,
                description="Small 2D identity-constrained toy problem for direct contour and surface visualization.",
            ),
            None,
        )
    if dataset_name == "Identity toy problem":
        D = np.eye(2, dtype=float)
        xtrue = np.array([0.8, 0.2], dtype=float)
        y = np.array([0.45, 0.05], dtype=float)
        xi = float(toy_xi)
        return (
            ProblemData(
                name=dataset_name,
                xtrue=xtrue,
                D=D,
                y=y,
                xi=xi,
                is_2d=True,
                description="2D identity operator with fixed target values.",
            ),
            None,
        )
    if dataset_name == "Diagonal operator toy problem":
        D = np.diag([float(diag_a), float(diag_b)]).astype(float)
        xtrue = np.array([0.8, 0.35], dtype=float)
        y = D @ xtrue
        xi = float(toy_xi)
        return (
            ProblemData(
                name=dataset_name,
                xtrue=xtrue,
                D=D,
                y=y,
                xi=xi,
                is_2d=True,
                description="2D diagonal operator toy problem with exact synthetic observation.",
            ),
            None,
        )
    if dataset_name == "Simulated toolbox dataset":
        sim = load_spoq_data_simulated(seed=simulated_seed)
        return (
            ProblemData(
                name=dataset_name,
                xtrue=sim.xtrue,
                D=sim.K,
                y=sim.y,
                xi=sim.xi,
                is_2d=False,
                description="MATLAB-style simulated sparse signal dataset with Toeplitz convolution matrix.",
            ),
            sim,
        )
    if dataset_name == DATASET_PAPER_STYLE_A:
        sim = load_paper_style_dataset_a(seed=simulated_seed)
        return (
            ProblemData(
                name=dataset_name,
                xtrue=sim.xtrue,
                D=sim.K,
                y=sim.y,
                xi=sim.xi,
                is_2d=False,
                description=(
                    "Paper-style synthetic instance (not an exact paper experiment): N=1000, P=48, "
                    "sparse nonnegative spikes, Gaussian noise, ξ = √N σ (Toeplitz kernel as in the simulated loader)."
                ),
            ),
            sim,
        )
    if dataset_name == DATASET_PAPER_STYLE_B:
        sim = load_paper_style_dataset_b(seed=simulated_seed)
        return (
            ProblemData(
                name=dataset_name,
                xtrue=sim.xtrue,
                D=sim.K,
                y=sim.y,
                xi=sim.xi,
                is_2d=False,
                description=(
                    "Paper-style synthetic instance (not an exact paper experiment): N=1000, P=94, "
                    "sparse nonnegative spikes, Gaussian noise, ξ = √N σ (Toeplitz kernel as in the simulated loader)."
                ),
            ),
            sim,
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def run_full_problem(
    problem: ProblemData,
    params: SpoqParams,
    *,
    solver_mode: int,
    gamma: float,
    max_iter: int,
    use_warm_start: bool,
    warm_start_iter: int,
    prox_max_iter: int = 5000,
    prox_prec: float = 1e-12,
) -> SolverRunResult:
    """Run the validated SPOQ solver on a chosen problem instance."""
    if use_warm_start:
        warm = pds_warmstart(problem.D, problem.y, problem.xi, warm_start_iter)
        x0 = warm.x.copy()
    else:
        warm = None
        x0 = np.zeros(problem.D.shape[1], dtype=float)

    x_final, history = run_spoq_solver(
        x0=x0,
        D=problem.D,
        y=problem.y,
        eta=problem.xi,
        params=params,
        max_iter=max_iter,
        gamma=gamma,
        metric_mode=solver_mode,
        prox_max_iter=prox_max_iter,
        prox_prec=prox_prec,
    )
    snr_db = None if problem.xtrue is None else _snr_db(problem.xtrue, x_final)
    return SolverRunResult(x_init=x0, x_final=x_final, history=history, warmstart=warm, snr_db=snr_db)
