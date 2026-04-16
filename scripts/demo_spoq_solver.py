"""Small demo for the minimal SPOQ forward-backward solver."""

from __future__ import annotations

import numpy as np

from spoq_solver import plot_solver_trajectory_on_spoq_contour, run_spoq_solver
from spoq_viz import SOOT_PRESET


def main() -> None:
    D = np.eye(2, dtype=float)
    y = np.array([0.45, 0.0], dtype=float)
    x0 = np.array([0.9, 0.7], dtype=float)
    eta = 0.3

    sol, history = run_spoq_solver(
        x0=x0,
        D=D,
        y=y,
        eta=eta,
        params=SOOT_PRESET,
        max_iter=20,
        gamma=1.0,
        prox_max_iter=3000,
        prox_prec=1e-12,
    )

    print("Final solution:", sol)
    print("Initial Psi:", history.psi_values[0])
    print("Final Psi:", history.psi_values[-1])
    print("Max feasibility residual:", max(history.feasibility_residuals))

    fig, _ = plot_solver_trajectory_on_spoq_contour(
        history,
        SOOT_PRESET,
        x_range=(-0.2, 1.1),
        y_range=(-0.2, 1.1),
        title="SPOQ solver trajectory",
    )
    fig.tight_layout()
    fig.savefig("spoq_solver_trajectory.png", dpi=220)
    print("Saved: spoq_solver_trajectory.png")


if __name__ == "__main__":
    main()
