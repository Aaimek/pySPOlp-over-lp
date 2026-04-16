"""Tests for MATLAB-style SPOQ warm start."""

from __future__ import annotations

import numpy as np

from load_spoq_data_simulated import load_spoq_data_simulated
from spoq_warmstart import norm2_estimate, pds_warmstart, proxl1


def test_proxl1_matches_matlab_threshold_then_nonnegativity() -> None:
    x = np.array([-2.0, -0.25, 0.25, 2.0], dtype=float)
    p = proxl1(x, 0.5)
    expected = np.array([0.0, 0.0, 0.0, 1.5], dtype=float)
    assert np.allclose(p, expected, atol=1e-12, rtol=1e-12)


def test_norm2_estimate_is_deterministic() -> None:
    K = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float)
    n1 = norm2_estimate(K, 2, seed=0)
    n2 = norm2_estimate(K, 2, seed=0)
    assert np.isclose(n1, n2, atol=1e-14, rtol=1e-14)


def test_pds_warmstart_shapes_and_repeatability() -> None:
    data = load_spoq_data_simulated(seed=0)
    res1 = pds_warmstart(data.K, data.y, data.xi, nbiter=10, norm_seed=0)
    res2 = pds_warmstart(data.K, data.y, data.xi, nbiter=10, norm_seed=0)
    assert res1.x.shape == data.xtrue.shape
    assert res1.refspec.shape == (10,)
    assert np.allclose(res1.x, res2.x, atol=1e-12, rtol=1e-12)
    assert np.allclose(res1.refspec, res2.refspec, atol=1e-12, rtol=1e-12)


def test_pds_warmstart_stays_nonnegative_and_improves_feasibility_over_ones() -> None:
    data = load_spoq_data_simulated(seed=0)
    res = pds_warmstart(data.K, data.y, data.xi, nbiter=10, norm_seed=0)
    x_ones = np.ones_like(data.xtrue)
    feas_ones = np.linalg.norm(data.K.dot(x_ones) - data.y) - data.xi
    feas_warm = np.linalg.norm(data.K.dot(res.x) - data.y) - data.xi
    assert np.min(res.x) >= -1e-10
    assert feas_warm <= feas_ones + 1e-8
