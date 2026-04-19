"""Tests for paper-style synthetic loaders (N=1000, P=48 / P=94)."""

from __future__ import annotations

import numpy as np

from load_spoq_data_paper_style import load_paper_style_dataset_a, load_paper_style_dataset_b, load_paper_style_spoq_data
from webapp.app_utils import DATASET_PAPER_STYLE_A, DATASET_PAPER_STYLE_B, make_problem_data


def test_paper_style_xi_rule_and_sparsity() -> None:
    for loader in (load_paper_style_dataset_a, load_paper_style_dataset_b):
        data = loader(seed=2)
        n = data.n_sample
        assert data.xtrue.shape == (n,)
        assert data.K.shape == (n, n)
        assert data.y.shape == (n,)
        assert np.count_nonzero(data.xtrue) == data.n_peak
        assert np.min(data.xtrue) >= 0.0
        expected_xi = float(np.sqrt(float(n)) * data.sigma)
        assert np.isclose(data.xi, expected_xi, rtol=0.0, atol=1e-12)


def test_paper_style_preset_sizes() -> None:
    a = load_paper_style_dataset_a(seed=0)
    b = load_paper_style_dataset_b(seed=0)
    assert a.n_sample == 1000 and a.n_peak == 48
    assert b.n_sample == 1000 and b.n_peak == 94


def test_make_problem_data_paper_style() -> None:
    prob_a, raw_a = make_problem_data(DATASET_PAPER_STYLE_A, simulated_seed=1)
    assert prob_a.xtrue is not None
    assert prob_a.xtrue.shape == (1000,)
    assert prob_a is not None and raw_a is not None
    assert prob_a.xi == raw_a.xi
    assert "not an exact paper experimental instance" in prob_a.description.lower() or "not an exact paper experiment" in prob_a.description.lower()

    prob_b, raw_b = make_problem_data(DATASET_PAPER_STYLE_B, simulated_seed=1)
    assert prob_b.xtrue.shape == (1000,)
    assert prob_b.xi == raw_b.xi


def test_custom_paper_style_n_peak() -> None:
    data = load_paper_style_spoq_data(n_sample=100, n_peak=10, seed=0, label="test")
    assert data.n_peak == 10
    assert np.isclose(data.xi, np.sqrt(100.0) * data.sigma)
