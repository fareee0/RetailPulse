"""
Unit Tests for Data & Concept Drift Detection (PSI and KS-Test)
"""

import pytest
import numpy as np
import pandas as pd
from src.monitoring.drift import calculate_psi, calculate_ks_test, evaluate_dataset_drift


def test_psi_identical_distributions():
    # If distributions are identical, PSI should be very close to 0
    np.random.seed(42)
    ref = np.random.normal(loc=100, scale=15, size=1000)
    cur = np.random.normal(loc=100, scale=15, size=1000)

    psi = calculate_psi(ref, cur, num_bins=10)
    assert psi < 0.10, f"Expected stable PSI < 0.10, got {psi}"


def test_psi_drifted_distribution():
    # Significant shift in mean and spread
    np.random.seed(42)
    ref = np.random.normal(loc=100, scale=10, size=1000)
    cur = np.random.normal(loc=160, scale=35, size=1000)

    psi = calculate_psi(ref, cur, num_bins=10)
    assert psi >= 0.25, f"Expected drifted PSI >= 0.25, got {psi}"


def test_ks_test():
    np.random.seed(42)
    ref = np.random.normal(0, 1, 500)
    cur_drifted = np.random.normal(3, 1, 500)

    ks_stat, p_val = calculate_ks_test(ref, cur_drifted)
    assert ks_stat > 0.5
    assert p_val < 0.05


def test_evaluate_dataset_drift():
    df_ref = pd.DataFrame({
        "feat1": np.random.normal(50, 5, 200),
        "feat2": np.random.normal(10, 2, 200)
    })
    df_cur = pd.DataFrame({
        "feat1": np.random.normal(50, 5, 200),
        "feat2": np.random.normal(50, 10, 200)  # heavily drifted
    })

    report = evaluate_dataset_drift(df_ref, df_cur, feature_cols=["feat1", "feat2"])
    assert "dataset_drift_detected" in report
    assert "feat1" in report["features"]
    assert "feat2" in report["features"]
    assert report["features"]["feat1"]["psi"] < report["features"]["feat2"]["psi"]
