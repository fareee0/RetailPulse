"""
Unit Tests for RetailPulse Forecasting Engine (Prophet, LSTM sequence, Hybrid metrics)
"""

import pytest
import numpy as np
import pandas as pd
from src.models.prophet_model import compute_forecast_metrics, prepare_prophet_data
from src.models.lstm_model import create_sequences


def test_prepare_prophet_data():
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({"Date": dates, "total_sales": np.arange(10) * 100.0})
    prophet_df = prepare_prophet_data(df)

    assert "ds" in prophet_df.columns
    assert "y" in prophet_df.columns
    assert len(prophet_df) == 10
    assert prophet_df["y"].iloc[-1] == 900.0


def test_compute_forecast_metrics():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])

    metrics = compute_forecast_metrics(y_true, y_pred, model_name="TestModel")
    assert metrics["mae"] == pytest.approx(6.67, 0.05)
    assert metrics["rmse"] > 0
    assert metrics["mape_pct"] > 0
    assert metrics["smape_pct"] > 0


def test_create_sequences():
    data = np.arange(20, dtype=float)
    seq_len = 5
    X, y = create_sequences(data, seq_length=seq_len)

    assert X.shape == (15, 5)
    assert y.shape == (15,)
    assert np.array_equal(X[0], np.array([0, 1, 2, 3, 4]))
    assert y[0] == 5.0
