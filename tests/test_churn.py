"""
Unit Tests for RetailPulse Churn Prediction Logic
"""

import pytest
import numpy as np
import pandas as pd
from src.features.churn_features import build_churn_features
from src.models.churn import prepare_churn_data


def test_build_churn_features_leakage_prevention():
    # Construct mock dataset across 180 days
    dates = pd.date_range("2023-01-01", periods=180, freq="D")
    records = []
    
    # Customer 1: active throughout (retained)
    for d in dates[::10]:
        records.append({"Invoice": "1", "StockCode": "A", "Quantity": 2, "InvoiceDate": d, "Price": 10.0, "CustomerID": "C1", "Country": "UK", "TotalPrice": 20.0})

    # Customer 2: active only in first 60 days (churned in last 90 days)
    for d in dates[:60:10]:
        records.append({"Invoice": "2", "StockCode": "B", "Quantity": 1, "InvoiceDate": d, "Price": 15.0, "CustomerID": "C2", "Country": "UK", "TotalPrice": 15.0})

    df = pd.DataFrame(records)
    churn_df, cutoff = build_churn_features(df, churn_window_days=90)

    assert "Churn" in churn_df.columns
    c1_row = churn_df[churn_df["CustomerID"] == "C1"].iloc[0]
    c2_row = churn_df[churn_df["CustomerID"] == "C2"].iloc[0]

    assert c1_row["Churn"] == 0  # active in evaluation window
    assert c2_row["Churn"] == 1  # 0 orders in evaluation window


def test_prepare_churn_data():
    df = pd.DataFrame({
        "CustomerID": ["C1", "C2", "C3", "C4"],
        "Recency": [10, 40, 90, 120],
        "Frequency": [10, 5, 2, 1],
        "Monetary": [1000, 500, 200, 100],
        "Churn": [0, 0, 1, 1]
    })
    X_train, X_test, y_train, y_test, scaler, feats = prepare_churn_data(
        df, feature_cols=["Recency", "Frequency", "Monetary"], test_size=0.5, random_seed=42
    )

    assert len(X_train) == 2
    assert len(X_test) == 2
    assert len(feats) == 3
