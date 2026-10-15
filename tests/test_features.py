"""
Unit Tests for RetailPulse Phase 2 Feature Engineering
"""

import pytest
import pandas as pd
import numpy as np
from src.features.rfm import calculate_rfm_features
from src.features.time_features import extract_calendar_features
from src.features.lag_features import build_daily_time_series, build_lag_features
from src.features.churn_features import build_churn_features

@pytest.fixture
def clean_transactions_sample():
    """Generates synthetic cleaned transaction records for feature testing."""
    dates = pd.date_range("2010-01-01", periods=100, freq="D")
    data = []
    customers = ["C101", "C102", "C103", "C104", "C105"]
    
    for i, d in enumerate(dates):
        cust = customers[i % len(customers)]
        data.append({
            "Invoice": f"INV{1000 + i}",
            "StockCode": f"SKU_{i % 10}",
            "Description": "SAMPLE PRODUCT",
            "Quantity": (i % 5) + 1,
            "InvoiceDate": d,
            "Price": 10.0,
            "CustomerID": cust,
            "Country": "United Kingdom",
            "TotalPrice": ((i % 5) + 1) * 10.0
        })
    return pd.DataFrame(data)

def test_rfm_calculation(clean_transactions_sample):
    """Test Recency, Frequency, Monetary, and AOV calculations."""
    rfm = calculate_rfm_features(clean_transactions_sample)
    
    assert len(rfm) == 5
    assert "Recency" in rfm.columns
    assert "Frequency" in rfm.columns
    assert "Monetary" in rfm.columns
    assert "AvgOrderValue" in rfm.columns
    assert "AvgPurchaseInterval" in rfm.columns
    assert (rfm["Recency"] >= 0).all()
    assert (rfm["Frequency"] >= 1).all()
    assert (rfm["Monetary"] > 0).all()
    
    # Mathematical identity check
    assert np.allclose(rfm["AvgOrderValue"], (rfm["Monetary"] / rfm["Frequency"]).round(2))

def test_purchase_interval_logic():
    """Test purchase interval for single-time vs repeat purchasers."""
    df_single = pd.DataFrame([{
        "Invoice": "INV1", "StockCode": "S1", "Description": "A", "Quantity": 1,
        "InvoiceDate": pd.Timestamp("2010-01-01"), "Price": 10.0, "CustomerID": "C999",
        "Country": "United Kingdom", "TotalPrice": 10.0
    }])
    rfm_single = calculate_rfm_features(df_single)
    assert rfm_single.loc[0, "AvgPurchaseInterval"] == 0.0

def test_calendar_and_holiday_extraction():
    """Test extraction of calendar properties and UK holidays."""
    dates = pd.Series(pd.to_datetime(["2010-12-25", "2010-12-26", "2010-06-15"]))
    feats = extract_calendar_features(dates, country_holiday="UK")
    
    assert "is_weekend" in feats.columns
    assert "is_uk_holiday" in feats.columns
    assert "sin_month" in feats.columns
    assert "trend_day_index" in feats.columns
    
    # Christmas in UK should be recognized as holiday
    assert feats.loc[0, "is_uk_holiday"] == 1

def test_lag_features_leakage_free(clean_transactions_sample):
    """Test that lag and rolling statistics do not leak future information."""
    daily = build_daily_time_series(clean_transactions_sample)
    lag_df = build_lag_features(daily, lag_days=[7, 14], rolling_windows=[7])
    
    assert "total_sales_lag_7d" in lag_df.columns
    assert "total_sales_rolling_mean_7d" in lag_df.columns
    
    # Check that lag_7d at index 10 equals raw sales at index 3
    assert lag_df.loc[10, "total_sales_lag_7d"] == lag_df.loc[3, "total_sales"]
    
    # Check that rolling mean at index 7 does NOT include index 7 sales (leakage prevention via shift(1))
    manual_mean_prev_7 = lag_df.loc[0:6, "total_sales"].mean().round(2)
    assert np.isclose(lag_df.loc[7, "total_sales_rolling_mean_7d"], manual_mean_prev_7)

def test_churn_feature_engineering_integrity(clean_transactions_sample):
    """Test churn cohort derivation and target generation without leakage."""
    churn_df, cutoff = build_churn_features(clean_transactions_sample, churn_window_days=30)
    
    assert "Churn" in churn_df.columns
    assert set(churn_df["Churn"].unique()).issubset({0, 1})
    assert "Recency" in churn_df.columns
    assert "Monetary" in churn_df.columns
    
    # Ensure no customer has LastOrderDate after cutoff
    assert (churn_df["LastOrderDate"] < cutoff).all()
