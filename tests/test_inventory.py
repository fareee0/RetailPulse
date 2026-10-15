"""
Unit Tests for Inventory Optimization Formulas (Safety Stock, ROP, EOQ)
"""

import pytest
import numpy as np
import pandas as pd
from src.inventory.optimizer import compute_inventory_metrics


def test_inventory_formulas():
    # Build a small dummy transaction dataset
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    records = []
    for d in dates:
        records.append({
            "Invoice": "INV1",
            "StockCode": "SKU_TOP",
            "Description": "Test Product",
            "Quantity": 10,
            "InvoiceDate": d,
            "Price": 25.0,
            "CustomerID": "C1",
            "Country": "UK",
            "TotalPrice": 250.0
        })

    df = pd.DataFrame(records)
    inv_df = compute_inventory_metrics(daily_df=pd.DataFrame(), transactions_df=df)

    assert not inv_df.empty
    row = inv_df.iloc[0]

    assert "safety_stock" in inv_df.columns
    assert "reorder_point" in inv_df.columns
    assert "eoq" in inv_df.columns
    assert "risk_status" in inv_df.columns

    # avg_daily_demand should be 10 units
    assert row["avg_daily_demand"] == pytest.approx(10.0, 0.1)
    # ROP should be >= 0
    assert row["reorder_point"] >= 0
    # EOQ should be > 0
    assert row["eoq"] > 0
