"""
Unit Tests for RetailPulse Phase 1 Data Cleaning Pipeline
"""

import pytest
import pandas as pd
import numpy as np
from src.data.clean import clean_retail_data, standardize_columns

@pytest.fixture
def sample_raw_data():
    """Generates a representative synthetic raw batch with edge cases."""
    data = {
        "Invoice": ["536365", "C536366", "536367", "536368", "536369", "536370", "536371", "536371"],
        "StockCode": ["85123A", "85123A", "71053", "84029G", "84029E", "22752", "21730", "21730"],
        "Description": ["WHITE HANGING HEART", "WHITE HANGING HEART", "WHITE METAL LANTERN", 
                        "KNITTED UNION FLAG", "RED WOOLLY HOTTIE", "SET 7 BABUSHKA", "GLASS STAR", "GLASS STAR"],
        "Quantity": [6, -1, 0, 8, 4, 2, 10, 10],   # Note: -1 (cancel), 0 (invalid)
        "InvoiceDate": ["2010-12-01 08:26:00", "2010-12-01 08:28:00", "2010-12-01 08:34:00",
                        "2010-12-01 08:34:00", "2010-12-01 08:35:00", "2010-12-01 08:45:00",
                        "2010-12-01 09:00:00", "2010-12-01 09:00:00"],
        "Price": [2.55, 2.55, 3.39, -1.0, 0.0, 7.65, 4.25, 4.25], # Note: -1.0 (invalid), 0.0 (invalid)
        "Customer ID": [17850, 17850, 17850, 13047, 12583, np.nan, 15808, 15808], # Note: np.nan (null customer)
        "Country": ["United Kingdom", "United Kingdom", "United Kingdom", 
                    "United Kingdom", "France", "Germany", "United Kingdom", "United Kingdom"]
    }
    return pd.DataFrame(data)

def test_standardize_columns(sample_raw_data):
    """Test column normalization to standard names."""
    df_std = standardize_columns(sample_raw_data)
    assert "CustomerID" in df_std.columns
    assert "Customer ID" not in df_std.columns
    assert "Price" in df_std.columns

def test_remove_null_customers(sample_raw_data):
    """Test dropping of rows with missing CustomerID."""
    clean_df, audit = clean_retail_data(sample_raw_data, save_output=False)
    assert not clean_df["CustomerID"].isna().any()
    step_audit = next(s for s in audit["steps"] if s["step"] == "remove_null_customer_id")
    assert step_audit["rows_removed"] == 1  # Row with np.nan Customer ID

def test_filter_cancellations(sample_raw_data):
    """Test removal of cancellation invoices starting with 'C'."""
    clean_df, audit = clean_retail_data(sample_raw_data, save_output=False)
    assert not clean_df["Invoice"].str.startswith("C").any()
    step_audit = next(s for s in audit["steps"] if s["step"] == "filter_cancellations")
    assert step_audit["rows_removed"] == 1  # C536366

def test_filter_non_positive_quantity_and_price(sample_raw_data):
    """Test filtering of non-positive quantities and prices."""
    clean_df, audit = clean_retail_data(sample_raw_data, save_output=False)
    assert (clean_df["Quantity"] > 0).all()
    assert (clean_df["Price"] > 0).all()

def test_drop_duplicates(sample_raw_data):
    """Test removal of exact duplicate transactions."""
    clean_df, audit = clean_retail_data(sample_raw_data, save_output=False)
    step_audit = next(s for s in audit["steps"] if s["step"] == "drop_duplicate_rows")
    assert step_audit["rows_removed"] == 1  # 536371 was duplicated

def test_total_price_calculation(sample_raw_data):
    """Test TotalPrice calculation."""
    clean_df, _ = clean_retail_data(sample_raw_data, save_output=False)
    expected = (clean_df["Quantity"] * clean_df["Price"]).round(4)
    assert np.allclose(clean_df["TotalPrice"], expected)

def test_data_types_and_chronology(sample_raw_data):
    """Test resulting data types and chronological sorting."""
    clean_df, _ = clean_retail_data(sample_raw_data, save_output=False)
    assert pd.api.types.is_datetime64_any_dtype(clean_df["InvoiceDate"])
    assert pd.api.types.is_string_dtype(clean_df["CustomerID"]) or clean_df["CustomerID"].dtype == object
    assert clean_df["InvoiceDate"].is_monotonic_increasing
