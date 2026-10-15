"""
RetailPulse - Data Cleaning Pipeline Module
Implements reusable, production-grade cleaning rules with detailed audit step logging.
"""

import os
from typing import Tuple, Dict, Any, Optional
import pandas as pd
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger("data_cleaner")

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column names across dataset variants."""
    rename_map = {
        "Customer ID": "CustomerID",
        "customer_id": "CustomerID",
        "UnitPrice": "Price",
        "unit_price": "Price",
        "InvoiceNo": "Invoice",
        "invoice": "Invoice",
        "StockCode": "StockCode",
        "stock_code": "StockCode",
        "Description": "Description",
        "description": "Description",
        "Quantity": "Quantity",
        "quantity": "Quantity",
        "InvoiceDate": "InvoiceDate",
        "invoice_date": "InvoiceDate",
        "Price": "Price",
        "price": "Price",
        "Country": "Country",
        "country": "Country"
    }
    return df.rename(columns=rename_map)

def clean_retail_data(
    df: pd.DataFrame, 
    config: Optional[Dict[str, Any]] = None,
    save_output: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the multi-stage retail data cleaning pipeline.
    
    Cleaning Stages:
      1. Standardize column names.
      2. Remove rows where CustomerID is null/NaN.
      3. Identify and isolate cancellations (invoices starting with 'C').
      4. Filter out non-positive quantities (Quantity <= 0) and non-positive prices (Price <= 0).
      5. Deduplicate transaction records.
      6. Enforce strict data types.
      7. Calculate TotalPrice (Quantity * Price).
      8. Persist cleaned data to data/processed/.

    Args:
        df: Raw combined transactions DataFrame.
        config: Optional configuration dictionary.
        save_output: Whether to persist results to disk.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: (Cleaned DataFrame, Audit Report Dict).
    """
    if config is None:
        config = load_config()

    audit: Dict[str, Any] = {
        "raw_rows": len(df),
        "steps": []
    }

    logger.info("=" * 60)
    logger.info("Starting RetailPulse Data Cleaning Pipeline")
    logger.info(f"Raw Input Row Count: {audit['raw_rows']:,}")
    logger.info("=" * 60)

    # Step 0: Column Normalization
    df_clean = standardize_columns(df.copy())
    current_count = len(df_clean)

    # Step 1: Remove Null Customer IDs
    if config.get("cleaning", {}).get("drop_null_customers", True):
        null_cust_mask = df_clean["CustomerID"].isna()
        null_cust_count = int(null_cust_mask.sum())
        df_clean = df_clean[~null_cust_mask]
        audit["steps"].append({
            "step": "remove_null_customer_id",
            "rows_removed": null_cust_count,
            "remaining_rows": len(df_clean),
            "pct_removed": round((null_cust_count / current_count) * 100, 2)
        })
        logger.info(f"[Step 1] Removed Null CustomerIDs: {null_cust_count:,} rows (Remaining: {len(df_clean):,})")
        current_count = len(df_clean)

    # Step 2: Handle Cancellations (Invoices starting with 'C')
    if config.get("cleaning", {}).get("filter_cancellations", True):
        # Invoices can be integers or strings, convert to string for checking
        inv_str = df_clean["Invoice"].astype(str).str.strip().str.upper()
        cancellation_mask = inv_str.str.startswith("C") | (df_clean["Quantity"] < 0)
        cancellation_count = int(cancellation_mask.sum())
        
        # Save cancellations separately for audit and churn analysis
        df_cancellations = df_clean[cancellation_mask].copy()
        audit["cancellation_rows"] = cancellation_count
        
        df_clean = df_clean[~cancellation_mask]
        audit["steps"].append({
            "step": "filter_cancellations",
            "rows_removed": cancellation_count,
            "remaining_rows": len(df_clean),
            "pct_removed": round((cancellation_count / current_count) * 100, 2)
        })
        logger.info(f"[Step 2] Filtered Cancellation Invoices (starting with 'C'): {cancellation_count:,} rows (Remaining: {len(df_clean):,})")
        current_count = len(df_clean)

    # Step 3: Remove Non-Positive Quantity (Quantity <= 0)
    min_qty = config.get("cleaning", {}).get("min_quantity", 1)
    non_pos_qty_mask = df_clean["Quantity"] < min_qty
    qty_removed = int(non_pos_qty_mask.sum())
    df_clean = df_clean[~non_pos_qty_mask]
    audit["steps"].append({
        "step": "filter_non_positive_quantity",
        "rows_removed": qty_removed,
        "remaining_rows": len(df_clean),
        "pct_removed": round((qty_removed / current_count) * 100, 2) if current_count else 0
    })
    logger.info(f"[Step 3] Filtered Non-Positive Quantities (< {min_qty}): {qty_removed:,} rows (Remaining: {len(df_clean):,})")
    current_count = len(df_clean)

    # Step 4: Remove Non-Positive Price (Price <= 0)
    min_price = config.get("cleaning", {}).get("min_price", 0.0001)
    non_pos_price_mask = df_clean["Price"] < min_price
    price_removed = int(non_pos_price_mask.sum())
    df_clean = df_clean[~non_pos_price_mask]
    audit["steps"].append({
        "step": "filter_non_positive_price",
        "rows_removed": price_removed,
        "remaining_rows": len(df_clean),
        "pct_removed": round((price_removed / current_count) * 100, 2) if current_count else 0
    })
    logger.info(f"[Step 4] Filtered Non-Positive Prices (< {min_price}): {price_removed:,} rows (Remaining: {len(df_clean):,})")
    current_count = len(df_clean)

    # Step 5: Drop Duplicates
    if config.get("cleaning", {}).get("drop_duplicates", True):
        initial_dup = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        dups_removed = initial_dup - len(df_clean)
        audit["steps"].append({
            "step": "drop_duplicate_rows",
            "rows_removed": dups_removed,
            "remaining_rows": len(df_clean),
            "pct_removed": round((dups_removed / current_count) * 100, 2) if current_count else 0
        })
        logger.info(f"[Step 5] Dropped Duplicate Rows: {dups_removed:,} rows (Remaining: {len(df_clean):,})")
        current_count = len(df_clean)

    # Step 6: Fix Data Types and Format Columns
    logger.info("[Step 6] Enforcing strict dtypes and standard formats...")
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
    df_clean["CustomerID"] = df_clean["CustomerID"].astype(int).astype(str)
    df_clean["Quantity"] = df_clean["Quantity"].astype(int)
    df_clean["Price"] = df_clean["Price"].astype(float)
    df_clean["Invoice"] = df_clean["Invoice"].astype(str)
    df_clean["StockCode"] = df_clean["StockCode"].astype(str)
    df_clean["Country"] = df_clean["Country"].astype(str).str.strip()
    if "Description" in df_clean.columns:
        df_clean["Description"] = df_clean["Description"].fillna("").astype(str).str.strip()

    # Step 7: Add TotalPrice Column
    logger.info("[Step 7] Calculating 'TotalPrice' (Quantity * Price)...")
    df_clean["TotalPrice"] = (df_clean["Quantity"] * df_clean["Price"]).round(4)

    # Sort Chronologically
    df_clean = df_clean.sort_values(by=["InvoiceDate", "Invoice"]).reset_index(drop=True)

    # Final Audit Summary
    audit["final_rows"] = len(df_clean)
    audit["total_rows_removed"] = audit["raw_rows"] - audit["final_rows"]
    audit["retention_rate_pct"] = round((audit["final_rows"] / audit["raw_rows"]) * 100, 2) if audit["raw_rows"] else 0.0
    audit["unique_customers"] = int(df_clean["CustomerID"].nunique())
    audit["unique_products"] = int(df_clean["StockCode"].nunique())
    audit["unique_countries"] = int(df_clean["Country"].nunique())
    audit["total_revenue"] = round(float(df_clean["TotalPrice"].sum()), 2)
    audit["date_min"] = df_clean["InvoiceDate"].min().strftime("%Y-%m-%d")
    audit["date_max"] = df_clean["InvoiceDate"].max().strftime("%Y-%m-%d")

    logger.info("-" * 60)
    logger.info("Cleaning Pipeline Completed Summary:")
    logger.info(f"  Final Valid Records:  {audit['final_rows']:,} / {audit['raw_rows']:,} ({audit['retention_rate_pct']}%)")
    logger.info(f"  Unique Customers:     {audit['unique_customers']:,}")
    logger.info(f"  Unique SKUs:          {audit['unique_products']:,}")
    logger.info(f"  Countries:            {audit['unique_countries']}")
    logger.info(f"  Total Revenue:        GBP {audit['total_revenue']:,.2f}")
    logger.info(f"  Date Horizon:         {audit['date_min']} to {audit['date_max']}")
    logger.info("-" * 60)

    # Step 8: Save to data/processed
    if save_output:
        proc_dir = config.get("paths", {}).get("processed_data_dir", "data/processed")
        os.makedirs(proc_dir, exist_ok=True)
        parquet_path = config.get("data", {}).get("cleaned_parquet", os.path.join(proc_dir, "online_retail_cleaned.parquet"))
        csv_path = config.get("data", {}).get("cleaned_csv", os.path.join(proc_dir, "online_retail_cleaned.csv"))
        
        logger.info(f"Saving cleaned dataset to Parquet: {parquet_path}")
        df_clean.to_parquet(parquet_path, index=False)
        
        logger.info(f"Saving cleaned dataset sample/full to CSV: {csv_path}")
        df_clean.to_csv(csv_path, index=False)

    return df_clean, audit
