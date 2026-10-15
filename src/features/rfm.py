"""
RetailPulse - Customer RFM & Behavioral Feature Engineering
Calculates Recency, Frequency, Monetary value, Average Order Value, Purchase Interval, and Tenure.
"""

from typing import Optional
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("rfm_features")

def calculate_rfm_features(
    df: pd.DataFrame, 
    reference_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Computes customer-level RFM, basket metrics, tenure, and purchase intervals.
    
    Args:
        df: Cleaned transaction DataFrame containing [Invoice, StockCode, Quantity, InvoiceDate, Price, CustomerID, TotalPrice, Country].
        reference_date: Cutoff date for recency calculation (defaults to max(InvoiceDate) + 1 day).
        
    Returns:
        pd.DataFrame: Customer-level feature table.
    """
    logger.info("Computing customer-level RFM and behavioral metrics...")
    
    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    logger.info(f"Using RFM reference snapshot date: {reference_date.strftime('%Y-%m-%d %H:%M:%S')}")

    # Group by CustomerID
    cust_group = df.groupby("CustomerID")

    # Order-level aggregation first (for exact order-level basket calculations)
    order_agg = df.groupby(["CustomerID", "Invoice"]).agg(
        OrderDate=("InvoiceDate", "min"),
        OrderTotal=("TotalPrice", "sum"),
        OrderItems=("Quantity", "sum")
    ).reset_index()

    # Customer aggregates
    rfm_df = cust_group.agg(
        FirstOrderDate=("InvoiceDate", "min"),
        LastOrderDate=("InvoiceDate", "max"),
        Frequency=("Invoice", "nunique"),
        TotalItems=("Quantity", "sum"),
        Monetary=("TotalPrice", "sum"),
        UniqueSKUs=("StockCode", "nunique")
    ).reset_index()

    # Calculate Recency in days
    rfm_df["Recency"] = (reference_date - rfm_df["LastOrderDate"]).dt.days

    # Calculate Tenure in days (distance between first purchase and reference date or last purchase)
    # Total customer lifespan from first purchase:
    rfm_df["TenureDays"] = (reference_date - rfm_df["FirstOrderDate"]).dt.days
    rfm_df["ActiveLifespanDays"] = (rfm_df["LastOrderDate"] - rfm_df["FirstOrderDate"]).dt.days

    # Calculate Average Order Value (AOV)
    rfm_df["AvgOrderValue"] = (rfm_df["Monetary"] / rfm_df["Frequency"]).round(2)

    # Calculate Average Basket Size (items per order)
    rfm_df["AvgBasketSize"] = (rfm_df["TotalItems"] / rfm_df["Frequency"]).round(2)

    # Calculate Average Purchase Interval (days between successive orders)
    # For Frequency > 1: ActiveLifespanDays / (Frequency - 1)
    rfm_df["AvgPurchaseInterval"] = np.where(
        rfm_df["Frequency"] > 1,
        (rfm_df["ActiveLifespanDays"] / (rfm_df["Frequency"] - 1)).round(1),
        0.0
    )

    # Attach primary country
    primary_country = cust_group["Country"].agg(lambda s: s.mode().iloc[0] if not s.empty else "Unknown").reset_index()
    rfm_df = rfm_df.merge(primary_country, on="CustomerID", how="left")

    logger.info(f"Derived RFM features for {len(rfm_df):,} distinct customers.")
    logger.info(f"Summary: Mean Recency={rfm_df['Recency'].mean():.1f}d | Mean Frequency={rfm_df['Frequency'].mean():.1f} | Mean Monetary=GBP {rfm_df['Monetary'].mean():,.2f}")
    
    return rfm_df
