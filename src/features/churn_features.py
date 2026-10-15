"""
RetailPulse - Leakage-Free Churn Feature Engineering
Defines customer churn over a fixed prediction window and derives features strictly 
from historical observation data prior to the cutoff date.
"""

from typing import Tuple, Optional
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger("churn_features")

def build_churn_features(
    df: pd.DataFrame, 
    churn_window_days: int = 90,
    cutoff_date: Optional[pd.Timestamp] = None
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    Constructs a leakage-free churn modeling dataset.
    
    Architecture:
      - Cutoff Date = T_max - churn_window_days.
      - Observation Window = [T_min, Cutoff Date): features engineered exclusively here.
      - Evaluation Window = [Cutoff Date, T_max]: target label determined here.
      - Churn = 1 if customer had 0 transactions in Evaluation Window, else 0.
      
    Args:
        df: Cleaned transactions DataFrame.
        churn_window_days: Prediction horizon length (default: 90 days).
        cutoff_date: Optional manual cutoff date override.
        
    Returns:
        Tuple[pd.DataFrame, pd.Timestamp]: (Enriched churn feature dataset, cutoff date).
    """
    logger.info(f"Building churn dataset with {churn_window_days}-day evaluation window...")
    
    max_date = pd.to_datetime(df["InvoiceDate"]).max()
    if cutoff_date is None:
        cutoff_date = max_date - pd.Timedelta(days=churn_window_days)
        
    logger.info(f"Timeline: Total Range: {df['InvoiceDate'].min().strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    logger.info(f"Timeline: Observation Split Cutoff: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}")

    # Split into historical observation and future evaluation sets
    df_obs = df[df["InvoiceDate"] < cutoff_date].copy()
    df_eval = df[df["InvoiceDate"] >= cutoff_date].copy()

    logger.info(f"Observation transactions: {len(df_obs):,} | Evaluation transactions: {len(df_eval):,}")

    # Eligible cohort: customers active before the cutoff date
    active_customers_before_cutoff = df_obs["CustomerID"].unique()
    retained_customers = set(df_eval["CustomerID"].unique())

    logger.info(f"Eligible customer cohort size: {len(active_customers_before_cutoff):,} customers.")

    # 1. Feature Engineering on df_obs ONLY
    cust_obs_group = df_obs.groupby("CustomerID")

    churn_df = cust_obs_group.agg(
        FirstOrderDate=("InvoiceDate", "min"),
        LastOrderDate=("InvoiceDate", "max"),
        Frequency=("Invoice", "nunique"),
        TotalItems=("Quantity", "sum"),
        Monetary=("TotalPrice", "sum"),
        UniqueSKUs=("StockCode", "nunique")
    ).reset_index()

    # Recency relative to Cutoff Date
    churn_df["Recency"] = (cutoff_date - churn_df["LastOrderDate"]).dt.days
    churn_df["TenureDays"] = (cutoff_date - churn_df["FirstOrderDate"]).dt.days
    churn_df["ActiveLifespanDays"] = (churn_df["LastOrderDate"] - churn_df["FirstOrderDate"]).dt.days

    # Average Order Value and Basket Size
    churn_df["AvgOrderValue"] = (churn_df["Monetary"] / churn_df["Frequency"]).round(2)
    churn_df["AvgBasketSize"] = (churn_df["TotalItems"] / churn_df["Frequency"]).round(2)

    # Average Purchase Interval before Cutoff
    churn_df["AvgPurchaseInterval"] = np.where(
        churn_df["Frequency"] > 1,
        (churn_df["ActiveLifespanDays"] / (churn_df["Frequency"] - 1)).round(1),
        0.0
    )

    # Recent Trend Velocities (last 30d vs last 90d before cutoff)
    t_30d = cutoff_date - pd.Timedelta(days=30)
    t_90d = cutoff_date - pd.Timedelta(days=90)
    
    orders_30d = df_obs[df_obs["InvoiceDate"] >= t_30d].groupby("CustomerID")["Invoice"].nunique().rename("OrdersLast30d")
    spend_30d = df_obs[df_obs["InvoiceDate"] >= t_30d].groupby("CustomerID")["TotalPrice"].sum().rename("SpendLast30d")
    orders_90d = df_obs[df_obs["InvoiceDate"] >= t_90d].groupby("CustomerID")["Invoice"].nunique().rename("OrdersLast90d")
    spend_90d = df_obs[df_obs["InvoiceDate"] >= t_90d].groupby("CustomerID")["TotalPrice"].sum().rename("SpendLast90d")

    churn_df = churn_df.merge(orders_30d, on="CustomerID", how="left").fillna({"OrdersLast30d": 0})
    churn_df = churn_df.merge(spend_30d, on="CustomerID", how="left").fillna({"SpendLast30d": 0.0})
    churn_df = churn_df.merge(orders_90d, on="CustomerID", how="left").fillna({"OrdersLast90d": 0})
    churn_df = churn_df.merge(spend_90d, on="CustomerID", how="left").fillna({"SpendLast90d": 0.0})

    # Primary Country
    country_map = cust_obs_group["Country"].agg(lambda s: s.mode().iloc[0] if not s.empty else "Unknown").reset_index()
    churn_df = churn_df.merge(country_map, on="CustomerID", how="left")

    # 2. Derive Ground Truth Target
    # Churn = 1 if customer NOT present in evaluation window, else 0
    churn_df["Churn"] = (~churn_df["CustomerID"].isin(retained_customers)).astype(int)

    churn_rate = churn_df["Churn"].mean() * 100
    logger.info(f"Derived Churn Dataset: {len(churn_df):,} customers. Churn Rate: {churn_rate:.2f}% ({churn_df['Churn'].sum()} churned, {(churn_df['Churn']==0).sum()} retained).")

    return churn_df, cutoff_date
