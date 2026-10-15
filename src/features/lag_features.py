"""
RetailPulse - Time Series Aggregation & Lag Feature Engineering
Builds daily and weekly sales aggregates with leakage-free lag and rolling statistics.
"""

from typing import List, Tuple
import pandas as pd
import numpy as np
from src.features.time_features import extract_calendar_features
from src.utils.logger import get_logger

logger = get_logger("lag_features")

def build_daily_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transactions to a continuous daily timeline with zero-fill for closed/idle days.
    """
    logger.info("Aggregating transactions to continuous daily timeline...")
    df_temp = df.copy()
    df_temp["Date"] = pd.to_datetime(df_temp["InvoiceDate"]).dt.normalize()

    daily_agg = df_temp.groupby("Date").agg(
        total_sales=("TotalPrice", "sum"),
        total_quantity=("Quantity", "sum"),
        num_orders=("Invoice", "nunique"),
        num_customers=("CustomerID", "nunique")
    ).reset_index()

    # Re-index to continuous calendar so that lag(7) is exactly 7 calendar days
    full_range = pd.date_range(start=daily_agg["Date"].min(), end=daily_agg["Date"].max(), freq="D", name="Date")
    daily_continuous = daily_agg.set_index("Date").reindex(full_range, fill_value=0.0).reset_index()

    # Derived daily metrics
    daily_continuous["avg_order_value"] = np.where(
        daily_continuous["num_orders"] > 0,
        daily_continuous["total_sales"] / daily_continuous["num_orders"],
        0.0
    ).round(2)
    daily_continuous["avg_basket_size"] = np.where(
        daily_continuous["num_orders"] > 0,
        daily_continuous["total_quantity"] / daily_continuous["num_orders"],
        0.0
    ).round(2)

    return daily_continuous

def build_lag_features(
    daily_df: pd.DataFrame,
    target_cols: List[str] = ["total_sales", "total_quantity"],
    lag_days: List[int] = [1, 2, 3, 7, 14, 28],
    rolling_windows: List[int] = [7, 14, 28]
) -> pd.DataFrame:
    """
    Constructs historical lag features and rolling aggregations.
    
    IMPORTANT LEAKAGE PREVENTION:
    All rolling statistics use `shift(1)` so the current day's value is never 
    included in its own rolling mean/std.
    
    Args:
        daily_df: Continuous daily transactions DataFrame.
        target_cols: Columns on which to compute lags and rolling windows.
        lag_days: Offsets in days for lag extraction.
        rolling_windows: Window sizes in days for rolling stats.
        
    Returns:
        pd.DataFrame: Feature matrix enriched with lags, rolling statistics, and calendar attributes.
    """
    logger.info("Extracting leakage-free lag and rolling statistics...")
    df_feat = daily_df.copy().sort_values("Date").reset_index(drop=True)

    # 1. Attach Calendar and Holiday Features
    calendar_feats = extract_calendar_features(df_feat["Date"], country_holiday="UK")
    for col in calendar_feats.columns:
        if col != "date":
            df_feat[col] = calendar_feats[col].values

    # 2. Add Lag Features
    for target in target_cols:
        for lag in lag_days:
            col_name = f"{target}_lag_{lag}d"
            df_feat[col_name] = df_feat[target].shift(lag)

    # 3. Add Rolling Mean and Std (Shifted by 1 to strictly avoid lookahead leakage)
    for target in target_cols:
        for window in rolling_windows:
            shifted = df_feat[target].shift(1)
            df_feat[f"{target}_rolling_mean_{window}d"] = shifted.rolling(window=window, min_periods=1).mean().round(2)
            df_feat[f"{target}_rolling_std_{window}d"] = shifted.rolling(window=window, min_periods=1).std().fillna(0.0).round(2)
            df_feat[f"{target}_rolling_min_{window}d"] = shifted.rolling(window=window, min_periods=1).min().round(2)
            df_feat[f"{target}_rolling_max_{window}d"] = shifted.rolling(window=window, min_periods=1).max().round(2)

    # 4. Exponentially Weighted Moving Averages (EWMA)
    for target in target_cols:
        shifted = df_feat[target].shift(1)
        df_feat[f"{target}_ewma_7d"] = shifted.ewm(span=7, min_periods=1).mean().round(2)
        df_feat[f"{target}_ewma_28d"] = shifted.ewm(span=28, min_periods=1).mean().round(2)

    logger.info(f"Built daily time-series matrix: {len(df_feat)} days with {len(df_feat.columns)} features.")
    return df_feat

def build_weekly_time_series(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates daily sales to weekly frequency with lag and rolling features.
    """
    logger.info("Constructing weekly aggregated sales series...")
    df_w = daily_df.set_index("Date").resample("W-MON").agg(
        weekly_sales=("total_sales", "sum"),
        weekly_quantity=("total_quantity", "sum"),
        weekly_orders=("num_orders", "sum"),
        weekly_customers=("num_customers", "sum")
    ).reset_index()

    # Weekly lags (1, 2, 4, 12 weeks)
    for lag in [1, 2, 4, 8]:
        df_w[f"weekly_sales_lag_{lag}w"] = df_w["weekly_sales"].shift(lag)
        df_w[f"weekly_qty_lag_{lag}w"] = df_w["weekly_quantity"].shift(lag)

    # Weekly rolling stats
    for win in [4, 8, 12]:
        shifted = df_w["weekly_sales"].shift(1)
        df_w[f"weekly_sales_rolling_mean_{win}w"] = shifted.rolling(win, min_periods=1).mean().round(2)
        df_w[f"weekly_sales_rolling_std_{win}w"] = shifted.rolling(win, min_periods=1).std().fillna(0).round(2)

    return df_w
