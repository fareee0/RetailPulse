"""
RetailPulse - Phase 2 Feature Pipeline Runner
Builds and saves customer RFM features, daily/weekly lag matrices, and churn datasets.
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.utils.config import load_config
from src.features.rfm import calculate_rfm_features
from src.features.lag_features import (
    build_daily_time_series, 
    build_lag_features, 
    build_weekly_time_series
)
from src.features.churn_features import build_churn_features

logger = get_logger("run_features")

def main():
    logger.info("=" * 65)
    logger.info("Executing Phase 2: Feature Engineering Pipeline")
    logger.info("=" * 65)

    cfg = load_config()
    proc_dir = cfg["paths"]["processed_data_dir"]
    cleaned_path = cfg["data"]["cleaned_parquet"]

    if not os.path.exists(cleaned_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_path}. Run Phase 1 first.")

    logger.info(f"Loading cleaned transactions from: {cleaned_path}")
    df = pd.read_parquet(cleaned_path)
    logger.info(f"Loaded {len(df):,} cleaned transactions.")

    # 1. Customer RFM & Behavioral Features
    logger.info(">>> 1. Engineering Customer RFM & Behavioral Features...")
    df_rfm = calculate_rfm_features(df)
    rfm_path = os.path.join(proc_dir, "customer_features.parquet")
    df_rfm.to_parquet(rfm_path, index=False)
    logger.info(f"Saved Customer Features to: {rfm_path} ({len(df_rfm):,} rows)")

    # 2. Time-Series Aggregates and Lag Features
    logger.info(">>> 2. Engineering Daily Time Series & Lag Features...")
    df_daily = build_daily_time_series(df)
    df_daily_features = build_lag_features(
        df_daily,
        target_cols=["total_sales", "total_quantity"],
        lag_days=cfg["features"].get("lag_windows", [7, 14, 28]),
        rolling_windows=cfg["features"].get("rolling_windows", [7, 14, 28])
    )
    daily_path = os.path.join(proc_dir, "daily_sales_features.parquet")
    df_daily_features.to_parquet(daily_path, index=False)
    logger.info(f"Saved Daily Sales Features to: {daily_path} ({len(df_daily_features):,} days)")

    # 3. Weekly Aggregates
    logger.info(">>> 3. Engineering Weekly Time Series...")
    df_weekly = build_weekly_time_series(df_daily)
    weekly_path = os.path.join(proc_dir, "weekly_sales_features.parquet")
    df_weekly.to_parquet(weekly_path, index=False)
    logger.info(f"Saved Weekly Sales Features to: {weekly_path} ({len(df_weekly):,} weeks)")

    # 4. Leakage-Free Churn Features (90-day evaluation horizon)
    logger.info(">>> 4. Engineering Leakage-Free Churn Features...")
    churn_window = cfg["features"].get("churn_cutoff_days", 90)
    df_churn, cutoff_date = build_churn_features(df, churn_window_days=churn_window)
    churn_path = os.path.join(proc_dir, "churn_features.parquet")
    df_churn.to_parquet(churn_path, index=False)
    logger.info(f"Saved Churn Modeling Dataset to: {churn_path} ({len(df_churn):,} customers)")

    print("\n" + "=" * 65)
    print("PHASE 2 - FEATURE ENGINEERING COMPLETE")
    print("=" * 65)
    print(f"Customer Profiles:      {len(df_rfm):,} profiles")
    print(f"Daily Time Horizon:     {len(df_daily_features):,} days ({df_daily_features['Date'].min().strftime('%Y-%m-%d')} to {df_daily_features['Date'].max().strftime('%Y-%m-%d')})")
    print(f"Weekly Time Horizon:    {len(df_weekly):,} weeks")
    print(f"Churn Cohort Size:      {len(df_churn):,} customers (Cutoff: {cutoff_date.strftime('%Y-%m-%d')})")
    print(f"Overall Churn Rate:     {df_churn['Churn'].mean()*100:.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    main()
