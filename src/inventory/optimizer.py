"""
RetailPulse - Inventory Optimization Engine (F5)
Computes safety stock, reorder points, and economic order quantity (EOQ)
per SKU using demand forecasts, configurable service levels, and lead times.
Flags overstock and stockout risks with actionable recommendations.
"""

import os
import json
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger("inventory_optimizer")


def compute_inventory_metrics(
    daily_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    forecast_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Computes inventory optimization metrics for top SKUs.

    Formulas:
        - Safety Stock = Z * σ_demand * √(lead_time_days)
        - Reorder Point = (avg_daily_demand * lead_time_days) + safety_stock
        - EOQ = √((2 * annual_demand * ordering_cost) / holding_cost_per_unit)

    Args:
        daily_df: Daily aggregated sales DataFrame (for demand stats).
        transactions_df: Cleaned transaction-level DataFrame (for SKU-level detail).
        forecast_df: Optional forecasted demand DataFrame.

    Returns:
        pd.DataFrame: SKU-level inventory recommendations.
    """
    cfg = load_config()
    inv_cfg = cfg.get("inventory", {})
    z_score = inv_cfg.get("z_score", 1.645)
    lead_time = inv_cfg.get("lead_time_days", 7)
    holding_pct = inv_cfg.get("holding_cost_pct", 0.20)
    ordering_cost = inv_cfg.get("ordering_cost", 50.0)
    top_n = inv_cfg.get("top_n_skus", 50)
    models_dir = cfg.get("paths", {}).get("models_dir", "models")
    proc_dir = cfg.get("paths", {}).get("processed_data_dir", "data/processed")
    os.makedirs(models_dir, exist_ok=True)

    logger.info(f"Computing inventory metrics: Z={z_score}, Lead Time={lead_time}d, Top {top_n} SKUs")

    # 1. SKU-level daily demand statistics
    sku_daily = transactions_df.copy()
    sku_daily["Date"] = pd.to_datetime(sku_daily["InvoiceDate"]).dt.normalize()

    # Get top N SKUs by total revenue
    top_skus = (
        sku_daily.groupby("StockCode")
        .agg(total_revenue=("TotalPrice", "sum"), total_qty=("Quantity", "sum"))
        .nlargest(top_n, "total_revenue")
        .reset_index()
    )
    top_sku_list = top_skus["StockCode"].tolist()

    # Filter to top SKUs
    sku_daily_filtered = sku_daily[sku_daily["StockCode"].isin(top_sku_list)]

    # Daily demand per SKU
    daily_demand = (
        sku_daily_filtered.groupby(["StockCode", "Date"])
        .agg(daily_qty=("Quantity", "sum"), daily_revenue=("TotalPrice", "sum"))
        .reset_index()
    )

    # Compute stats per SKU
    sku_stats = daily_demand.groupby("StockCode").agg(
        avg_daily_demand=("daily_qty", "mean"),
        std_daily_demand=("daily_qty", "std"),
        max_daily_demand=("daily_qty", "max"),
        total_demand=("daily_qty", "sum"),
        active_days=("Date", "nunique"),
        avg_daily_revenue=("daily_revenue", "mean"),
    ).reset_index()

    sku_stats["std_daily_demand"] = sku_stats["std_daily_demand"].fillna(0)

    # Get descriptions for top SKUs
    desc_map = (
        transactions_df[transactions_df["StockCode"].isin(top_sku_list)]
        .groupby("StockCode")["Description"]
        .first()
        .reset_index()
    )
    sku_stats = sku_stats.merge(desc_map, on="StockCode", how="left")

    # Get avg price per SKU
    price_map = (
        transactions_df[transactions_df["StockCode"].isin(top_sku_list)]
        .groupby("StockCode")["Price"]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={"Price": "avg_unit_price"})
    )
    sku_stats = sku_stats.merge(price_map, on="StockCode", how="left")

    # 2. Compute safety stock, reorder point, EOQ
    sku_stats["safety_stock"] = (
        z_score * sku_stats["std_daily_demand"] * np.sqrt(lead_time)
    ).round(0).astype(int)

    sku_stats["reorder_point"] = (
        (sku_stats["avg_daily_demand"] * lead_time) + sku_stats["safety_stock"]
    ).round(0).astype(int)

    # EOQ
    annual_demand = sku_stats["avg_daily_demand"] * 365
    holding_cost = sku_stats["avg_unit_price"].fillna(1) * holding_pct
    sku_stats["eoq"] = np.sqrt(
        (2 * annual_demand * ordering_cost) / holding_cost.clip(lower=0.01)
    ).round(0).astype(int)

    # 3. Risk assessment
    # Estimated current stock (simplified: assume last 7 days average * 3 as proxy)
    sku_stats["est_days_of_stock"] = np.where(
        sku_stats["avg_daily_demand"] > 0,
        (sku_stats["safety_stock"] * 2) / sku_stats["avg_daily_demand"],
        999,
    ).round(1)

    def classify_risk(row):
        if row["est_days_of_stock"] < lead_time:
            return "🔴 Stockout Risk"
        elif row["est_days_of_stock"] > lead_time * 4:
            return "🟡 Overstock Risk"
        else:
            return "🟢 Optimal"

    sku_stats["risk_status"] = sku_stats.apply(classify_risk, axis=1)

    # 4. Recommendations
    def recommend(row):
        if "Stockout" in row["risk_status"]:
            return f"URGENT: Reorder {row['eoq']} units immediately. Below reorder point ({row['reorder_point']} units)."
        elif "Overstock" in row["risk_status"]:
            return f"Reduce orders. Current surplus exceeds {lead_time * 3}d of demand. Consider promotions."
        else:
            return f"Maintain EOQ of {row['eoq']} units with reorder at {row['reorder_point']} units."

    sku_stats["recommendation"] = sku_stats.apply(recommend, axis=1)

    # Sort by risk priority
    risk_order = {"🔴 Stockout Risk": 0, "🟡 Overstock Risk": 1, "🟢 Optimal": 2}
    sku_stats["risk_sort"] = sku_stats["risk_status"].map(risk_order)
    sku_stats = sku_stats.sort_values("risk_sort").reset_index(drop=True)
    sku_stats.drop(columns=["risk_sort"], inplace=True)

    # Save
    output_path = os.path.join(proc_dir, "inventory_recommendations.parquet")
    sku_stats.to_parquet(output_path, index=False)

    csv_path = os.path.join(proc_dir, "inventory_recommendations.csv")
    sku_stats.to_csv(csv_path, index=False)

    logger.info(f"Inventory recommendations saved: {len(sku_stats)} SKUs")
    logger.info(f"Risk distribution:\n{sku_stats['risk_status'].value_counts().to_string()}")

    return sku_stats
