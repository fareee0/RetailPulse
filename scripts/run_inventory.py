"""
RetailPulse - Phase 6 Inventory Optimization Runner (F5)
Computes Safety Stock, Reorder Point, and Economic Order Quantity (EOQ) for top SKUs.
Identifies overstock and stockout risks and generates interactive dashboard charts.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.utils.config import load_config
from src.inventory.optimizer import compute_inventory_metrics

logger = get_logger("run_inventory")


def main():
    logger.info("=" * 65)
    logger.info("Executing Phase 6: Inventory Optimization Engine (F5)")
    logger.info("=" * 65)

    cfg = load_config()
    proc_dir = cfg["paths"]["processed_data_dir"]
    fig_dir = os.path.join(cfg["paths"].get("reports_dir", "reports"), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Load Cleaned Transactions & Daily Features
    clean_path = cfg["data"]["cleaned_parquet"]
    logger.info(f"Loading cleaned transactions from: {clean_path}")
    df_clean = pd.read_parquet(clean_path)

    daily_path = os.path.join(proc_dir, "daily_sales_features.parquet")
    daily_df = pd.read_parquet(daily_path) if os.path.exists(daily_path) else None

    # 2. Compute Inventory Optimization Metrics
    logger.info("\n>>> Running Inventory Optimization Algorithms...")
    inv_df = compute_inventory_metrics(
        daily_df=daily_df,
        transactions_df=df_clean
    )

    # 3. Visualizations
    logger.info("\n>>> Generating Inventory Visualizations...")
    
    # 3A: Risk Status Distribution Donut Chart
    status_counts = inv_df["risk_status"].value_counts().reset_index()
    status_counts.columns = ["Risk_Status", "Count"]
    color_map = {
        "🔴 Stockout Risk": "#EF4444",
        "🟡 Overstock Risk": "#F59E0B",
        "🟢 Optimal": "#10B981"
    }

    fig_donut = px.pie(
        status_counts,
        names="Risk_Status",
        values="Count",
        hole=0.5,
        title="Inventory Health: Top SKUs Stockout vs Overstock Risk Distribution",
        color="Risk_Status",
        color_discrete_map=color_map,
        template="plotly_white"
    )
    fig_donut.update_traces(textinfo="label+percent+value")
    fig_donut.update_layout(height=420)
    fig_donut.write_html(os.path.join(fig_dir, "inventory_risk_donut.html"))

    # 3B: Reorder Point vs Safety Stock vs EOQ Bar Chart
    top15 = inv_df.head(15)
    fig_bars = go.Figure(data=[
        go.Bar(name="Safety Stock", x=top15["StockCode"], y=top15["safety_stock"], marker_color="#3B82F6"),
        go.Bar(name="Reorder Point", x=top15["StockCode"], y=top15["reorder_point"], marker_color="#F59E0B"),
        go.Bar(name="EOQ", x=top15["StockCode"], y=top15["eoq"], marker_color="#10B981"),
    ])
    fig_bars.update_layout(
        barmode="group",
        title="Inventory Thresholds by SKU (Top 15 Products by Priority)",
        xaxis_title="Stock Code",
        yaxis_title="Units",
        template="plotly_white",
        height=480
    )
    fig_bars.write_html(os.path.join(fig_dir, "inventory_thresholds.html"))

    # Print Summary
    print("\n" + "=" * 65)
    print("PHASE 6 - INVENTORY OPTIMIZATION RECOMMENDATIONS REPORT")
    print("=" * 65)
    print(f"Total Evaluated SKUs: {len(inv_df)}")
    print(f"Risk Breakdown:")
    for status, count in inv_df["risk_status"].value_counts().items():
        print(f"  {status}: {count} SKUs ({count/len(inv_df)*100:.1f}%)")
    print("-" * 65)
    print(f"{'SKU':<10} | {'Description':<28} | {'Safety':<7} | {'ROP':<7} | {'EOQ':<7} | {'Status':<16}")
    print("-" * 65)
    for _, row in inv_df.head(10).iterrows():
        desc = (str(row['Description'])[:25] + "...") if len(str(row['Description'])) > 25 else str(row['Description'])
        print(f"{str(row['StockCode']):<10} | {desc:<28} | {row['safety_stock']:<7} | {row['reorder_point']:<7} | {row['eoq']:<7} | {row['risk_status']:<16}")
    print("=" * 65)


if __name__ == "__main__":
    main()
