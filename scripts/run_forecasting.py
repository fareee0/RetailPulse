"""
RetailPulse - Phase 4 Demand Forecasting Runner (F3, F4)
Executes Prophet, LSTM, and Hybrid forecasting models.
Performs model evaluation (MAE, RMSE, MAPE, sMAPE), compares performance,
and records artifacts in MLflow.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.utils.config import load_config
from src.features.lag_features import build_daily_time_series
from src.models.prophet_model import prepare_prophet_data, train_prophet
from src.models.lstm_model import train_lstm
from src.models.hybrid import train_hybrid

logger = get_logger("run_forecasting")


def main():
    logger.info("=" * 65)
    logger.info("Executing Phase 4: Demand Forecasting Engine (F3, F4)")
    logger.info("=" * 65)

    cfg = load_config()
    proc_dir = cfg["paths"]["processed_data_dir"]
    fig_dir = os.path.join(cfg["paths"].get("reports_dir", "reports"), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Load Daily Sales data (or generate from cleaned transactions)
    daily_path = os.path.join(proc_dir, "daily_sales_features.parquet")
    if os.path.exists(daily_path):
        logger.info(f"Loading daily time series from: {daily_path}")
        daily_df = pd.read_parquet(daily_path)
    else:
        logger.info("Daily sales features not found. Building from cleaned transactions...")
        clean_path = cfg["data"]["cleaned_parquet"]
        df_clean = pd.read_parquet(clean_path)
        daily_df = build_daily_time_series(df_clean)
        daily_df.to_parquet(daily_path, index=False)

    # 2. Train Prophet
    logger.info("\n>>> Step 1: Training Prophet Model with UK Holidays...")
    df_prophet = prepare_prophet_data(daily_df)
    prophet_model, prophet_forecast, prophet_test, prophet_metrics = train_prophet(
        df_prophet=df_prophet,
        log_to_mlflow=True
    )

    # Format prophet predictions table for hybrid alignment
    prophet_preds = prophet_forecast[prophet_forecast["ds"].isin(prophet_test["ds"])][["ds", "yhat"]].copy()
    prophet_preds = prophet_preds.merge(prophet_test[["ds", "y"]], on="ds", how="inner")

    # 3. Train LSTM
    logger.info("\n>>> Step 2: Training LSTM Sequence-to-One Model...")
    lstm_model, lstm_scaler, lstm_preds, lstm_metrics = train_lstm(
        daily_df=daily_df,
        log_to_mlflow=True
    )

    # 4. Train Hybrid Ensemble
    logger.info("\n>>> Step 3: Constructing Hybrid Model...")
    hybrid_preds, hybrid_metrics, comparison = train_hybrid(
        prophet_preds=prophet_preds,
        lstm_preds=lstm_preds,
        prophet_metrics=prophet_metrics,
        lstm_metrics=lstm_metrics,
        log_to_mlflow=True
    )

    # 5. Visualizations
    logger.info("\n>>> Generating Forecasting Comparison Charts...")
    fig = go.Figure()
    
    # Actuals
    fig.add_trace(go.Scatter(
        x=hybrid_preds["ds"], y=hybrid_preds["y"],
        mode="lines+markers", name="Actual Demand",
        line=dict(color="#1E293B", width=2),
        marker=dict(size=4)
    ))
    # Prophet
    fig.add_trace(go.Scatter(
        x=hybrid_preds["ds"], y=hybrid_preds["yhat_prophet"],
        mode="lines", name=f"Prophet (MAE: £{prophet_metrics['mae']:,.0f})",
        line=dict(color="#3B82F6", width=2, dash="dot")
    ))
    # LSTM
    fig.add_trace(go.Scatter(
        x=hybrid_preds["ds"], y=hybrid_preds["yhat_lstm"],
        mode="lines", name=f"LSTM (MAE: £{lstm_metrics['mae']:,.0f})",
        line=dict(color="#10B981", width=2, dash="dash")
    ))
    # Hybrid
    fig.add_trace(go.Scatter(
        x=hybrid_preds["ds"], y=hybrid_preds["yhat_hybrid"],
        mode="lines", name=f"Hybrid Ensemble (MAE: £{hybrid_metrics['mae']:,.0f})",
        line=dict(color="#8B5CF6", width=3)
    ))

    fig.update_layout(
        title="Walk-Forward Demand Forecasting Comparison (Prophet vs LSTM vs Hybrid)",
        xaxis_title="Date",
        yaxis_title="Daily Revenue (£)",
        template="plotly_white",
        hovermode="x unified",
        height=540,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    chart_path = os.path.join(fig_dir, "forecasting_comparison.html")
    fig.write_html(chart_path)
    logger.info(f"Saved forecasting comparison chart to: {chart_path}")

    # Print Summary
    print("\n" + "=" * 65)
    print("PHASE 4 - DEMAND FORECASTING EVALUATION REPORT")
    print("=" * 65)
    print(f"{'Model':<15} | {'MAE (£)':<10} | {'RMSE (£)':<10} | {'MAPE (%)':<10} | {'sMAPE (%)':<10}")
    print("-" * 65)
    print(f"{'Prophet':<15} | {prophet_metrics['mae']:<10.2f} | {prophet_metrics['rmse']:<10.2f} | {prophet_metrics['mape_pct']:<10.2f} | {prophet_metrics['smape_pct']:<10.2f}")
    print(f"{'LSTM':<15} | {lstm_metrics['mae']:<10.2f} | {lstm_metrics['rmse']:<10.2f} | {lstm_metrics['mape_pct']:<10.2f} | {lstm_metrics['smape_pct']:<10.2f}")
    print(f"{'Hybrid Ensemble':<15} | {hybrid_metrics['mae']:<10.2f} | {hybrid_metrics['rmse']:<10.2f} | {hybrid_metrics['mape_pct']:<10.2f} | {hybrid_metrics['smape_pct']:<10.2f}")
    print("-" * 65)
    print(f"Hybrid Improvement over Prophet: {comparison['improvement_over_prophet_mae_pct']:.2f}%")
    print(f"Hybrid Improvement over LSTM:    {comparison['improvement_over_lstm_mae_pct']:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
