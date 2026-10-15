"""
RetailPulse - Hybrid Demand Forecasting Model (F3/F4)
Combines Prophet (trend + seasonality) with LSTM (residual patterns)
via a weighted-average ensemble. Falls back to simple average if weights
are equal. Reports metrics and compares against individual models.
"""

import os
import json
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import joblib

from src.models.prophet_model import compute_forecast_metrics
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger("hybrid_model")


def train_hybrid(
    prophet_preds: pd.DataFrame,
    lstm_preds: pd.DataFrame,
    prophet_metrics: Dict[str, float],
    lstm_metrics: Dict[str, float],
    log_to_mlflow: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Any]]:
    """
    Builds a hybrid forecast by weighted-averaging Prophet and LSTM predictions.

    Strategy:
        1. Align Prophet and LSTM predictions on the same date range.
        2. Weighted average: hybrid = w_p * prophet + w_l * lstm.
        3. Optionally derive weights from inverse MAE (better model gets more weight).
        4. Report combined metrics and model comparison.

    Args:
        prophet_preds: DataFrame with columns [ds, y, yhat] from Prophet.
        lstm_preds: DataFrame with columns [ds, y, yhat] from LSTM.
        prophet_metrics: Prophet evaluation metrics dict.
        lstm_metrics: LSTM evaluation metrics dict.
        log_to_mlflow: Whether to log to MLflow.

    Returns:
        Tuple: (hybrid predictions DataFrame, hybrid metrics, full comparison dict).
    """
    cfg = load_config()
    hcfg = cfg.get("forecasting", {}).get("hybrid", {})
    models_dir = cfg.get("paths", {}).get("models_dir", "models")
    os.makedirs(models_dir, exist_ok=True)

    # Align predictions on common dates
    prophet_df = prophet_preds[["ds", "y", "yhat"]].copy()
    prophet_df.columns = ["ds", "y", "yhat_prophet"]

    lstm_df = lstm_preds[["ds", "yhat"]].copy()
    lstm_df.columns = ["ds", "yhat_lstm"]

    merged = prophet_df.merge(lstm_df, on="ds", how="inner")
    logger.info(f"Hybrid model: {len(merged)} aligned test days for ensemble")

    if len(merged) == 0:
        logger.warning("No overlapping dates between Prophet and LSTM predictions!")
        return pd.DataFrame(), {}, {}

    # Compute weights (inverse-MAE weighting or config-defined)
    p_w = hcfg.get("prophet_weight", None)
    l_w = hcfg.get("lstm_weight", None)

    if p_w is None or l_w is None:
        # Inverse-MAE weighting: better model (lower MAE) gets higher weight
        p_mae = max(prophet_metrics.get("mae", 1.0), 0.01)
        l_mae = max(lstm_metrics.get("mae", 1.0), 0.01)
        inv_total = (1.0 / p_mae) + (1.0 / l_mae)
        p_w = (1.0 / p_mae) / inv_total
        l_w = (1.0 / l_mae) / inv_total
        logger.info(f"Auto-derived ensemble weights (inverse-MAE): Prophet={p_w:.3f}, LSTM={l_w:.3f}")
    else:
        # Normalize
        total = p_w + l_w
        p_w, l_w = p_w / total, l_w / total
        logger.info(f"Config-defined ensemble weights: Prophet={p_w:.3f}, LSTM={l_w:.3f}")

    # Weighted average
    merged["yhat_hybrid"] = (p_w * merged["yhat_prophet"] + l_w * merged["yhat_lstm"]).round(2)
    merged["yhat_hybrid"] = merged["yhat_hybrid"].clip(lower=0)

    # Evaluate hybrid
    hybrid_metrics = compute_forecast_metrics(
        merged["y"].values, merged["yhat_hybrid"].values, "Hybrid_Ensemble"
    )
    hybrid_metrics["prophet_weight"] = round(p_w, 4)
    hybrid_metrics["lstm_weight"] = round(l_w, 4)

    # Full comparison
    comparison = {
        "prophet": prophet_metrics,
        "lstm": lstm_metrics,
        "hybrid": hybrid_metrics,
        "improvement_over_prophet_mae_pct": round(
            (1 - hybrid_metrics["mae"] / max(prophet_metrics.get("mae", 1), 0.01)) * 100, 2
        ),
        "improvement_over_lstm_mae_pct": round(
            (1 - hybrid_metrics["mae"] / max(lstm_metrics.get("mae", 1), 0.01)) * 100, 2
        ),
    }

    # Report
    logger.info("=" * 60)
    logger.info("FORECAST MODEL COMPARISON")
    logger.info("=" * 60)
    for name, m in [("Prophet", prophet_metrics), ("LSTM", lstm_metrics), ("Hybrid", hybrid_metrics)]:
        logger.info(f"  {name:12s} | MAE: {m['mae']:>10.2f} | RMSE: {m['rmse']:>10.2f} | MAPE: {m['mape_pct']:>7.2f}%")
    logger.info(f"  Hybrid improvement over Prophet: {comparison['improvement_over_prophet_mae_pct']:.2f}%")
    logger.info(f"  Hybrid improvement over LSTM:    {comparison['improvement_over_lstm_mae_pct']:.2f}%")
    logger.info("=" * 60)

    # Save
    comp_path = os.path.join(models_dir, "forecast_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)

    merged.to_parquet(os.path.join(cfg["paths"]["processed_data_dir"], "hybrid_forecast.parquet"), index=False)
    logger.info(f"Saved hybrid predictions and comparison to {models_dir}/")

    # MLflow
    if log_to_mlflow:
        import mlflow
        tracking_uri = cfg.get("mlops", {}).get("mlflow_tracking_uri", "sqlite:///mlflow.db")
        exp_name = cfg.get("mlops", {}).get("experiment_name", "RetailPulse_Analytics")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(exp_name)
        with mlflow.start_run(run_name="Hybrid_Ensemble_Forecast") as run:
            mlflow.log_param("model_type", "Hybrid_Ensemble")
            mlflow.log_param("prophet_weight", round(p_w, 4))
            mlflow.log_param("lstm_weight", round(l_w, 4))
            for k, v in hybrid_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            mlflow.log_artifact(comp_path)
            logger.info(f"MLflow Hybrid run logged: {run.info.run_id}")

    return merged, hybrid_metrics, comparison
