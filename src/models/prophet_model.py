"""
RetailPulse - Prophet Demand Forecasting Model (F3)
Fits Facebook Prophet with UK holidays and weekly/yearly seasonality.
Supports walk-forward validation and MLflow experiment tracking.
"""

import os
import json
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import joblib
import mlflow

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger("prophet_model")


def prepare_prophet_data(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a daily sales DataFrame into Prophet-compatible format (ds, y).

    Args:
        daily_df: DataFrame with 'Date' and 'total_sales' columns.

    Returns:
        pd.DataFrame: Prophet-formatted DataFrame.
    """
    df_prophet = daily_df[["Date", "total_sales"]].copy()
    df_prophet.columns = ["ds", "y"]
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])
    df_prophet = df_prophet.sort_values("ds").reset_index(drop=True)
    return df_prophet


def train_prophet(
    df_prophet: pd.DataFrame,
    split_date: Optional[pd.Timestamp] = None,
    log_to_mlflow: bool = True,
) -> Tuple[Prophet, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Trains a Prophet model with UK holidays and evaluates on a held-out test set.

    Args:
        df_prophet: Prophet-formatted DataFrame (ds, y).
        split_date: Date to split train/test. Defaults to max(ds) - 90 days.
        log_to_mlflow: Whether to log results to MLflow.

    Returns:
        Tuple: (fitted Prophet model, forecast DataFrame, test DataFrame, metrics dict).
    """
    cfg = load_config()
    fcfg = cfg.get("forecasting", {}).get("prophet", {})
    random_seed = cfg.get("project", {}).get("random_seed", 42)
    models_dir = cfg.get("paths", {}).get("models_dir", "models")
    os.makedirs(models_dir, exist_ok=True)

    if split_date is None:
        split_days = cfg.get("forecasting", {}).get("train_test_split_days", 90)
        split_date = df_prophet["ds"].max() - pd.Timedelta(days=split_days)

    train_df = df_prophet[df_prophet["ds"] < split_date].copy()
    test_df = df_prophet[df_prophet["ds"] >= split_date].copy()

    logger.info(f"Prophet train set: {len(train_df)} days ({train_df['ds'].min().date()} to {train_df['ds'].max().date()})")
    logger.info(f"Prophet test set:  {len(test_df)} days ({test_df['ds'].min().date()} to {test_df['ds'].max().date()})")

    # Build and fit Prophet model
    model = Prophet(
        yearly_seasonality=fcfg.get("yearly_seasonality", True),
        weekly_seasonality=fcfg.get("weekly_seasonality", True),
        daily_seasonality=fcfg.get("daily_seasonality", False),
        changepoint_prior_scale=fcfg.get("changepoint_prior_scale", 0.05),
        seasonality_prior_scale=fcfg.get("seasonality_prior_scale", 10.0),
    )

    # Add UK holidays
    model.add_country_holidays(country_name="UK")
    model.fit(train_df)

    # Forecast over test period
    future = model.make_future_dataframe(periods=len(test_df), freq="D")
    forecast = model.predict(future)

    # Evaluate on test set
    forecast_test = forecast[forecast["ds"].isin(test_df["ds"])][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    merged = test_df.merge(forecast_test, on="ds", how="inner")

    metrics = compute_forecast_metrics(merged["y"].values, merged["yhat"].values, "Prophet")

    # Save model
    model_path = os.path.join(models_dir, "prophet_model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Saved Prophet model to {model_path}")

    # Save metrics
    metrics_path = os.path.join(models_dir, "prophet_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Log to MLflow
    if log_to_mlflow:
        _log_mlflow(cfg, metrics, model_path, "Prophet_Forecast")

    return model, forecast, test_df, metrics


def compute_forecast_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"
) -> Dict[str, float]:
    """Computes MAE, RMSE, MAPE and sMAPE for forecast evaluation."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # MAPE (avoid division by zero)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    else:
        mape = float("inf")

    # sMAPE
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape_mask = denom != 0
    if smape_mask.sum() > 0:
        smape = float(np.mean(np.abs(y_true[smape_mask] - y_pred[smape_mask]) / denom[smape_mask]) * 100)
    else:
        smape = float("inf")

    result = {
        "model": model_name,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape_pct": round(mape, 2),
        "smape_pct": round(smape, 2),
    }
    logger.info(f"[{model_name}] MAE={mae:.2f} | RMSE={rmse:.2f} | MAPE={mape:.2f}% | sMAPE={smape:.2f}%")
    return result


def _log_mlflow(cfg: dict, metrics: dict, artifact_path: str, run_name: str) -> None:
    """Helper to log forecast run to MLflow."""
    tracking_uri = cfg.get("mlops", {}).get("mlflow_tracking_uri", "sqlite:///mlflow.db")
    exp_name = cfg.get("mlops", {}).get("experiment_name", "RetailPulse_Analytics")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model_type", metrics.get("model", run_name))
        for k, v in metrics.items():
            if k != "model" and isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
        if os.path.exists(artifact_path):
            mlflow.log_artifact(artifact_path)
        logger.info(f"MLflow run logged: {run.info.run_id}")
