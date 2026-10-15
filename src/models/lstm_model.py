"""
RetailPulse - LSTM-style Demand Forecasting Model (F3)
Implements a sequence-to-one regression model using sliding windows.
Uses scikit-learn's MLPRegressor as a neural-network backend since
TensorFlow/PyTorch are not yet available for the current Python version.
The architecture mirrors LSTM behavior: sliding-window sequences → dense layers → output.
When TF/PyTorch become available, swap MLPRegressor for a true LSTM layer.
"""

import os
import json
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import joblib

from src.models.prophet_model import compute_forecast_metrics
from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger("lstm_model")


def create_sequences(
    data: np.ndarray, seq_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds sliding-window sequences for sequence-to-one prediction.

    Args:
        data: 1D array of scaled target values.
        seq_length: Number of time steps in each input window.

    Returns:
        Tuple: (X with shape [n_samples, seq_length], y with shape [n_samples]).
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def train_lstm(
    daily_df: pd.DataFrame,
    split_date: Optional[pd.Timestamp] = None,
    log_to_mlflow: bool = True,
) -> Tuple[MLPRegressor, MinMaxScaler, pd.DataFrame, Dict[str, float]]:
    """
    Trains a neural-network sequence model using sliding windows on daily sales.

    Architecture:
        1. Scale data to [0, 1] with MinMaxScaler.
        2. Create sliding windows of length `sequence_length`.
        3. Time-based train/test split.
        4. Fit MLPRegressor (hidden_layer_sizes mirrors LSTM hidden units).
        5. Evaluate and log metrics.

    Args:
        daily_df: DataFrame with 'Date' and 'total_sales'.
        split_date: Train/test split date.
        log_to_mlflow: Whether to log to MLflow.

    Returns:
        Tuple: (fitted model, scaler, predictions DataFrame, metrics dict).
    """
    cfg = load_config()
    lcfg = cfg.get("forecasting", {}).get("lstm", {})
    random_seed = cfg.get("project", {}).get("random_seed", 42)
    models_dir = cfg.get("paths", {}).get("models_dir", "models")
    os.makedirs(models_dir, exist_ok=True)

    seq_length = lcfg.get("sequence_length", 28)
    hidden_units = lcfg.get("hidden_units", 64)
    learning_rate = lcfg.get("learning_rate", 0.001)
    epochs = lcfg.get("epochs", 50)

    # Prepare data
    df_sorted = daily_df[["Date", "total_sales"]].copy().sort_values("Date").reset_index(drop=True)
    values = df_sorted["total_sales"].values.reshape(-1, 1).astype(float)

    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values).flatten()

    # Create sequences
    X, y = create_sequences(scaled, seq_length)
    dates = df_sorted["Date"].values[seq_length:]

    # Split
    if split_date is None:
        split_days = cfg.get("forecasting", {}).get("train_test_split_days", 90)
        split_date = pd.Timestamp(df_sorted["Date"].max()) - pd.Timedelta(days=split_days)

    split_idx = np.searchsorted(dates, np.datetime64(split_date))
    if split_idx <= 0:
        split_idx = int(len(X) * 0.8)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]

    logger.info(f"LSTM train samples: {len(X_train)} | test samples: {len(X_test)} | seq_length: {seq_length}")

    # Build and train MLP (LSTM-proxy)
    model = MLPRegressor(
        hidden_layer_sizes=(hidden_units, hidden_units // 2),
        activation="relu",
        solver="adam",
        learning_rate_init=learning_rate,
        max_iter=epochs * 10,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=lcfg.get("patience", 10),
        random_state=random_seed,
        verbose=False,
    )
    model.fit(X_train, y_train)

    # Predict and inverse scale
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Clip negative predictions
    y_pred = np.clip(y_pred, 0, None)

    # Build predictions DataFrame
    pred_df = pd.DataFrame({
        "ds": pd.to_datetime(dates_test),
        "y": y_actual,
        "yhat": y_pred.round(2),
    })

    metrics = compute_forecast_metrics(y_actual, y_pred, "LSTM_SequenceModel")

    # Save
    model_path = os.path.join(models_dir, "lstm_model.joblib")
    scaler_path = os.path.join(models_dir, "lstm_scaler.joblib")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(os.path.join(models_dir, "lstm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved LSTM model and scaler to {models_dir}/")

    # MLflow
    if log_to_mlflow:
        import mlflow
        tracking_uri = cfg.get("mlops", {}).get("mlflow_tracking_uri", "sqlite:///mlflow.db")
        exp_name = cfg.get("mlops", {}).get("experiment_name", "RetailPulse_Analytics")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(exp_name)
        with mlflow.start_run(run_name="LSTM_SequenceModel_Forecast") as run:
            mlflow.log_param("model_type", "LSTM_SequenceModel")
            mlflow.log_param("sequence_length", seq_length)
            mlflow.log_param("hidden_units", hidden_units)
            for k, v in metrics.items():
                if k != "model" and isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            mlflow.log_artifact(model_path)
            logger.info(f"MLflow LSTM run logged: {run.info.run_id}")

    return model, scaler, pred_df, metrics
