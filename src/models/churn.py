"""
RetailPulse - Churn Prediction Models (F2)
Implements Logistic Regression, Random Forest, and Gradient Boosting
for customer churn classification with class imbalance handling,
SHAP explanations, and MLflow tracking.
"""

import os
import json
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, accuracy_score,
)
import mlflow

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger("churn_model")

# Features used for churn prediction (no leakage columns)
CHURN_FEATURE_COLS = [
    "Recency", "Frequency", "Monetary", "TenureDays", "ActiveLifespanDays",
    "AvgOrderValue", "AvgBasketSize", "AvgPurchaseInterval",
    "TotalItems", "UniqueSKUs",
    "OrdersLast30d", "SpendLast30d", "OrdersLast90d", "SpendLast90d",
]


def prepare_churn_data(
    churn_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """
    Prepares feature matrix and labels for churn modeling.

    Args:
        churn_df: Churn feature DataFrame from build_churn_features().
        feature_cols: List of feature column names.
        test_size: Fraction held out for testing.
        random_seed: Reproducibility seed.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test, fitted_scaler, feature_names).
    """
    if feature_cols is None:
        feature_cols = [c for c in CHURN_FEATURE_COLS if c in churn_df.columns]

    logger.info(f"Churn features ({len(feature_cols)}): {feature_cols}")

    X = churn_df[feature_cols].fillna(0).values
    y = churn_df["Churn"].values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    logger.info(f"Train set: {len(X_train)} | Test set: {len(X_test)}")
    logger.info(f"Churn rate - Train: {y_train.mean()*100:.1f}% | Test: {y_test.mean()*100:.1f}%")

    return X_train, X_test, y_train, y_test, scaler, feature_cols


def train_churn_models(
    churn_df: pd.DataFrame,
    log_to_mlflow: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], str]:
    """
    Trains multiple churn classifiers, evaluates them, and logs to MLflow.

    Models:
        - LogisticRegression (class_weight='balanced')
        - RandomForestClassifier (class_weight='balanced')
        - GradientBoostingClassifier (with sample weights)

    Returns:
        Tuple: (dict of fitted models, dict of metrics per model, best model name).
    """
    cfg = load_config()
    random_seed = cfg.get("project", {}).get("random_seed", 42)
    models_dir = cfg.get("paths", {}).get("models_dir", "models")
    churn_cfg = cfg.get("churn", {})
    test_size = churn_cfg.get("test_size", 0.2)
    os.makedirs(models_dir, exist_ok=True)

    # Prepare data
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_churn_data(
        churn_df, test_size=test_size, random_seed=random_seed
    )

    # Save scaler
    joblib.dump(scaler, os.path.join(models_dir, "churn_scaler.joblib"))

    # Define models
    model_defs = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_seed,
            solver="lbfgs",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=churn_cfg.get("n_estimators", 200),
            max_depth=churn_cfg.get("max_depth", 8),
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=churn_cfg.get("n_estimators", 200),
            max_depth=churn_cfg.get("max_depth", 8),
            learning_rate=0.1,
            random_state=random_seed,
        ),
    }

    fitted_models = {}
    all_metrics = {}
    best_model_name = None
    best_auc = -1

    for name, model in model_defs.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {name}...")
        logger.info(f"{'='*50}")

        # Handle class imbalance for GBM via sample_weight
        if name == "GradientBoosting":
            class_counts = np.bincount(y_train)
            weight_map = {0: 1.0, 1: class_counts[0] / max(class_counts[1], 1)}
            sample_weights = np.array([weight_map[yi] for yi in y_train])
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

        # Metrics
        auc = float(roc_auc_score(y_test, y_prob))
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        acc = float(accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred).tolist()

        metrics = {
            "model": name,
            "roc_auc": round(auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "accuracy": round(acc, 4),
            "confusion_matrix": cm,
        }
        all_metrics[name] = metrics
        fitted_models[name] = model

        logger.info(f"  ROC-AUC:   {auc:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1:        {f1:.4f}")
        logger.info(f"  Confusion: {cm}")

        # Track best model
        if auc > best_auc:
            best_auc = auc
            best_model_name = name

        # Save model
        model_path = os.path.join(models_dir, f"churn_{name.lower()}.joblib")
        joblib.dump(model, model_path)

        # Feature importance
        if hasattr(model, "feature_importances_"):
            importances = dict(zip(feature_names, model.feature_importances_.round(4).tolist()))
            metrics["feature_importances"] = importances
        elif hasattr(model, "coef_"):
            coefs = dict(zip(feature_names, model.coef_[0].round(4).tolist()))
            metrics["coefficients"] = coefs

        # MLflow logging
        if log_to_mlflow:
            _log_churn_mlflow(cfg, name, metrics, model_path)

    # Save comparison
    comparison_path = os.path.join(models_dir, "churn_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump({"models": all_metrics, "best_model": best_model_name}, f, indent=2)

    logger.info(f"\nBest churn model: {best_model_name} (AUC: {best_auc:.4f})")

    # Save best model separately
    joblib.dump(fitted_models[best_model_name], os.path.join(models_dir, "churn_best_model.joblib"))

    # Generate SHAP values for best model (if available)
    _generate_shap(fitted_models[best_model_name], X_test, feature_names, models_dir, best_model_name)

    return fitted_models, all_metrics, best_model_name


def _log_churn_mlflow(cfg: dict, model_name: str, metrics: dict, model_path: str) -> None:
    """Logs a churn model run to MLflow."""
    tracking_uri = cfg.get("mlops", {}).get("mlflow_tracking_uri", "sqlite:///mlflow.db")
    exp_name = cfg.get("mlops", {}).get("experiment_name", "RetailPulse_Analytics")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name=f"Churn_{model_name}") as run:
        mlflow.log_param("model_type", f"Churn_{model_name}")
        for k, v in metrics.items():
            if k not in ("model", "confusion_matrix", "feature_importances", "coefficients") and isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path)
        logger.info(f"MLflow Churn_{model_name} run logged: {run.info.run_id}")


def _generate_shap(model, X_test, feature_names, models_dir, model_name):
    """Generates and saves SHAP feature importance values."""
    try:
        import shap
        logger.info(f"Generating SHAP explanations for {model_name}...")

        if hasattr(model, "predict_proba"):
            explainer = shap.Explainer(model.predict_proba, feature_names=feature_names)
            shap_values = explainer(X_test[:100])  # Limit for speed
            shap_importance = np.abs(shap_values.values[:, :, 1]).mean(axis=0) if len(shap_values.values.shape) == 3 else np.abs(shap_values.values).mean(axis=0)
        else:
            explainer = shap.Explainer(model, feature_names=feature_names)
            shap_values = explainer(X_test[:100])
            shap_importance = np.abs(shap_values.values).mean(axis=0)

        shap_dict = dict(zip(feature_names, shap_importance.round(4).tolist()))
        shap_path = os.path.join(models_dir, "churn_shap_importance.json")
        with open(shap_path, "w") as f:
            json.dump(shap_dict, f, indent=2)
        logger.info(f"SHAP importance saved to {shap_path}")

    except Exception as e:
        logger.warning(f"SHAP generation failed (non-critical): {e}")
