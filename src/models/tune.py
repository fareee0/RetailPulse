"""
RetailPulse - Hyperparameter Tuning with Optuna (F4)
Provides Optuna-based tuning for churn and forecasting models.
"""

import os
import json
from typing import Dict, Any, Optional
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
import joblib

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger("optuna_tuner")

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_churn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 30,
    cv_folds: int = 5,
    log_to_mlflow: bool = True,
) -> Dict[str, Any]:
    """
    Tunes a GradientBoostingClassifier for churn prediction using Optuna.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        n_trials: Number of Optuna trials.
        cv_folds: Number of cross-validation folds.
        log_to_mlflow: Whether to log best run to MLflow.

    Returns:
        Dict with best params, best score, and the study object.
    """
    cfg = load_config()
    random_seed = cfg.get("project", {}).get("random_seed", 42)
    models_dir = cfg.get("paths", {}).get("models_dir", "models")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
        model = GradientBoostingClassifier(
            **params, random_state=random_seed
        )
        scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction="maximize", study_name="churn_gb_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best_score = study.best_value

    logger.info(f"Optuna churn tuning complete: Best AUC={best_score:.4f}")
    logger.info(f"Best params: {best}")

    # Save results
    result = {"best_params": best, "best_auc": round(best_score, 4), "n_trials": n_trials}
    result_path = os.path.join(models_dir, "optuna_churn_results.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    # Train best model
    best_model = GradientBoostingClassifier(**best, random_state=random_seed)
    best_model.fit(X_train, y_train)
    joblib.dump(best_model, os.path.join(models_dir, "churn_tuned_best.joblib"))

    if log_to_mlflow:
        import mlflow
        tracking_uri = cfg.get("mlops", {}).get("mlflow_tracking_uri", "sqlite:///mlflow.db")
        exp_name = cfg.get("mlops", {}).get("experiment_name", "RetailPulse_Analytics")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(exp_name)
        with mlflow.start_run(run_name="Optuna_Churn_Tuning") as run:
            mlflow.log_params(best)
            mlflow.log_metric("best_cv_auc", best_score)
            mlflow.log_artifact(result_path)
            logger.info(f"MLflow Optuna run logged: {run.info.run_id}")

    return {"best_params": best, "best_score": best_score, "study": study, "best_model": best_model}
