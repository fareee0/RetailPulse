"""
RetailPulse - Model Performance Tracking & Degradation Engine (F8)
Tracks metric degradation over sliding evaluation windows against baseline.
Triggers automated retraining flags and logs health records to MLflow.
"""

import os
import json
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger("performance_monitor")


def evaluate_forecast_degradation(
    baseline_metrics: Dict[str, float],
    current_metrics: Dict[str, float],
    threshold_pct: float = 0.15
) -> Dict[str, Any]:
    """
    Checks if demand forecast error (MAE / RMSE / sMAPE) has degraded beyond tolerance.
    """
    degradations = {}
    triggered = False

    for metric in ["mae", "rmse", "smape_pct"]:
        if metric in baseline_metrics and metric in current_metrics:
            base_val = baseline_metrics[metric]
            curr_val = current_metrics[metric]
            if base_val > 0:
                pct_change = (curr_val - base_val) / base_val
            else:
                pct_change = 0.0

            is_degraded = pct_change > threshold_pct
            if is_degraded:
                triggered = True

            degradations[metric] = {
                "baseline": round(base_val, 2),
                "current": round(curr_val, 2),
                "relative_change_pct": round(pct_change * 100, 2),
                "threshold_pct": round(threshold_pct * 100, 2),
                "degraded": bool(is_degraded)
            }

    status = "RETRAIN_TRIGGERED" if triggered else "HEALTHY"
    logger.info(f"Forecast Performance Status: {status} | Triggered={triggered}")
    return {
        "status": status,
        "retrain_recommended": triggered,
        "metrics": degradations
    }


def evaluate_churn_degradation(
    baseline_metrics: Dict[str, float],
    current_metrics: Dict[str, float],
    threshold_pct: float = 0.10
) -> Dict[str, Any]:
    """
    Checks if classification metrics (ROC-AUC / F1) have dropped below acceptable margin.
    """
    degradations = {}
    triggered = False

    for metric in ["roc_auc", "f1_score"]:
        if metric in baseline_metrics and metric in current_metrics:
            base_val = baseline_metrics[metric]
            curr_val = current_metrics[metric]
            # Degradation in classification = current is lower than baseline
            if base_val > 0:
                pct_drop = (base_val - curr_val) / base_val
            else:
                pct_drop = 0.0

            is_degraded = pct_drop > threshold_pct
            if is_degraded:
                triggered = True

            degradations[metric] = {
                "baseline": round(base_val, 4),
                "current": round(curr_val, 4),
                "relative_drop_pct": round(pct_drop * 100, 2),
                "threshold_pct": round(threshold_pct * 100, 2),
                "degraded": bool(is_degraded)
            }

    status = "RETRAIN_TRIGGERED" if triggered else "HEALTHY"
    logger.info(f"Churn Classifier Status: {status} | Triggered={triggered}")
    return {
        "status": status,
        "retrain_recommended": triggered,
        "metrics": degradations
    }
