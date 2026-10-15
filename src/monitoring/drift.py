"""
RetailPulse - Data & Concept Drift Monitoring Engine (F7)
Implements:
  1. Population Stability Index (PSI) calculation for numerical/categorical features.
  2. Kolmogorov-Smirnov (KS) two-sample test for continuous distribution shift.
  3. Reference vs. Current sliding-window comparison with threshold-based alerting.
  4. Automatic drift audit report generation.
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from scipy import stats

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger("drift_monitor")


def calculate_psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    eps: float = 1e-4
) -> float:
    """
    Computes the Population Stability Index (PSI) between reference and current samples.
    
    Rule of Thumb:
      - PSI < 0.1: No significant shift (Stable).
      - 0.1 <= PSI < 0.25: Moderate shift (Monitor).
      - PSI >= 0.25: Significant shift (Drift detected, trigger alert/retrain).
    """
    ref = np.asarray(reference).dropna() if hasattr(reference, 'dropna') else np.asarray(reference)
    cur = np.asarray(current).dropna() if hasattr(current, 'dropna') else np.asarray(current)
    
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    
    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # Determine quantile bins based on reference population
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(ref, quantiles)
    bin_edges = np.unique(bin_edges)  # remove duplicates in case of repeated values
    
    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate frequencies in each bin
    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    # Convert to proportions with smoothing epsilon to prevent division by zero
    ref_pct = np.maximum(ref_counts / len(ref), eps)
    cur_pct = np.maximum(cur_counts / len(cur), eps)

    # Normalize proportions after epsilon adjustment
    ref_pct = ref_pct / np.sum(ref_pct)
    cur_pct = cur_pct / np.sum(cur_pct)

    # PSI Formula: sum((Actual% - Expected%) * ln(Actual% / Expected%))
    psi_val = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(np.round(psi_val, 4))


def calculate_ks_test(
    reference: np.ndarray,
    current: np.ndarray
) -> Tuple[float, float]:
    """
    Performs Kolmogorov-Smirnov two-sample test for distribution difference.
    
    Returns:
        Tuple[float, float]: (KS Statistic, p-value).
    """
    ref = np.asarray(reference).dropna() if hasattr(reference, 'dropna') else np.asarray(reference)
    cur = np.asarray(current).dropna() if hasattr(current, 'dropna') else np.asarray(current)
    
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    
    if len(ref) == 0 or len(cur) == 0:
        return 0.0, 1.0

    res = stats.ks_2samp(ref, cur)
    return float(np.round(res.statistic, 4)), float(np.round(res.pvalue, 6))


def evaluate_dataset_drift(
    df_reference: pd.DataFrame,
    df_current: pd.DataFrame,
    feature_cols: List[str],
    psi_threshold: float = 0.25,
    ks_alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Evaluates drift across multiple feature columns between reference and current windows.
    """
    logger.info(f"Running drift detection on {len(feature_cols)} features...")
    logger.info(f"Thresholds: PSI >= {psi_threshold} | KS alpha <= {ks_alpha}")

    feature_results = {}
    drifted_features = []

    for col in feature_cols:
        if col not in df_reference.columns or col not in df_current.columns:
            logger.warning(f"Feature '{col}' not present in both datasets. Skipping.")
            continue

        ref_vals = df_reference[col].values
        cur_vals = df_current[col].values

        psi = calculate_psi(ref_vals, cur_vals)
        ks_stat, ks_pval = calculate_ks_test(ref_vals, cur_vals)

        # Flag drift if PSI exceeds threshold or KS test rejects null hypothesis
        is_psi_drift = psi >= psi_threshold
        is_ks_drift = ks_pval < ks_alpha
        is_drifted = is_psi_drift or is_ks_drift

        if is_drifted:
            drifted_features.append(col)

        status = "CRITICAL DRIFT" if (is_psi_drift and is_ks_drift) else ("MODERATE DRIFT" if is_drifted else "STABLE")

        feature_results[col] = {
            "psi": psi,
            "psi_drift": is_psi_drift,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pval,
            "ks_drift": is_ks_drift,
            "status": status,
            "ref_mean": float(np.round(np.nanmean(ref_vals), 2)),
            "cur_mean": float(np.round(np.nanmean(cur_vals), 2)),
            "ref_std": float(np.round(np.nanstd(ref_vals), 2)),
            "cur_std": float(np.round(np.nanstd(cur_vals), 2))
        }

    drift_share = len(drifted_features) / max(len(feature_cols), 1)
    overall_drift_detected = drift_share >= 0.33 or any(
        res["psi"] >= psi_threshold for res in feature_results.values()
    )

    report = {
        "dataset_drift_detected": bool(overall_drift_detected),
        "drift_share_pct": round(drift_share * 100, 2),
        "drifted_features_count": len(drifted_features),
        "total_features_evaluated": len(feature_results),
        "drifted_features": drifted_features,
        "psi_threshold": psi_threshold,
        "ks_alpha": ks_alpha,
        "features": feature_results
    }

    logger.info(f"Drift Audit: Overall Drift Detected = {overall_drift_detected} "
                f"({len(drifted_features)}/{len(feature_results)} features flagged)")
    return report
