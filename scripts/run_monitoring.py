"""
RetailPulse - Phase 7 Monitoring & Drift Detection Runner (F7, F8)
Executes Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) drift checks
between historical reference and current observation windows.
Tracks forecast error and churn classifier degradation against threshold triggers.
Logs results and alerts to reports and MLflow.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import mlflow

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.utils.config import load_config
from src.monitoring.drift import evaluate_dataset_drift
from src.monitoring.performance import evaluate_forecast_degradation, evaluate_churn_degradation

logger = get_logger("run_monitoring")


def main():
    logger.info("=" * 65)
    logger.info("Executing Phase 7: Model & Data Drift Monitoring Pipeline (F7, F8)")
    logger.info("=" * 65)

    cfg = load_config()
    proc_dir = cfg["paths"]["processed_data_dir"]
    models_dir = cfg["paths"]["models_dir"]
    fig_dir = os.path.join(cfg["paths"].get("reports_dir", "reports"), "figures")
    reports_dir = cfg["paths"].get("reports_dir", "reports")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Feature Drift Detection (Reference vs Current Window)
    logger.info("\n>>> Step 1: Evaluating Feature Distribution Drift...")
    churn_path = os.path.join(proc_dir, "churn_features.parquet")
    if not os.path.exists(churn_path):
        raise FileNotFoundError(f"Customer churn features missing at {churn_path}")

    df_features = pd.read_parquet(churn_path)

    # Split into Reference Window (first 70% chronologically) and Current Window (recent 30%)
    split_point = int(len(df_features) * 0.70)
    df_ref = df_features.iloc[:split_point]
    df_cur = df_features.iloc[split_point:]

    monitored_features = [
        "Recency", "Frequency", "Monetary", "TenureDays",
        "AvgOrderValue", "AvgBasketSize", "OrdersLast30d", "SpendLast30d"
    ]
    monitored_features = [f for f in monitored_features if f in df_features.columns]

    psi_thresh = cfg.get("monitoring", {}).get("drift_psi_threshold", 0.25)
    ks_alpha = cfg.get("monitoring", {}).get("drift_ks_alpha", 0.05)

    drift_report = evaluate_dataset_drift(
        df_reference=df_ref,
        df_current=df_cur,
        feature_cols=monitored_features,
        psi_threshold=psi_thresh,
        ks_alpha=ks_alpha
    )

    drift_report_path = os.path.join(reports_dir, "drift_audit_report.json")
    with open(drift_report_path, "w") as f:
        json.dump(drift_report, f, indent=2)
    logger.info(f"Drift audit report saved to: {drift_report_path}")

    # 2. Performance Tracking & Degradation Check
    logger.info("\n>>> Step 2: Evaluating Model Performance Degradation...")
    
    # Check forecast degradation
    forecast_comp_path = os.path.join(models_dir, "forecast_comparison.json")
    forecast_status = {"status": "NO_PREVIOUS_RECORD"}
    if os.path.exists(forecast_comp_path):
        with open(forecast_comp_path, "r") as f:
            f_comp = json.load(f)
        baseline_forecast = f_comp.get("prophet", {})
        current_forecast = f_comp.get("hybrid", {})
        forecast_status = evaluate_forecast_degradation(
            baseline_metrics=baseline_forecast,
            current_metrics=current_forecast,
            threshold_pct=0.15
        )

    # Check churn degradation
    churn_comp_path = os.path.join(models_dir, "churn_comparison.json")
    churn_status = {"status": "NO_PREVIOUS_RECORD"}
    if os.path.exists(churn_comp_path):
        with open(churn_comp_path, "r") as f:
            c_comp = json.load(f)
        models_data = c_comp.get("models", {})
        if "GradientBoosting" in models_data and "LogisticRegression" in models_data:
            churn_status = evaluate_churn_degradation(
                baseline_metrics=models_data["GradientBoosting"],
                current_metrics=models_data["LogisticRegression"],
                threshold_pct=0.10
            )

    health_summary = {
        "drift_detected": drift_report["dataset_drift_detected"],
        "drift_share_pct": drift_report["drift_share_pct"],
        "drifted_features": drift_report["drifted_features"],
        "forecast_health": forecast_status,
        "churn_health": churn_status,
        "retrain_required": drift_report["dataset_drift_detected"] or forecast_status.get("retrain_recommended", False)
    }

    health_path = os.path.join(reports_dir, "model_health_summary.json")
    with open(health_path, "w") as f:
        json.dump(health_summary, f, indent=2)

    # 3. Visualizations
    logger.info("\n>>> Generating Drift & Health Visualizations...")
    
    # 3A: PSI Score Bar Chart
    psi_data = [
        {"Feature": feat, "PSI": data["psi"], "Status": data["status"]}
        for feat, data in drift_report["features"].items()
    ]
    df_psi = pd.DataFrame(psi_data).sort_values("PSI", ascending=True)

    fig_psi = px.bar(
        df_psi,
        x="PSI",
        y="Feature",
        orientation="h",
        color="Status",
        color_discrete_map={
            "STABLE": "#10B981",
            "MODERATE DRIFT": "#F59E0B",
            "CRITICAL DRIFT": "#EF4444"
        },
        title=f"Population Stability Index (PSI) by Feature (Threshold: {psi_thresh})",
        template="plotly_white"
    )
    fig_psi.add_vline(x=psi_thresh, line_dash="dash", line_color="#EF4444", annotation_text="Drift Alert Threshold (0.25)")
    fig_psi.update_layout(height=450)
    fig_psi.write_html(os.path.join(fig_dir, "drift_psi_scores.html"))

    # 4. Log to MLflow
    tracking_uri = cfg.get("mlops", {}).get("mlflow_tracking_uri", "sqlite:///mlflow.db")
    exp_name = cfg.get("mlops", {}).get("experiment_name", "RetailPulse_Analytics")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name="Monitoring_Drift_Audit") as run:
        mlflow.log_metric("drift_share_pct", drift_report["drift_share_pct"])
        mlflow.log_metric("drifted_features_count", drift_report["drifted_features_count"])
        mlflow.log_metric("retrain_required", int(health_summary["retrain_required"]))
        for feat, data in drift_report["features"].items():
            mlflow.log_metric(f"psi_{feat}", data["psi"])
            mlflow.log_metric(f"ks_stat_{feat}", data["ks_statistic"])
        mlflow.log_artifact(drift_report_path)
        mlflow.log_artifact(health_path)
        logger.info(f"Monitoring MLflow run logged: {run.info.run_id}")

    # Print Summary
    print("\n" + "=" * 65)
    print("PHASE 7 - DRIFT & PERFORMANCE MONITORING AUDIT REPORT")
    print("=" * 65)
    print(f"Dataset Drift Flag:     {'🔴 DRIFT DETECTED' if drift_report['dataset_drift_detected'] else '🟢 STABLE'}")
    print(f"Drift Share:            {drift_report['drift_share_pct']}% of features ({len(drift_report['drifted_features'])}/{len(drift_report['features'])})")
    print(f"Retrain Triggered:      {health_summary['retrain_required']}")
    print("-" * 65)
    print(f"{'Feature':<22} | {'PSI':<8} | {'KS Stat':<8} | {'p-value':<10} | {'Status':<16}")
    print("-" * 65)
    for feat, data in drift_report["features"].items():
        print(f"{feat:<22} | {data['psi']:<8.4f} | {data['ks_statistic']:<8.4f} | {data['ks_pvalue']:<10.4f} | {data['status']:<16}")
    print("=" * 65)


if __name__ == "__main__":
    main()
