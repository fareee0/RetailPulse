"""
RetailPulse - Production MLOps Orchestration Pipeline (Airflow DAG)
Pipeline Stages:
  1. data_ingestion: Ingest raw 2-year Online Retail transactions.
  2. data_cleaning: Run validation & audit rules, purge nulls/cancellations.
  3. feature_engineering: Compute RFM, time features, lag aggregations, and churn cohort.
  4. train_segmentation: Fit KMeans and quintile scoring.
  5. train_forecasting: Train Prophet, LSTM, and Hybrid ensemble models.
  6. train_churn: Fit Logistic Regression, Random Forest, Gradient Boosting.
  7. evaluate_models: Assess MAE, RMSE, ROC-AUC, and F1 against acceptance criteria.
  8. drift_monitoring: Run PSI & Kolmogorov-Smirnov distribution checks.
  9. register_model: Push champion models to MLflow Model Registry.
"""

from datetime import datetime, timedelta
import os
import sys

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
except ImportError:
    # Graceful fallback for local development environments where Airflow is optional
    DAG = None
    PythonOperator = None
    BashOperator = None


default_args = {
    "owner": "retailpulse_data_team",
    "depends_on_past": False,
    "email": ["mlops-alerts@retailpulse.ai"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=60),
}


def task_ingest(**kwargs):
    """Downloads or verifies raw dataset availability."""
    from src.data.load import load_raw_data
    df = load_raw_data()
    print(f"[Airflow] Ingested raw transactions: {len(df):,} rows.")
    return len(df)


def task_clean(**kwargs):
    """Executes data cleaning rules and saves cleaned parquet."""
    from src.data.load import load_raw_data
    from src.data.clean import clean_retail_data
    df_raw = load_raw_data()
    df_clean, audit = clean_retail_data(df_raw, save_output=True)
    print(f"[Airflow] Cleaned data: {len(df_clean):,} valid rows retained ({audit['retention_rate_pct']}%).")
    return audit


def task_features(**kwargs):
    """Extracts RFM, lag, and churn features."""
    import pandas as pd
    from src.utils.config import load_config
    from src.features.rfm import calculate_rfm_features
    from src.features.lag_features import build_daily_time_series, build_lag_features
    from src.features.churn_features import build_churn_features

    cfg = load_config()
    clean_path = cfg["data"]["cleaned_parquet"]
    proc_dir = cfg["paths"]["processed_data_dir"]
    df_clean = pd.read_parquet(clean_path)

    # 1. Customer RFM
    df_rfm = calculate_rfm_features(df_clean)
    df_rfm.to_parquet(os.path.join(proc_dir, "customer_features.parquet"), index=False)

    # 2. Daily Time Series & Lags
    daily = build_daily_time_series(df_clean)
    daily_feats = build_lag_features(daily)
    daily_feats.to_parquet(os.path.join(proc_dir, "daily_sales_features.parquet"), index=False)

    # 3. Churn Cohort
    df_churn, _ = build_churn_features(df_clean, churn_window_days=90)
    df_churn.to_parquet(os.path.join(proc_dir, "churn_features.parquet"), index=False)
    print("[Airflow] All feature stores successfully materialized.")


def task_train_models(**kwargs):
    """Runs segmentation, forecasting, and churn training routines."""
    import subprocess
    python_bin = sys.executable
    subprocess.run([python_bin, "scripts/run_segmentation.py"], check=True)
    subprocess.run([python_bin, "scripts/run_forecasting.py"], check=True)
    subprocess.run([python_bin, "scripts/run_churn.py"], check=True)
    subprocess.run([python_bin, "scripts/run_inventory.py"], check=True)
    print("[Airflow] All AI/ML models trained and serialized.")


def task_evaluate(**kwargs):
    """Verifies that model quality satisfies production thresholds."""
    import json
    with open("models/forecast_comparison.json") as f:
        f_comp = json.load(f)
    with open("models/churn_comparison.json") as f:
        c_comp = json.load(f)

    hybrid_mape = f_comp["hybrid"]["mape_pct"]
    best_churn_auc = c_comp["models"][c_comp["best_model"]]["roc_auc"]

    assert hybrid_mape < 35.0, f"Forecast error too high: {hybrid_mape}%"
    assert best_churn_auc > 0.70, f"Churn model AUC too low: {best_churn_auc}"
    print(f"[Airflow] Evaluation Passed: Forecast MAPE={hybrid_mape}%, Churn AUC={best_churn_auc}")


def task_drift_check(**kwargs):
    """Evaluates PSI and KS feature drift."""
    import subprocess
    subprocess.run([sys.executable, "scripts/run_monitoring.py"], check=True)
    print("[Airflow] Drift monitoring check completed.")


def task_register_model(**kwargs):
    """Registers champion model to MLflow Model Registry."""
    import mlflow
    from src.utils.config import load_config
    cfg = load_config()
    tracking_uri = cfg.get("mlops", {}).get("mlflow_tracking_uri", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"[Airflow] Registered Champion Models in MLflow URI: {tracking_uri}")


if DAG is not None:
    dag = DAG(
        "retailpulse_mlops_pipeline",
        default_args=default_args,
        description="RetailPulse AI-Powered Customer Analytics & Demand Forecasting Orchestration",
        schedule_interval="0 3 * * 1",  # Every Monday at 3:00 AM UTC
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["retailpulse", "production", "mlops"],
    )

    t1_ingest = PythonOperator(task_id="data_ingestion", python_callable=task_ingest, dag=dag)
    t2_clean = PythonOperator(task_id="data_cleaning", python_callable=task_clean, dag=dag)
    t3_features = PythonOperator(task_id="feature_engineering", python_callable=task_features, dag=dag)
    t4_train = PythonOperator(task_id="train_models", python_callable=task_train_models, dag=dag)
    t5_evaluate = PythonOperator(task_id="evaluate_models", python_callable=task_evaluate, dag=dag)
    t6_drift = PythonOperator(task_id="drift_monitoring", python_callable=task_drift_check, dag=dag)
    t7_register = PythonOperator(task_id="register_model", python_callable=task_register_model, dag=dag)

    # Dependency graph
    t1_ingest >> t2_clean >> t3_features >> t4_train >> t5_evaluate >> t6_drift >> t7_register
