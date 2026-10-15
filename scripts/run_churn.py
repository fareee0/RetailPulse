"""
RetailPulse - Phase 5 Churn Prediction Runner (F2)
Executes leakage-free churn feature construction, trains multiple classifiers,
handles class imbalance, computes ROC-AUC / Precision / Recall / F1 / Confusion Matrix,
generates SHAP values and feature importance plots, and logs to MLflow.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.utils.config import load_config
from src.features.churn_features import build_churn_features
from src.models.churn import train_churn_models, CHURN_FEATURE_COLS
from src.models.tune import tune_churn_model

logger = get_logger("run_churn")


def main():
    logger.info("=" * 65)
    logger.info("Executing Phase 5: Customer Churn Prediction Engine (F2)")
    logger.info("=" * 65)

    cfg = load_config()
    proc_dir = cfg["paths"]["processed_data_dir"]
    fig_dir = os.path.join(cfg["paths"].get("reports_dir", "reports"), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Load or build Churn Features
    churn_path = os.path.join(proc_dir, "churn_features.parquet")
    if os.path.exists(churn_path):
        logger.info(f"Loading churn feature dataset from: {churn_path}")
        df_churn = pd.read_parquet(churn_path)
    else:
        logger.info("Churn features not found. Building leakage-free dataset from cleaned records...")
        clean_path = cfg["data"]["cleaned_parquet"]
        df_clean = pd.read_parquet(clean_path)
        cutoff_days = cfg.get("features", {}).get("churn_cutoff_days", 90)
        df_churn, cutoff_date = build_churn_features(df_clean, churn_window_days=cutoff_days)
        df_churn.to_parquet(churn_path, index=False)

    # 2. Train and Evaluate Models (Logistic Regression, Random Forest, Gradient Boosting)
    logger.info("\n>>> Training Churn Classifiers...")
    fitted_models, all_metrics, best_model_name = train_churn_models(
        churn_df=df_churn,
        log_to_mlflow=True
    )

    # 3. Visualizations
    logger.info("\n>>> Generating Churn Model Visualizations...")
    
    # 3A: Model Comparison Bar Chart
    model_names = list(all_metrics.keys())
    roc_aucs = [all_metrics[m]["roc_auc"] for m in model_names]
    f1s = [all_metrics[m]["f1_score"] for m in model_names]
    recalls = [all_metrics[m]["recall"] for m in model_names]
    precisions = [all_metrics[m]["precision"] for m in model_names]

    fig_cmp = go.Figure(data=[
        go.Bar(name="ROC-AUC", x=model_names, y=roc_aucs, marker_color="#3B82F6"),
        go.Bar(name="F1 Score", x=model_names, y=f1s, marker_color="#10B981"),
        go.Bar(name="Recall", x=model_names, y=recalls, marker_color="#F59E0B"),
        go.Bar(name="Precision", x=model_names, y=precisions, marker_color="#8B5CF6"),
    ])
    fig_cmp.update_layout(
        barmode="group",
        title="Churn Model Performance Comparison (Test Set)",
        yaxis=dict(range=[0, 1.05], title="Metric Score"),
        template="plotly_white",
        height=450
    )
    fig_cmp.write_html(os.path.join(fig_dir, "churn_model_comparison.html"))

    # 3B: Best Model Confusion Matrix Heatmap
    best_cm = all_metrics[best_model_name]["confusion_matrix"]
    cm_df = pd.DataFrame(
        best_cm,
        index=["Actual Retained (0)", "Actual Churned (1)"],
        columns=["Pred Retained (0)", "Pred Churned (1)"]
    )
    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Reds",
        title=f"Confusion Matrix: Best Churn Model ({best_model_name})",
        template="plotly_white"
    )
    fig_cm.update_layout(height=420)
    fig_cm.write_html(os.path.join(fig_dir, "churn_confusion_matrix.html"))

    # 3C: Feature Importance Chart
    best_model = fitted_models[best_model_name]
    if hasattr(best_model, "feature_importances_"):
        feats = [c for c in CHURN_FEATURE_COLS if c in df_churn.columns]
        feat_imp_df = pd.DataFrame({
            "Feature": feats,
            "Importance": best_model.feature_importances_
        }).sort_values("Importance", ascending=True)

        fig_imp = px.bar(
            feat_imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"Feature Importances ({best_model_name})",
            color="Importance",
            color_continuous_scale="Viridis",
            template="plotly_white"
        )
        fig_imp.update_layout(height=480)
        fig_imp.write_html(os.path.join(fig_dir, "churn_feature_importance.html"))

    # 4. Save Customer-Level Predictions
    scaler_path = os.path.join(cfg["paths"]["models_dir"], "churn_scaler.joblib")
    import joblib
    scaler = joblib.load(scaler_path)
    X_all = scaler.transform(df_churn[[c for c in CHURN_FEATURE_COLS if c in df_churn.columns]].fillna(0).values)
    best_clf = fitted_models[best_model_name]
    df_churn["Churn_Probability"] = best_clf.predict_proba(X_all)[:, 1].round(4)
    df_churn["Churn_Predicted"] = (df_churn["Churn_Probability"] >= 0.5).astype(int)

    def churn_risk_tier(p):
        if p >= 0.75:
            return "🔴 High Risk"
        elif p >= 0.40:
            return "🟡 Medium Risk"
        else:
            return "🟢 Low Risk"

    df_churn["Churn_Risk_Tier"] = df_churn["Churn_Probability"].apply(churn_risk_tier)
    df_churn.to_parquet(os.path.join(proc_dir, "customer_churn_predictions.parquet"), index=False)

    # Print Summary
    print("\n" + "=" * 65)
    print("PHASE 5 - CHURN PREDICTION EVALUATION REPORT")
    print("=" * 65)
    print(f"{'Model':<22} | {'ROC-AUC':<9} | {'F1 Score':<9} | {'Recall':<9} | {'Precision':<9}")
    print("-" * 65)
    for m_name, m_data in all_metrics.items():
        star = " *" if m_name == best_model_name else ""
        print(f"{m_name + star:<22} | {m_data['roc_auc']:<9.4f} | {m_data['f1_score']:<9.4f} | {m_data['recall']:<9.4f} | {m_data['precision']:<9.4f}")
    print("-" * 65)
    print(f"Champion Model Selected: {best_model_name} (ROC-AUC: {all_metrics[best_model_name]['roc_auc']:.4f})")
    print(f"Customer Risk Distribution:")
    for tier, count in df_churn["Churn_Risk_Tier"].value_counts().items():
        print(f"  {tier}: {count:,} customers ({count/len(df_churn)*100:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    main()
