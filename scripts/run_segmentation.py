"""
RetailPulse - Phase 3 Customer Segmentation Runner
Executes RFM scoring, KMeans clustering, PCA projections, MLflow tracking, and chart generation.
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
from src.models.segmentation import train_customer_segmentation

logger = get_logger("run_segmentation")

def main():
    logger.info("=" * 65)
    logger.info("Executing Phase 3: Customer Segmentation Pipeline (F1)")
    logger.info("=" * 65)

    cfg = load_config()
    proc_dir = cfg["paths"]["processed_data_dir"]
    fig_dir = os.path.join(cfg["paths"].get("reports_dir", "reports"), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    rfm_path = os.path.join(proc_dir, "customer_features.parquet")
    if not os.path.exists(rfm_path):
        raise FileNotFoundError(f"Customer features not found at {rfm_path}. Run Phase 2 first.")

    logger.info(f"Loading customer RFM features from: {rfm_path}")
    df_rfm = pd.read_parquet(rfm_path)

    # Execute Segmentation Pipeline
    df_segmented, kmeans, scaler, pca, metrics = train_customer_segmentation(
        df_rfm=df_rfm,
        features=["Recency", "Frequency", "Monetary"],
        log_to_mlflow=True
    )

    # Save Enriched Segmented Dataset
    seg_parquet = os.path.join(proc_dir, "customer_segments.parquet")
    seg_csv = os.path.join(proc_dir, "customer_segments.csv")
    df_segmented.to_parquet(seg_parquet, index=False)
    df_segmented.to_csv(seg_csv, index=False)
    logger.info(f"Saved Customer Segments to {seg_parquet} and {seg_csv}")

    # Generate Visualization 1: Elbow Curve & Silhouette Curve
    logger.info("Generating Segmentation Charts: Elbow & Silhouette...")
    k_eval = metrics["k_evaluation"]
    k_vals = list(k_eval["inertias"].keys())
    inertias = list(k_eval["inertias"].values())
    silhouettes = list(k_eval["silhouettes"].values())

    fig_eval = make_subplots(rows=1, cols=2, subplot_titles=("Elbow Method (Inertia)", "Silhouette Score vs k"))
    fig_eval.add_trace(
        go.Scatter(x=k_vals, y=inertias, mode="lines+markers", name="Inertia", marker=dict(size=8, color="#2563EB")),
        row=1, col=1
    )
    fig_eval.add_trace(
        go.Scatter(x=k_vals, y=silhouettes, mode="lines+markers", name="Silhouette", marker=dict(size=8, color="#10B981")),
        row=1, col=2
    )
    # Highlight chosen k
    opt_k = metrics["k"]
    fig_eval.add_vline(x=opt_k, line_dash="dash", line_color="#EF4444", annotation_text=f"Selected k={opt_k}", row=1, col=2)
    fig_eval.update_layout(template="plotly_white", height=420, showlegend=False, title=f"KMeans Cluster Optimization Analysis (Optimal k={opt_k})")
    fig_eval.write_html(os.path.join(fig_dir, "clustering_elbow_silhouette.html"))

    # Generate Visualization 2: 2D PCA Cluster Projection
    logger.info("Generating Segmentation Charts: 2D PCA Cluster Map...")
    fig_pca = px.scatter(
        df_segmented,
        x="PCA1",
        y="PCA2",
        color=df_segmented["KMeans_Cluster"].astype(str),
        symbol="RFM_Segment",
        hover_data=["CustomerID", "Recency", "Frequency", "Monetary", "RFM_Segment"],
        title=f"2D PCA Customer Projection (KMeans Clusters & Heuristic Segment Overlay)",
        labels={"color": "Cluster", "symbol": "RFM Segment"},
        template="plotly_white"
    )
    fig_pca.update_layout(height=560)
    fig_pca.write_html(os.path.join(fig_dir, "clustering_pca_2d.html"))

    # Generate Visualization 3: Cross-Comparison Heatmap
    logger.info("Generating Segmentation Charts: RFM vs KMeans Cross-Tab Heatmap...")
    ct_df = pd.crosstab(df_segmented["RFM_Segment"], df_segmented["KMeans_Cluster"])
    fig_ct = px.imshow(
        ct_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Cross-Tabulation: Heuristic RFM Segments vs Unsupervised KMeans Clusters",
        labels=dict(x="KMeans Cluster ID", y="Heuristic RFM Segment", color="Customer Count"),
        template="plotly_white"
    )
    fig_ct.update_layout(height=500)
    fig_ct.write_html(os.path.join(fig_dir, "clustering_crosstab_heatmap.html"))

    print("\n" + "=" * 65)
    print("PHASE 3 - CUSTOMER SEGMENTATION COMPLETE")
    print("=" * 65)
    print(f"Optimal Cluster Count:  k = {metrics['k']}")
    print(f"Silhouette Score:       {metrics['silhouette_score']:.4f}")
    print(f"Davies-Bouldin Score:   {metrics['davies_bouldin_score']:.4f}")
    print(f"PCA Variance Explained: {sum(metrics['pca_variance_ratio'])*100:.1f}%")
    print("\nCluster Archetype Summary:")
    for prof in metrics["cluster_profiles"]:
        print(f"  Cluster {prof['cluster_id']}: {prof['size']:,} customers ({prof['pct_of_customers']}%) | "
              f"Avg Recency: {prof['avg_recency_days']}d | Avg Freq: {prof['avg_frequency_orders']} | "
              f"Avg Spend: £{prof['avg_monetary_gbp']:,.0f} | Dominant: '{prof['dominant_rfm_segment']}'")
    print("=" * 65)

if __name__ == "__main__":
    main()
