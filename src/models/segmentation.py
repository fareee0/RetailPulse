"""
RetailPulse - Customer Segmentation Engine (F1)
Implements:
  1. Heuristic RFM Quintile Scoring with Named Personas.
  2. Unsupervised KMeans Clustering on log-transformed & scaled RFM metrics.
  3. Elbow method and Silhouette score optimization.
  4. Cross-comparison matrix between RFM heuristic segments and KMeans clusters.
  5. MLflow experiment tracking and model serialization.
"""

import os
import json
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import mlflow

from src.utils.logger import get_logger
from src.utils.config import load_config

logger = get_logger("segmentation")

def assign_rfm_quintiles(df_rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Computes RFM scores (1 to 5) using quintiles and maps named customer segments.
    
    Scoring:
      - Recency: Lower days -> Higher Score (5 = most recent).
      - Frequency: Higher orders -> Higher Score (5 = most frequent).
      - Monetary: Higher spend -> Higher Score (5 = highest spend).
    """
    logger.info("Computing RFM quintile scores and business segment archetypes...")
    df = df_rfm.copy()

    # Recency: 5 is best (lowest days) -> reverse quintile ranking
    df["R_Score"] = pd.qcut(df["Recency"], q=5, labels=[5, 4, 3, 2, 1]).astype(int)

    # Frequency: higher is better (handle duplicates with rank method 'first')
    df["F_Score"] = pd.qcut(df["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    # Monetary: higher is better
    df["M_Score"] = pd.qcut(df["Monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)

    df["RFM_Score"] = df["R_Score"].astype(str) + df["F_Score"].astype(str) + df["M_Score"].astype(str)
    df["RFM_Mean"] = ((df["R_Score"] + df["F_Score"] + df["M_Score"]) / 3.0).round(2)

    # Heuristic Rule-based Segmentation
    def map_segment(row) -> str:
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        rf = (r + f) / 2.0
        
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3 and m >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "New / Promising"
        elif r >= 3 and f >= 3:
            return "Potential Loyalists"
        elif r in [2, 3] and f in [2, 3]:
            return "Need Attention"
        elif r <= 2 and f >= 3 and m >= 3:
            return "At Risk"
        elif r == 1 and f >= 4:
            return "Can't Lose Them"
        elif r <= 2 and f <= 2 and m >= 3:
            return "About to Sleep"
        elif r <= 2 and f <= 2 and m <= 2:
            return "Hibernating / Lost"
        else:
            return "Standard Customers"

    df["RFM_Segment"] = df.apply(map_segment, axis=1)
    logger.info(f"RFM Segmentation distribution:\n{df['RFM_Segment'].value_counts()}")
    return df

def find_optimal_k(
    X_scaled: np.ndarray, 
    k_range: range = range(2, 9), 
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Evaluates KMeans across k_range to compute Inertia (Elbow) and Silhouette Scores.
    """
    logger.info(f"Evaluating optimal k for KMeans across {list(k_range)}...")
    inertias = {}
    silhouettes = {}
    davies_scores = {}

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels))
        db = float(davies_bouldin_score(X_scaled, labels))
        
        inertias[k] = round(float(km.inertia_), 2)
        silhouettes[k] = round(sil, 4)
        davies_scores[k] = round(db, 4)
        logger.info(f"  k={k} -> Silhouette: {sil:.4f} | Inertia: {km.inertia_:,.1f} | Davies-Bouldin: {db:.4f}")

    # Optimal k chosen by highest silhouette score
    optimal_k = max(silhouettes, key=silhouettes.get)
    logger.info(f"Optimal cluster count selected: k={optimal_k} (Silhouette: {silhouettes[optimal_k]})")

    return {
        "optimal_k": optimal_k,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "davies_scores": davies_scores
    }

def train_customer_segmentation(
    df_rfm: pd.DataFrame,
    features: List[str] = ["Recency", "Frequency", "Monetary"],
    k: Optional[int] = None,
    log_to_mlflow: bool = True
) -> Tuple[pd.DataFrame, KMeans, StandardScaler, PCA, Dict[str, Any]]:
    """
    Executes full customer segmentation pipeline:
      - Log1p transformation for skewness.
      - Standard scaling.
      - KMeans clustering & PCA 2D projection.
      - Cross-comparison matrix against RFM quintile segments.
      - MLflow experiment logging.
      
    Returns:
        Tuple: (Segmented DataFrame, fitted KMeans, fitted Scaler, fitted PCA, Evaluation Metrics).
    """
    cfg = load_config()
    random_state = cfg.get("project", {}).get("random_seed", 42)
    models_dir = cfg.get("paths", {}).get("models_dir", "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Heuristic RFM Quintiles
    df_scored = assign_rfm_quintiles(df_rfm)

    # 2. Log-transformation of skewed features (Recency, Frequency, Monetary)
    logger.info(f"Applying log1p transformation to features: {features}...")
    X_log = np.log1p(df_scored[features].values)

    # 3. Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    # 4. Find optimal k (or use specified k)
    k_eval = find_optimal_k(X_scaled, k_range=range(2, 9), random_state=random_state)
    if k is None:
        k = k_eval["optimal_k"]

    # 5. Fit Champion KMeans Model
    logger.info(f"Fitting final KMeans model with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=15)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_scored["KMeans_Cluster"] = cluster_labels

    # 6. Dimensionality Reduction (PCA to 2 components for visualization)
    pca = PCA(n_components=2, random_state=random_state)
    pca_2d = pca.fit_transform(X_scaled)
    df_scored["PCA1"] = pca_2d[:, 0].round(4)
    df_scored["PCA2"] = pca_2d[:, 1].round(4)
    explained_variance = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    logger.info(f"PCA 2D explained variance ratios: {explained_variance} (Total: {sum(explained_variance)*100:.1f}%)")

    # 7. Cluster Persona Characterization
    cluster_profiles = []
    for c in range(k):
        sub = df_scored[df_scored["KMeans_Cluster"] == c]
        profile = {
            "cluster_id": c,
            "size": len(sub),
            "pct_of_customers": round(len(sub) / len(df_scored) * 100, 2),
            "avg_recency_days": round(float(sub["Recency"].mean()), 1),
            "avg_frequency_orders": round(float(sub["Frequency"].mean()), 1),
            "avg_monetary_gbp": round(float(sub["Monetary"].mean()), 2),
            "avg_aov_gbp": round(float(sub["AvgOrderValue"].mean()), 2),
            "dominant_rfm_segment": sub["RFM_Segment"].mode().iloc[0] if not sub.empty else "N/A"
        }
        cluster_profiles.append(profile)

    # 8. Cross-Tabulation: RFM Segments vs KMeans Clusters
    cross_tab = pd.crosstab(df_scored["RFM_Segment"], df_scored["KMeans_Cluster"]).to_dict()

    # Evaluation Metrics
    sil_score = float(silhouette_score(X_scaled, cluster_labels))
    db_score = float(davies_bouldin_score(X_scaled, cluster_labels))
    ch_score = float(calinski_harabasz_score(X_scaled, cluster_labels))

    metrics = {
        "k": k,
        "silhouette_score": round(sil_score, 4),
        "davies_bouldin_score": round(db_score, 4),
        "calinski_harabasz_score": round(ch_score, 2),
        "inertia": round(float(kmeans.inertia_), 2),
        "pca_variance_ratio": explained_variance,
        "k_evaluation": k_eval,
        "cluster_profiles": cluster_profiles,
        "cross_tabulation": cross_tab
    }

    # 9. Save Models & Artifacts
    joblib.dump(kmeans, os.path.join(models_dir, "kmeans_model.joblib"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    joblib.dump(pca, os.path.join(models_dir, "pca.joblib"))
    with open(os.path.join(models_dir, "clustering_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved clustering models and metrics to {models_dir}/")

    # 10. Log to MLflow
    if log_to_mlflow:
        tracking_uri = cfg.get("mlops", {}).get("mlflow_tracking_uri", "sqlite:///mlflow.db")
        exp_name = cfg.get("mlops", {}).get("experiment_name", "RetailPulse_Analytics")
        logger.info(f"Logging clustering experiment to MLflow (URI: {tracking_uri}, Exp: {exp_name})...")
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(exp_name)

        with mlflow.start_run(run_name="Customer_Segmentation_KMeans") as run:
            mlflow.log_param("model_type", "KMeans_Clustering")
            mlflow.log_param("k_clusters", k)
            mlflow.log_param("features", ",".join(features))
            mlflow.log_param("scaling", "StandardScaler_Log1p")
            mlflow.log_param("random_seed", random_state)
            
            mlflow.log_metric("silhouette_score", sil_score)
            mlflow.log_metric("davies_bouldin_score", db_score)
            mlflow.log_metric("calinski_harabasz_score", ch_score)
            mlflow.log_metric("inertia", float(kmeans.inertia_))
            mlflow.log_metric("pca_var_component_1", explained_variance[0])
            mlflow.log_metric("pca_var_component_2", explained_variance[1])

            mlflow.log_artifact(os.path.join(models_dir, "clustering_metrics.json"))
            mlflow.log_artifact(os.path.join(models_dir, "kmeans_model.joblib"))

            logger.info(f"MLflow Run completed. Run ID: {run.info.run_id}")

    return df_scored, kmeans, scaler, pca, metrics
