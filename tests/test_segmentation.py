"""
Unit Tests for RetailPulse Phase 3 Customer Segmentation Engine (F1)
"""

import pytest
import pandas as pd
import numpy as np
from src.models.segmentation import (
    assign_rfm_quintiles,
    find_optimal_k,
    train_customer_segmentation
)

@pytest.fixture
def synthetic_rfm():
    """Generates synthetic RFM dataset spanning different customer tiers."""
    np.random.seed(42)
    n = 150
    data = {
        "CustomerID": [f"C{1000 + i}" for i in range(n)],
        "Recency": np.random.randint(1, 400, size=n),
        "Frequency": np.random.geometric(p=0.2, size=n),
        "Monetary": np.random.exponential(scale=500, size=n) + 10.0,
        "AvgOrderValue": np.random.uniform(20, 200, size=n),
        "TotalItems": np.random.randint(5, 500, size=n),
        "TenureDays": np.random.randint(50, 700, size=n),
        "AvgPurchaseInterval": np.random.uniform(5, 60, size=n),
        "Country": ["United Kingdom"] * n
    }
    return pd.DataFrame(data)

def test_assign_rfm_quintiles(synthetic_rfm):
    """Test quintile scoring logic and segment labeling."""
    df_scored = assign_rfm_quintiles(synthetic_rfm)
    
    assert "R_Score" in df_scored.columns
    assert "F_Score" in df_scored.columns
    assert "M_Score" in df_scored.columns
    assert "RFM_Score" in df_scored.columns
    assert "RFM_Segment" in df_scored.columns
    
    assert df_scored["R_Score"].between(1, 5).all()
    assert df_scored["F_Score"].between(1, 5).all()
    assert df_scored["M_Score"].between(1, 5).all()
    assert not df_scored["RFM_Segment"].isna().any()

def test_find_optimal_k(synthetic_rfm):
    """Test finding optimal k across a given range."""
    X_log = np.log1p(synthetic_rfm[["Recency", "Frequency", "Monetary"]].values)
    k_eval = find_optimal_k(X_log, k_range=range(2, 5))
    
    assert "optimal_k" in k_eval
    assert k_eval["optimal_k"] in range(2, 5)
    assert len(k_eval["silhouettes"]) == 3
    assert all(isinstance(v, float) for v in k_eval["silhouettes"].values())

def test_train_customer_segmentation(synthetic_rfm):
    """Test end-to-end segmentation pipeline."""
    df_seg, km, scaler, pca, metrics = train_customer_segmentation(
        synthetic_rfm,
        features=["Recency", "Frequency", "Monetary"],
        k=3,
        log_to_mlflow=False
    )
    
    assert "KMeans_Cluster" in df_seg.columns
    assert set(df_seg["KMeans_Cluster"].unique()).issubset({0, 1, 2})
    assert "PCA1" in df_seg.columns
    assert "PCA2" in df_seg.columns
    assert metrics["k"] == 3
    assert -1.0 <= metrics["silhouette_score"] <= 1.0
    assert len(metrics["cluster_profiles"]) == 3
