"""
RetailPulse - Production FastAPI Serving Layer (F6)
Exposes endpoints for:
  - GET  /health: Service and model registry health check
  - POST /predict/churn: Real-time customer churn probability & risk tier
  - POST /forecast: Multi-step future sales demand forecast
  - POST /segment: Customer RFM scoring and KMeans cluster classification
  - GET  /inventory/recommendations: SKU-level safety stock, ROP, and EOQ alerts
  - GET  /monitoring/drift: Data & concept drift audit report
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.config import load_config
from src.models.segmentation import assign_rfm_quintiles
from src.models.churn import CHURN_FEATURE_COLS

logger = get_logger("api")
cfg = load_config()

app = FastAPI(
    title="RetailPulse AI Analytics & Forecasting API",
    description="Production-grade RESTful API for customer analytics, churn risk, demand forecasting, and inventory optimization.",
    version="1.0.0"
)

MODELS_DIR = cfg.get("paths", {}).get("models_dir", "models")
PROC_DIR = cfg.get("paths", {}).get("processed_data_dir", "data/processed")
REPORTS_DIR = cfg.get("paths", {}).get("reports_dir", "reports")

# In-memory artifact cache
_MODELS_CACHE: Dict[str, Any] = {}


def get_cached_model(filename: str):
    if filename not in _MODELS_CACHE:
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model artifact '{filename}' not found at {path}. Run training pipelines first.")
        _MODELS_CACHE[filename] = joblib.load(path)
    return _MODELS_CACHE[filename]


# --- Request & Response Schemas ---

class ChurnPredictionRequest(BaseModel):
    Recency: float = Field(..., description="Days since last purchase", example=45.0)
    Frequency: int = Field(..., description="Total lifetime distinct orders", example=6)
    Monetary: float = Field(..., description="Total lifetime spend in GBP", example=1250.50)
    TenureDays: float = Field(..., description="Days since first customer order", example=320.0)
    ActiveLifespanDays: float = Field(..., description="Days between first and last order", example=275.0)
    AvgOrderValue: float = Field(..., description="Average spend per order in GBP", example=208.41)
    AvgBasketSize: float = Field(..., description="Average items per order", example=14.2)
    AvgPurchaseInterval: float = Field(..., description="Average days between consecutive orders", example=55.0)
    TotalItems: int = Field(..., description="Total items bought lifetime", example=85)
    UniqueSKUs: int = Field(..., description="Distinct SKUs purchased", example=28)
    OrdersLast30d: int = Field(default=0, description="Orders in the last 30 days", example=1)
    SpendLast30d: float = Field(default=0.0, description="Spend in the last 30 days in GBP", example=150.0)
    OrdersLast90d: int = Field(default=0, description="Orders in the last 90 days", example=2)
    SpendLast90d: float = Field(default=0.0, description="Spend in the last 90 days in GBP", example=380.0)


class ChurnPredictionResponse(BaseModel):
    churn_probability: float
    churn_predicted: int
    risk_tier: str
    actionable_recommendation: str


class SegmentationRequest(BaseModel):
    Recency: float = Field(..., description="Days since last purchase", example=12.0)
    Frequency: int = Field(..., description="Total lifetime order count", example=15)
    Monetary: float = Field(..., description="Total spend in GBP", example=4500.0)
    AvgOrderValue: Optional[float] = Field(default=None, description="Average order value")


class SegmentationResponse(BaseModel):
    rfm_score: str
    rfm_segment: str
    kmeans_cluster: int
    archetype: str


class ForecastRequest(BaseModel):
    horizon_days: int = Field(default=30, ge=1, le=180, description="Number of future days to forecast")


class ForecastDataPoint(BaseModel):
    date: str
    predicted_sales: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class ForecastResponse(BaseModel):
    model_type: str
    horizon_days: int
    forecast: List[ForecastDataPoint]


# --- Endpoints ---

@app.get("/health", tags=["System"])
def health_check():
    """Returns system status, active version, and model availability."""
    models_ready = {
        "kmeans": os.path.exists(os.path.join(MODELS_DIR, "kmeans_model.joblib")),
        "churn": os.path.exists(os.path.join(MODELS_DIR, "churn_best_model.joblib")),
        "prophet": os.path.exists(os.path.join(MODELS_DIR, "prophet_model.joblib")),
        "lstm": os.path.exists(os.path.join(MODELS_DIR, "lstm_model.joblib")),
    }
    all_ready = all(models_ready.values())
    return {
        "status": "HEALTHY" if all_ready else "PARTIAL",
        "service": "RetailPulse AI Engine",
        "version": cfg.get("project", {}).get("version", "1.0.0"),
        "models_available": models_ready
    }


@app.post("/predict/churn", response_model=ChurnPredictionResponse, tags=["Customer Analytics"])
def predict_churn(payload: ChurnPredictionRequest):
    """Predicts customer churn probability and returns retention guidance."""
    try:
        model = get_cached_model("churn_best_model.joblib")
        scaler = get_cached_model("churn_scaler.joblib")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {str(e)}")

    feat_dict = payload.model_dump()
    vector = np.array([[feat_dict.get(c, 0.0) for c in CHURN_FEATURE_COLS]])
    vector_scaled = scaler.transform(vector)

    prob = float(model.predict_proba(vector_scaled)[0, 1])
    is_churn = int(prob >= 0.5)

    if prob >= 0.75:
        tier = "🔴 High Risk"
        rec = "Immediate automated win-back discount offer (20%) + personalized account manager outreach."
    elif prob >= 0.40:
        tier = "🟡 Medium Risk"
        rec = "Engage with personalized product recommendations and loyalty tier incentives."
    else:
        tier = "🟢 Low Risk"
        rec = "Maintain standard automated nurture cadence; customer engagement is strong."

    return ChurnPredictionResponse(
        churn_probability=round(prob, 4),
        churn_predicted=is_churn,
        risk_tier=tier,
        actionable_recommendation=rec
    )


@app.post("/segment", response_model=SegmentationResponse, tags=["Customer Analytics"])
def segment_customer(payload: SegmentationRequest):
    """Assigns RFM quintile persona and unsupervised KMeans cluster."""
    try:
        kmeans = get_cached_model("kmeans_model.joblib")
        scaler = get_cached_model("scaler.joblib")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {str(e)}")

    # Prepare single row DataFrame for RFM calculation
    aov = payload.AvgOrderValue or (payload.Monetary / max(payload.Frequency, 1))
    df_row = pd.DataFrame([{
        "Recency": payload.Recency,
        "Frequency": payload.Frequency,
        "Monetary": payload.Monetary,
        "AvgOrderValue": aov
    }])

    # Quintile mapping heuristics
    r = 5 if payload.Recency <= 15 else (4 if payload.Recency <= 45 else (3 if payload.Recency <= 90 else (2 if payload.Recency <= 180 else 1)))
    f = 5 if payload.Frequency >= 10 else (4 if payload.Frequency >= 6 else (3 if payload.Frequency >= 3 else (2 if payload.Frequency >= 2 else 1)))
    m = 5 if payload.Monetary >= 3000 else (4 if payload.Monetary >= 1500 else (3 if payload.Monetary >= 600 else (2 if payload.Monetary >= 250 else 1)))
    
    rfm_score = f"{r}{f}{m}"
    if r >= 4 and f >= 4 and m >= 4:
        rfm_seg = "Champions"
    elif r >= 3 and f >= 3 and m >= 3:
        rfm_seg = "Loyal Customers"
    elif r >= 4 and f <= 2:
        rfm_seg = "New / Promising"
    elif r <= 2 and f >= 3:
        rfm_seg = "At Risk"
    elif r == 1 and f >= 4:
        rfm_seg = "Can't Lose Them"
    elif r <= 2 and f <= 2:
        rfm_seg = "Hibernating / Lost"
    else:
        rfm_seg = "Standard Customers"

    # KMeans cluster
    X_log = np.log1p([[payload.Recency, payload.Frequency, payload.Monetary]])
    X_scaled = scaler.transform(X_log)
    cluster = int(kmeans.predict(X_scaled)[0])

    cluster_archetypes = {
        0: "Core Active Buyers",
        1: "High-Value VIPs",
        2: "Lapsed / Infrequent",
        3: "At-Risk Spenders",
        4: "Recent Trial Customers"
    }
    archetype = cluster_archetypes.get(cluster, f"Cluster {cluster}")

    return SegmentationResponse(
        rfm_score=rfm_score,
        rfm_segment=rfm_seg,
        kmeans_cluster=cluster,
        archetype=archetype
    )


@app.post("/forecast", response_model=ForecastResponse, tags=["Forecasting"])
def get_demand_forecast(payload: ForecastRequest):
    """Returns future sales demand predictions with confidence bounds."""
    try:
        prophet_model = get_cached_model("prophet_model.joblib")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Forecasting model not loaded: {str(e)}")

    future = prophet_model.make_future_dataframe(periods=payload.horizon_days, freq="D")
    forecast_df = prophet_model.predict(future)
    future_tail = forecast_df.tail(payload.horizon_days)

    results = []
    for _, row in future_tail.iterrows():
        results.append(ForecastDataPoint(
            date=pd.to_datetime(row["ds"]).strftime("%Y-%m-%d"),
            predicted_sales=round(float(max(row["yhat"], 0.0)), 2),
            lower_bound=round(float(max(row["yhat_lower"], 0.0)), 2),
            upper_bound=round(float(max(row["yhat_upper"], 0.0)), 2)
        ))

    return ForecastResponse(
        model_type="Prophet_with_UK_Holidays",
        horizon_days=payload.horizon_days,
        forecast=results
    )


@app.get("/inventory/recommendations", tags=["Inventory Optimization"])
def get_inventory_recommendations(
    risk_filter: Optional[str] = Query(None, description="Filter by risk tier: 'Stockout', 'Overstock', 'Optimal'"),
    limit: int = Query(50, ge=1, le=100)
):
    """Retrieves SKU-level safety stock, reorder point, and EOQ recommendations."""
    path = os.path.join(PROC_DIR, "inventory_recommendations.parquet")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Inventory recommendations not generated yet.")

    df_inv = pd.read_parquet(path)
    if risk_filter:
        df_inv = df_inv[df_inv["risk_status"].str.contains(risk_filter, case=False, na=False)]

    df_inv = df_inv.head(limit)
    return {
        "total_records": len(df_inv),
        "items": df_inv.to_dict(orient="records")
    }


@app.get("/monitoring/drift", tags=["Monitoring & Governance"])
def get_drift_report():
    """Retrieves latest feature drift and model health audit report."""
    path = os.path.join(REPORTS_DIR, "drift_audit_report.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Drift audit report not generated yet.")

    with open(path, "r") as f:
        data = json.load(f)
    return data
