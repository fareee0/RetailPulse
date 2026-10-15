"""
Unit Tests for RetailPulse FastAPI Endpoints
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_api_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert data["service"] == "RetailPulse AI Engine"


def test_api_segment_endpoint():
    res = client.post("/segment", json={
        "Recency": 10.0,
        "Frequency": 12,
        "Monetary": 3500.0
    })
    assert res.status_code == 200
    data = res.json()
    assert "rfm_segment" in data
    assert "kmeans_cluster" in data


def test_api_predict_churn_endpoint():
    res = client.post("/predict/churn", json={
        "Recency": 60.0, "Frequency": 3, "Monetary": 450.0,
        "TenureDays": 200.0, "ActiveLifespanDays": 140.0,
        "AvgOrderValue": 150.0, "AvgBasketSize": 8.0, "AvgPurchaseInterval": 45.0,
        "TotalItems": 24, "UniqueSKUs": 12, "OrdersLast30d": 0, "SpendLast30d": 0.0,
        "OrdersLast90d": 1, "SpendLast90d": 150.0
    })
    assert res.status_code == 200
    data = res.json()
    assert "churn_probability" in data
    assert "risk_tier" in data


def test_api_inventory_recommendations():
    res = client.get("/inventory/recommendations?limit=5")
    assert res.status_code == 200
    data = res.json()
    assert "items" in data
    assert len(data["items"]) <= 5


def test_api_monitoring_drift():
    res = client.get("/monitoring/drift")
    assert res.status_code == 200
    data = res.json()
    assert "dataset_drift_detected" in data
