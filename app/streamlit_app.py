"""
RetailPulse – AI-Powered Customer Analytics & Demand Forecasting Platform
Streamlit Multi-Page Executive Hub (F6)
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="RetailPulse – Executive Overview",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark glass styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1e1b4b 100%);
    color: #f8fafc;
}
.metric-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.2rem;
    backdrop-filter: blur(12px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 179, 237, 0.4);
}
.metric-title {
    color: #94a3b8;
    font-size: 0.82rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-value {
    color: #f8fafc;
    font-size: 1.85rem;
    font-weight: 700;
    margin-top: 0.2rem;
}
.metric-sub {
    color: #10b981;
    font-size: 0.78rem;
    font-weight: 500;
    margin-top: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

PROC_DIR = "data/processed"
REPORTS_DIR = "reports"
MODELS_DIR = "models"


@st.cache_data(ttl=600)
def load_data():
    seg_path = os.path.join(PROC_DIR, "customer_segments.parquet")
    churn_path = os.path.join(PROC_DIR, "customer_churn_predictions.parquet")
    daily_path = os.path.join(PROC_DIR, "daily_sales_features.parquet")
    inv_path = os.path.join(PROC_DIR, "inventory_recommendations.parquet")
    
    df_seg = pd.read_parquet(seg_path) if os.path.exists(seg_path) else pd.DataFrame()
    df_churn = pd.read_parquet(churn_path) if os.path.exists(churn_path) else pd.DataFrame()
    df_daily = pd.read_parquet(daily_path) if os.path.exists(daily_path) else pd.DataFrame()
    df_inv = pd.read_parquet(inv_path) if os.path.exists(inv_path) else pd.DataFrame()
    return df_seg, df_churn, df_daily, df_inv


@st.cache_data(ttl=600)
def load_health():
    path = os.path.join(REPORTS_DIR, "model_health_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


df_seg, df_churn, df_daily, df_inv = load_data()
health_info = load_health()

# Header
st.markdown("""
<div style="padding: 1rem 0 1.5rem 0;">
    <div style="font-size: 2.2rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">
        🛒 RetailPulse Executive Command Center
    </div>
    <div style="color: #94a3b8; font-size: 1rem; margin-top: 0.2rem;">
        AI-Powered Customer Intelligence, Demand Forecasting & Inventory Optimization
    </div>
</div>
""", unsafe_allow_html=True)

# KPI Row
c1, c2, c3, c4, c5 = st.columns(5)
total_rev = df_daily["total_sales"].sum() if not df_daily.empty else 0.0
total_cust = len(df_seg) if not df_seg.empty else 0
avg_aov = df_seg["AvgOrderValue"].mean() if not df_seg.empty else 0.0
high_risk_churn = (df_churn["Churn_Risk_Tier"] == "🔴 High Risk").sum() if not df_churn.empty else 0
stockout_items = (df_inv["risk_status"] == "🔴 Stockout Risk").sum() if not df_inv.empty else 0

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Revenue</div>
        <div class="metric-value">£{total_rev:,.0f}</div>
        <div class="metric-sub">2-Year Transaction Horizon</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Customer Base</div>
        <div class="metric-value">{total_cust:,}</div>
        <div class="metric-sub">100% Segmented</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Average AOV</div>
        <div class="metric-value">£{avg_aov:.2f}</div>
        <div class="metric-sub">Across All Cohorts</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">High Churn Risk</div>
        <div class="metric-value" style="color: #f87171;">{high_risk_churn:,}</div>
        <div class="metric-sub" style="color: #f87171;">Requires Immediate Win-Back</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Stockout Alerts</div>
        <div class="metric-value" style="color: #fbbf24;">{stockout_items} SKUs</div>
        <div class="metric-sub" style="color: #fbbf24;">Reorder Point Breached</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main Grid: Revenue Trend & Customer Persona Mix
col_left, col_right = st.columns([6, 4])

with col_left:
    st.markdown("### 📈 Revenue Velocity & Seasonal Trajectory")
    if not df_daily.empty:
        fig_trend = px.line(
            df_daily,
            x="Date",
            y="total_sales",
            title="Daily Gross Merchandise Value (£)",
            template="plotly_dark",
            color_discrete_sequence=["#38bdf8"]
        )
        fig_trend.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

with col_right:
    st.markdown("### 👥 Customer Persona Breakdown")
    if not df_seg.empty:
        seg_counts = df_seg["RFM_Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        fig_donut = px.pie(
            seg_counts,
            names="Segment",
            values="Count",
            hole=0.55,
            template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.Teal_r
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

# MLOps System Health Banner
st.markdown("---")
st.markdown("### 🛡️ Production Health & Drift Governance Panel")
h1, h2, h3, h4 = st.columns(4)
drift_detected = health_info.get("drift_detected", False)
drift_share = health_info.get("drift_share_pct", 0.0)
retrain = health_info.get("retrain_required", False)

with h1:
    status_icon = "🔴 DRIFT DETECTED" if drift_detected else "🟢 STABLE"
    st.metric("Feature Drift Status", status_icon)
with h2:
    st.metric("Drifted Features Share", f"{drift_share}%")
with h3:
    st.metric("Forecast Model Status", health_info.get("forecast_health", {}).get("status", "HEALTHY"))
with h4:
    st.metric("Auto-Retraining Flag", "TRIGGERED" if retrain else "IDLE")

st.info("💡 Use the sidebar navigation to drill down into detailed Sales, Customer Intelligence, Forecasting, and Inventory pages.")
