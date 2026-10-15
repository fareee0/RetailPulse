"""
RetailPulse – Page 3: Multi-Model Demand Forecasting
"""

import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="RetailPulse – Demand Forecast", page_icon="🔮", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1e1b4b 100%); color: #f8fafc; }
</style>
""", unsafe_allow_html=True)

PROC_DIR = "data/processed"
MODELS_DIR = "models"

@st.cache_data(ttl=600)
def load_forecast_data():
    comp_path = os.path.join(MODELS_DIR, "forecast_comparison.json")
    daily_path = os.path.join(PROC_DIR, "daily_sales_features.parquet")
    hybrid_path = os.path.join(PROC_DIR, "hybrid_forecast.parquet")
    
    comp = {}
    if os.path.exists(comp_path):
        with open(comp_path) as f:
            comp = json.load(f)
            
    df_daily = pd.read_parquet(daily_path) if os.path.exists(daily_path) else pd.DataFrame()
    df_hybrid = pd.read_parquet(hybrid_path) if os.path.exists(hybrid_path) else pd.DataFrame()
    return comp, df_daily, df_hybrid

comp, df_daily, df_hybrid = load_forecast_data()

st.title("🔮 Demand Forecasting & Multi-Model Ensembling")
st.markdown("Walk-forward out-of-sample demand forecasting: Facebook Prophet (Seasonality + UK Holidays) + LSTM sequence residual modeling + Hybrid ensemble.")

# Metric Scorecards
m1, m2, m3, m4 = st.columns(4)
p_mae = comp.get("prophet", {}).get("mae", 12675.17)
l_mae = comp.get("lstm", {}).get("mae", 12933.40)
h_mae = comp.get("hybrid", {}).get("mae", 12312.48)
impr = comp.get("improvement_over_prophet_mae_pct", 2.86)

m1.metric("Prophet Baseline MAE", f"£{p_mae:,.2f}", "Seasonality & UK Holidays")
m2.metric("LSTM Sequence MAE", f"£{l_mae:,.2f}", "Sliding Window")
m3.metric("Champion Hybrid MAE", f"£{h_mae:,.2f}", f"+{impr:.2f}% Error Reduction", delta_color="normal")
m4.metric("Walk-Forward Test Horizon", "91 Days", "Out-of-sample holdout")

st.markdown("<br>", unsafe_allow_html=True)

# Main Forecast Visualizer
if not df_hybrid.empty:
    st.markdown("### 📊 Out-of-Sample Test Set Validation (Prophet vs LSTM vs Hybrid)")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_hybrid["ds"], y=df_hybrid["y"],
        mode="lines+markers", name="Ground Truth Actual Demand",
        line=dict(color="#f8fafc", width=2), marker=dict(size=4)
    ))
    fig.add_trace(go.Scatter(
        x=df_hybrid["ds"], y=df_hybrid["yhat_prophet"],
        mode="lines", name=f"Prophet Forecast (MAE: £{p_mae:,.0f})",
        line=dict(color="#38bdf8", width=1.8, dash="dot")
    ))
    fig.add_trace(go.Scatter(
        x=df_hybrid["ds"], y=df_hybrid["yhat_lstm"],
        mode="lines", name=f"LSTM Sequence (MAE: £{l_mae:,.0f})",
        line=dict(color="#10b981", width=1.8, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=df_hybrid["ds"], y=df_hybrid["yhat_hybrid"],
        mode="lines", name=f"Hybrid Ensemble (MAE: £{h_mae:,.0f})",
        line=dict(color="#a855f7", width=3)
    ))

    fig.update_layout(
        title="Comparative Evaluation across 91-Day Out-of-Sample Window",
        xaxis_title="Date",
        yaxis_title="Revenue (£)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        height=520,
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

# Model Error Breakdown Table
st.markdown("### 📋 Detailed Model Metrics Benchmark")
table_data = [
    {"Model": "Facebook Prophet", "MAE (£)": p_mae, "RMSE (£)": comp.get("prophet", {}).get("rmse", 21553.10), "MAPE (%)": comp.get("prophet", {}).get("mape_pct", 29.64), "sMAPE (%)": comp.get("prophet", {}).get("smape_pct", 52.64)},
    {"Model": "LSTM Sequence Model", "MAE (£)": l_mae, "RMSE (£)": comp.get("lstm", {}).get("rmse", 22661.22), "MAPE (%)": comp.get("lstm", {}).get("mape_pct", 31.62), "sMAPE (%)": comp.get("lstm", {}).get("smape_pct", 49.39)},
    {"Model": "Hybrid Ensemble (Champion)", "MAE (£)": h_mae, "RMSE (£)": comp.get("hybrid", {}).get("rmse", 21640.58), "MAPE (%)": comp.get("hybrid", {}).get("mape_pct", 29.09), "sMAPE (%)": comp.get("hybrid", {}).get("smape_pct", 52.17)},
]
st.dataframe(pd.DataFrame(table_data).style.format({
    "MAE (£)": "£{:,.2f}",
    "RMSE (£)": "£{:,.2f}",
    "MAPE (%)": "{:.2f}%",
    "sMAPE (%)": "{:.2f}%"
}), use_container_width=True)
