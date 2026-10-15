"""
RetailPulse – Page 4: Inventory Intelligence & Optimization
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="RetailPulse – Inventory Intelligence", page_icon="📦", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1e1b4b 100%); color: #f8fafc; }
</style>
""", unsafe_allow_html=True)

PROC_DIR = "data/processed"

@st.cache_data(ttl=600)
def load_inv_data():
    path = os.path.join(PROC_DIR, "inventory_recommendations.parquet")
    return pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame()

df_inv = load_inv_data()

st.title("📦 Inventory Optimization & Stockout Prevention Engine")
st.markdown("Automated computation of Safety Stock (Z=1.645, 95% SLA), Reorder Points (ROP), and Economic Order Quantity (EOQ) across top SKUs.")

# KPI summary
if not df_inv.empty:
    k1, k2, k3, k4 = st.columns(4)
    total_skus = len(df_inv)
    stockouts = (df_inv["risk_status"] == "🔴 Stockout Risk").sum()
    overstock = (df_inv["risk_status"] == "🟡 Overstock Risk").sum()
    optimal = (df_inv["risk_status"] == "🟢 Optimal").sum()

    k1.metric("Monitored Top SKUs", total_skus)
    k2.metric("Critical Stockout Alerts", f"{stockouts} SKUs", "Order Immediately", delta_color="inverse")
    k3.metric("Overstock Warnings", f"{overstock} SKUs", "Excess Holding Cost", delta_color="off")
    k4.metric("Optimal Stock Health", f"{optimal} SKUs", "Within Tolerance")

st.markdown("<br>", unsafe_allow_html=True)

# Risk Filter
if not df_inv.empty:
    risk_options = ["All Risks", "🔴 Stockout Risk", "🟡 Overstock Risk", "🟢 Optimal"]
    sel_risk = st.selectbox("🚦 Filter by Risk Status", risk_options)
    
    filtered_inv = df_inv.copy()
    if sel_risk != "All Risks":
        filtered_inv = filtered_inv[filtered_inv["risk_status"] == sel_risk]

    # Visualizations
    col1, col2 = st.columns([5, 5])
    with col1:
        st.markdown("#### Inventory Risk Ratio")
        counts = df_inv["risk_status"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        fig_donut = px.pie(
            counts,
            names="Status",
            values="Count",
            hole=0.55,
            template="plotly_dark",
            color="Status",
            color_discrete_map={
                "🔴 Stockout Risk": "#ef4444",
                "🟡 Overstock Risk": "#f59e0b",
                "🟢 Optimal": "#10b981"
            }
        )
        fig_donut.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=380)
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        st.markdown("#### Reorder Point vs Safety Stock (Top SKUs)")
        top12 = filtered_inv.head(12)
        fig_bar = go.Figure(data=[
            go.Bar(name="Safety Stock", x=top12["StockCode"], y=top12["safety_stock"], marker_color="#38bdf8"),
            go.Bar(name="Reorder Point", x=top12["StockCode"], y=top12["reorder_point"], marker_color="#f59e0b"),
            go.Bar(name="EOQ Batch", x=top12["StockCode"], y=top12["eoq"], marker_color="#10b981"),
        ])
        fig_bar.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=380
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### 📋 Automated SKU Reorder Action Schedule")
    cols_display = [
        "StockCode", "Description", "avg_daily_demand", "safety_stock",
        "reorder_point", "eoq", "risk_status", "recommendation"
    ]
    st.dataframe(
        filtered_inv[[c for c in cols_display if c in filtered_inv.columns]].style.format({
            "avg_daily_demand": "{:.1f}",
            "safety_stock": "{:,.0f}",
            "reorder_point": "{:,.0f}",
            "eoq": "{:,.0f}"
        }),
        use_container_width=True,
        height=450
    )
