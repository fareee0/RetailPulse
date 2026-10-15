"""
RetailPulse – Page 1: Sales & Revenue Analytics
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="RetailPulse – Sales Analytics", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1e1b4b 100%); color: #f8fafc; }
</style>
""", unsafe_allow_html=True)

PROC_DIR = "data/processed"

@st.cache_data(ttl=600)
def load_sales_data():
    daily_path = os.path.join(PROC_DIR, "daily_sales_features.parquet")
    clean_path = os.path.join(PROC_DIR, "online_retail_cleaned.parquet")
    df_daily = pd.read_parquet(daily_path) if os.path.exists(daily_path) else pd.DataFrame()
    df_clean = pd.read_parquet(clean_path) if os.path.exists(clean_path) else pd.DataFrame()
    return df_daily, df_clean

df_daily, df_clean = load_sales_data()

st.title("📊 Sales & Revenue Velocity Analytics")
st.markdown("Explore macroeconomic sales volume, geographic penetration, seasonality, and product concentration.")

# Global Filters
with st.expander("🎛️ Interactive Filters", expanded=True):
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        if not df_clean.empty:
            countries = ["All Countries"] + sorted(df_clean["Country"].dropna().unique().tolist())
            sel_country = st.selectbox("🌍 Filter by Country", countries)
        else:
            sel_country = "All Countries"
    with col_f2:
        if not df_daily.empty:
            df_daily["Date"] = pd.to_datetime(df_daily["Date"])
            min_date = df_daily["Date"].min().date()
            max_date = df_daily["Date"].max().date()
            date_range = st.date_input("📅 Date Range", (min_date, max_date), min_value=min_date, max_value=max_date)
        else:
            date_range = None

# Filter data
filtered_daily = df_daily.copy()
filtered_clean = df_clean.copy()

if date_range and len(date_range) == 2 and not filtered_daily.empty:
    filtered_daily = filtered_daily[(filtered_daily["Date"].dt.date >= date_range[0]) & (filtered_daily["Date"].dt.date <= date_range[1])]

if sel_country != "All Countries" and not filtered_clean.empty:
    filtered_clean = filtered_clean[filtered_clean["Country"] == sel_country]

# KPIs
k1, k2, k3, k4 = st.columns(4)
rev = filtered_daily["total_sales"].sum() if not filtered_daily.empty else 0
orders = filtered_daily["num_orders"].sum() if not filtered_daily.empty else 0
avg_day_rev = filtered_daily["total_sales"].mean() if not filtered_daily.empty else 0
top_sku = filtered_clean["Description"].value_counts().index[0] if not filtered_clean.empty else "N/A"

k1.metric("Selected Period Revenue", f"£{rev:,.0f}")
k2.metric("Total Transactions", f"{orders:,.0f}")
k3.metric("Daily Avg Run Rate", f"£{avg_day_rev:,.0f}/day")
k4.metric("Top Demand Product", str(top_sku)[:25] + "...")

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Sales Trajectory", "🌍 Geographic Spread", "🏆 Top Performing SKUs"])

with tab1:
    if not filtered_daily.empty:
        fig_line = px.line(
            filtered_daily,
            x="Date",
            y=["total_sales", "total_sales_rolling_mean_7d", "total_sales_rolling_mean_28d"],
            labels={"value": "Revenue (£)", "variable": "Metric"},
            title="Daily Revenue with 7-Day & 28-Day Moving Average Trends",
            template="plotly_dark",
            color_discrete_sequence=["#38bdf8", "#f59e0b", "#10b981"]
        )
        fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=450)
        st.plotly_chart(fig_line, use_container_width=True)

with tab2:
    if not filtered_clean.empty:
        geo = filtered_clean.groupby("Country").agg(
            revenue=("TotalPrice", "sum"),
            orders=("Invoice", "nunique")
        ).reset_index().sort_values("revenue", ascending=False).head(15)

        fig_geo = px.bar(
            geo,
            x="revenue",
            y="Country",
            orientation="h",
            title="Top 15 Countries by Generated Revenue (£)",
            template="plotly_dark",
            color="revenue",
            color_continuous_scale="Viridis"
        )
        fig_geo.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=450)
        st.plotly_chart(fig_geo, use_container_width=True)

with tab3:
    if not filtered_clean.empty:
        top_prods = filtered_clean.groupby(["StockCode", "Description"]).agg(
            revenue=("TotalPrice", "sum"),
            units_sold=("Quantity", "sum")
        ).reset_index().sort_values("revenue", ascending=False).head(20)

        st.dataframe(
            top_prods.style.format({"revenue": "£{:,.2f}", "units_sold": "{:,}"}),
            use_container_width=True,
            height=400
        )
