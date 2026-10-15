"""
RetailPulse - Exploratory Data Analysis (EDA) Report Generator
Produces publication-grade interactive Plotly visualizations across sales trends, 
seasonality, geographic distribution, and customer RFM behaviors.
"""

import os
import sys
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

logger = get_logger("eda_generator")

def generate_eda_visualizations():
    logger.info("Generating interactive Plotly EDA charts...")
    cfg = load_config()
    proc_dir = cfg["paths"]["processed_data_dir"]
    fig_dir = os.path.join(cfg["paths"].get("reports_dir", "reports"), "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Load Cleaned Transactions and Features
    df_sales = pd.read_parquet(cfg["data"]["cleaned_parquet"])
    df_daily = pd.read_parquet(os.path.join(proc_dir, "daily_sales_features.parquet"))
    df_rfm = pd.read_parquet(os.path.join(proc_dir, "customer_features.parquet"))

    # Chart 1: Top 10 Products by Revenue & Quantity Sold
    logger.info("Creating Chart 1: Top Products...")
    prod_agg = df_sales.groupby(["StockCode", "Description"]).agg(
        TotalRevenue=("TotalPrice", "sum"),
        TotalQuantity=("Quantity", "sum")
    ).reset_index().sort_values("TotalRevenue", ascending=False).head(10)
    
    # Format labels
    prod_agg["Label"] = prod_agg["Description"].str[:28] + " (" + prod_agg["StockCode"] + ")"
    
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(
        go.Bar(
            x=prod_agg["Label"], 
            y=prod_agg["TotalRevenue"], 
            name="Revenue (£)", 
            marker_color="#2563EB",
            text=prod_agg["TotalRevenue"].apply(lambda v: f"£{v:,.0f}"),
            textposition="auto"
        ),
        secondary_y=False
    )
    fig1.add_trace(
        go.Scatter(
            x=prod_agg["Label"], 
            y=prod_agg["TotalQuantity"], 
            name="Units Sold", 
            mode="lines+markers+text",
            marker=dict(size=9, color="#F59E0B"),
            text=prod_agg["TotalQuantity"].apply(lambda v: f"{v:,}"),
            textposition="top center"
        ),
        secondary_y=True
    )
    fig1.update_layout(
        title="Top 10 Revenue-Generating Products & Volume Sold",
        template="plotly_white",
        hovermode="x unified",
        xaxis_tickangle=-25,
        height=520,
        margin=dict(l=40, r=40, t=60, b=100)
    )
    fig1.write_html(os.path.join(fig_dir, "top_products.html"))

    # Chart 2: Top Countries by Revenue
    logger.info("Creating Chart 2: Top Countries by Revenue...")
    country_agg = df_sales.groupby("Country").agg(
        Revenue=("TotalPrice", "sum"),
        Customers=("CustomerID", "nunique"),
        Orders=("Invoice", "nunique")
    ).reset_index().sort_values("Revenue", ascending=False)
    
    top_countries = country_agg.head(10)
    fig2 = px.bar(
        top_countries,
        x="Country",
        y="Revenue",
        color="Customers",
        color_continuous_scale="Viridis",
        title="Top 10 Global Markets by Net Sales (£)",
        text="Revenue",
        labels={"Revenue": "Total Revenue (£)", "Customers": "Active Customers"},
        template="plotly_white"
    )
    fig2.update_traces(texttemplate="£%{text:,.0f}", textposition="outside")
    fig2.update_layout(height=480, margin=dict(l=40, r=40, t=60, b=60))
    fig2.write_html(os.path.join(fig_dir, "top_countries.html"))

    # Chart 3: 2-Year Daily Sales Timeline with Moving Averages
    logger.info("Creating Chart 3: Sales Trend & Seasonality Timeline...")
    df_daily["Date"] = pd.to_datetime(df_daily["Date"])
    df_daily["sales_ma7"] = df_daily["total_sales"].rolling(7, min_periods=1).mean()
    df_daily["sales_ma30"] = df_daily["total_sales"].rolling(30, min_periods=1).mean()

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=df_daily["Date"], y=df_daily["total_sales"],
        mode="lines", name="Daily Net Sales",
        line=dict(color="#93C5FD", width=1),
        opacity=0.6
    ))
    fig3.add_trace(go.Scatter(
        x=df_daily["Date"], y=df_daily["sales_ma7"],
        mode="lines", name="7-Day Moving Avg",
        line=dict(color="#2563EB", width=2)
    ))
    fig3.add_trace(go.Scatter(
        x=df_daily["Date"], y=df_daily["sales_ma30"],
        mode="lines", name="30-Day Trend (Monthly MA)",
        line=dict(color="#DC2626", width=2.5)
    ))
    fig3.update_layout(
        title="2-Year Daily Retail Sales Trajectory (Dec 2009 – Dec 2011)",
        xaxis_title="Date",
        yaxis_title="Net Sales (£)",
        template="plotly_white",
        hovermode="x unified",
        height=500
    )
    fig3.write_html(os.path.join(fig_dir, "sales_trend.html"))

    # Chart 4: Day-of-Week and Monthly Seasonality
    logger.info("Creating Chart 4: Seasonality Analysis...")
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dow_agg = df_daily.groupby("day_name")["total_sales"].mean().reindex(dow_order).reset_index()

    month_order = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]
    month_agg = df_daily.groupby("month_name")["total_sales"].mean().reindex(month_order).reset_index()

    fig4 = make_subplots(rows=1, cols=2, subplot_titles=("Average Daily Sales by Day of Week", "Average Daily Sales by Month"))
    fig4.add_trace(
        go.Bar(x=dow_agg["day_name"], y=dow_agg["total_sales"], marker_color="#3B82F6", name="Day of Week"),
        row=1, col=1
    )
    fig4.add_trace(
        go.Bar(x=month_agg["month_name"], y=month_agg["total_sales"], marker_color="#10B981", name="Month"),
        row=1, col=2
    )
    fig4.update_layout(template="plotly_white", height=450, showlegend=False)
    fig4.update_xaxes(tickangle=-30, row=1, col=2)
    fig4.write_html(os.path.join(fig_dir, "seasonality.html"))

    # Chart 5: Customer RFM Distributions (Log Transformed)
    logger.info("Creating Chart 5: RFM Distributions...")
    fig5 = make_subplots(rows=1, cols=3, subplot_titles=("Recency (Days)", "Frequency (Orders, Log10)", "Monetary (£ Spend, Log10)"))
    fig5.add_trace(go.Histogram(x=df_rfm["Recency"], nbinsx=35, marker_color="#6366F1", name="Recency"), row=1, col=1)
    fig5.add_trace(go.Histogram(x=np.log10(df_rfm["Frequency"]), nbinsx=30, marker_color="#8B5CF6", name="Log10(Freq)"), row=1, col=2)
    fig5.add_trace(go.Histogram(x=np.log10(df_rfm["Monetary"].clip(lower=1)), nbinsx=35, marker_color="#EC4899", name="Log10(Monetary)"), row=1, col=3)
    fig5.update_layout(template="plotly_white", height=420, showlegend=False, title="Customer RFM Behavioral Distributions")
    fig5.write_html(os.path.join(fig_dir, "rfm_distributions.html"))

    # Chart 6: Customer Tenure vs Monetary Spend Scatter
    logger.info("Creating Chart 6: Tenure vs Monetary Scatter...")
    fig6 = px.scatter(
        df_rfm,
        x="TenureDays",
        y="Monetary",
        size="Frequency",
        color="Recency",
        hover_data=["CustomerID", "AvgOrderValue", "Country"],
        color_continuous_scale="Plasma_r",
        log_y=True,
        title="Customer Lifetime Value vs Tenure (Bubble Size = Order Frequency, Color = Recency)",
        labels={"TenureDays": "Customer Tenure (Days)", "Monetary": "Total Spend (£, Log Scale)", "Recency": "Recency (Days)"},
        template="plotly_white"
    )
    fig6.update_layout(height=520)
    fig6.write_html(os.path.join(fig_dir, "tenure_vs_monetary.html"))

    logger.info(f"All 6 interactive Plotly figures saved successfully to: {fig_dir}")

if __name__ == "__main__":
    generate_eda_visualizations()
