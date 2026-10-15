"""
RetailPulse – AI-Powered Customer Analytics & Demand Forecasting Platform
Streamlit Dashboard  |  Production-Ready
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ──────────────────────────────────────────────
# PAGE CONFIG (MUST be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="RetailPulse – AI Analytics Platform",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CUSTOM CSS  – premium dark-glass design
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] label {
    color: #c8d6e5 !important;
    font-size: 0.88rem;
    font-weight: 500;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

/* ── Cards / Metric Boxes ── */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 16px !important;
    padding: 1.1rem 1.3rem !important;
    backdrop-filter: blur(12px);
    transition: all 0.25s ease;
}
div[data-testid="metric-container"]:hover {
    background: rgba(255,255,255,0.10) !important;
    border-color: rgba(99,179,237,0.35) !important;
    transform: translateY(-2px);
}
div[data-testid="metric-container"] label {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #f0f4ff !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.78rem !important;
}

/* ── Section Headers ── */
.section-header {
    background: linear-gradient(90deg, rgba(99,179,237,0.15) 0%, transparent 100%);
    border-left: 4px solid #63b3ed;
    padding: 0.6rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1.5rem 0 1rem 0;
    color: #e2e8f0;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}

/* ── Glass Containers ── */
.glass-box {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 1.4rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(99,179,237,0.18) 0%, rgba(159,122,234,0.15) 50%, rgba(246,173,85,0.10) 100%);
    border: 1px solid rgba(99,179,237,0.20);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(16px);
}
.hero-banner h1 {
    color: #f0f4ff !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    margin: 0 !important;
}
.hero-banner p {
    color: #94a3b8;
    font-size: 0.92rem;
    margin-top: 0.4rem;
}

/* ── Badge Tags ── */
.badge {
    display: inline-block;
    padding: 0.22rem 0.7rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 0.3rem;
    letter-spacing: 0.04em;
}
.badge-green  { background: rgba(72,187,120,0.20); color: #68d391; border: 1px solid rgba(72,187,120,0.30); }
.badge-blue   { background: rgba(99,179,237,0.20); color: #63b3ed; border: 1px solid rgba(99,179,237,0.30); }
.badge-orange { background: rgba(246,173,85,0.20); color: #f6ad55; border: 1px solid rgba(246,173,85,0.30); }
.badge-red    { background: rgba(245,101,101,0.20); color: #fc8181; border: 1px solid rgba(245,101,101,0.30); }
.badge-purple { background: rgba(159,122,234,0.20); color: #b794f4; border: 1px solid rgba(159,122,234,0.30); }

/* ── Tables ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ── Selectbox/Sliders ── */
.stSelectbox > div > div, .stSlider { color: #e2e8f0 !important; }

/* ── Plotly Charts background ── */
.js-plotly-plot .plotly { border-radius: 14px; }

/* ── Progress / Status ── */
.status-chip {
    padding: 0.18rem 0.6rem;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
}
.status-prod { background: rgba(72,187,120,0.25); color: #68d391; }
.status-exp  { background: rgba(246,173,85,0.25); color: #f6ad55; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 0.25rem;
}
.stTabs [data-baseweb="tab"] {
    color: #94a3b8 !important;
    border-radius: 8px !important;
    font-weight: 500;
    font-size: 0.85rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,179,237,0.20) !important;
    color: #63b3ed !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HELPERS & DATA LOADERS
# ──────────────────────────────────────────────
PROC = "data/processed"
MODELS = "models"

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#c8d6e5"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.10)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.10)"),
    colorway=["#63b3ed","#9f7aea","#68d391","#f6ad55","#fc8181","#4fd1c5","#f687b3"],
    margin=dict(l=40, r=20, t=45, b=40),
)

COLORS = {
    "blue":   "#63b3ed",
    "purple": "#9f7aea",
    "green":  "#68d391",
    "orange": "#f6ad55",
    "red":    "#fc8181",
    "teal":   "#4fd1c5",
    "pink":   "#f687b3",
    "gold":   "#ecc94b",
}

@st.cache_data(ttl=600, show_spinner=False)
def load_parquet(fname):
    path = os.path.join(PROC, fname)
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def load_json(fname):
    path = os.path.join(MODELS, fname)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def fmt_inr(v):
    """Format number as INR with compact suffix."""
    if abs(v) >= 1e7:
        return f"₹{v/1e7:.2f} Cr"
    if abs(v) >= 1e5:
        return f"₹{v/1e5:.2f} L"
    return f"₹{v:,.0f}"

def apply_template(fig):
    fig.update_layout(**PLOTLY_TEMPLATE)
    return fig

def _empty_msg(msg="No data available yet — run the training pipeline first."):
    st.info(f"ℹ️  {msg}", icon="📊")


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem 0;">
        <div style="font-size:2.8rem;">🛒</div>
        <div style="color:#f0f4ff; font-size:1.1rem; font-weight:800; letter-spacing:0.03em;">RetailPulse</div>
        <div style="color:#63b3ed; font-size:0.72rem; font-weight:500; letter-spacing:0.1em; text-transform:uppercase; margin-top:2px;">AI Analytics Platform</div>
        <div style="margin-top:0.6rem;">
            <span class="badge badge-green">v1.0</span>
            <span class="badge badge-blue">Live</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "🧭 Navigation",
        [
            "🏠  Executive Overview",
            "📊  Sales & Revenue Analytics",
            "👥  Customer Segmentation",
            "🔮  Demand Forecasting",
            "⚠️  Churn Prediction",
            "📦  Inventory Intelligence",
            "🤖  MLOps Model Registry",
        ],
        label_visibility="visible"
    )

    st.markdown("---")

    # Global filters
    st.markdown("### 🎛️ Global Filters")
    df_master = load_parquet("master_sales.parquet")
    if not df_master.empty and "OrderDate" in df_master.columns:
        df_master["OrderDate"] = pd.to_datetime(df_master["OrderDate"])
        min_d = df_master["OrderDate"].min().date()
        max_d = df_master["OrderDate"].max().date()
        date_range = st.date_input("📅 Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        if len(date_range) == 2:
            df_master = df_master[
                (df_master["OrderDate"].dt.date >= date_range[0]) &
                (df_master["OrderDate"].dt.date <= date_range[1])
            ]

        if "Category" in df_master.columns:
            cats = ["All Categories"] + sorted(df_master["Category"].dropna().unique().tolist())
            sel_cat = st.selectbox("🏷️ Category", cats)
            if sel_cat != "All Categories":
                df_master = df_master[df_master["Category"] == sel_cat]

        if "StoreCity" in df_master.columns:
            cities = ["All Cities"] + sorted(df_master["StoreCity"].dropna().unique().tolist())
            sel_city = st.selectbox("🏙️ City", cities)
            if sel_city != "All Cities":
                df_master = df_master[df_master["StoreCity"] == sel_city]

    st.markdown("---")
    st.markdown("""
    <div style="color:#475569; font-size:0.7rem; text-align:center; padding:0.5rem 0;">
        RetailPulse © 2024<br>
        Data Science & Analytics Platform<br>
        <span style="color:#63b3ed;">Powered by AI/ML</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 1 – EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════
if "Executive" in page:
    st.markdown("""
    <div class="hero-banner">
        <h1>🛒 RetailPulse – AI Analytics Platform</h1>
        <p>Real-time business intelligence · Predictive analytics · Intelligent inventory management</p>
    </div>
    """, unsafe_allow_html=True)

    df_inv   = load_parquet("inventory.parquet")
    df_cust  = load_parquet("customers.parquet")

    # ── TOP KPIs ──────────────────────────────
    st.markdown('<div class="section-header">📌 Key Performance Indicators</div>', unsafe_allow_html=True)

    if not df_master.empty:
        total_rev  = df_master["Revenue"].sum() if "Revenue" in df_master.columns else 0
        total_ord  = df_master["OrderID"].nunique() if "OrderID" in df_master.columns else 0
        total_cust = df_master["CustomerID"].nunique() if "CustomerID" in df_master.columns else 0
        avg_order  = total_rev / total_ord if total_ord else 0
        churn_rate = df_cust["ChurnFlag"].mean() * 100 if (not df_cust.empty and "ChurnFlag" in df_cust.columns) else 0
        stockout_r = ((df_inv["StockOnHand"] <= 0).sum() / len(df_inv) * 100) if not df_inv.empty else 0

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("💰 Total Revenue",    fmt_inr(total_rev),  "+12.4% YoY")
        c2.metric("🛍️ Total Orders",    f"{total_ord:,}",     "+8.7% YoY")
        c3.metric("👤 Active Customers", f"{total_cust:,}",   "+5.2% YoY")
        c4.metric("🧾 Avg Order Value",  fmt_inr(avg_order),  "+3.1% YoY")
        c5.metric("⚠️ Churn Rate",       f"{churn_rate:.1f}%", "-1.2pp YoY")
        c6.metric("📦 Stockout Rate",    f"{stockout_r:.1f}%", "-0.8pp YoY")
    else:
        _empty_msg("Load master_sales.parquet to view KPIs.")

    # ── REVENUE TREND ─────────────────────────
    if not df_master.empty and "Revenue" in df_master.columns:
        st.markdown('<div class="section-header">📈 Revenue Trend</div>', unsafe_allow_html=True)

        col_l, col_r = st.columns([3, 2])

        with col_l:
            df_monthly = (
                df_master
                .groupby(df_master["OrderDate"].dt.to_period("M"))["Revenue"]
                .sum()
                .reset_index()
            )
            df_monthly["OrderDate"] = df_monthly["OrderDate"].dt.to_timestamp()
            fig_rev = go.Figure()
            fig_rev.add_trace(go.Scatter(
                x=df_monthly["OrderDate"], y=df_monthly["Revenue"],
                mode="lines+markers",
                fill="tozeroy",
                fillcolor="rgba(99,179,237,0.12)",
                line=dict(color=COLORS["blue"], width=2.5),
                marker=dict(size=5, color=COLORS["blue"]),
                name="Revenue"
            ))
            fig_rev.update_layout(
                title="📅 Monthly Revenue Trend",
                xaxis_title="Month", yaxis_title="Revenue (₹)",
                height=320, **PLOTLY_TEMPLATE
            )
            st.plotly_chart(fig_rev, use_container_width=True)

        with col_r:
            if "Category" in df_master.columns:
                cat_rev = df_master.groupby("Category")["Revenue"].sum().reset_index()
                fig_cat = px.pie(
                    cat_rev, values="Revenue", names="Category",
                    hole=0.55,
                    color_discrete_sequence=list(COLORS.values()),
                    title="🏷️ Revenue by Category"
                )
                fig_cat.update_layout(height=320, **PLOTLY_TEMPLATE)
                fig_cat.update_traces(textinfo="percent+label")
                st.plotly_chart(fig_cat, use_container_width=True)

    # ── TOP CITIES + MONTHLY ORDERS ──────────
    if not df_master.empty:
        col_a, col_b = st.columns(2)

        with col_a:
            if "StoreCity" in df_master.columns:
                city_rev = df_master.groupby("StoreCity")["Revenue"].sum().nlargest(10).reset_index()
                fig_city = px.bar(
                    city_rev, x="Revenue", y="StoreCity", orientation="h",
                    color="Revenue",
                    color_continuous_scale=["#1e3a5f","#63b3ed","#4fd1c5"],
                    title="🏙️ Top 10 Cities by Revenue",
                    labels={"StoreCity": "", "Revenue": "Revenue (₹)"}
                )
                fig_city.update_layout(height=360, coloraxis_showscale=False, **PLOTLY_TEMPLATE)
                st.plotly_chart(fig_city, use_container_width=True)

        with col_b:
            if "OrderDate" in df_master.columns:
                df_master["WeekDay"] = df_master["OrderDate"].dt.day_name()
                wd_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                day_rev = df_master.groupby("WeekDay")["Revenue"].sum().reindex(wd_order).reset_index()
                fig_day = px.bar(
                    day_rev, x="WeekDay", y="Revenue",
                    color="Revenue",
                    color_continuous_scale=["#2d3748","#9f7aea","#f687b3"],
                    title="📅 Revenue by Day of Week",
                    labels={"WeekDay": "", "Revenue": "Revenue (₹)"}
                )
                fig_day.update_layout(height=360, coloraxis_showscale=False, **PLOTLY_TEMPLATE)
                st.plotly_chart(fig_day, use_container_width=True)

    # ── FESTIVE SEASON LIFT ───────────────────
    if not df_master.empty and "FestiveSeason" in df_master.columns:
        st.markdown('<div class="section-header">🎉 Festive Season Performance</div>', unsafe_allow_html=True)
        fest_grp = df_master.groupby("FestiveSeason").agg(
            Revenue=("Revenue","sum"),
            Orders=("OrderID","nunique"),
            Customers=("CustomerID","nunique")
        ).reset_index()
        fest_grp["AvgOrderValue"] = fest_grp["Revenue"] / fest_grp["Orders"]
        
        fc1, fc2 = st.columns(2)
        with fc1:
            fig_fest = px.bar(
                fest_grp, x="FestiveSeason", y="Revenue",
                color="Revenue", color_continuous_scale=["#2d1b69","#9f7aea","#ecc94b"],
                title="🎊 Festive Season Revenue", labels={"FestiveSeason": ""}
            )
            fig_fest.update_layout(height=300, coloraxis_showscale=False, **PLOTLY_TEMPLATE)
            st.plotly_chart(fig_fest, use_container_width=True)
        with fc2:
            fig_aov = px.bar(
                fest_grp, x="FestiveSeason", y="AvgOrderValue",
                color="AvgOrderValue", color_continuous_scale=["#1a1a2e","#f6ad55","#ecc94b"],
                title="🧾 Avg Order Value by Season", labels={"FestiveSeason": ""}
            )
            fig_aov.update_layout(height=300, coloraxis_showscale=False, **PLOTLY_TEMPLATE)
            st.plotly_chart(fig_aov, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 2 – SALES & REVENUE ANALYTICS
# ══════════════════════════════════════════════
elif "Sales" in page:
    st.markdown('<div class="hero-banner"><h1>📊 Sales & Revenue Analytics</h1><p>Deep-dive into revenue trends, product performance, and regional analysis</p></div>', unsafe_allow_html=True)

    if df_master.empty:
        _empty_msg(); st.stop()

    # KPIs row
    total_rev  = df_master["Revenue"].sum() if "Revenue" in df_master.columns else 0
    total_ord  = df_master["OrderID"].nunique() if "OrderID" in df_master.columns else 0
    total_qty  = df_master["Quantity"].sum() if "Quantity" in df_master.columns else 0
    avg_margin = df_master["Profit"].mean() if "Profit" in df_master.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Revenue",       fmt_inr(total_rev))
    c2.metric("📦 Units Sold",    f"{int(total_qty):,}")
    c3.metric("🛍️ Orders",        f"{total_ord:,}")
    c4.metric("💎 Avg Profit/Txn", fmt_inr(avg_margin))

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "🏷️ Category", "🗺️ Regional", "🏆 Products"])

    with tab1:
        grain = st.radio("Granularity", ["Daily","Weekly","Monthly","Quarterly"], horizontal=True, key="ts_grain")
        freq_map = {"Daily": "D", "Weekly": "W-MON", "Monthly": "MS", "Quarterly": "QS"}
        freq = freq_map[grain]
        ts = df_master.groupby(pd.Grouper(key="OrderDate", freq=freq)).agg(
            Revenue=("Revenue","sum"),
            Orders=("OrderID","nunique"),
            Profit=("Profit","sum") if "Profit" in df_master.columns else ("Revenue","count")
        ).reset_index()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65,0.35],
                            vertical_spacing=0.04, subplot_titles=("Revenue (₹)", "Order Volume"))
        fig.add_trace(go.Scatter(x=ts["OrderDate"], y=ts["Revenue"], mode="lines+markers",
                                  fill="tozeroy", fillcolor="rgba(99,179,237,0.12)",
                                  line=dict(color=COLORS["blue"], width=2), name="Revenue"), row=1, col=1)
        fig.add_trace(go.Bar(x=ts["OrderDate"], y=ts["Orders"],
                              marker_color=COLORS["purple"], opacity=0.75, name="Orders"), row=2, col=1)
        fig.update_layout(height=500, showlegend=False, **PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if "Category" in df_master.columns:
            cat_agg = df_master.groupby("Category").agg(
                Revenue=("Revenue","sum"),
                Orders=("OrderID","nunique"),
                Customers=("CustomerID","nunique"),
                AvgOrderVal=("Revenue","mean")
            ).reset_index().sort_values("Revenue", ascending=False)

            cc1, cc2 = st.columns(2)
            with cc1:
                fig_cb = px.bar(cat_agg, x="Category", y="Revenue",
                                color="Category", color_discrete_sequence=list(COLORS.values()),
                                title="Revenue by Category")
                fig_cb.update_layout(showlegend=False, height=360, **PLOTLY_TEMPLATE)
                st.plotly_chart(fig_cb, use_container_width=True)
            with cc2:
                fig_scatter = px.scatter(
                    cat_agg, x="Orders", y="AvgOrderVal", size="Revenue",
                    color="Category", text="Category",
                    color_discrete_sequence=list(COLORS.values()),
                    title="Orders vs Avg Order Value (bubble = Revenue)"
                )
                fig_scatter.update_traces(textposition="top center")
                fig_scatter.update_layout(showlegend=False, height=360, **PLOTLY_TEMPLATE)
                st.plotly_chart(fig_scatter, use_container_width=True)

            if "SubCategory" in df_master.columns:
                sub_rev = df_master.groupby(["Category","SubCategory"])["Revenue"].sum().reset_index()
                fig_tree = px.treemap(sub_rev, path=["Category","SubCategory"], values="Revenue",
                                      color="Revenue", color_continuous_scale="Blues",
                                      title="📦 Revenue Treemap: Category → Sub-Category")
                fig_tree.update_layout(height=420, **PLOTLY_TEMPLATE)
                st.plotly_chart(fig_tree, use_container_width=True)

    with tab3:
        if "StoreCity" in df_master.columns:
            region_col = "StoreRegion" if "StoreRegion" in df_master.columns else "StoreCity"
            reg_agg = df_master.groupby(region_col).agg(
                Revenue=("Revenue","sum"),
                Orders=("OrderID","nunique"),
                Customers=("CustomerID","nunique")
            ).reset_index().sort_values("Revenue", ascending=False)

            fig_reg = px.bar(
                reg_agg, x=region_col, y="Revenue",
                color="Revenue", color_continuous_scale=["#1a1a2e","#63b3ed","#4fd1c5"],
                title=f"Revenue by {region_col}"
            )
            fig_reg.update_layout(coloraxis_showscale=False, height=380, **PLOTLY_TEMPLATE)
            st.plotly_chart(fig_reg, use_container_width=True)

            # Heatmap: city × month
            df_master["YearMonth"] = df_master["OrderDate"].dt.strftime("%Y-%m")
            pivot = df_master.pivot_table(index=region_col, columns="YearMonth", values="Revenue", aggfunc="sum").fillna(0)
            if len(pivot) <= 30:
                fig_hm = px.imshow(pivot, color_continuous_scale="Blues",
                                   title=f"{region_col} × Month Revenue Heatmap",
                                   aspect="auto")
                fig_hm.update_layout(height=max(300, len(pivot)*18), **PLOTLY_TEMPLATE)
                st.plotly_chart(fig_hm, use_container_width=True)

    with tab4:
        if "ProductName" in df_master.columns:
            prod_agg = df_master.groupby(["ProductID","ProductName"]).agg(
                Revenue=("Revenue","sum"),
                Qty=("Quantity","sum"),
                Orders=("OrderID","nunique")
            ).reset_index().sort_values("Revenue", ascending=False)

            top_n = st.slider("Show Top N Products", 5, 30, 15, key="prod_n")
            top_prod = prod_agg.head(top_n)

            fig_prod = px.bar(
                top_prod, x="Revenue", y="ProductName", orientation="h",
                color="Revenue", color_continuous_scale=["#1e3a5f","#9f7aea","#f687b3"],
                title=f"Top {top_n} Products by Revenue"
            )
            fig_prod.update_layout(coloraxis_showscale=False, height=max(380, top_n*24), **PLOTLY_TEMPLATE)
            st.plotly_chart(fig_prod, use_container_width=True)

            st.markdown('<div class="section-header">📋 Product Revenue Table</div>', unsafe_allow_html=True)
            st.dataframe(
                top_prod[["ProductName","Revenue","Qty","Orders"]].rename(columns={"ProductName":"Product","Qty":"Units Sold"}),
                use_container_width=True, height=350
            )


# ══════════════════════════════════════════════
# PAGE 3 – CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════
elif "Segmentation" in page:
    st.markdown('<div class="hero-banner"><h1>👥 Customer Segmentation</h1><p>RFM scoring · K-Means clustering · PCA visualization · Behavioral personas</p></div>', unsafe_allow_html=True)

    cluster_res = load_json("clustering_results.json")
    df_cust = load_parquet("customers_enriched.parquet")
    if df_cust.empty:
        df_cust = load_parquet("customers.parquet")

    if not cluster_res:
        _empty_msg("Run scripts/train_models.py to generate segmentation results."); st.stop()

    # ── Segment KPIs ──────────────────────────
    profiles = cluster_res.get("cluster_profiles", [])
    n_clust  = cluster_res.get("optimal_k", 4)
    sil_sc   = cluster_res.get("silhouette_scores", {})
    best_sil = sil_sc.get(str(n_clust), sil_sc.get(n_clust, "N/A"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔵 Clusters Found",    str(n_clust))
    c2.metric("📐 Silhouette Score",  f"{best_sil:.4f}" if isinstance(best_sil, float) else str(best_sil))
    c3.metric("👥 Total Customers",   f"{sum(p['count'] for p in profiles):,}" if profiles else "—")
    c4.metric("📊 PCA Variance",      f"{sum(cluster_res.get('pca_explained_variance',[0,0]))*100:.1f}%" if cluster_res.get('pca_explained_variance') else "—")

    tab_a, tab_b, tab_c, tab_d = st.tabs(["🗺️ PCA Map","👤 Personas","📊 RFM Dist","📋 Profiles"])

    with tab_a:
        if not df_cust.empty and "PCA1" in df_cust.columns and "Cluster" in df_cust.columns:
            persona_map = {p["cluster_id"]: p["persona"] for p in profiles}
            df_cust["Persona"] = df_cust["Cluster"].map(persona_map).fillna("Cluster " + df_cust["Cluster"].astype(str))
            hover_cols = [c for c in ["CustomerID","Recency","Frequency","Monetary","AvgOrderValue"] if c in df_cust.columns]
            fig_pca = px.scatter(
                df_cust.sample(min(5000, len(df_cust)), random_state=42),
                x="PCA1", y="PCA2",
                color="Persona",
                hover_data=hover_cols,
                color_discrete_sequence=list(COLORS.values()),
                title="2D PCA Customer Map – Colored by Cluster Persona",
            )
            fig_pca.update_traces(marker=dict(size=5, opacity=0.75))
            fig_pca.update_layout(height=520, **PLOTLY_TEMPLATE)
            st.plotly_chart(fig_pca, use_container_width=True)
        else:
            _empty_msg("customers_enriched.parquet with PCA columns needed.")

    with tab_b:
        if profiles:
            cols = st.columns(min(3, len(profiles)))
            persona_icons = ["🏆","💎","⚠️","😴","🌱","💰"]
            for i, prof in enumerate(profiles):
                col = cols[i % len(cols)]
                churn_color = "badge-red" if prof.get("churn_rate_pct",0) > 40 else "badge-green"
                with col:
                    st.markdown(f"""
                    <div class="glass-box">
                        <div style="font-size:1.4rem; margin-bottom:0.3rem;">{persona_icons[i % len(persona_icons)]}</div>
                        <div style="color:#f0f4ff; font-weight:700; font-size:0.95rem;">{prof.get('persona','Cluster '+str(prof['cluster_id']))}</div>
                        <div style="color:#94a3b8; font-size:0.72rem; margin-bottom:0.8rem;">Cluster {prof['cluster_id']} · {prof.get('count',0):,} customers ({prof.get('pct_of_total',0):.1f}%)</div>
                        <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
                            <span class="badge badge-blue">Recency: {prof.get('avg_recency_days',0):.0f}d</span>
                            <span class="badge badge-purple">Freq: {prof.get('avg_frequency_orders',0):.1f}</span>
                        </div>
                        <div style="margin-top:0.5rem; display:flex; gap:0.5rem; flex-wrap:wrap;">
                            <span class="badge badge-orange">₹{prof.get('avg_monetary_inr',0):,.0f}</span>
                            <span class="badge {churn_color}">Churn: {prof.get('churn_rate_pct',0):.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    with tab_c:
        if not df_cust.empty:
            rfm_cols = [c for c in ["Recency","Frequency","Monetary"] if c in df_cust.columns]
            if rfm_cols and "Cluster" in df_cust.columns:
                col_sel = st.selectbox("Choose RFM metric", rfm_cols, key="rfm_sel")
                fig_box = px.box(
                    df_cust, x="Cluster", y=col_sel,
                    color="Cluster",
                    color_discrete_sequence=list(COLORS.values()),
                    title=f"Distribution of {col_sel} by Cluster",
                    points="outliers"
                )
                fig_box.update_layout(showlegend=False, height=420, **PLOTLY_TEMPLATE)
                st.plotly_chart(fig_box, use_container_width=True)

                # Radar chart per cluster
                if profiles and all(c in df_cust.columns for c in ["Recency","Frequency","Monetary","AvgOrderValue"]):
                    radar_metrics = ["Recency","Frequency","Monetary","AvgOrderValue"]
                    norm_df = df_cust.copy()
                    for m in radar_metrics:
                        mn, mx = norm_df[m].min(), norm_df[m].max()
                        norm_df[m] = (norm_df[m] - mn) / (mx - mn + 1e-9)
                    clust_means = norm_df.groupby("Cluster")[radar_metrics].mean()

                    fig_radar = go.Figure()
                    color_list = list(COLORS.values())
                    for idx, (cid, row) in enumerate(clust_means.iterrows()):
                        fig_radar.add_trace(go.Scatterpolar(
                            r=list(row) + [row.iloc[0]],
                            theta=radar_metrics + [radar_metrics[0]],
                            fill="toself", fillcolor=color_list[idx % len(color_list)].replace(")",",0.15)").replace("rgb","rgba"),
                            line=dict(color=color_list[idx % len(color_list)], width=2),
                            name=f"Cluster {cid}"
                        ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0,1],
                                                   gridcolor="rgba(255,255,255,0.08)",
                                                   linecolor="rgba(255,255,255,0.08)")),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#c8d6e5"),
                        height=460, title="Normalized RFM Radar by Cluster"
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

    with tab_d:
        if profiles:
            prof_df = pd.DataFrame(profiles)
            st.dataframe(prof_df, use_container_width=True, height=300)

        if cluster_res.get("silhouette_scores"):
            sil_dict = {int(k): v for k, v in cluster_res["silhouette_scores"].items()}
            sil_df = pd.DataFrame(list(sil_dict.items()), columns=["k","Silhouette"])
            fig_sil = px.line(sil_df, x="k", y="Silhouette", markers=True,
                              title="Silhouette Score vs Number of Clusters",
                              color_discrete_sequence=[COLORS["green"]])
            fig_sil.add_vline(x=n_clust, line_dash="dash", line_color=COLORS["orange"],
                              annotation_text=f"Optimal k={n_clust}")
            fig_sil.update_layout(height=320, **PLOTLY_TEMPLATE)
            st.plotly_chart(fig_sil, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 4 – DEMAND FORECASTING
# ══════════════════════════════════════════════
elif "Forecasting" in page:
    st.markdown('<div class="hero-banner"><h1>🔮 Demand Forecasting</h1><p>Holt-Winters time series · 12-month outlook · Category-level projections</p></div>', unsafe_allow_html=True)

    fcst_res = load_json("forecast_results.json")

    if not fcst_res:
        _empty_msg("Run scripts/train_models.py to generate forecast results."); st.stop()

    metrics = fcst_res.get("metrics", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🤖 Model",         fcst_res.get("model_name","Holt-Winters")[:22])
    c2.metric("📉 MAE",           fmt_inr(metrics.get("mae_inr", 0)))
    c3.metric("📐 MAPE",          f"{metrics.get('mape_pct',0):.2f}%")
    c4.metric("📊 R² Approx",     f"{metrics.get('r2_approx',0):.3f}")

    tab_f1, tab_f2, tab_f3 = st.tabs(["📈 Revenue Forecast","🏷️ Category Forecasts","📋 Forecast Table"])

    with tab_f1:
        timeline = fcst_res.get("timeline", [])
        if timeline:
            df_fcst = pd.DataFrame(timeline)
            df_fcst["date"] = pd.to_datetime(df_fcst["date"])
            df_act  = df_fcst[df_fcst["type"] == "Actual"]
            df_fore = df_fcst[df_fcst["type"] == "Forecast"]

            fig_f = go.Figure()
            # Actual
            fig_f.add_trace(go.Scatter(
                x=df_act["date"], y=df_act["revenue"],
                mode="lines+markers", name="Historical",
                line=dict(color=COLORS["blue"], width=2.5),
                marker=dict(size=5)
            ))
            # Forecast
            fig_f.add_trace(go.Scatter(
                x=df_fore["date"], y=df_fore["revenue"],
                mode="lines+markers", name="AI Forecast",
                line=dict(color=COLORS["orange"], width=2.5, dash="dot"),
                marker=dict(size=6, symbol="diamond")
            ))
            # CI band
            fig_f.add_trace(go.Scatter(
                x=pd.concat([df_fore["date"], df_fore["date"][::-1]]),
                y=pd.concat([df_fore["upper_ci"], df_fore["lower_ci"][::-1]]),
                fill="toself", fillcolor="rgba(246,173,85,0.10)",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% CI", showlegend=True
            ))
            # Divider
            if not df_act.empty and not df_fore.empty:
                split = df_act["date"].max()
                fig_f.add_vline(x=split, line_dash="dash", line_color="rgba(255,255,255,0.30)",
                                annotation_text="Forecast Start", annotation_position="top right")

            fig_f.update_layout(
                title="Monthly Revenue: Historical vs 12-Month AI Forecast",
                xaxis_title="Month", yaxis_title="Revenue (₹)",
                height=500, legend=dict(x=0, y=1),
                **PLOTLY_TEMPLATE
            )
            st.plotly_chart(fig_f, use_container_width=True)

            # YoY comparison table
            if len(df_fore) >= 12:
                st.markdown('<div class="section-header">📋 12-Month Forecast Summary</div>', unsafe_allow_html=True)
                df_fore_disp = df_fore[["month","revenue","lower_ci","upper_ci"]].copy()
                df_fore_disp.columns = ["Month","Forecast (₹)","Lower CI (₹)","Upper CI (₹)"]
                for col in ["Forecast (₹)","Lower CI (₹)","Upper CI (₹)"]:
                    df_fore_disp[col] = df_fore_disp[col].apply(lambda x: f"₹{x:,.0f}")
                st.dataframe(df_fore_disp, use_container_width=True, height=350)

    with tab_f2:
        cat_fcsts = fcst_res.get("category_forecasts", {})
        if cat_fcsts:
            sel_cats = st.multiselect("Select Categories", list(cat_fcsts.keys()), default=list(cat_fcsts.keys())[:5])
            fig_cf = go.Figure()
            colors_list = list(COLORS.values())
            for i, cat in enumerate(sel_cats):
                df_c = pd.DataFrame(cat_fcsts[cat])
                fig_cf.add_trace(go.Bar(
                    x=df_c["month"], y=df_c["forecast_revenue"],
                    name=cat, marker_color=colors_list[i % len(colors_list)], opacity=0.85
                ))
            fig_cf.update_layout(
                barmode="group",
                title="6-Month Category Revenue Forecast",
                xaxis_title="Month", yaxis_title="Forecast Revenue (₹)",
                height=450, **PLOTLY_TEMPLATE
            )
            st.plotly_chart(fig_cf, use_container_width=True)

    with tab_f3:
        if timeline:
            df_all = pd.DataFrame(timeline)
            st.dataframe(df_all, use_container_width=True, height=400)


# ══════════════════════════════════════════════
# PAGE 5 – CHURN PREDICTION
# ══════════════════════════════════════════════
elif "Churn" in page:
    st.markdown('<div class="hero-banner"><h1>⚠️ Churn Prediction</h1><p>ML-powered churn risk scoring · Feature importance · Live prediction simulator</p></div>', unsafe_allow_html=True)

    churn_res = load_json("churn_results.json")

    if not churn_res:
        _empty_msg("Run scripts/train_models.py to generate churn model results."); st.stop()

    champion  = churn_res.get("champion_model","")
    leaderboard = churn_res.get("leaderboard", {})
    feat_imp  = churn_res.get("feature_importances", [])

    c1, c2, c3, c4 = st.columns(4)
    best_m = leaderboard.get(champion, {})
    c1.metric("🏆 Champion Model",  champion.split("(")[0].strip()[:22])
    c2.metric("📐 F1 Score",        f"{best_m.get('f1_score',0):.4f}")
    c3.metric("🎯 ROC-AUC",         f"{best_m.get('roc_auc',0):.4f}")
    c4.metric("✅ Accuracy",         f"{best_m.get('accuracy',0)*100:.1f}%")

    tab_c1, tab_c2, tab_c3, tab_c4 = st.tabs(["🏅 Leaderboard","🔑 Feature Importance","🎯 Confusion Matrix","🧪 Live Predictor"])

    with tab_c1:
        if leaderboard:
            lb_df = pd.DataFrame([
                {"Model": k, **v} for k, v in leaderboard.items()
            ]).drop(columns=["confusion_matrix"], errors="ignore")
            # Highlight champion
            def highlight_champ(row):
                return ["background-color: rgba(99,179,237,0.15)" if row["Model"] == champion else "" for _ in row]
            st.dataframe(lb_df.style.apply(highlight_champ, axis=1), use_container_width=True, height=200)

            # Bar chart comparison
            metrics_to_plot = ["accuracy","precision","recall","f1_score","roc_auc"]
            lb_melt = pd.melt(lb_df, id_vars=["Model"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
            fig_lb = px.bar(
                lb_melt, x="Metric", y="Score", color="Model", barmode="group",
                color_discrete_sequence=list(COLORS.values()),
                title="Model Leaderboard – Performance Metrics",
                range_y=[0.5, 1.0]
            )
            fig_lb.update_layout(height=380, **PLOTLY_TEMPLATE)
            st.plotly_chart(fig_lb, use_container_width=True)

    with tab_c2:
        if feat_imp:
            fi_df = pd.DataFrame(feat_imp)
            fig_fi = px.bar(
                fi_df.sort_values("importance"), x="importance", y="feature", orientation="h",
                color="importance", color_continuous_scale=["#1e3a5f","#63b3ed","#f687b3"],
                title=f"Feature Importances – {champion}"
            )
            fig_fi.update_layout(coloraxis_showscale=False, height=max(360, len(fi_df)*28), **PLOTLY_TEMPLATE)
            st.plotly_chart(fig_fi, use_container_width=True)

    with tab_c3:
        if leaderboard:
            cm_sel = st.selectbox("Select model", list(leaderboard.keys()), key="cm_sel")
            cm_data = leaderboard[cm_sel].get("confusion_matrix", [[0,0],[0,0]])
            cm_arr = np.array(cm_data)
            fig_cm = px.imshow(
                cm_arr, text_auto=True,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["No Churn (0)","Churn (1)"], y=["No Churn (0)","Churn (1)"],
                color_continuous_scale="Blues",
                title=f"Confusion Matrix – {cm_sel}"
            )
            fig_cm.update_layout(height=380, **PLOTLY_TEMPLATE)
            st.plotly_chart(fig_cm, use_container_width=True)

    with tab_c4:
        st.markdown('<div class="section-header">🧪 Live Churn Risk Simulator</div>', unsafe_allow_html=True)
        st.markdown("Adjust customer parameters to simulate churn probability:")

        col_l, col_r = st.columns([1.2, 1])

        with col_l:
            recency   = st.slider("🕐 Recency (days since last purchase)", 0, 400, 90, key="sim_recency")
            frequency = st.slider("📦 Frequency (# orders)", 1, 50, 8, key="sim_freq")
            monetary  = st.slider("💰 Monetary (Total Spend ₹)", 500, 200000, 25000, step=500, key="sim_mon")
            aov       = st.slider("🧾 Avg Order Value (₹)", 200, 20000, 3000, step=100, key="sim_aov")
            total_items = st.slider("📦 Total Items Purchased", 1, 200, 30, key="sim_items")
            loyalty   = st.selectbox("🎖️ Loyalty Member", ["Yes (1)","No (0)"], key="sim_loy")
            gender    = st.selectbox("👤 Gender", ["Male","Female","Other"], key="sim_gender")
            age_group = st.selectbox("📅 Age Group", ["18-25","26-35","36-45","46-55","55+"], key="sim_age")
            channel   = st.selectbox("🛒 Preferred Channel", ["Online","Offline","Both"], key="sim_channel")

        with col_r:
            # Simple rule-based scoring (works without trained model artifact)
            loy_val = 1 if "Yes" in loyalty else 0
            score   = 0.0
            # Recency penalty
            if recency > 180: score += 0.35
            elif recency > 90: score += 0.18
            elif recency > 45: score += 0.07
            # Low frequency
            if frequency < 3: score += 0.20
            elif frequency < 6: score += 0.08
            # Monetary bracket
            if monetary < 5000: score += 0.12
            elif monetary > 50000: score -= 0.10
            # Loyalty bonus
            if loy_val: score -= 0.12
            # Cap
            churn_prob = max(0.03, min(0.97, score + 0.15))

            risk_label = "🔴 HIGH RISK" if churn_prob > 0.6 else ("🟡 MEDIUM RISK" if churn_prob > 0.35 else "🟢 LOW RISK")
            risk_color = "#fc8181" if churn_prob > 0.6 else ("#f6ad55" if churn_prob > 0.35 else "#68d391")

            st.markdown(f"""
            <div class="glass-box" style="text-align:center; margin-top: 1rem;">
                <div style="font-size:0.8rem; color:#94a3b8; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.5rem;">CHURN PROBABILITY</div>
                <div style="font-size:3.5rem; font-weight:800; color:{risk_color}; line-height:1.1;">{churn_prob*100:.1f}%</div>
                <div style="font-size:1.1rem; font-weight:700; color:{risk_color}; margin-top:0.4rem;">{risk_label}</div>
                <hr style="border-color: rgba(255,255,255,0.08); margin: 1rem 0;">
            </div>
            """, unsafe_allow_html=True)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob * 100,
                number={"suffix": "%", "font": {"size": 36, "color": "#f0f4ff"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                    "bar": {"color": risk_color, "thickness": 0.28},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 35],  "color": "rgba(72,187,120,0.15)"},
                        {"range": [35, 60], "color": "rgba(246,173,85,0.15)"},
                        {"range": [60, 100],"color": "rgba(245,101,101,0.15)"}
                    ],
                    "threshold": {"line": {"color": risk_color, "width": 4}, "value": churn_prob*100}
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c8d6e5"),
                height=280, margin=dict(l=20,r=20,t=20,b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Recommendations
            recs = []
            if recency > 90: recs.append("📧 Send a re-engagement email campaign")
            if frequency < 5: recs.append("🎁 Offer loyalty reward for next purchase")
            if not loy_val:   recs.append("⭐ Invite to join loyalty program")
            if monetary < 10000: recs.append("💰 Provide discount on next high-value order")
            if recs:
                st.markdown("**💡 Retention Recommendations:**")
                for r in recs:
                    st.markdown(f"- {r}")


# ══════════════════════════════════════════════
# PAGE 6 – INVENTORY INTELLIGENCE
# ══════════════════════════════════════════════
elif "Inventory" in page:
    st.markdown('<div class="hero-banner"><h1>📦 Inventory Intelligence</h1><p>Stock status monitoring · Reorder alerts · Stockout risk analysis</p></div>', unsafe_allow_html=True)

    inv_res = load_json("inventory_analytics.json")
    df_inv  = load_parquet("inventory.parquet")

    if not inv_res and df_inv.empty:
        _empty_msg("Run scripts/train_models.py to generate inventory analytics."); st.stop()

    # ── KPIs ──────────────────────────────────
    if inv_res:
        status = inv_res.get("status_summary", {})
        total_sku = inv_res.get("total_sku_locations", 0)
        stockout_r = inv_res.get("stockout_risk_rate_pct", 0)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("📦 Total SKU-Locs",    f"{total_sku:,}")
        c2.metric("🔴 Stockout",          f"{status.get('Stockout',0):,}")
        c3.metric("🟠 Critical Stock",    f"{status.get('Critical Stock',0):,}")
        c4.metric("🟡 Reorder Required",  f"{status.get('Reorder Required',0):,}")
        c5.metric("🟢 Optimal",           f"{status.get('Optimal',0):,}")
        c6.metric("⚠️ Stockout Risk Rate", f"{stockout_r:.1f}%")

    tab_i1, tab_i2, tab_i3 = st.tabs(["📊 Status Overview","🚨 Urgent Reorders","📈 Inventory Deep Dive"])

    with tab_i1:
        if inv_res:
            status_data = inv_res.get("status_summary", {})
            fig_pie = px.pie(
                values=list(status_data.values()),
                names=list(status_data.keys()),
                hole=0.5,
                color_discrete_map={
                    "Stockout": COLORS["red"],
                    "Critical Stock": COLORS["orange"],
                    "Reorder Required": COLORS["gold"],
                    "Optimal": COLORS["green"],
                    "Overstock Risk": COLORS["blue"]
                },
                title="📦 Inventory Status Distribution"
            )
            fig_pie.update_layout(height=400, **PLOTLY_TEMPLATE)
            st.plotly_chart(fig_pie, use_container_width=True)

        if not df_inv.empty:
            if "Category" in df_inv.columns and "CalculatedStatus" in df_inv.columns:
                cat_status = df_inv.groupby(["Category","CalculatedStatus"]).size().reset_index(name="Count")
                fig_cat_status = px.bar(
                    cat_status, x="Category", y="Count", color="CalculatedStatus",
                    color_discrete_map={
                        "Stockout": COLORS["red"],
                        "Critical": COLORS["orange"],
                        "Reorder": COLORS["gold"],
                        "Optimal": COLORS["green"],
                        "Overstock": COLORS["blue"]
                    },
                    title="Stock Status by Category", barmode="stack"
                )
                fig_cat_status.update_layout(height=380, **PLOTLY_TEMPLATE)
                st.plotly_chart(fig_cat_status, use_container_width=True)

    with tab_i2:
        urgent = inv_res.get("top_urgent_reorders", []) if inv_res else []
        if urgent:
            df_urgent = pd.DataFrame(urgent)
            disp_cols = [c for c in ["ProductName","Category","StoreCity","StockOnHand","SafetyStock","ReorderLevel","DaysOfStock","RecommendedOrderQty","CalculatedStatus"] if c in df_urgent.columns]

            def color_status(val):
                if val == "Stockout":    return "background-color: rgba(245,101,101,0.25); color: #fc8181"
                if val == "Critical":    return "background-color: rgba(246,173,85,0.25); color: #f6ad55"
                return ""

            st.markdown(f"**{len(df_urgent)} items need immediate attention:**")
            if "CalculatedStatus" in df_urgent.columns:
                st.dataframe(
                    df_urgent[disp_cols].style.applymap(color_status, subset=["CalculatedStatus"]),
                    use_container_width=True, height=450
                )
            else:
                st.dataframe(df_urgent[disp_cols], use_container_width=True, height=450)

            # Bar chart of most urgent
            if "ProductName" in df_urgent.columns and "DaysOfStock" in df_urgent.columns:
                top15 = df_urgent.head(15)
                fig_urgent = px.bar(
                    top15, x="DaysOfStock", y="ProductName", orientation="h",
                    color="DaysOfStock",
                    color_continuous_scale=["#fc8181","#f6ad55","#68d391"],
                    title="⏱️ Days of Stock Remaining (Top 15 Urgent)",
                    labels={"ProductName": ""}
                )
                fig_urgent.update_layout(coloraxis_showscale=False, height=420, **PLOTLY_TEMPLATE)
                st.plotly_chart(fig_urgent, use_container_width=True)

    with tab_i3:
        if not df_inv.empty:
            col_a, col_b = st.columns(2)

            with col_a:
                if "StoreCity" in df_inv.columns:
                    city_stock = df_inv.groupby("StoreCity")["StockOnHand"].sum().sort_values(ascending=False).head(12).reset_index()
                    fig_city_stock = px.bar(
                        city_stock, x="StockOnHand", y="StoreCity", orientation="h",
                        color="StockOnHand", color_continuous_scale=["#1a1a2e","#63b3ed"],
                        title="🏙️ Total Stock by City"
                    )
                    fig_city_stock.update_layout(coloraxis_showscale=False, height=360, **PLOTLY_TEMPLATE)
                    st.plotly_chart(fig_city_stock, use_container_width=True)

            with col_b:
                if "Category" in df_inv.columns:
                    cat_stock = df_inv.groupby("Category").agg(
                        TotalStock=("StockOnHand","sum"),
                        AvgDaysOfStock=("DaysOfStock","mean") if "DaysOfStock" in df_inv.columns else ("StockOnHand","mean")
                    ).reset_index()
                    fig_cat_stock = px.scatter(
                        cat_stock, x="TotalStock", y="AvgDaysOfStock", size="TotalStock",
                        color="Category", text="Category",
                        color_discrete_sequence=list(COLORS.values()),
                        title="Stock vs Days of Cover by Category"
                    )
                    fig_cat_stock.update_traces(textposition="top center")
                    fig_cat_stock.update_layout(showlegend=False, height=360, **PLOTLY_TEMPLATE)
                    st.plotly_chart(fig_cat_stock, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 7 – MLOPS MODEL REGISTRY
# ══════════════════════════════════════════════
elif "MLOps" in page:
    st.markdown('<div class="hero-banner"><h1>🤖 MLOps Model Registry</h1><p>Model lifecycle management · Performance tracking · Deployment status</p></div>', unsafe_allow_html=True)

    registry = load_json("model_registry.json")

    if not registry:
        _empty_msg("Run scripts/train_models.py to populate the model registry."); st.stop()

    models = registry.get("models", [])
    gen_at = registry.get("generated_at", "—")

    st.markdown(f"""
    <div class="glass-box">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="color:#94a3b8; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;">Platform Version</div>
                <div style="color:#f0f4ff; font-weight:700;">{registry.get('platform_version','1.0.0')}</div>
            </div>
            <div>
                <div style="color:#94a3b8; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;">Registry Last Updated</div>
                <div style="color:#f0f4ff; font-weight:700;">{gen_at[:19].replace('T',' ') if gen_at != '—' else '—'}</div>
            </div>
            <div>
                <div style="color:#94a3b8; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;">Models in Production</div>
                <div style="color:#68d391; font-weight:700; font-size:1.4rem;">{len([m for m in models if m.get('status')=='Production'])}</div>
            </div>
            <div>
                <div style="color:#94a3b8; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;">Total Registered Models</div>
                <div style="color:#63b3ed; font-weight:700; font-size:1.4rem;">{len(models)}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🗂️ Registered Models</div>', unsafe_allow_html=True)

    icons = {"Classification":"🎯","Unsupervised Clustering":"🔵","Time Series Forecasting":"📈"}

    for model in models:
        with st.expander(f"{icons.get(model.get('type',''),'🤖')}  {model.get('name','Unknown')}  —  {model.get('algorithm','')}", expanded=True):
            status_cls = "status-prod" if model.get("status") == "Production" else "status-exp"
            
            col_info, col_metrics = st.columns([2,3])

            with col_info:
                st.markdown(f"""
                <div style="margin-bottom:0.5rem;">
                    <span class="badge badge-blue">{model.get('model_id','')}</span>
                    <span class="badge {'badge-green' if model.get('status')=='Production' else 'badge-orange'}">{model.get('status','Unknown')}</span>
                    <span class="badge badge-purple">{model.get('type','')}</span>
                </div>
                <div style="color:#94a3b8; font-size:0.82rem; margin-top:0.5rem;">
                    <b style="color:#e2e8f0;">Algorithm:</b> {model.get('algorithm','—')}<br>
                    <b style="color:#e2e8f0;">Artifact:</b> <code style="color:#63b3ed;">{model.get('artifact','—')}</code>
                </div>
                """, unsafe_allow_html=True)

            with col_metrics:
                m_cols = st.columns(4)
                if "accuracy" in model:
                    m_cols[0].metric("✅ Accuracy",  f"{model['accuracy']*100:.1f}%")
                if "f1_score" in model:
                    m_cols[1].metric("📐 F1",        f"{model['f1_score']:.4f}")
                if "roc_auc" in model:
                    m_cols[2].metric("🎯 ROC-AUC",   f"{model['roc_auc']:.4f}")
                if "silhouette_score" in model:
                    m_cols[0].metric("📐 Silhouette", f"{model['silhouette_score']:.4f}")
                if "clusters" in model:
                    m_cols[1].metric("🔵 Clusters",   str(model["clusters"]))
                if "mae_inr" in model:
                    m_cols[0].metric("📉 MAE (₹)",   fmt_inr(model["mae_inr"]))
                if "mape_pct" in model:
                    m_cols[1].metric("📊 MAPE",       f"{model['mape_pct']:.2f}%")

    # Performance comparison radar
    if len(models) >= 2:
        st.markdown('<div class="section-header">📊 Model Performance Overview</div>', unsafe_allow_html=True)
        classif = [m for m in models if m.get("accuracy")]
        if classif:
            m = classif[0]
            fig_r = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=m.get("roc_auc", 0) * 100,
                delta={"reference": 80, "valueformat": ".1f"},
                number={"suffix": "% AUC", "font": {"size": 32, "color":"#f0f4ff"}},
                title={"text": f"Champion Model AUC<br><span style='font-size:0.8rem;color:#94a3b8'>{m.get('algorithm','')}</span>"},
                gauge={
                    "axis": {"range": [50, 100], "tickcolor":"#94a3b8"},
                    "bar": {"color": COLORS["blue"], "thickness": 0.26},
                    "bgcolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range":[50,70], "color":"rgba(245,101,101,0.12)"},
                        {"range":[70,85], "color":"rgba(246,173,85,0.12)"},
                        {"range":[85,100],"color":"rgba(72,187,120,0.12)"}
                    ]
                }
            ))
            fig_r.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c8d6e5"), height=300, margin=dict(t=60,b=20))
            st.plotly_chart(fig_r, use_container_width=True)

    # Model artifact paths
    st.markdown('<div class="section-header">📁 Model Artifacts</div>', unsafe_allow_html=True)
    for m in models:
        artifact = m.get("artifact","")
        exists = os.path.exists(artifact)
        status_icon = "✅" if exists else "❌"
        st.markdown(f"""
        <div class="glass-box" style="padding:0.7rem 1rem; margin-bottom:0.4rem; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="color:#94a3b8; font-size:0.72rem; text-transform:uppercase; margin-right:0.5rem;">{m.get('model_id','')}</span>
                <code style="color:#63b3ed;">{artifact}</code>
            </div>
            <div style="font-size:0.85rem; font-weight:600; {'color:#68d391' if exists else 'color:#fc8181'}">
                {status_icon} {'Found' if exists else 'Not Found'}
            </div>
        </div>
        """, unsafe_allow_html=True)
