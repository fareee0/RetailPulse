"""
RetailPulse – Page 2: Customer Segmentation & Churn Risk Analytics
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="RetailPulse – Customer Analytics", page_icon="👥", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1e1b4b 100%); color: #f8fafc; }
</style>
""", unsafe_allow_html=True)

PROC_DIR = "data/processed"

@st.cache_data(ttl=600)
def load_cust_data():
    seg_path = os.path.join(PROC_DIR, "customer_segments.parquet")
    churn_path = os.path.join(PROC_DIR, "customer_churn_predictions.parquet")
    df_seg = pd.read_parquet(seg_path) if os.path.exists(seg_path) else pd.DataFrame()
    df_churn = pd.read_parquet(churn_path) if os.path.exists(churn_path) else pd.DataFrame()
    return df_seg, df_churn

df_seg, df_churn = load_cust_data()

st.title("👥 Customer Intelligence, Personas & Churn Risk")
st.markdown("Deep dive into customer RFM archetypes, KMeans cluster mapping, and machine-learning churn risk scores.")

# Filter by Segment
if not df_seg.empty:
    segments = ["All Segments"] + sorted(df_seg["RFM_Segment"].dropna().unique().tolist())
    sel_seg = st.selectbox("🎯 Filter by RFM Persona", segments)
    if sel_seg != "All Segments":
        df_seg = df_seg[df_seg["RFM_Segment"] == sel_seg]

tab_seg, tab_churn, tab_predict = st.tabs(["🏷️ RFM & Clusters", "⚠️ Churn Risk Tiers", "🔮 Real-Time Customer Scoring"])

with tab_seg:
    col1, col2 = st.columns([6, 4])
    with col1:
        if not df_seg.empty and "PCA1" in df_seg.columns:
            fig_pca = px.scatter(
                df_seg,
                x="PCA1",
                y="PCA2",
                color="KMeans_Cluster",
                symbol="RFM_Segment",
                hover_data=["CustomerID", "Recency", "Frequency", "Monetary"],
                title="2D PCA Customer Projection (KMeans Cluster vs RFM)",
                template="plotly_dark",
                color_continuous_scale="Turbo"
            )
            fig_pca.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=450)
            st.plotly_chart(fig_pca, use_container_width=True)

    with col2:
        if not df_seg.empty:
            st.markdown("#### Persona Financial Profile")
            profile = df_seg.groupby("RFM_Segment").agg(
                Customers=("CustomerID", "count"),
                Avg_Recency=("Recency", "mean"),
                Avg_Freq=("Frequency", "mean"),
                Avg_Spend=("Monetary", "mean")
            ).reset_index().sort_values("Avg_Spend", ascending=False)
            st.dataframe(profile.style.format({
                "Avg_Recency": "{:.1f}d",
                "Avg_Freq": "{:.1f}",
                "Avg_Spend": "£{:,.0f}",
                "Customers": "{:,}"
            }), use_container_width=True, height=400)

with tab_churn:
    if not df_churn.empty:
        c1, c2 = st.columns([5, 5])
        with c1:
            risk_counts = df_churn["Churn_Risk_Tier"].value_counts().reset_index()
            risk_counts.columns = ["Risk_Tier", "Count"]
            fig_pie = px.pie(
                risk_counts,
                names="Risk_Tier",
                values="Count",
                hole=0.5,
                title="Customer Cohort Churn Risk Distribution",
                template="plotly_dark",
                color="Risk_Tier",
                color_discrete_map={
                    "🔴 High Risk": "#ef4444",
                    "🟡 Medium Risk": "#f59e0b",
                    "🟢 Low Risk": "#10b981"
                }
            )
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown("#### Top At-Risk High-Value Customers (Immediate Retention Targets)")
            high_value_at_risk = df_churn[df_churn["Churn_Risk_Tier"] == "🔴 High Risk"].sort_values("Monetary", ascending=False).head(10)
            cols_show = ["CustomerID", "Recency", "Frequency", "Monetary", "Churn_Probability"]
            st.dataframe(
                high_value_at_risk[[c for c in cols_show if c in high_value_at_risk.columns]].style.format({
                    "Monetary": "£{:,.2f}",
                    "Churn_Probability": "{:.1%}",
                    "Recency": "{:.0f}d"
                }),
                use_container_width=True,
                height=380
            )

with tab_predict:
    st.markdown("#### Live Interactive Customer Scoring Calculator")
    st.markdown("Enter customer metrics to evaluate real-time churn probability and RFM classification.")
    p1, p2, p3 = st.columns(3)
    with p1:
        in_r = st.number_input("Recency (Days since last order)", min_value=1, max_value=730, value=30)
    with p2:
        in_f = st.number_input("Frequency (Lifetime Orders)", min_value=1, max_value=200, value=5)
    with p3:
        in_m = st.number_input("Monetary (Lifetime Spend £)", min_value=1.0, max_value=50000.0, value=1200.0)

    if st.button("Calculate Churn Probability & Persona", type="primary"):
        # Heuristic scoring
        r_score = 5 if in_r <= 15 else (4 if in_r <= 45 else (3 if in_r <= 90 else (2 if in_r <= 180 else 1)))
        f_score = 5 if in_f >= 10 else (4 if in_f >= 6 else (3 if in_f >= 3 else (2 if in_f >= 2 else 1)))
        m_score = 5 if in_m >= 3000 else (4 if in_m >= 1500 else (3 if in_m >= 600 else (2 if in_m >= 250 else 1)))
        
        # Risk estimation proxy
        risk_score = min(max((in_r / 120.0) * 0.6 + (1.0 / in_f) * 0.3 - (in_m / 10000.0) * 0.1, 0.05), 0.95)
        
        res1, res2 = st.columns(2)
        with res1:
            st.metric("Estimated Churn Probability", f"{risk_score:.1%}")
            if risk_score >= 0.7:
                st.error("🔴 High Churn Risk — Priority intervention needed.")
            elif risk_score >= 0.4:
                st.warning("🟡 Medium Churn Risk — Send engagement nurture.")
            else:
                st.success("🟢 Low Churn Risk — Highly engaged buyer.")
        with res2:
            st.metric("RFM Composite Score", f"{r_score}{f_score}{m_score}")
            st.info("Persona: Champion / Loyal Customer" if (r_score>=3 and f_score>=3) else "Persona: At Risk / Dormant")
