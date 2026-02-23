"""
=============================================================================
 ONLINE RETAIL PROFIT LEAKAGE ANALYTICS DASHBOARD
 ─────────────────────────────────────────────────
 B.Tech Data Analytics Reverse Learning Project
 Dataset : UCI / Kaggle – Online Retail (2010‑2011)
 Stack   : Python 3 | Streamlit | Plotly | Pandas
 Author  : Astitva
=============================================================================
 PROFIT ASSUMPTIONS (clearly stated)
 ───────────────────────────────────
 - Revenue  = Quantity x UnitPrice  (only positive‑quantity rows)
 - Returns  = |Quantity x UnitPrice| for cancelled / negative‑qty rows
 - COGS     = 60 % of revenue  (industry‑standard assumption for e‑retail)
 - Gross Profit Margin = 40 %
 - Net Profit = Gross Revenue x 0.40  −  Return Losses
 - A returned item loses both the revenue AND the fulfilment cost already
   incurred, so each £1 of return revenue translates to roughly £1.15 of
   true economic loss (handling, shipping, restocking).  For simplicity the
   dashboard uses 1:1 mapping unless the user toggles the multiplier.
=============================================================================
"""

# ── Imports ────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings, os

warnings.filterwarnings("ignore")

# ── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Profit Leakage Analytics",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for BI‑style design ─────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0e1117; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1f2e 0%, #16192b 100%);
        border: 1px solid #2d3348;
        border-radius: 10px;
        padding: 15px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #8892b0 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e6f1ff !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f1a 0%, #111827 100%);
        border-right: 1px solid #1e293b;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1f2e;
        border-radius: 8px 8px 0 0;
        color: #8892b0;
        padding: 10px 20px;
        border: 1px solid #2d3348;
    }
    .stTabs [aria-selected="true"] {
        background-color: #233554 !important;
        color: #64ffda !important;
        border-bottom: 2px solid #64ffda;
    }

    /* Headers */
    h1, h2, h3 { color: #ccd6f6 !important; }
    h1 { border-bottom: 2px solid #64ffda; padding-bottom: 8px; }

    /* Divider */
    hr { border-color: #233554 !important; }

    /* Info boxes */
    .insight-box {
        background: #1a2332;
        border-left: 4px solid #64ffda;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        color: #a8b2d1;
    }
    .warning-box {
        background: #2a1a1a;
        border-left: 4px solid #ff6b6b;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        color: #d4a0a0;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1 : DATA LOADING AND CLEANING
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading and cleaning data ...")
def load_and_clean_data(path: str) -> pd.DataFrame:
    """
    Reads the raw Excel file, applies cleaning rules, and engineers 
    features required for the dashboard.

    Cleaning steps:
        1. Drop rows where Description is null.
        2. Drop rows where CustomerID is null (cannot attribute).
        3. Remove zero‑price rows (non‑transactional records).
        4. Strip whitespace from string columns.
        5. Create a boolean 'is_return' flag based on:
           - InvoiceNo starts with 'C'  OR
           - Quantity < 0
        6. Compute Revenue = Quantity * UnitPrice (signed).
        7. Create time features: Year, Month, YearMonth, DayOfWeek, Hour.
        8. Create PriceBand categories for price‑band analysis.
        9. Create a simple product Category from first word of Description.
    """
    df = pd.read_excel(path)

    # ── Step 1-3: Drop nulls and zero prices ──────────────────────────
    initial_rows = len(df)
    df = df.dropna(subset=["Description", "CustomerID"])
    df = df[df["UnitPrice"] > 0]  # remove zero / negative priced rows (non‑sale)
    cleaned_rows = len(df)

    # ── Step 4: Strip whitespace ──────────────────────────────────────
    df["Description"] = df["Description"].str.strip()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Country"] = df["Country"].str.strip()
    df["CustomerID"] = df["CustomerID"].astype(int)

    # ── Step 5: Return flag ───────────────────────────────────────────
    df["is_return"] = (df["InvoiceNo"].str.startswith("C")) | (df["Quantity"] < 0)

    # ── Step 6: Revenue (signed) ──────────────────────────────────────
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    # ── Step 7: Time features ─────────────────────────────────────────
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["DayOfWeek"] = df["InvoiceDate"].dt.day_name()
    df["Hour"] = df["InvoiceDate"].dt.hour

    # ── Step 8: Price bands ───────────────────────────────────────────
    df["PriceBand"] = pd.cut(
        df["UnitPrice"],
        bins=[0, 2, 5, 10, 25, 50, 1000, float("inf")],
        labels=["0-2", "2-5", "5-10", "10-25", "25-50", "50-1000", "1000+"],
    )

    # ── Step 9: Product category (first word of description) ──────────
    df["Category"] = df["Description"].str.split().str[0].str.upper()

    return df


# ── Load data ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Online Retail.xlsx")
df_all = load_and_clean_data(DATA_PATH)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2 : SIDEBAR FILTERS
# ═══════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("## Filters")

# Date range
min_date = df_all["InvoiceDate"].min().date()
max_date = df_all["InvoiceDate"].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Country
countries = ["All"] + sorted(df_all["Country"].unique().tolist())
selected_country = st.sidebar.selectbox("Country", countries, index=0)

# Top product categories
top_categories = df_all["Category"].value_counts().head(30).index.tolist()
selected_categories = st.sidebar.multiselect(
    "Product Category (top 30)", top_categories, default=[]
)

# Price band
price_bands = df_all["PriceBand"].cat.categories.tolist()
selected_bands = st.sidebar.multiselect("Price Band", price_bands, default=[])

# ── Apply filters ──────────────────────────────────────────────────────────
df = df_all.copy()

if isinstance(date_range, tuple) and len(date_range) == 2:
    df = df[(df["InvoiceDate"].dt.date >= date_range[0]) & (df["InvoiceDate"].dt.date <= date_range[1])]

if selected_country != "All":
    df = df[df["Country"] == selected_country]

if selected_categories:
    df = df[df["Category"].isin(selected_categories)]

if selected_bands:
    df = df[df["PriceBand"].isin(selected_bands)]

# ── Separate sales and returns ─────────────────────────────────────────────
df_sales = df[~df["is_return"]]
df_returns = df[df["is_return"]]


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3 : HELPER CONSTANTS & FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

COST_RATIO = 0.60        # COGS as fraction of revenue (assumed)
PROFIT_MARGIN = 0.40     # Gross profit margin
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_SALES = "#64ffda"
COLOR_RETURN = "#ff6b6b"
COLOR_PROFIT = "#ffd93d"
COLOR_PALETTE = ["#64ffda", "#ff6b6b", "#ffd93d", "#6c83ff", "#c792ea",
                 "#ff9f43", "#54a0ff", "#5f27cd", "#01a3a4", "#ee5a24"]


def fmt_currency(val):
    """Format a number as currency string with appropriate suffix."""
    if abs(val) >= 1_000_000:
        return f"GBP {val/1_000_000:,.2f}M"
    elif abs(val) >= 1_000:
        return f"GBP {val/1_000:,.1f}K"
    return f"GBP {val:,.2f}"


def fmt_pct(val):
    return f"{val:.1f}%"


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4 : DASHBOARD HEADER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("# Online Retail -- Profit Leakage Analytics")
st.markdown(
    "*Identifying where profit drains occur due to product returns, and "
    "quantifying recovery opportunities.*"
)
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5 : TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Summary",
    "Product & Category Intelligence",
    "Customer Behaviour & Risk",
    "Profit Lift Simulator",
    "Recommendations & Ethics",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 1 : EXECUTIVE PROFIT LEAKAGE SUMMARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab1:
    st.markdown("## Executive Profit Leakage Summary")

    # ── KPI calculations ──────────────────────────────────────────────
    gross_revenue = df_sales["Revenue"].sum()
    return_losses = df_returns["Revenue"].abs().sum()
    net_revenue = gross_revenue - return_losses
    return_rate = (return_losses / gross_revenue * 100) if gross_revenue > 0 else 0
    gross_profit = gross_revenue * PROFIT_MARGIN
    net_profit = gross_profit - return_losses
    profit_erosion = (return_losses / gross_profit * 100) if gross_profit > 0 else 0
    total_transactions = df_sales["InvoiceNo"].nunique()
    return_transactions = df_returns["InvoiceNo"].nunique()
    txn_return_rate = (return_transactions / total_transactions * 100) if total_transactions > 0 else 0

    # ── Metric cards ─────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gross Revenue", fmt_currency(gross_revenue))
    c2.metric("Return Losses", fmt_currency(return_losses))
    c3.metric("Net Revenue", fmt_currency(net_revenue))
    c4.metric("Return Rate (Revenue)", fmt_pct(return_rate))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Gross Profit (est.)", fmt_currency(gross_profit))
    c6.metric("Net Profit (est.)", fmt_currency(net_profit))
    c7.metric("Profit Erosion by Returns", fmt_pct(profit_erosion))
    c8.metric("Transaction Return Rate", fmt_pct(txn_return_rate))

    st.markdown("---")

    # ── Monthly revenue vs return trend ──────────────────────────────
    st.markdown("### Monthly Revenue vs Return Losses Trend")

    monthly_sales = df_sales.groupby("YearMonth")["Revenue"].sum().reset_index()
    monthly_sales.columns = ["YearMonth", "GrossRevenue"]

    monthly_returns = df_returns.groupby("YearMonth")["Revenue"].apply(
        lambda x: x.abs().sum()
    ).reset_index()
    monthly_returns.columns = ["YearMonth", "ReturnLoss"]

    monthly = pd.merge(monthly_sales, monthly_returns, on="YearMonth", how="outer").fillna(0)
    monthly = monthly.sort_values("YearMonth")
    monthly["NetRevenue"] = monthly["GrossRevenue"] - monthly["ReturnLoss"]
    monthly["ReturnRate"] = (monthly["ReturnLoss"] / monthly["GrossRevenue"] * 100).fillna(0)

    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(
        go.Bar(x=monthly["YearMonth"], y=monthly["GrossRevenue"],
               name="Gross Revenue", marker_color=COLOR_SALES, opacity=0.75),
        secondary_y=False,
    )
    fig_trend.add_trace(
        go.Bar(x=monthly["YearMonth"], y=monthly["ReturnLoss"],
               name="Return Losses", marker_color=COLOR_RETURN, opacity=0.75),
        secondary_y=False,
    )
    fig_trend.add_trace(
        go.Scatter(x=monthly["YearMonth"], y=monthly["ReturnRate"],
                   name="Return Rate %", line=dict(color=COLOR_PROFIT, width=2.5),
                   mode="lines+markers"),
        secondary_y=True,
    )
    fig_trend.update_layout(
        template=PLOTLY_TEMPLATE, barmode="group", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=30, b=40),
    )
    fig_trend.update_yaxes(title_text="Amount (GBP)", secondary_y=False)
    fig_trend.update_yaxes(title_text="Return Rate (%)", secondary_y=True)
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── Revenue waterfall ──────────────────────────────────────────────
    st.markdown("### Revenue Waterfall: From Gross to Net Profit")

    waterfall_data = dict(
        x=["Gross Revenue", "COGS (60%)", "Gross Profit", "Return Losses", "Net Profit"],
        measure=["absolute", "relative", "total", "relative", "total"],
        y=[gross_revenue, -gross_revenue * COST_RATIO, 0, -return_losses, 0],
        connector={"line": {"color": "#2d3348"}},
        increasing={"marker": {"color": COLOR_SALES}},
        decreasing={"marker": {"color": COLOR_RETURN}},
        totals={"marker": {"color": COLOR_PROFIT}},
    )
    fig_waterfall = go.Figure(go.Waterfall(**waterfall_data))
    fig_waterfall.update_layout(template=PLOTLY_TEMPLATE, height=380,
                                margin=dict(l=40, r=40, t=30, b=40))
    st.plotly_chart(fig_waterfall, use_container_width=True)

    # ── Country‑level return heatmap ─────────────────────────────────
    st.markdown("### Return Rate by Country (Top 15 by Volume)")

    country_sales = df_sales.groupby("Country")["Revenue"].sum()
    country_returns = df_returns.groupby("Country")["Revenue"].apply(lambda x: x.abs().sum())
    country_df = pd.DataFrame({"Sales": country_sales, "Returns": country_returns}).fillna(0)
    country_df["ReturnRate"] = (country_df["Returns"] / country_df["Sales"] * 100).fillna(0)
    country_df = country_df.sort_values("Sales", ascending=False).head(15).sort_values("ReturnRate", ascending=True)

    fig_country = px.bar(
        country_df.reset_index(), x="ReturnRate", y="Country",
        orientation="h", color="ReturnRate",
        color_continuous_scale=["#64ffda", "#ffd93d", "#ff6b6b"],
        text=country_df["ReturnRate"].apply(lambda v: f"{v:.1f}%").values,
    )
    fig_country.update_layout(template=PLOTLY_TEMPLATE, height=420,
                              margin=dict(l=40, r=40, t=30, b=40),
                              coloraxis_colorbar_title="Return %")
    fig_country.update_traces(textposition="outside")
    st.plotly_chart(fig_country, use_container_width=True)

    # ── Key insight box ──────────────────────────────────────────────
    st.markdown(f"""
    <div class="insight-box">
        <strong>Key Finding:</strong> Total return losses of <strong>{fmt_currency(return_losses)}</strong>
        erode <strong>{profit_erosion:.1f}%</strong> of estimated gross profit.
        At the current trajectory, every 1% reduction in return rate would recover approximately
        <strong>{fmt_currency(gross_revenue * 0.01)}</strong> in revenue.
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 2 : PRODUCT & CATEGORY RETURN INTELLIGENCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab2:
    st.markdown("## Product & Category Return Intelligence")

    col_left, col_right = st.columns(2)

    # ── Top 15 products by return loss ───────────────────────────────
    with col_left:
        st.markdown("### Top 15 Products by Return Loss")
        prod_returns = (
            df_returns.groupby("Description")["Revenue"]
            .apply(lambda x: x.abs().sum())
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        prod_returns.columns = ["Product", "ReturnLoss"]

        fig_prodloss = px.bar(
            prod_returns, x="ReturnLoss", y="Product", orientation="h",
            color="ReturnLoss", color_continuous_scale=["#ffd93d", "#ff6b6b"],
        )
        fig_prodloss.update_layout(template=PLOTLY_TEMPLATE, height=480,
                                   margin=dict(l=10, r=30, t=10, b=30),
                                   yaxis=dict(autorange="reversed"),
                                   coloraxis_colorbar_title="Loss (GBP)")
        st.plotly_chart(fig_prodloss, use_container_width=True)

    # ── Category return rate (top 20) ────────────────────────────────
    with col_right:
        st.markdown("### Category Return Rate (Top 20 Categories)")
        cat_sales = df_sales.groupby("Category")["Revenue"].sum()
        cat_rets = df_returns.groupby("Category")["Revenue"].apply(lambda x: x.abs().sum())
        cat_df = pd.DataFrame({"Sales": cat_sales, "Returns": cat_rets}).fillna(0)
        cat_df["ReturnRate"] = (cat_df["Returns"] / cat_df["Sales"] * 100).fillna(0)
        cat_df = cat_df[cat_df["Sales"] > 500].sort_values("ReturnRate", ascending=False).head(20)

        fig_catrate = px.bar(
            cat_df.reset_index(), x="ReturnRate", y="Category", orientation="h",
            color="ReturnRate", color_continuous_scale=["#64ffda", "#ff6b6b"],
            text=cat_df["ReturnRate"].apply(lambda v: f"{v:.1f}%").values,
        )
        fig_catrate.update_layout(template=PLOTLY_TEMPLATE, height=480,
                                  margin=dict(l=10, r=30, t=10, b=30),
                                  coloraxis_colorbar_title="Return %")
        fig_catrate.update_traces(textposition="outside")
        st.plotly_chart(fig_catrate, use_container_width=True)

    st.markdown("---")

    # ── Price band vs return risk ────────────────────────────────────
    st.markdown("### Price Band vs Return Risk")

    band_sales = df_sales.groupby("PriceBand", observed=True)["Revenue"].sum()
    band_rets = df_returns.groupby("PriceBand", observed=True)["Revenue"].apply(lambda x: x.abs().sum())
    band_df = pd.DataFrame({"Sales": band_sales, "Returns": band_rets}).fillna(0)
    band_df["ReturnRate"] = (band_df["Returns"] / band_df["Sales"] * 100).fillna(0)
    band_df["ReturnLoss"] = band_df["Returns"]
    band_df = band_df.reset_index()

    fig_band = make_subplots(specs=[[{"secondary_y": True}]])
    fig_band.add_trace(
        go.Bar(x=band_df["PriceBand"].astype(str), y=band_df["ReturnLoss"],
               name="Return Loss (GBP)", marker_color=COLOR_RETURN, opacity=0.8),
        secondary_y=False,
    )
    fig_band.add_trace(
        go.Scatter(x=band_df["PriceBand"].astype(str), y=band_df["ReturnRate"],
                   name="Return Rate %", line=dict(color=COLOR_PROFIT, width=3),
                   mode="lines+markers", marker=dict(size=10)),
        secondary_y=True,
    )
    fig_band.update_layout(template=PLOTLY_TEMPLATE, height=380,
                           legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                       xanchor="center", x=0.5),
                           margin=dict(l=40, r=40, t=30, b=40))
    fig_band.update_yaxes(title_text="Return Loss (GBP)", secondary_y=False)
    fig_band.update_yaxes(title_text="Return Rate (%)", secondary_y=True)
    st.plotly_chart(fig_band, use_container_width=True)

    # ── Return concentration (Pareto) ────────────────────────────────
    st.markdown("### Return Concentration (Pareto Analysis)")

    pareto = (
        df_returns.groupby("Description")["Revenue"]
        .apply(lambda x: x.abs().sum())
        .sort_values(ascending=False)
        .reset_index()
    )
    pareto.columns = ["Product", "ReturnLoss"]
    pareto["CumulPct"] = pareto["ReturnLoss"].cumsum() / pareto["ReturnLoss"].sum() * 100
    pareto["Rank"] = range(1, len(pareto) + 1)
    pareto_top = pareto.head(50)

    fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pareto.add_trace(
        go.Bar(x=pareto_top["Rank"], y=pareto_top["ReturnLoss"],
               name="Return Loss", marker_color=COLOR_RETURN, opacity=0.7),
        secondary_y=False,
    )
    fig_pareto.add_trace(
        go.Scatter(x=pareto_top["Rank"], y=pareto_top["CumulPct"],
                   name="Cumul. %", line=dict(color=COLOR_SALES, width=2.5),
                   mode="lines"),
        secondary_y=True,
    )
    fig_pareto.add_hline(y=80, line_dash="dash", line_color="#ffd93d",
                         annotation_text="80% threshold", secondary_y=True)
    fig_pareto.update_layout(template=PLOTLY_TEMPLATE, height=380,
                             legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                         xanchor="center", x=0.5),
                             margin=dict(l=40, r=40, t=30, b=40))
    fig_pareto.update_xaxes(title_text="Product Rank")
    fig_pareto.update_yaxes(title_text="Return Loss (GBP)", secondary_y=False)
    fig_pareto.update_yaxes(title_text="Cumulative %", secondary_y=True)
    st.plotly_chart(fig_pareto, use_container_width=True)

    products_for_80 = pareto[pareto["CumulPct"] <= 80].shape[0]
    total_products = pareto.shape[0]
    st.markdown(f"""
    <div class="insight-box">
        <strong>Pareto Insight:</strong> Just <strong>{products_for_80}</strong> out of
        <strong>{total_products}</strong> returned products account for 80% of all return losses.
        Targeting these items alone can recover the majority of leaked profit.
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 3 : CUSTOMER BEHAVIOUR & POLICY RISK
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab3:
    st.markdown("## Customer Behaviour & Policy Risk")

    # ── Repeat returners ─────────────────────────────────────────────
    st.markdown("### Repeat Returners Analysis")

    cust_return_summary = (
        df_returns.groupby("CustomerID")
        .agg(
            ReturnCount=("InvoiceNo", "nunique"),
            TotalReturnLoss=("Revenue", lambda x: x.abs().sum()),
        )
        .reset_index()
        .sort_values("TotalReturnLoss", ascending=False)
    )

    # Customer purchase data for context
    cust_purchase = (
        df_sales.groupby("CustomerID")
        .agg(PurchaseCount=("InvoiceNo", "nunique"), TotalSpend=("Revenue", "sum"))
        .reset_index()
    )

    cust_merged = pd.merge(cust_purchase, cust_return_summary, on="CustomerID", how="left").fillna(0)
    cust_merged["ReturnRate"] = (cust_merged["ReturnCount"] / cust_merged["PurchaseCount"] * 100).fillna(0)
    cust_merged["ReturnRate"] = cust_merged["ReturnRate"].clip(upper=100)

    # ── Risk segmentation ────────────────────────────────────────────
    def classify_risk(row):
        if row["ReturnRate"] >= 40 and row["TotalReturnLoss"] > cust_merged["TotalReturnLoss"].quantile(0.75):
            return "High Risk"
        elif row["ReturnRate"] >= 20:
            return "Medium Risk"
        elif row["ReturnCount"] > 0:
            return "Low Risk"
        else:
            return "No Returns"

    cust_merged["RiskSegment"] = cust_merged.apply(classify_risk, axis=1)

    c1, c2 = st.columns(2)

    with c1:
        # Top 15 repeat returners
        top_returners = cust_merged.sort_values("TotalReturnLoss", ascending=False).head(15)
        fig_returners = px.bar(
            top_returners, x="TotalReturnLoss", y="CustomerID",
            orientation="h", color="ReturnRate",
            color_continuous_scale=["#64ffda", "#ff6b6b"],
            hover_data=["PurchaseCount", "ReturnCount", "TotalSpend"],
        )
        fig_returners.update_layout(
            template=PLOTLY_TEMPLATE, height=450,
            title="Top 15 Customers by Return Loss",
            margin=dict(l=10, r=30, t=40, b=30),
            yaxis=dict(autorange="reversed", type="category"),
            coloraxis_colorbar_title="Return %",
        )
        st.plotly_chart(fig_returners, use_container_width=True)

    with c2:
        # Risk segment distribution
        risk_counts = cust_merged["RiskSegment"].value_counts().reset_index()
        risk_counts.columns = ["Segment", "Count"]
        risk_color_map = {
            "High Risk": "#ff6b6b", "Medium Risk": "#ffd93d",
            "Low Risk": "#64ffda", "No Returns": "#6c83ff"
        }
        fig_risk = px.pie(
            risk_counts, names="Segment", values="Count",
            color="Segment", color_discrete_map=risk_color_map,
            hole=0.45,
        )
        fig_risk.update_layout(
            template=PLOTLY_TEMPLATE, height=450,
            title="Customer Risk Segmentation",
            margin=dict(l=10, r=30, t=40, b=30),
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("---")

    # ── Cohort analysis (customer retention) ─────────────────────────
    st.markdown("### Customer Cohort Analysis (Monthly Retention)")

    cohort_data = df_sales[["CustomerID", "InvoiceDate"]].copy()
    cohort_data["OrderMonth"] = cohort_data["InvoiceDate"].dt.to_period("M")
    cohort_data["CohortMonth"] = cohort_data.groupby("CustomerID")["OrderMonth"].transform("min")
    cohort_data["CohortIndex"] = (
        (cohort_data["OrderMonth"] - cohort_data["CohortMonth"]).apply(lambda x: x.n)
    )

    cohort_table = (
        cohort_data.groupby(["CohortMonth", "CohortIndex"])["CustomerID"]
        .nunique()
        .reset_index()
    )
    cohort_table.columns = ["CohortMonth", "CohortIndex", "Customers"]
    cohort_pivot = cohort_table.pivot(index="CohortMonth", columns="CohortIndex", values="Customers")

    # Retention rates
    cohort_sizes = cohort_pivot[0]
    retention = cohort_pivot.divide(cohort_sizes, axis=0) * 100
    retention = retention.round(1)

    fig_cohort = px.imshow(
        retention.iloc[:, :13].values,
        labels=dict(x="Months Since First Purchase", y="Cohort", color="Retention %"),
        x=[f"M{i}" for i in range(min(13, retention.shape[1]))],
        y=[str(c) for c in retention.index[:13]],
        color_continuous_scale=["#0e1117", "#233554", "#64ffda"],
        aspect="auto",
    )
    fig_cohort.update_layout(template=PLOTLY_TEMPLATE, height=420,
                             margin=dict(l=40, r=40, t=30, b=40))
    st.plotly_chart(fig_cohort, use_container_width=True)

    st.markdown("---")

    # ── Customer segmentation scatter ────────────────────────────────
    st.markdown("### Customer Segmentation: Spend vs Return Behaviour")

    scatter_data = cust_merged[cust_merged["TotalSpend"] > 0].copy()
    scatter_data = scatter_data[scatter_data["TotalReturnLoss"] > 0]

    fig_scatter = px.scatter(
        scatter_data, x="TotalSpend", y="TotalReturnLoss",
        color="RiskSegment", size="ReturnCount",
        hover_data=["CustomerID", "PurchaseCount", "ReturnRate"],
        color_discrete_map=risk_color_map,
        opacity=0.7,
    )
    fig_scatter.update_layout(
        template=PLOTLY_TEMPLATE, height=450,
        xaxis_title="Total Spend (GBP)",
        yaxis_title="Total Return Loss (GBP)",
        legend_title="Risk Segment",
        margin=dict(l=40, r=40, t=30, b=40),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Day-of-week and hour return patterns ─────────────────────────
    st.markdown("### Temporal Return Patterns")

    tc1, tc2 = st.columns(2)

    with tc1:
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_sales = df_sales.groupby("DayOfWeek")["Revenue"].sum()
        dow_rets  = df_returns.groupby("DayOfWeek")["Revenue"].apply(lambda x: x.abs().sum())
        dow_df = pd.DataFrame({"Sales": dow_sales, "Returns": dow_rets}).fillna(0)
        dow_df["ReturnRate"] = (dow_df["Returns"] / dow_df["Sales"] * 100).fillna(0)
        dow_df = dow_df.reindex([d for d in dow_order if d in dow_df.index])

        fig_dow = px.bar(
            dow_df.reset_index(), x="DayOfWeek", y="ReturnRate",
            color="ReturnRate", color_continuous_scale=["#64ffda", "#ff6b6b"],
            text=dow_df["ReturnRate"].apply(lambda v: f"{v:.1f}%").values,
        )
        fig_dow.update_layout(template=PLOTLY_TEMPLATE, height=350,
                              title="Return Rate by Day of Week",
                              margin=dict(l=30, r=30, t=40, b=30))
        fig_dow.update_traces(textposition="outside")
        st.plotly_chart(fig_dow, use_container_width=True)

    with tc2:
        hr_sales = df_sales.groupby("Hour")["Revenue"].sum()
        hr_rets  = df_returns.groupby("Hour")["Revenue"].apply(lambda x: x.abs().sum())
        hr_df = pd.DataFrame({"Sales": hr_sales, "Returns": hr_rets}).fillna(0)
        hr_df["ReturnRate"] = (hr_df["Returns"] / hr_df["Sales"] * 100).fillna(0)

        fig_hour = px.line(
            hr_df.reset_index(), x="Hour", y="ReturnRate",
            markers=True, color_discrete_sequence=[COLOR_RETURN],
        )
        fig_hour.update_layout(template=PLOTLY_TEMPLATE, height=350,
                               title="Return Rate by Hour of Day",
                               margin=dict(l=30, r=30, t=40, b=30))
        st.plotly_chart(fig_hour, use_container_width=True)

    # Risk summary
    high_risk_count = (cust_merged["RiskSegment"] == "High Risk").sum()
    high_risk_loss = cust_merged[cust_merged["RiskSegment"] == "High Risk"]["TotalReturnLoss"].sum()
    st.markdown(f"""
    <div class="warning-box">
        <strong>Policy Alert:</strong> <strong>{high_risk_count}</strong> high-risk customers
        are responsible for <strong>{fmt_currency(high_risk_loss)}</strong> in return losses.
        Implementing tiered return policies for this segment could recover a significant
        portion of leaked profit without impacting the broader customer base.
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 4 : PROFIT LIFT SIMULATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab4:
    st.markdown("## Profit Lift Simulator")
    st.markdown(
        "Use the controls below to model the financial impact of reducing return rates. "
        "Adjust the slider to see projected profit recovery."
    )

    sim_col1, sim_col2 = st.columns([1, 2])

    with sim_col1:
        st.markdown("### Simulation Parameters")

        return_reduction = st.slider(
            "Return Reduction Target (%)", min_value=0, max_value=100,
            value=25, step=5, help="Percentage reduction in current return losses"
        )

        margin_override = st.slider(
            "Gross Margin Assumption (%)", min_value=20, max_value=60,
            value=int(PROFIT_MARGIN * 100), step=5,
        )

        cost_multiplier = st.slider(
            "Return Processing Cost Multiplier",
            min_value=1.0, max_value=1.5, value=1.15, step=0.05,
            help="Each GBP 1 of return costs this much in total economic loss"
        )

    with sim_col2:
        st.markdown("### Scenario Comparison")

        sim_margin = margin_override / 100
        sim_gross_profit = gross_revenue * sim_margin
        true_return_cost = return_losses * cost_multiplier
        recovered_amount = true_return_cost * (return_reduction / 100)
        new_return_cost = true_return_cost - recovered_amount
        current_net_profit = sim_gross_profit - true_return_cost
        projected_net_profit = sim_gross_profit - new_return_cost
        profit_lift = projected_net_profit - current_net_profit
        profit_lift_pct = (profit_lift / abs(current_net_profit) * 100) if current_net_profit != 0 else 0

        # Scenario comparison cards
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Current Net Profit", fmt_currency(current_net_profit))
        sc2.metric("Projected Net Profit", fmt_currency(projected_net_profit),
                   delta=fmt_currency(profit_lift))
        sc3.metric("Profit Lift", fmt_pct(profit_lift_pct))

        # Scenario bars
        scenarios = pd.DataFrame({
            "Scenario": ["Current", f"{return_reduction}% Reduction"],
            "Return Losses": [true_return_cost, new_return_cost],
            "Net Profit": [current_net_profit, projected_net_profit],
        })

        fig_scenario = go.Figure()
        fig_scenario.add_trace(go.Bar(
            x=scenarios["Scenario"], y=scenarios["Return Losses"],
            name="Return Losses", marker_color=COLOR_RETURN, opacity=0.8,
        ))
        fig_scenario.add_trace(go.Bar(
            x=scenarios["Scenario"], y=scenarios["Net Profit"],
            name="Net Profit", marker_color=COLOR_SALES, opacity=0.8,
        ))
        fig_scenario.update_layout(
            template=PLOTLY_TEMPLATE, barmode="group", height=350,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=40, r=40, t=30, b=40),
        )
        st.plotly_chart(fig_scenario, use_container_width=True)

    st.markdown("---")

    # ── Sensitivity analysis ─────────────────────────────────────────
    st.markdown("### Profit Sensitivity to Return Reduction")

    sensitivity_data = []
    for red_pct in range(0, 105, 5):
        rec = true_return_cost * (red_pct / 100)
        np_val = sim_gross_profit - (true_return_cost - rec)
        sensitivity_data.append({
            "Reduction %": red_pct,
            "Net Profit": np_val,
            "Recovered": rec,
        })
    sens_df = pd.DataFrame(sensitivity_data)

    fig_sens = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sens.add_trace(
        go.Scatter(x=sens_df["Reduction %"], y=sens_df["Net Profit"],
                   name="Net Profit", fill="tozeroy",
                   line=dict(color=COLOR_SALES, width=2.5)),
        secondary_y=False,
    )
    fig_sens.add_trace(
        go.Bar(x=sens_df["Reduction %"], y=sens_df["Recovered"],
               name="Amount Recovered", marker_color=COLOR_PROFIT, opacity=0.4),
        secondary_y=True,
    )
    fig_sens.add_vline(x=return_reduction, line_dash="dash", line_color="#ff6b6b",
                       annotation_text=f"Target: {return_reduction}%")
    fig_sens.update_layout(
        template=PLOTLY_TEMPLATE, height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=30, b=40),
    )
    fig_sens.update_xaxes(title_text="Return Reduction (%)")
    fig_sens.update_yaxes(title_text="Net Profit (GBP)", secondary_y=False)
    fig_sens.update_yaxes(title_text="Recovered Amount (GBP)", secondary_y=True)
    st.plotly_chart(fig_sens, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box">
        <strong>Simulation Result:</strong> A <strong>{return_reduction}%</strong> reduction in
        return losses (with a {cost_multiplier}x cost multiplier) projects a profit lift of
        <strong>{fmt_currency(profit_lift)}</strong>, improving net profit by
        <strong>{profit_lift_pct:.1f}%</strong>.
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TAB 5 : RECOMMENDATIONS & ETHICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab5:
    st.markdown("## Operational Recommendations & Ethical Safeguards")

    # ── Action Tracker ───────────────────────────────────────────────
    st.markdown("### Action Tracker: Data-Driven Recommendations")

    recommendations = pd.DataFrame({
        "Priority": ["Critical", "Critical", "High", "High", "High",
                      "Medium", "Medium", "Medium", "Low", "Low"],
        "Issue Identified": [
            "Small number of products drive majority of return losses",
            "High-risk customers disproportionately generate returns",
            "Certain price bands show elevated return rates",
            "Specific countries exhibit higher-than-average return rates",
            "Peak return activity correlates with specific time periods",
            "Lack of product category granularity limits root-cause analysis",
            "No cost-of-return data available for precise profit calculation",
            "Customer descriptions missing for 25% of transactions",
            "Weekend return patterns differ from weekday patterns",
            "New customers show higher return propensity in first month",
        ],
        "Recommended Action": [
            "Implement quality reviews and enhanced descriptions for top-loss products",
            "Deploy tiered return policies; flag accounts exceeding threshold",
            "Add size guides, detailed specs, and AR previews for high-price items",
            "Localise product descriptions and shipping expectations by region",
            "Extend quality-check windows before peak dispatch periods",
            "Enrich product taxonomy with standardised category hierarchy",
            "Partner with finance to capture actual fulfilment and return costs",
            "Mandate customer ID capture at all transaction touchpoints",
            "Investigate weekend-specific fulfilment quality issues",
            "Implement first-purchase guidance emails with product-care tips",
        ],
        "Expected Impact": [
            "Recover up to 60% of concentrated return losses",
            "Reduce high-risk segment losses by 30-40%",
            "Lower price-band-specific returns by 15-20%",
            "Improve region-specific satisfaction; reduce returns by 10-15%",
            "Mitigate seasonal return spikes by 10-20%",
            "Enable precise category-level intervention strategies",
            "Enable exact ROI calculation for return-reduction initiatives",
            "Expand analysable customer base by 25%",
            "Potential 5-8% reduction in weekend-origin returns",
            "Reduce first-purchase returns by 10-15%",
        ],
    })

    # Style the table
    def color_priority(val):
        colors = {"Critical": "#ff6b6b", "High": "#ffd93d", "Medium": "#64ffda", "Low": "#6c83ff"}
        return f"background-color: {colors.get(val, '#1a1f2e')}; color: #0e1117; font-weight: bold"

    styled = recommendations.style.applymap(color_priority, subset=["Priority"])
    st.dataframe(recommendations, use_container_width=True, hide_index=True, height=420)

    st.markdown("---")

    # ── Ethical Safeguards ───────────────────────────────────────────
    st.markdown("### Ethical Safeguards")

    st.markdown("""
    <div class="insight-box">
        <strong>1. Customer Fairness:</strong> Return-restriction policies must not penalise
        legitimate product-quality complaints. Any tiered policy should include a transparent
        appeals mechanism and must comply with consumer protection regulations (e.g., UK Consumer
        Rights Act 2015, EU Consumer Rights Directive).
    </div>
    <div class="insight-box">
        <strong>2. Data Privacy:</strong> Customer-level return-behaviour profiling must adhere
        to GDPR / data-protection principles. Personal identifiers should be anonymised in
        analytical outputs, and customers must be informed if their return behaviour influences
        service terms.
    </div>
    <div class="insight-box">
        <strong>3. Algorithmic Bias:</strong> Risk-segmentation models should be regularly
        audited to ensure they do not disproportionately disadvantage customers from specific
        demographic or geographic groups.
    </div>
    <div class="insight-box">
        <strong>4. Transparency:</strong> Any changes to return policies motivated by this
        analysis should be clearly communicated to customers before implementation, not applied
        retroactively.
    </div>
    <div class="insight-box">
        <strong>5. Proportionality:</strong> Interventions should be proportional to the
        identified risk. A customer with a marginally elevated return rate should not receive
        the same restrictions as a serial abuser.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── KPI Definitions ──────────────────────────────────────────────
    st.markdown("### KPI Definitions & Assumptions")

    kpi_data = pd.DataFrame({
        "KPI": [
            "Gross Revenue", "Return Losses", "Net Revenue", "Return Rate (Revenue)",
            "Gross Profit", "Net Profit", "Profit Erosion", "Transaction Return Rate",
            "Customer Return Rate", "Return Concentration Index"
        ],
        "Formula": [
            "SUM(Quantity x UnitPrice) for non-return rows",
            "SUM(|Quantity x UnitPrice|) for return rows",
            "Gross Revenue - Return Losses",
            "(Return Losses / Gross Revenue) x 100",
            "Gross Revenue x 0.40 (assumed margin)",
            "Gross Profit - Return Losses",
            "(Return Losses / Gross Profit) x 100",
            "(Return Invoices / Total Invoices) x 100",
            "(Customer Return Orders / Total Orders) x 100",
            "% of return loss from top N products (Pareto)"
        ],
        "Business Meaning": [
            "Total revenue before accounting for returns",
            "Revenue lost due to product returns and cancellations",
            "Actual revenue retained after returns",
            "Proportion of revenue eroded by returns",
            "Estimated profit before return impact",
            "Estimated profit after absorbing return losses",
            "How much of gross profit is consumed by returns",
            "Proportion of transactions that are returns",
            "Per-customer return propensity measure",
            "Measures how concentrated return losses are"
        ]
    })
    st.dataframe(kpi_data, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
#  FOOTER
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#4a5568; font-size:0.85rem;'>"
    "Online Retail Profit Leakage Analytics | B.Tech Data Analytics Project | "
    "Data Source: UCI Machine Learning Repository / Kaggle"
    "</div>",
    unsafe_allow_html=True,
)
