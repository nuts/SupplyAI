"""
app.py — SupplyAI F&B Demand & Supply Planner
Main Streamlit entry point with 9 pages.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io, zipfile
from datetime import datetime
from pathlib import Path

from utils.data_loader import (
    load_sales, load_inventory, load_bom_excel,
    load_production_lines, load_marketing_calendar,
    make_placeholder_production_lines, make_placeholder_marketing_calendar,
    capture_bytes, validate_data,
    WAREHOUSE_LABELS, REGIONAL_WAREHOUSES, PRIMARY_WAREHOUSES, DC_SHIP_CADENCE
)
from utils.formatters import (
    fmt_units, fmt_lbs, fmt_pct, fmt_weeks, fmt_currency,
    tag_pill, confidence_tag, status_tag, trend_tag, mini_bar,
    allergen_badges, week_keys, week_short_labels
)
from utils.persistence import (
    save_to_cache, load_from_cache, get_cache_metadata, clear_cache, list_cached_files,
    save_forecast_snapshot, list_snapshots, load_snapshot, delete_snapshot,
    save_scenario, list_scenarios, load_scenario, delete_scenario,
)
from modules.forecast import (
    run_forecast, build_dc_forecast,
    CURRENT_WEEK, CURRENT_YEAR, get_current_week_year
)
from modules.production import (
    build_production_schedule, detect_changeover_conflicts, build_capacity_utilization
)
from modules.purchasing import (
    build_purchasing_plan, flag_bom_gaps, build_supplier_pos
)
from modules.dc_network import (
    build_dc_inventory_summary, build_network_summary, build_replenishment_transfers
)
from modules.accuracy import (
    compute_accuracy, aggregate_accuracy_by_category, overall_accuracy_metrics
)
from modules.alerts import generate_alerts


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG + GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SupplyAI Planner", page_icon="🌿",
    layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

* { font-family: 'DM Mono', monospace; }
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; }

.block-container { padding-top: 1.2rem !important; padding-bottom: 1rem !important; max-width: 1500px; }
[data-testid="stSidebar"] { background: #141614; }
[data-testid="stSidebar"] * { font-family: 'DM Mono', monospace; }

div[data-testid="stSidebarNav"] { display: none; }
[data-testid="stHeader"] { background: transparent; }

/* Metrics — denser */
[data-testid="metric-container"] {
  background: #141614; border: 1px solid #2a2e2a;
  border-radius: 8px; padding: 10px 14px;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
  font-size: 10px; color: #6b7a6b; text-transform: uppercase; letter-spacing: 0.07em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-family: 'Syne', sans-serif; font-size: 22px; color: #e8ede8;
}

/* Tables — dense */
.dense-table {
  width: 100%; border-collapse: collapse; font-size: 11.5px;
  background: #141614; border: 1px solid #2a2e2a; border-radius: 8px; overflow: hidden;
}
.dense-table thead th {
  background: #1a1d1a; color: #6b7a6b; font-size: 9.5px;
  text-transform: uppercase; letter-spacing: 0.07em;
  padding: 8px 10px; text-align: left; border-bottom: 1px solid #2a2e2a;
  white-space: nowrap; position: sticky; top: 0;
}
.dense-table tbody tr { border-bottom: 1px solid #2a2e2a; transition: background 0.12s; }
.dense-table tbody tr:hover { background: #1a1d1a; }
.dense-table tbody tr:last-child { border-bottom: none; }
.dense-table td { padding: 7px 10px; vertical-align: middle; color: #e8ede8; }
.dense-table tfoot { background: #1a1d1a; }
.dense-table tfoot td {
  padding: 9px 10px; font-weight: 600; color: #b8f542; border-top: 1px solid #2a2e2a;
}
.num { text-align: right; font-variant-numeric: tabular-nums; }

/* Wrap dense tables in scrollable container */
.tbl-wrap { max-height: 560px; overflow: auto; border-radius: 8px; }

/* Alert cards */
.alert-card {
  background: #141614; border: 1px solid #2a2e2a; border-radius: 7px;
  padding: 12px 14px; margin-bottom: 8px; display: flex; gap: 11px; align-items: flex-start;
}
.alert-card.crit { border-left: 3px solid #f54242; }
.alert-card.warn { border-left: 3px solid #f5a842; }
.alert-card.info { border-left: 3px solid #42f5a8; }
.alert-card .ic { font-size: 16px; line-height: 1; }
.alert-card .ttl { font-family: 'Syne', sans-serif; font-weight: 600; font-size: 12.5px; margin-bottom: 3px; color: #e8ede8; }
.alert-card .body { color: #9aaa9a; font-size: 11.5px; line-height: 1.5; }

/* Section headers — tighter */
.sh { display: flex; align-items: baseline; gap: 10px; margin-bottom: 12px; margin-top: 18px; }
.sh:first-child { margin-top: 0; }
.sh .t { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 15px; color: #e8ede8; }
.sh .s { color: #6b7a6b; font-size: 11px; }

/* Buttons */
.stButton > button {
  background: #b8f542; color: #0d0f0e; border: none; font-family: 'Syne', sans-serif;
  font-weight: 700; font-size: 13px; border-radius: 6px; transition: all 0.15s;
}
.stButton > button:hover { background: #ceff5a; transform: translateY(-1px); }

/* Selectbox / inputs */
[data-baseweb="select"] > div { background: #141614 !important; border-color: #2a2e2a !important; }
[data-baseweb="input"] > div  { background: #141614 !important; border-color: #2a2e2a !important; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
  font-family: 'DM Mono', monospace; font-size: 12px; padding: 8px 16px;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0d0f0e; }
::-webkit-scrollbar-thumb { background: #2a2e2a; border-radius: 3px; }

/* Reduce paragraph spacing globally */
p { margin-bottom: 0.4rem; }
hr { margin: 1rem 0 !important; border-color: #2a2e2a !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE + CACHE BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "sales": None, "inventory": None, "bom_data": None,
        "prod_lines": None, "marketing_cal": None,
        "forecast": None, "dc_forecast": None,
        "prod_df": None, "purch_detail": None, "purch_agg": None,
        "dc_summary": None, "transfers": None, "transfer_summary": None,
        "capacity_df": None, "alerts": [],
        "ran": False,
        "inv_max_seasons": {"Q1": 12, "Q2": 12, "Q3": 14, "Q4": 20},
        "scenario_params": {
            "forecast_uplift_pct": 0,
            "marketing_uplift_factor": 1.0,
        },
        "_initial_load_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _load_cached_data():
    """On first session boot, restore cached uploads from disk."""
    if st.session_state["_initial_load_done"]:
        return

    if st.session_state.sales is None:
        cached = load_from_cache("sales_raw")
        if cached is not None:
            try:
                st.session_state.sales = load_sales(cached)
            except Exception: pass

    if st.session_state.inventory is None:
        cached = load_from_cache("inventory_raw")
        if cached is not None:
            try:
                st.session_state.inventory = load_inventory(cached)
            except Exception: pass

    if st.session_state.bom_data is None:
        cached = load_from_cache("bom_raw")
        if cached is not None:
            try:
                st.session_state.bom_data = load_bom_excel(cached)
            except Exception: pass

    if st.session_state.prod_lines is None:
        cached = load_from_cache("prodlines_raw")
        if cached is not None:
            try:
                st.session_state.prod_lines = load_production_lines(cached)
            except Exception: pass

    if st.session_state.marketing_cal is None:
        cached = load_from_cache("mktg_raw")
        if cached is not None:
            try:
                st.session_state.marketing_cal = load_marketing_calendar(cached)
            except Exception: pass

    st.session_state["_initial_load_done"] = True


_init_state()
_load_cached_data()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<h2 style='color:#b8f542;font-family:Syne,sans-serif;margin-bottom:4px'>🌿 SupplyAI</h2>"
        "<div style='color:#6b7a6b;font-size:10px;margin-bottom:14px'>F&B Supply Planner</div>",
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "📈 Demand Forecast",
         "🏭 Production Plan",
         "📊 Capacity",
         "🛒 Purchasing Plan",
         "🏪 DC Network",
         "🚚 DC Transfers",
         "🎯 Forecast Accuracy",
         "🧪 Scenarios",
         "🚨 Alerts",
         "⚙️ Settings"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Data Status**")
    def _dot(b): return "🟢" if b else "🔴"
    st.markdown(
        f"{_dot(st.session_state.sales is not None)} Sales  \n"
        f"{_dot(st.session_state.inventory is not None)} Inventory  \n"
        f"{_dot(st.session_state.bom_data is not None)} BOM  \n"
        f"{_dot(st.session_state.prod_lines is not None)} Lines (placeholder OK)  \n"
        f"{_dot(st.session_state.marketing_cal is not None)} Mktg (placeholder OK)"
    )

    st.markdown("---")
    if st.session_state.ran:
        crit = sum(1 for a in st.session_state.alerts if a["severity"]=="critical")
        warn = sum(1 for a in st.session_state.alerts if a["severity"]=="warning")
        st.markdown(f"**Alerts:** 🔴 {crit} &nbsp; 🟡 {warn}")

    st.markdown(
        f"<div style='color:#3a3e3a;font-size:9px;margin-top:20px'>Current week: W{CURRENT_WEEK} · {CURRENT_YEAR}</div>",
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS — HTML TABLE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════
def render_html_table(headers, rows, footer=None):
    """Render a dense HTML table with optional totals footer."""
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        body += f"<tr>{cells}</tr>"
    foot = ""
    if footer:
        foot_cells = "".join(f"<td>{cell}</td>" for cell in footer)
        foot = f"<tfoot><tr>{foot_cells}</tr></tfoot>"

    html = f"""
    <div class="tbl-wrap">
      <table class="dense-table">
        <thead><tr>{head}</tr></thead>
        <tbody>{body}</tbody>
        {foot}
      </table>
    </div>
    """
    return html


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ANALYSIS (CORE PIPELINE)
# ═══════════════════════════════════════════════════════════════════════════════
def run_analysis(scenario_params=None):
    if (st.session_state.sales is None or st.session_state.inventory is None
        or st.session_state.bom_data is None):
        st.error("Sales, Inventory, and BOM are required.")
        return

    # Use placeholder production lines if not uploaded
    prod_lines = st.session_state.prod_lines
    if prod_lines is None:
        prod_lines = make_placeholder_production_lines(
            st.session_state.bom_data["sku_master"]
        )

    sp = scenario_params or st.session_state.scenario_params

    with st.spinner("Building demand forecast…"):
        fc = run_forecast(
            sales=st.session_state.sales,
            sku_master=st.session_state.bom_data["sku_master"],
            inventory=st.session_state.inventory,
            marketing_cal=st.session_state.marketing_cal,
            scenario_params=sp,
        )
        st.session_state.forecast = fc
        st.session_state.dc_forecast = build_dc_forecast(fc, st.session_state.inventory)

    with st.spinner("Building production schedule…"):
        prod = build_production_schedule(
            forecast=fc, inventory=st.session_state.inventory,
            production_lines=prod_lines, bom_data=st.session_state.bom_data,
            inv_max_by_season=st.session_state.inv_max_seasons,
        )
        st.session_state.prod_df = prod
        st.session_state.capacity_df = build_capacity_utilization(prod, prod_lines)

    with st.spinner("Building purchasing plan (multi-level BOM)…"):
        detail, agg = build_purchasing_plan(prod, st.session_state.bom_data)
        st.session_state.purch_detail = detail
        st.session_state.purch_agg = agg

    with st.spinner("Analysing DC network…"):
        dc_sum = build_dc_inventory_summary(
            inventory=st.session_state.inventory,
            forecast=fc, bom_data=st.session_state.bom_data,
        )
        st.session_state.dc_summary = dc_sum
        transfers, transfer_sum = build_replenishment_transfers(
            dc_sum, st.session_state.inventory, fc
        )
        st.session_state.transfers = transfers
        st.session_state.transfer_summary = transfer_sum

    with st.spinner("Generating alerts…"):
        gaps = flag_bom_gaps(fc, st.session_state.bom_data)
        conflicts = detect_changeover_conflicts(prod)
        alerts = generate_alerts(
            forecast=fc, prod_df=prod,
            dc_summary=dc_sum, transfers=transfers,
            bom_gaps=gaps, changeover_conflicts=conflicts,
            capacity_df=st.session_state.capacity_df,
            marketing_cal=st.session_state.marketing_cal,
        )
        st.session_state.alerts = alerts

    st.session_state.ran = True
    st.success(f"✓ Analysis complete · {len(fc)} variants · {len(alerts)} alerts")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown(
        "<h1 style='color:#b8f542;margin-bottom:4px'>SupplyAI</h1>"
        "<div style='color:#6b7a6b;font-size:12px;margin-bottom:18px'>"
        "F&B Demand &amp; Supply Planning Overview</div>",
        unsafe_allow_html=True
    )

    col_run, col_lock, col_info = st.columns([1, 1, 3])
    with col_run:
        if st.button("▶ Run Analysis", type="primary", use_container_width=True):
            run_analysis()
            st.rerun()
    with col_lock:
        if st.session_state.ran:
            if st.button("📌 Lock Forecast", use_container_width=True,
                            help="Save current forecast as snapshot for accuracy tracking"):
                # Add snapshot metadata
                snap_df = st.session_state.forecast.copy()
                snap_df["snapshot_week"] = CURRENT_WEEK
                snap_df["snapshot_year"] = CURRENT_YEAR
                snap_id = save_forecast_snapshot(snap_df, label=f"W{CURRENT_WEEK}")
                st.success(f"Locked: {snap_id}")

    if not st.session_state.ran:
        if st.session_state.sales is not None:
            st.info(f"📁 Cached data loaded · {len(st.session_state.sales):,} sales rows ready · "
                    "Click Run Analysis to refresh.")
        else:
            st.info("Upload data files in **⚙️ Settings**, then click **Run Analysis**.")
        st.stop()

    fc = st.session_state.forecast
    inv = st.session_state.inventory

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    total_8wk  = fc["total_8wk"].sum()
    high_conf  = (fc["confidence"] == "High").sum()
    new_skus   = (fc["yoy_status"] == "New").sum()
    crit_alerts = sum(1 for a in st.session_state.alerts if a["severity"]=="critical")
    primary_inv = inv[inv["warehouse"].isin(PRIMARY_WAREHOUSES)]["units_on_hand"].sum()
    avg_cover  = fc[fc["avg_weekly"] > 0]["weeks_cover"].clip(upper=20).mean()

    with c1: st.metric("8wk Forecast",   f"{int(total_8wk):,}")
    with c2: st.metric("Variants",        f"{len(fc):,}")
    with c3: st.metric("High Confidence", f"{high_conf}/{len(fc)}")
    with c4: st.metric("New SKUs",        f"{new_skus}")
    with c5: st.metric("NJ Inventory",    f"{int(primary_inv):,}")
    with c6: st.metric("Critical Alerts", str(crit_alerts),
                          delta_color="inverse" if crit_alerts > 0 else "normal")

    st.markdown("---")

    # Charts row
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='sh'><div class='t'>Top 12 SKUs by 8wk Forecast</div></div>", unsafe_allow_html=True)
        top12 = fc.nlargest(12, "total_8wk")
        fig = go.Figure(go.Bar(
            x=top12["total_8wk"], y=top12["product_name"].str[:35],
            orientation="h", marker_color="#b8f542",
            text=top12["total_8wk"].apply(lambda x: f"{int(x):,}"),
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="#0d0f0e", plot_bgcolor="#141614", font_color="#e8ede8",
            height=380, margin=dict(l=10, r=40, t=10, b=10),
            xaxis=dict(gridcolor="#2a2e2a"), yaxis=dict(gridcolor="#2a2e2a", autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='sh'><div class='t'>Forecast by Category L1</div></div>", unsafe_allow_html=True)
        by_cat = fc.groupby("cat_l1")["total_8wk"].sum().reset_index().sort_values("total_8wk", ascending=False)
        fig2 = go.Figure(go.Bar(
            x=by_cat["cat_l1"], y=by_cat["total_8wk"],
            marker_color=["#b8f542","#42f5a8","#f5a842","#f54242","#42b8f5","#a842f5"][:len(by_cat)],
            text=by_cat["total_8wk"].apply(lambda x: f"{int(x):,}"), textposition="outside",
        ))
        fig2.update_layout(
            paper_bgcolor="#0d0f0e", plot_bgcolor="#141614", font_color="#e8ede8",
            height=380, margin=dict(l=10, r=20, t=10, b=10),
            xaxis=dict(gridcolor="#2a2e2a"), yaxis=dict(gridcolor="#2a2e2a"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Top alerts
    if st.session_state.alerts:
        st.markdown("<div class='sh'><div class='t'>Top Alerts</div></div>", unsafe_allow_html=True)
        for a in st.session_state.alerts[:5]:
            sev = a["severity"]
            icon = "🔴" if sev=="critical" else "🟡" if sev=="warning" else "🔵"
            st.markdown(
                f'<div class="alert-card {sev[:4]}"><div class="ic">{icon}</div>'
                f'<div><div class="ttl">{a["title"]}</div><div class="body">{a["body"]}</div></div></div>',
                unsafe_allow_html=True
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DEMAND FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Demand Forecast":
    st.markdown("<h2 style='color:#b8f542;margin-bottom:6px'>📈 Demand Forecast</h2>", unsafe_allow_html=True)

    if not st.session_state.ran:
        st.warning("Run Analysis from the Overview page first.")
        st.stop()

    fc = st.session_state.forecast
    wk_short = week_short_labels(CURRENT_WEEK)

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Total 8wk Units", f"{int(fc['total_8wk'].sum()):,}")
    with c2: st.metric("Variants", f"{len(fc):,}")
    with c3: st.metric("New SKUs", f"{(fc['yoy_status']=='New').sum()}")
    with c4: st.metric("Promo-Affected", f"{fc['has_promo'].sum()}")
    with c5:
        growing = (fc["yoy"].fillna(0) > 0.05).sum()
        st.metric("Growing", f"{growing}")

    # Filters
    st.markdown("<div class='sh'></div>", unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns([2, 2, 1.5, 2])
    with f1:
        cats = ["All"] + sorted(fc["cat_l1"].dropna().unique().tolist())
        cat_filter = st.selectbox("Category L1", cats, key="fc_cat")
    with f2:
        cat2s = ["All"] + sorted(fc["cat_l2"].dropna().unique().tolist())
        cat2_filter = st.selectbox("Category L2", cat2s, key="fc_cat2")
    with f3:
        conf_filter = st.selectbox("Confidence", ["All", "High", "Medium", "Low"])
    with f4:
        search = st.text_input("Search", placeholder="SKU or product name…")

    display = fc.copy()
    if cat_filter  != "All": display = display[display["cat_l1"] == cat_filter]
    if cat2_filter != "All": display = display[display["cat_l2"] == cat2_filter]
    if conf_filter != "All": display = display[display["confidence"] == conf_filter]
    if search:
        s = search.lower()
        display = display[
            display["variant_id"].str.lower().str.contains(s, na=False) |
            display["product_name"].str.lower().str.contains(s, na=False)
        ]

    st.markdown(f"<div style='color:#6b7a6b;font-size:11px;margin-bottom:8px'>"
                f"<b>{len(display)} variants</b> · Weeks {wk_short[0]}–{wk_short[-1]}</div>",
                unsafe_allow_html=True)

    # Build dense HTML table
    headers = ["Variant", "Product", "L1", "L2", "Recent 4wk Avg", "YoY"] + wk_short + ["8wk Total", "Conf"]
    rows = []
    max_total = display["total_8wk"].max() if len(display) else 1
    for _, r in display.iterrows():
        wfc = r["weekly_forecast"]
        yoy_html = trend_tag(r["yoy"]) if r["yoy_status"] == "OK" else trend_tag(None)
        wk_cells = [f"<span class='num'>{int(w):,}</span>" for w in wfc]
        total_html = (f"<span class='num' style='color:#b8f542;font-weight:600'>{int(r['total_8wk']):,}</span>"
                       f"{mini_bar(r['total_8wk'], max_total)}")
        rows.append([
            f"<span style='color:#6b7a6b;font-size:10.5px'>{r['variant_id']}</span>",
            f"<span style='font-weight:500'>{r['product_name'][:42]}</span>",
            tag_pill(r['cat_l1'], "#6b7a6b"),
            f"<span style='color:#6b7a6b;font-size:10.5px'>{r['cat_l2']}</span>",
            f"<span class='num'>{int(r['recent_4w_avg']):,}</span>",
            yoy_html,
            *wk_cells,
            total_html,
            confidence_tag(r["confidence"]),
        ])

    # Totals row
    if len(display) > 0:
        total_by_week = [display["weekly_forecast"].apply(lambda x: x[i] if i < len(x) else 0).sum()
                          for i in range(8)]
        footer = ["TOTAL", "", "", "", "",
                  f"<span class='num'>—</span>"] + \
                 [f"<span class='num'>{int(t):,}</span>" for t in total_by_week] + \
                 [f"<span class='num' style='color:#b8f542'>{int(display['total_8wk'].sum()):,}</span>", ""]
    else:
        footer = None

    st.markdown(render_html_table(headers, rows, footer), unsafe_allow_html=True)

    # Export
    csv_rows = []
    for _, r in display.iterrows():
        row = {
            "Variant ID": r["variant_id"], "Product": r["product_name"],
            "Cat L1": r["cat_l1"], "Cat L2": r["cat_l2"],
            "Recent 4wk Avg": r["recent_4w_avg"],
            "YoY (4w trailing)": (
                f"{round(r['yoy']*100,1)}%" if r["yoy_status"]=="OK" else "New"
            ),
            "Confidence": r["confidence"],
        }
        for i, wl in enumerate(wk_short):
            row[wl] = int(r["weekly_forecast"][i]) if i < len(r["weekly_forecast"]) else 0
        row["8wk Total"] = int(r["total_8wk"])
        csv_rows.append(row)
    csv = pd.DataFrame(csv_rows).to_csv(index=False)
    st.download_button("⬇ Export Forecast CSV", csv,
                          file_name=f"demand_forecast_W{CURRENT_WEEK}.csv", mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PRODUCTION PLAN
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏭 Production Plan":
    st.markdown("<h2 style='color:#b8f542;margin-bottom:6px'>🏭 Production Plan</h2>", unsafe_allow_html=True)

    if not st.session_state.ran:
        st.warning("Run Analysis from the Overview page first.")
        st.stop()

    prod = st.session_state.prod_df
    wk_short = week_short_labels(CURRENT_WEEK)

    needs = prod[prod["needs_production"] == True]
    total_lbs = prod["total_prod_lbs"].sum()
    n_runs = prod["n_runs"].sum()
    unassigned = (prod["line_name"].astype(str).str.contains("Unassigned")).sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Need Production", f"{len(needs)}")
    with c2: st.metric("Total Production", f"{int(total_lbs):,} lbs")
    with c3: st.metric("Runs Scheduled", f"{int(n_runs)}")
    with c4: st.metric("Unassigned to Line", f"{unassigned}")
    with c5: st.metric("Covered by Inv", f"{len(prod) - len(needs)}")

    # Filters
    f1, f2, f3 = st.columns([2, 2, 2])
    with f1:
        lines = ["All"] + sorted(prod["line_name"].dropna().unique().tolist())
        line_filter = st.selectbox("Line", lines)
    with f2:
        cats = ["All"] + sorted(prod["cat_l1"].dropna().unique().tolist())
        cat_filter = st.selectbox("Category", cats, key="prod_cat")
    with f3:
        only_need = st.checkbox("Show only needing production", value=True)

    display = prod.copy()
    if line_filter != "All": display = display[display["line_name"] == line_filter]
    if cat_filter  != "All": display = display[display["cat_l1"] == cat_filter]
    if only_need: display = display[display["needs_production"] == True]

    # Table
    headers = ["Variant", "Product", "Line", "Allergens", "On Hand", "Runs"] + wk_short + ["Total lbs", "End Inv"]
    rows = []
    max_lbs = max([max(p) for p in display["prod_by_week_lbs"] if p]) if len(display) else 1
    for _, r in display.iterrows():
        wk_cells = []
        for lbs in r["prod_by_week_lbs"]:
            if lbs > 0:
                wk_cells.append(
                    f"<span class='num'>{int(lbs):,}</span>{mini_bar(lbs, max_lbs)}"
                )
            else:
                wk_cells.append("<span style='color:#3a3e3a;font-size:10px'>—</span>")
        rows.append([
            f"<span style='color:#6b7a6b;font-size:10.5px'>{r['variant_id']}</span>",
            f"<span style='font-weight:500'>{r['product_name'][:38]}</span>",
            tag_pill(r["line_name"][:25], "#42b8f5"),
            allergen_badges(r["allergens"]),
            f"<span class='num'>{int(r['on_hand_units']):,}</span>",
            f"<span class='num'>{int(r['n_runs'])}</span>",
            *wk_cells,
            f"<span class='num' style='color:#b8f542;font-weight:600'>{int(r['total_prod_lbs']):,}</span>",
            f"<span class='num'>{int(r['ending_inv_units']):,}</span>",
        ])

    # Totals row
    if len(display) > 0:
        total_by_week = [
            display["prod_by_week_lbs"].apply(lambda x: x[i] if i < len(x) else 0).sum()
            for i in range(8)
        ]
        footer = ["TOTAL", "", "", "",
                  f"<span class='num'>{int(display['on_hand_units'].sum()):,}</span>",
                  f"<span class='num'>{int(display['n_runs'].sum())}</span>"] + \
                 [f"<span class='num'>{int(t):,}</span>" for t in total_by_week] + \
                 [f"<span class='num' style='color:#b8f542'>{int(display['total_prod_lbs'].sum()):,}</span>",
                  f"<span class='num'>{int(display['ending_inv_units'].sum()):,}</span>"]
    else:
        footer = None

    st.markdown(render_html_table(headers, rows, footer), unsafe_allow_html=True)

    # Run detail
    st.markdown("<div class='sh' style='margin-top:18px'><div class='t'>Run Detail</div></div>", unsafe_allow_html=True)
    selected_sku = st.selectbox("Select SKU", display["variant_id"].tolist() if len(display) else [])
    if selected_sku:
        sku_row = display[display["variant_id"] == selected_sku].iloc[0]
        runs = sku_row.get("runs", [])
        if runs:
            run_data = []
            for run in runs:
                run_data.append({
                    "Week": wk_short[run["week_idx"]] if run["week_idx"] < 8 else f"W+{run['week_idx']}",
                    "Run (lbs)": int(run["run_lbs"]),
                    "Run (units)": int(run["run_units"]),
                    "Covers": f"{run['covers_weeks']} wks",
                    "Inv After (lbs)": int(run["inv_after_lbs"]),
                    "Flag": run.get("flag") or "✓ OK",
                })
            st.dataframe(pd.DataFrame(run_data), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CAPACITY UTILIZATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Capacity":
    st.markdown("<h2 style='color:#b8f542;margin-bottom:6px'>📊 Capacity Utilization</h2>", unsafe_allow_html=True)

    if not st.session_state.ran:
        st.warning("Run Analysis from the Overview page first.")
        st.stop()

    cap = st.session_state.capacity_df
    wk_short = week_short_labels(CURRENT_WEEK)

    if cap is None or cap.empty:
        st.info("No production lines configured yet.")
        st.stop()

    # KPIs
    over_cap = (cap["peak_util"] > 100).sum()
    high_cap = ((cap["peak_util"] >= 85) & (cap["peak_util"] <= 100)).sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Lines", f"{len(cap)}")
    with c2: st.metric("Avg Utilization", f"{cap['avg_util'].mean():.0f}%")
    with c3: st.metric("Peak >100%", f"{over_cap}", delta_color="inverse")
    with c4: st.metric("Peak 85-100%", f"{high_cap}")

    # Heatmap
    st.markdown("<div class='sh'><div class='t'>Utilization Heatmap</div></div>", unsafe_allow_html=True)

    # Build matrix
    matrix = []
    for _, r in cap.iterrows():
        matrix.append(r["utilization_pct"])

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=wk_short,
        y=cap["line_name"].tolist(),
        colorscale=[
            [0.0, "#1a1d1a"], [0.3, "#42f5a8"], [0.6, "#b8f542"],
            [0.85, "#f5a842"], [1.0, "#f54242"]
        ],
        zmin=0, zmax=120,
        text=[[f"{v:.0f}%" for v in r] for r in matrix],
        texttemplate="%{text}",
        hovertemplate="Line: %{y}<br>Week: %{x}<br>Utilization: %{z}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#0d0f0e", plot_bgcolor="#141614", font_color="#e8ede8",
        height=max(280, 60 * len(cap)),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.markdown("<div class='sh'><div class='t'>Line Detail</div></div>", unsafe_allow_html=True)
    headers = ["Line", "Max lbs/wk", "# SKUs"] + wk_short + ["Avg Util", "Peak Util"]
    rows = []
    for _, r in cap.iterrows():
        cells = []
        for util in r["utilization_pct"]:
            colour = "#f54242" if util > 100 else "#f5a842" if util >= 85 else "#b8f542" if util >= 50 else "#6b7a6b"
            cells.append(f"<span class='num' style='color:{colour}'>{util:.0f}%</span>")
        peak_colour = "#f54242" if r["peak_util"] > 100 else "#f5a842" if r["peak_util"] >= 85 else "#b8f542"
        rows.append([
            f"<b>{r['line_name']}</b>",
            f"<span class='num'>{int(r['max_run_lbs']):,}</span>",
            f"<span class='num'>{r['n_skus']}</span>",
            *cells,
            f"<span class='num'>{r['avg_util']:.0f}%</span>",
            f"<span class='num' style='color:{peak_colour};font-weight:600'>{r['peak_util']:.0f}%</span>",
        ])
    st.markdown(render_html_table(headers, rows), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PURCHASING PLAN
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🛒 Purchasing Plan":
    st.markdown("<h2 style='color:#b8f542;margin-bottom:6px'>🛒 Purchasing Plan</h2>", unsafe_allow_html=True)

    if not st.session_state.ran:
        st.warning("Run Analysis from the Overview page first.")
        st.stop()

    agg = st.session_state.purch_agg
    wk_short = week_short_labels(CURRENT_WEEK)

    if agg is None or agg.empty:
        st.info("No purchasing data.")
        st.stop()

    raw = agg[agg["ingredient_type"] == "raw_material"]
    pkg = agg[agg["ingredient_type"] == "packaging"]

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Raw Material Lines", f"{len(raw)}")
    with c2: st.metric("Packaging Lines", f"{len(pkg)}")
    with c3: st.metric("Total Raw lbs", f"{int(raw['total_need'].sum()):,}")
    with c4: st.metric("Total Packaging", f"{int(pkg['total_need'].sum()):,}")

    type_filter = st.radio("Type", ["All", "Raw Material", "Packaging"], horizontal=True)
    display = agg.copy()
    if type_filter == "Raw Material": display = display[display["ingredient_type"] == "raw_material"]
    if type_filter == "Packaging":    display = display[display["ingredient_type"] == "packaging"]

    headers = ["Ingredient", "Type", "Unit", "# SKUs"] + wk_short + ["Total"]
    rows = []
    for _, r in display.iterrows():
        wn = r["weekly_need"]
        type_tag = tag_pill("RAW", "#b8f542") if r["ingredient_type"] == "raw_material" \
                                                else tag_pill("PKG", "#42b8f5")
        cells = [f"<span class='num'>{int(w):,}</span>" if w > 0
                 else "<span style='color:#3a3e3a'>—</span>" for w in wn]
        rows.append([
            f"<span style='font-weight:500'>{r['ingredient_name'][:50]}</span>"
            f"<span style='color:#3a3e3a;font-size:9.5px'> · {r['ingredient_id']}</span>",
            type_tag,
            f"<span style='color:#6b7a6b'>{r['unit']}</span>",
            f"<span class='num'>{r['n_source_skus']}</span>",
            *cells,
            f"<span class='num' style='color:#b8f542;font-weight:600'>{int(r['total_need']):,}</span>",
        ])

    if len(display) > 0:
        # Compute totals — by unit type since lbs and eaches don't sum together
        footer_cells = ["TOTAL", "", "", ""]
        for i in range(8):
            wt = sum([r["weekly_need"][i] if i < len(r["weekly_need"]) else 0
                      for _, r in display.iterrows()])
            footer_cells.append(f"<span class='num'>{int(wt):,}</span>")
        footer_cells.append(f"<span class='num' style='color:#b8f542'>{int(display['total_need'].sum()):,}</span>")
        footer = footer_cells
    else:
        footer = None

    st.markdown(render_html_table(headers, rows, footer), unsafe_allow_html=True)

    # PO Export
    st.markdown("<div class='sh' style='margin-top:18px'><div class='t'>Export PO Templates by Supplier</div></div>",
                  unsafe_allow_html=True)
    if st.button("📦 Generate Supplier PO Pack (ZIP)"):
        wk_keys_list = week_keys(CURRENT_WEEK, CURRENT_YEAR)
        pos = build_supplier_pos(agg, week_keys=wk_keys_list)

        if not pos:
            st.warning("No POs to generate.")
        else:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for sup, df in pos.items():
                    csv_data = df.to_csv(index=False)
                    zf.writestr(f"PO_{sup}.csv", csv_data)
                # Combined file
                combined = pd.concat([df.assign(supplier=sup) for sup, df in pos.items()])
                zf.writestr("ALL_POs_combined.csv", combined.to_csv(index=False))

            buf.seek(0)
            st.download_button("⬇ Download PO Pack",
                                  buf, file_name=f"PO_Pack_W{CURRENT_WEEK}.zip",
                                  mime="application/zip")
            st.success(f"✓ {len(pos)} supplier POs generated")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DC NETWORK
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏪 DC Network":
    st.markdown("<h2 style='color:#b8f542;margin-bottom:6px'>🏪 DC Network</h2>", unsafe_allow_html=True)

    if not st.session_state.ran:
        st.warning("Run Analysis from the Overview page first.")
        st.stop()

    dc_sum = st.session_state.dc_summary
    if dc_sum is None or dc_sum.empty:
        st.info("No regional DC data.")
        st.stop()

    net_sum = build_network_summary(dc_sum)

    # DC summary cards
    cols = st.columns(len(net_sum))
    for col, (_, row) in zip(cols, net_sum.iterrows()):
        with col:
            oos  = int(row.get("out_of_stock", 0))
            crit = int(row.get("critical", 0))
            col_val = "#f54242" if (oos + crit) > 0 else "#b8f542"
            avg = row.get("avg_weeks_cover", 0) or 0
            st.markdown(
                f"<div style='background:#141614;border:1px solid #2a2e2a;"
                f"border-radius:8px;padding:13px 15px'>"
                f"<div style='color:#6b7a6b;font-size:10px;text-transform:uppercase;letter-spacing:.07em'>"
                f"{row['warehouse_label']}</div>"
                f"<div style='font-family:Syne,sans-serif;font-weight:700;font-size:22px;color:{col_val}'>"
                f"{avg:.1f}wk</div>"
                f"<div style='font-size:10px;color:#6b7a6b;margin-top:3px'>"
                f"🔴 {oos} OOS · 🟡 {int(row.get('below_reorder', 0))} reord</div></div>",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # Filters
    f1, f2, f3 = st.columns([2, 2, 2])
    with f1:
        whs = ["All"] + sorted(dc_sum["warehouse_label"].dropna().unique().tolist())
        wh_filter = st.selectbox("Warehouse", whs, key="dc_wh")
    with f2:
        cats = ["All"] + sorted(dc_sum["cat_l1"].dropna().unique().tolist())
        cat_filter = st.selectbox("Category", cats, key="dc_cat")
    with f3:
        statuses = ["All", "Out of Stock", "Critical", "Below Reorder", "Low", "Adequate", "Healthy"]
        stat_filter = st.selectbox("Status", statuses)

    display = dc_sum.copy()
    if wh_filter  != "All": display = display[display["warehouse_label"] == wh_filter]
    if cat_filter != "All": display = display[display["cat_l1"] == cat_filter]
    if stat_filter != "All": display = display[display["status"] == stat_filter]

    headers = ["Variant", "Product", "DC", "On Hand", "Avg/wk", "Cover", "Target", "Replen Need", "Status"]
    rows = []
    for _, r in display.iterrows():
        cover_color = "#f54242" if r["weeks_cover"] < 2 else \
                       "#f5a842" if r["weeks_cover"] < 4 else "#b8f542"
        rows.append([
            f"<span style='color:#6b7a6b;font-size:10.5px'>{r['variant_id']}</span>",
            f"<span style='font-weight:500'>{r.get('product_name','')[:40]}</span>",
            tag_pill(r["warehouse_label"][:12], "#42b8f5"),
            f"<span class='num'>{int(r['units_on_hand']):,}</span>",
            f"<span class='num'>{r['dc_avg_weekly']:.0f}</span>",
            f"<span class='num' style='color:{cover_color}'>{fmt_weeks(r['weeks_cover'])}</span>",
            f"<span class='num'>{int(r['target_units']):,}</span>" if pd.notna(r.get("target_units")) else "<span style='color:#3a3e3a'>—</span>",
            f"<span class='num'>{int(r['replen_need']):,}</span>" if pd.notna(r.get("replen_need")) else "<span style='color:#3a3e3a'>—</span>",
            status_tag(r["status"]),
        ])
    st.markdown(render_html_table(headers, rows), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DC TRANSFERS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚚 DC Transfers":
    st.markdown("<h2 style='color:#b8f542;margin-bottom:6px'>🚚 Replenishment Transfer Plan</h2>", unsafe_allow_html=True)

    if not st.session_state.ran:
        st.warning("Run Analysis from the Overview page first.")
        st.stop()

    transfers = st.session_state.transfers
    summary   = st.session_state.transfer_summary

    if transfers is None or transfers.empty:
        st.success("✓ No transfers needed — all DCs are at or above target stock.")
        st.stop()

    # Network KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Transfers", f"{len(transfers)}")
    with c2: st.metric("Total Units to Move", f"{int(transfers['can_fulfill'].sum()):,}")
    with c3: st.metric("Urgent", f"{(transfers['priority']=='Urgent').sum()}", delta_color="inverse")
    with c4: st.metric("Shortages at Primary", f"{(transfers['shortage']>0).sum()}", delta_color="inverse")

    # Per-DC summary cards
    st.markdown("<div class='sh'><div class='t'>By Destination</div><div class='s'>FL/TX: 1 ship/wk · NV/IN: 2 ship/wk</div></div>",
                unsafe_allow_html=True)
    cols = st.columns(len(summary))
    for col, (_, row) in zip(cols, summary.iterrows()):
        with col:
            st.markdown(
                f"<div style='background:#141614;border:1px solid #2a2e2a;border-radius:8px;padding:13px 15px'>"
                f"<div style='color:#6b7a6b;font-size:10px;text-transform:uppercase'>{row['destination_label']}</div>"
                f"<div style='font-family:Syne;font-weight:700;font-size:20px;color:#b8f542'>"
                f"{int(row['total_units']):,} units</div>"
                f"<div style='font-size:10px;color:#6b7a6b;margin-top:3px'>"
                f"{row['total_skus']} SKUs · {row['shipments_per_wk']}x/wk</div>"
                f"<div style='font-size:10px;color:#f54242;margin-top:2px'>"
                f"{int(row['urgent_count'])} urgent</div></div>",
                unsafe_allow_html=True
            )

    # Filters
    st.markdown("---")
    f1, f2 = st.columns(2)
    with f1:
        dests = ["All"] + sorted(transfers["destination_label"].unique().tolist())
        dest_filter = st.selectbox("Destination DC", dests)
    with f2:
        prios = ["All"] + ["Urgent", "High", "Medium", "Low"]
        prio_filter = st.selectbox("Priority", prios)

    display = transfers.copy()
    if dest_filter != "All": display = display[display["destination_label"] == dest_filter]
    if prio_filter != "All": display = display[display["priority"] == prio_filter]

    # Table
    headers = ["Variant", "Product", "Destination", "Priority", "On Hand",
                "Target", "Need", "Primary Avail", "Can Fulfill", "Per Shipment", "Cover"]
    rows = []
    prio_colours = {"Urgent": "#f54242", "High": "#f5a842", "Medium": "#42b8f5", "Low": "#6b7a6b"}
    for _, r in display.iterrows():
        shortage_html = ""
        if r["shortage"] > 0:
            shortage_html = f" <span style='color:#f54242;font-size:10px'>(-{r['shortage']})</span>"
        rows.append([
            f"<span style='color:#6b7a6b;font-size:10.5px'>{r['variant_id']}</span>",
            f"<span style='font-weight:500'>{r.get('product_name','')[:36]}</span>",
            tag_pill(r["destination_label"][:12], "#42b8f5"),
            tag_pill(r["priority"], prio_colours.get(r["priority"], "#6b7a6b")),
            f"<span class='num'>{r['current_on_hand']:,}</span>",
            f"<span class='num'>{r['target_units']:,}</span>",
            f"<span class='num' style='color:#f5a842'>{r['transfer_need']:,}</span>",
            f"<span class='num'>{r['primary_avail']:,}</span>",
            f"<span class='num' style='color:#b8f542'>{r['can_fulfill']:,}</span>{shortage_html}",
            f"<span class='num'>{r['per_shipment']:,}</span>",
            f"<span class='num'>{fmt_weeks(r['weeks_cover'])}</span>",
        ])

    if len(display) > 0:
        footer = ["TOTAL", "", "", "",
                  f"<span class='num'>{int(display['current_on_hand'].sum()):,}</span>",
                  f"<span class='num'>{int(display['target_units'].sum()):,}</span>",
                  f"<span class='num'>{int(display['transfer_need'].sum()):,}</span>",
                  "",
                  f"<span class='num' style='color:#b8f542'>{int(display['can_fulfill'].sum()):,}</span>",
                  f"<span class='num'>{int(display['per_shipment'].sum()):,}</span>", ""]
    else:
        footer = None

    st.markdown(render_html_table(headers, rows, footer), unsafe_allow_html=True)

    # Export
    csv = display.to_csv(index=False)
    st.download_button("⬇ Export Transfer Plan CSV", csv,
                          file_name=f"transfers_W{CURRENT_WEEK}.csv", mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FORECAST ACCURACY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Forecast Accuracy":
    st.markdown("<h2 style='color:#b8f542;margin-bottom:6px'>🎯 Forecast Accuracy</h2>", unsafe_allow_html=True)

    snaps = list_snapshots()
    if not snaps:
        st.info("No forecast snapshots yet. On the Overview page, click "
                 "**📌 Lock Forecast** to capture the current forecast as a baseline. "
                 "Once a snapshot has aged at least 1 week, accuracy metrics will appear here.")
        st.stop()

    st.markdown(f"<div style='color:#6b7a6b;font-size:11px'>{len(snaps)} snapshot(s) saved</div>",
                unsafe_allow_html=True)

    # Snapshot picker
    snap_labels = [f"{s['snapshot_id']} ({s.get('saved_at','')[:10]})" for s in snaps]
    selected_idx = st.selectbox("Select snapshot to analyse",
                                 range(len(snaps)),
                                 format_func=lambda i: snap_labels[i])
    snap = snaps[selected_idx]
    snap_df = load_snapshot(snap["snapshot_id"])

    if snap_df is None or st.session_state.sales is None:
        st.warning("Cannot load snapshot or sales data.")
        st.stop()

    # Compute accuracy
    acc_df = compute_accuracy(snap_df, st.session_state.sales)

    if acc_df.empty:
        st.info("This snapshot is too recent — no actuals yet for the forecasted weeks.")
        if st.button("🗑 Delete Snapshot"):
            delete_snapshot(snap["snapshot_id"])
            st.success("Snapshot deleted.")
            st.rerun()
        st.stop()

    overall = overall_accuracy_metrics(acc_df)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("SKUs with Actuals", f"{overall['n_variants']}")
    with c2: st.metric("Weighted MAPE", f"{overall['weighted_mape']:.1f}%" if overall['weighted_mape'] else "—")
    with c3: st.metric("Weighted Bias", f"{overall['weighted_bias']:+.1f}%" if overall['weighted_bias'] else "—")
    with c4: st.metric("Median MAPE", f"{overall['median_mape']:.1f}%" if overall['median_mape'] else "—")

    # By category
    st.markdown("<div class='sh'><div class='t'>Accuracy by Category</div></div>", unsafe_allow_html=True)
    by_cat = aggregate_accuracy_by_category(acc_df)
    if not by_cat.empty:
        st.dataframe(by_cat, use_container_width=True, hide_index=True)

    # Worst forecasted SKUs
    st.markdown("<div class='sh'><div class='t'>Top 20 Worst-Forecasted SKUs</div></div>", unsafe_allow_html=True)
    worst = acc_df.nlargest(20, "mape")
    st.dataframe(
        worst[["variant_id", "product_name", "cat_l1", "n_weeks",
               "total_forecast", "total_actual", "mape", "bias"]],
        use_container_width=True, hide_index=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Scenarios":
    st.markdown("<h2 style='color:#b8f542;margin-bottom:6px'>🧪 Scenario Modeling</h2>", unsafe_allow_html=True)

    if not st.session_state.ran:
        st.warning("Run a baseline analysis first from the Overview page.")
        st.stop()

    st.markdown("Adjust parameters, then click **Apply Scenario** to recompute the forecast. "
                 "Save scenarios for comparison.")

    # Capture baseline KPIs
    baseline_total  = st.session_state.forecast["total_8wk"].sum()
    baseline_alerts = sum(1 for a in st.session_state.alerts if a["severity"]=="critical")

    st.markdown("<div class='sh'><div class='t'>Scenario Parameters</div></div>", unsafe_allow_html=True)

    s1, s2 = st.columns(2)
    with s1:
        uplift = st.slider("Global Forecast Uplift (%)", -30, 50, 0, 5,
                            help="Adjust all forecasts by this percentage")
    with s2:
        mktg_factor = st.slider("Marketing Uplift Factor", 0.0, 2.0, 1.0, 0.1,
                                  help="Multiplier on marketing campaign uplifts (0 = ignore, 2 = double impact)")

    s3, s4 = st.columns(2)
    with s3:
        scenario_name = st.text_input("Scenario Name", value="My Scenario")
    with s4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▶ Apply Scenario", use_container_width=True):
            st.session_state.scenario_params = {
                "forecast_uplift_pct":      uplift,
                "marketing_uplift_factor":  mktg_factor,
            }
            run_analysis(st.session_state.scenario_params)
            st.rerun()

    # Show comparison
    if st.session_state.ran:
        st.markdown("---")
        st.markdown("<div class='sh'><div class='t'>Current vs Baseline</div></div>", unsafe_allow_html=True)
        current_total  = st.session_state.forecast["total_8wk"].sum()
        current_alerts = sum(1 for a in st.session_state.alerts if a["severity"]=="critical")

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("8wk Forecast", f"{int(current_total):,}",
                              delta=f"{int(current_total-baseline_total):+,} vs base")
        with c2: st.metric("Critical Alerts", current_alerts,
                              delta=f"{current_alerts - baseline_alerts:+d}",
                              delta_color="inverse")
        with c3:
            sp = st.session_state.scenario_params
            sp_summary = f"Uplift: {sp.get('forecast_uplift_pct',0):+}% · Mktg: {sp.get('marketing_uplift_factor',1):.1f}x"
            st.metric("Active Params", sp_summary)

    # Save / Load
    st.markdown("---")
    st.markdown("<div class='sh'><div class='t'>Saved Scenarios</div></div>", unsafe_allow_html=True)

    saved = list_scenarios()
    if not saved:
        st.markdown("<div style='color:#6b7a6b;font-size:11px'>No saved scenarios yet.</div>",
                      unsafe_allow_html=True)

    sa1, sa2 = st.columns([1, 3])
    with sa1:
        if st.button("💾 Save Current"):
            save_scenario(scenario_name, st.session_state.scenario_params,
                            {"total_8wk": int(st.session_state.forecast["total_8wk"].sum())})
            st.success(f"Saved: {scenario_name}")
            st.rerun()

    if saved:
        for s in saved:
            cols = st.columns([3, 2, 2, 1])
            cols[0].markdown(f"**{s['name']}**")
            cols[1].markdown(f"<span style='color:#6b7a6b;font-size:11px'>{s['saved_at'][:16]}</span>", unsafe_allow_html=True)
            cols[2].markdown(f"<span style='color:#b8f542;font-size:11px'>"
                              f"Uplift: {s['params'].get('forecast_uplift_pct',0):+}% · "
                              f"Mktg: {s['params'].get('marketing_uplift_factor',1):.1f}x</span>", unsafe_allow_html=True)
            with cols[3]:
                if st.button("Load", key=f"load_{s['name']}"):
                    st.session_state.scenario_params = s["params"]
                    run_analysis(s["params"])
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ALERTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Alerts":
    st.markdown("<h2 style='color:#b8f542;margin-bottom:6px'>🚨 Alerts &amp; Flags</h2>", unsafe_allow_html=True)

    if not st.session_state.ran:
        st.warning("Run Analysis from the Overview page first.")
        st.stop()

    alerts = st.session_state.alerts
    if not alerts:
        st.success("✓ No alerts — plan looks clean.")
        st.stop()

    crit  = [a for a in alerts if a["severity"]=="critical"]
    warns = [a for a in alerts if a["severity"]=="warning"]
    infos = [a for a in alerts if a["severity"]=="info"]

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("🔴 Critical", len(crit))
    with c2: st.metric("🟡 Warnings", len(warns))
    with c3: st.metric("🔵 Info", len(infos))

    cat_filter = st.selectbox("Filter by category",
                                ["All"] + sorted({a["category"] for a in alerts}))

    for a in alerts:
        if cat_filter != "All" and a["category"] != cat_filter:
            continue
        sev = a["severity"]
        icon = "🔴" if sev=="critical" else "🟡" if sev=="warning" else "🔵"
        st.markdown(
            f'<div class="alert-card {sev[:4]}"><div class="ic">{icon}</div>'
            f'<div><div class="ttl">[{a["category"]}] {a["title"]}</div>'
            f'<div class="body">{a["body"]}</div></div></div>',
            unsafe_allow_html=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.markdown("<h2 style='color:#b8f542;margin-bottom:6px'>⚙️ Settings &amp; Data</h2>", unsafe_allow_html=True)

    # Cache status
    st.markdown("<div class='sh'><div class='t'>Cached Data</div><div class='s'>Files persist across sessions</div></div>",
                unsafe_allow_html=True)

    cache_files = list_cached_files()
    if cache_files:
        for cf in cache_files:
            saved_at = cf.get("saved_at", "")
            filename = cf.get("filename", "")
            st.markdown(
                f"<div style='background:#141614;border:1px solid #2a2e2a;border-radius:6px;"
                f"padding:8px 12px;margin-bottom:5px;display:flex;justify-content:space-between;font-size:11px'>"
                f"<span style='color:#b8f542'>● {cf['key']}</span>"
                f"<span style='color:#6b7a6b'>{filename} · {saved_at[:16]}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        if st.button("🗑 Clear All Cache"):
            clear_cache()
            for k in ["sales", "inventory", "bom_data", "prod_lines", "marketing_cal",
                      "forecast", "prod_df", "purch_detail", "purch_agg", "dc_summary",
                      "transfers", "capacity_df", "alerts", "ran"]:
                if k in st.session_state:
                    if isinstance(st.session_state[k], (list, bool)):
                        st.session_state[k] = [] if isinstance(st.session_state[k], list) else False
                    else:
                        st.session_state[k] = None
            st.rerun()
    else:
        st.markdown("<div style='color:#6b7a6b;font-size:11px'>No cached files yet.</div>",
                      unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='sh'><div class='t'>Upload Files</div><div class='s'>Each upload is saved to cache</div></div>",
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Sales Export** (order_nexus CSV) ✳️ Required")
        sales_file = st.file_uploader("Sales", type=["csv"], label_visibility="collapsed", key="up_sales")
        if sales_file:
            try:
                raw_bytes = capture_bytes(sales_file)
                save_to_cache("sales_raw", raw_bytes, {"filename": sales_file.name})
                st.session_state.sales = load_sales(raw_bytes)
                st.success(f"✓ {len(st.session_state.sales):,} sales rows · cached")
            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("**Inventory Snapshot** (CSV) ✳️ Required")
        inv_file = st.file_uploader("Inventory", type=["csv"], label_visibility="collapsed", key="up_inv")
        if inv_file:
            try:
                raw_bytes = capture_bytes(inv_file)
                save_to_cache("inventory_raw", raw_bytes, {"filename": inv_file.name})
                st.session_state.inventory = load_inventory(raw_bytes)
                st.success(f"✓ {len(st.session_state.inventory):,} inventory rows · cached")
            except Exception as e:
                st.error(f"Error: {e}")

    with col2:
        st.markdown("**SKU Master with BOM** (.xlsx) ✳️ Required")
        bom_file = st.file_uploader("BOM", type=["xlsx"], label_visibility="collapsed", key="up_bom")
        if bom_file:
            try:
                raw_bytes = capture_bytes(bom_file)
                save_to_cache("bom_raw", raw_bytes, {"filename": bom_file.name})
                st.session_state.bom_data = load_bom_excel(raw_bytes)
                n_sku = len(st.session_state.bom_data["sku_master"])
                n_bom = len(st.session_state.bom_data["bom"])
                st.success(f"✓ {n_sku} SKUs, {n_bom} BOM lines · cached")
            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("**Production Lines** (CSV) — uses placeholder if missing")
        lines_file = st.file_uploader("Lines", type=["csv"], label_visibility="collapsed", key="up_lines")
        if lines_file:
            try:
                raw_bytes = capture_bytes(lines_file)
                save_to_cache("prodlines_raw", raw_bytes, {"filename": lines_file.name})
                st.session_state.prod_lines = load_production_lines(raw_bytes)
                st.success(f"✓ {len(st.session_state.prod_lines)} lines · cached")
            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("**Marketing Calendar** (CSV) — uses placeholder if missing")
        mktg_file = st.file_uploader("Mktg", type=["csv"], label_visibility="collapsed", key="up_mktg")
        if mktg_file:
            try:
                raw_bytes = capture_bytes(mktg_file)
                save_to_cache("mktg_raw", raw_bytes, {"filename": mktg_file.name})
                st.session_state.marketing_cal = load_marketing_calendar(raw_bytes)
                st.success(f"✓ {len(st.session_state.marketing_cal)} events · cached")
            except Exception as e:
                st.error(f"Error: {e}")

    # Validation
    if (st.session_state.sales is not None and st.session_state.inventory is not None
        and st.session_state.bom_data is not None):
        st.markdown("---")
        st.markdown("<div class='sh'><div class='t'>Data Validation</div></div>", unsafe_allow_html=True)
        issues = validate_data(st.session_state.sales, st.session_state.inventory, st.session_state.bom_data)
        if not issues:
            st.success("✓ All data validated.")
        for issue in issues:
            lvl = issue["level"]
            if lvl == "critical": st.error(f"🔴 {issue['message']}")
            elif lvl == "warning": st.warning(f"🟡 {issue['message']}")
            else: st.info(f"🔵 {issue['message']}")

    st.markdown("---")
    st.markdown("<div class='sh'><div class='t'>Seasonal Inventory Targets</div>"
                "<div class='s'>Max weeks of inventory by season</div></div>", unsafe_allow_html=True)
    sc1, sc2, sc3, sc4 = st.columns(4)
    seasons = st.session_state.inv_max_seasons
    with sc1: seasons["Q1"] = st.number_input("Q1 Jan-Mar (wks)", 4, 52, int(seasons["Q1"]))
    with sc2: seasons["Q2"] = st.number_input("Q2 Apr-Jun (wks)", 4, 52, int(seasons["Q2"]))
    with sc3: seasons["Q3"] = st.number_input("Q3 Jul-Sep (wks)", 4, 52, int(seasons["Q3"]))
    with sc4: seasons["Q4"] = st.number_input("Q4 Oct-Dec (wks)", 4, 52, int(seasons["Q4"]))
    st.session_state.inv_max_seasons = seasons

    # Templates
    st.markdown("---")
    st.markdown("<div class='sh'><div class='t'>File Templates</div></div>", unsafe_allow_html=True)

    line_template = pd.DataFrame({
        "line_name":    ["Line 1 - Bagging", "Line 2 - Mixing", "Line 3 - Chocolate"],
        "min_run_lbs":  [200, 300, 250],
        "max_run_lbs":  [3000, 5000, 2500],
        "sku_ids":      ["1007-013R;1015-010R", "9012-005R", "5067-030R"],
    })
    st.download_button("⬇ Production Lines Template", line_template.to_csv(index=False),
                          file_name="production_lines_template.csv", mime="text/csv")

    mktg_template = pd.DataFrame({
        "event_name":       ["Summer BBQ", "Holiday Gifting"],
        "event_type":       ["seasonal", "seasonal"],
        "start_date":       ["2026-05-25", "2026-11-01"],
        "end_date":         ["2026-07-04", "2026-12-26"],
        "affected_sku_ids": ["1007;1015", "1029;5067"],
        "uplift_percent":   [25, 50],
    })
    st.download_button("⬇ Marketing Calendar Template", mktg_template.to_csv(index=False),
                          file_name="marketing_calendar_template.csv", mime="text/csv")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;color:#2a2e2a;font-size:9px;margin-top:30px'>"
    f"SupplyAI · F&B Demand &amp; Supply Planner · v2 · W{CURRENT_WEEK} {CURRENT_YEAR}"
    "</div>",
    unsafe_allow_html=True
)
