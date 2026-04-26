"""
app.py
SupplyAI — F&B Demand & Supply Planner
Main Streamlit entry point.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from utils.data_loader import (
    load_sales, load_inventory, load_bom_excel,
    load_production_lines, load_marketing_calendar,
    validate_data, WAREHOUSE_LABELS, REGIONAL_WAREHOUSES
)
from utils.formatters import (
    fmt_units, fmt_lbs, fmt_pct, fmt_weeks, fmt_currency,
    coverage_colour, trend_colour, confidence_colour, cv_label,
    allergen_badges, delta_str, week_display_labels, week_labels,
    style_coverage, style_trend
)
from modules.forecast   import run_forecast, build_dc_forecast, CURRENT_WEEK, CURRENT_YEAR
from modules.production import build_production_schedule, detect_changeover_conflicts
from modules.purchasing import build_purchasing_plan, flag_bom_gaps
from modules.dc_network import build_dc_inventory_summary, build_network_summary
from modules.alerts     import generate_alerts

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SupplyAI Planner",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
  .metric-card {
    background: #141614; border: 1px solid #2a2e2a;
    border-radius: 8px; padding: 14px 16px; margin-bottom: 8px;
  }
  .metric-label { color: #6b7a6b; font-size: 11px; text-transform: uppercase;
                  letter-spacing: .07em; margin-bottom: 4px; }
  .metric-value { font-size: 22px; font-weight: 700; color: #e8ede8; }
  .metric-delta { font-size: 11px; margin-top: 2px; }
  .alert-crit { border-left: 3px solid #f54242; background: rgba(245,66,66,.06);
                border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; }
  .alert-warn { border-left: 3px solid #f5a842; background: rgba(245,168,66,.06);
                border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; }
  .alert-info { border-left: 3px solid #42f5a8; background: rgba(66,245,168,.06);
                border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; }
  .alert-title { font-weight: 600; font-size: 13px; margin-bottom: 3px; }
  .alert-body  { color: #9aaa9a; font-size: 12px; line-height: 1.5; }
  .status-pill {
    display: inline-block; border-radius: 12px; padding: 2px 10px;
    font-size: 11px; font-weight: 500;
  }
  div[data-testid="stSidebarNav"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "sales":           None,
        "inventory":       None,
        "bom_data":        None,
        "prod_lines":      None,
        "marketing_cal":   None,
        "forecast":        None,
        "dc_forecast":     None,
        "prod_df":         None,
        "purch_detail":    None,
        "purch_agg":       None,
        "dc_summary":      None,
        "alerts":          [],
        "ran":             False,
        "inv_max_seasons": {"Q1": 12, "Q2": 12, "Q3": 14, "Q4": 20},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 SupplyAI")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📈 Demand Forecast", "🏭 Production Plan",
         "🛒 Purchasing Plan", "🏪 DC Network", "🚨 Alerts", "⚙️ Settings"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Data status
    st.markdown("**Data Status**")
    def _dot(loaded): return "🟢" if loaded else "🔴"

    st.markdown(
        f"{_dot(st.session_state.sales is not None)} Sales  \n"
        f"{_dot(st.session_state.inventory is not None)} Inventory  \n"
        f"{_dot(st.session_state.bom_data is not None)} BOM / SKU Master  \n"
        f"{_dot(st.session_state.prod_lines is not None)} Production Lines  \n"
        f"{_dot(st.session_state.marketing_cal is not None)} Marketing Calendar"
    )

    st.markdown("---")

    if st.session_state.ran:
        crit = sum(1 for a in st.session_state.alerts if a["severity"] == "critical")
        warn = sum(1 for a in st.session_state.alerts if a["severity"] == "warning")
        st.markdown(f"**Alerts:** 🔴 {crit} critical  &nbsp; 🟡 {warn} warnings")


# ── Run analysis button (available on Overview) ────────────────────────────────
def run_analysis():
    if st.session_state.sales is None:
        st.error("Sales data required to run analysis.")
        return
    if st.session_state.inventory is None:
        st.error("Inventory data required to run analysis.")
        return
    if st.session_state.bom_data is None:
        st.error("BOM / SKU Master required to run analysis.")
        return

    with st.spinner("Building demand forecast…"):
        fc = run_forecast(
            sales         = st.session_state.sales,
            sku_master    = st.session_state.bom_data["sku_master"],
            inventory     = st.session_state.inventory,
            marketing_cal = st.session_state.marketing_cal,
            inv_max_weeks_by_season = st.session_state.inv_max_seasons,
        )
        st.session_state.forecast = fc
        dc_fc = build_dc_forecast(fc, st.session_state.inventory)
        st.session_state.dc_forecast = dc_fc

    with st.spinner("Building production schedule…"):
        prod_df = build_production_schedule(
            forecast          = fc,
            inventory         = st.session_state.inventory,
            production_lines  = st.session_state.prod_lines,
            bom_data          = st.session_state.bom_data,
            inv_max_by_season = st.session_state.inv_max_seasons,
        )
        st.session_state.prod_df = prod_df

    with st.spinner("Building purchasing plan…"):
        detail, agg = build_purchasing_plan(prod_df, st.session_state.bom_data)
        st.session_state.purch_detail = detail
        st.session_state.purch_agg   = agg

    with st.spinner("Analysing DC network…"):
        dc_sum = build_dc_inventory_summary(
            inventory  = st.session_state.inventory,
            forecast   = fc,
            bom_data   = st.session_state.bom_data,
        )
        st.session_state.dc_summary = dc_sum

    with st.spinner("Generating alerts…"):
        bom_gaps    = flag_bom_gaps(fc, st.session_state.bom_data)
        conflicts   = detect_changeover_conflicts(prod_df)
        alerts      = generate_alerts(
            forecast             = fc,
            prod_df              = prod_df,
            dc_summary           = dc_sum if dc_sum is not None else pd.DataFrame(),
            bom_gaps             = bom_gaps,
            changeover_conflicts = conflicts,
            marketing_cal        = st.session_state.marketing_cal,
        )
        st.session_state.alerts = alerts

    st.session_state.ran = True
    st.success(f"Analysis complete — {len(fc)} variants forecast, "
               f"{len(alerts)} alerts generated.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🌿 SupplyAI — Supply Planning Overview")

    col_run, col_info = st.columns([1, 3])
    with col_run:
        if st.button("▶ Run Analysis", type="primary", use_container_width=True):
            run_analysis()

    if not st.session_state.ran:
        st.info("Upload your data files in **⚙️ Settings**, then click **Run Analysis**.")

        # Show data requirements
        st.markdown("### Required Files")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**Core (required)**
- `order_nexus` sales export (CSV)
- Inventory snapshot (CSV)
- SKU master with BOM (Excel)
""")
        with col2:
            st.markdown("""
**Optional (enhances plan)**
- Production lines (CSV)
- Marketing calendar (CSV)
""")
        st.stop()

    fc  = st.session_state.forecast
    inv = st.session_state.inventory

    # ── KPI row ────────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    total_8wk   = fc["total_8wk"].sum()
    high_conf   = (fc["confidence"] == "High").sum()
    crit_alerts = sum(1 for a in st.session_state.alerts if a["severity"] == "critical")
    zero_inv    = (fc["inv_units"] == 0).sum()
    total_inv   = inv[~inv["warehouse"].isin(["none"])]["units_on_hand"].sum()
    avg_cover   = fc[fc["avg_weekly"] > 0]["weeks_cover"].clip(upper=20).mean()

    with c1:
        st.metric("8wk Forecast (units)", f"{int(total_8wk):,}")
    with c2:
        st.metric("SKUs Forecasted", f"{len(fc):,}")
    with c3:
        st.metric("High Confidence", f"{high_conf} / {len(fc)}")
    with c4:
        st.metric("Total Inventory", f"{int(total_inv):,}")
    with c5:
        st.metric("Avg Coverage", f"{avg_cover:.1f} wks")
    with c6:
        st.metric("Critical Alerts", str(crit_alerts),
                  delta="action needed" if crit_alerts > 0 else "all clear",
                  delta_color="inverse" if crit_alerts > 0 else "normal")

    st.markdown("---")

    # ── Top SKUs chart ─────────────────────────────────────────────────────────
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### Top 15 SKUs by 8wk Forecast")
        top15 = fc.nlargest(15, "total_8wk")
        fig = go.Figure(go.Bar(
            x=top15["total_8wk"],
            y=top15["variant_id"],
            orientation="h",
            marker_color="#b8f542",
            text=top15["total_8wk"].apply(lambda x: f"{int(x):,}"),
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="#0d0f0e", plot_bgcolor="#141614",
            font_color="#e8ede8", height=420,
            margin=dict(l=10, r=40, t=10, b=10),
            xaxis=dict(gridcolor="#2a2e2a"),
            yaxis=dict(gridcolor="#2a2e2a", autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_chart2:
        st.markdown("#### Forecast by Category L1")
        by_cat = fc.groupby("cat_l1")["total_8wk"].sum().reset_index()
        by_cat = by_cat.sort_values("total_8wk", ascending=False)
        fig2 = go.Figure(go.Bar(
            x=by_cat["cat_l1"],
            y=by_cat["total_8wk"],
            marker_color=["#b8f542","#42f5a8","#f5a842","#f54242","#42b8f5"],
            text=by_cat["total_8wk"].apply(lambda x: f"{int(x):,}"),
            textposition="outside",
        ))
        fig2.update_layout(
            paper_bgcolor="#0d0f0e", plot_bgcolor="#141614",
            font_color="#e8ede8", height=420,
            margin=dict(l=10, r=20, t=10, b=10),
            xaxis=dict(gridcolor="#2a2e2a"),
            yaxis=dict(gridcolor="#2a2e2a"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Recent alerts summary ──────────────────────────────────────────────────
    if st.session_state.alerts:
        st.markdown("#### Top Alerts")
        for alert in st.session_state.alerts[:5]:
            sev   = alert["severity"]
            cls   = f"alert-{sev[:4]}"
            icon  = "🔴" if sev == "critical" else "🟡" if sev == "warning" else "🔵"
            st.markdown(
                f'<div class="{cls}">'
                f'<div class="alert-title">{icon} {alert["title"]}</div>'
                f'<div class="alert-body">{alert["body"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DEMAND FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Demand Forecast":
    st.title("📈 Demand Forecast")

    if not st.session_state.ran:
        st.warning("Run Analysis from the Overview page first.")
        st.stop()

    fc = st.session_state.forecast
    wk_labels  = week_display_labels(CURRENT_WEEK, CURRENT_YEAR)
    wk_keys    = week_labels(CURRENT_WEEK, CURRENT_YEAR)

    # ── KPIs ───────────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("Total 8wk Units",    f"{int(fc['total_8wk'].sum()):,}")
    with c2: st.metric("Variants",           f"{len(fc):,}")
    with c3: st.metric("High Confidence",    f"{(fc['confidence']=='High').sum()}")
    with c4: st.metric("Promo-Affected",     f"{fc['has_promo'].sum()}")
    with c5:
        growing = (fc["yoy"] > 0.05).sum()
        st.metric("Growing Trend", f"{growing}", delta=f"{growing} variants >5% YoY")

    st.markdown("---")

    # ── Filters ────────────────────────────────────────────────────────────────
    f1, f2, f3, f4, f5 = st.columns([2, 2, 1.5, 1.5, 1])
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
    with f5:
        st.markdown("<br>", unsafe_allow_html=True)
        show_volatile = st.checkbox("Volatile only", value=False)

    # Apply filters
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
    if show_volatile:
        display = display[display["cv"] > 0.6]

    st.markdown(f"**{len(display)} variants** · Weeks {wk_labels[0]}–{wk_labels[-1]}")

    # ── Table ──────────────────────────────────────────────────────────────────
    # Build display dataframe
    rows = []
    for _, r in display.iterrows():
        yoy_pct = round(r["yoy"] * 100, 1)
        yoy_str = f"+{yoy_pct}%" if yoy_pct >= 0 else f"{yoy_pct}%"
        cv_lbl, _ = cv_label(r.get("cv", 0))
        row = {
            "Variant":    r["variant_id"],
            "Product":    r["product_name"],
            "L1":         r.get("cat_l1", ""),
            "L2":         r.get("cat_l2", ""),
            "Format":     r.get("variant_name", ""),
            "Baseline":   int(r["baseline"]),
            "YoY":        yoy_str,
            "Variability": cv_lbl,
            "Inv Units":  int(r["inv_units"]),
            "Cover":      fmt_weeks(r["weeks_cover"]),
            "Confidence": r["confidence"],
            "8wk Total":  int(r["total_8wk"]),
        }
        for i, wl in enumerate(wk_labels):
            wfc = r["weekly_forecast"]
            row[wl] = int(wfc[i]) if isinstance(wfc, list) and i < len(wfc) else 0
        rows.append(row)

    tbl = pd.DataFrame(rows)

    st.dataframe(
        tbl,
        use_container_width=True,
        height=500,
        column_config={
            "YoY":        st.column_config.TextColumn("YoY Trend"),
            "Cover":      st.column_config.TextColumn("Inv Cover"),
            "8wk Total":  st.column_config.NumberColumn("8wk Total", format="%d"),
            **{wl: st.column_config.NumberColumn(wl, format="%d") for wl in wk_labels},
        }
    )

    # ── Export ─────────────────────────────────────────────────────────────────
    csv = tbl.to_csv(index=False)
    st.download_button(
        "⬇ Export Forecast CSV",
        csv,
        file_name=f"demand_forecast_{wk_keys[0]}_{wk_keys[-1]}.csv",
        mime="text/csv"
    )

    # ── Variability chart ──────────────────────────────────────────────────────
    if "cv" in fc.columns:
        st.markdown("---")
        st.markdown("#### Demand Variability Distribution")
        cv_data = fc["cv"].clip(upper=2.0)
        fig_cv = go.Figure(go.Histogram(
            x=cv_data, nbinsx=30,
            marker_color="#b8f542", opacity=0.8
        ))
        fig_cv.add_vline(x=0.3, line_dash="dash", line_color="#f5a842",
                         annotation_text="Moderate threshold")
        fig_cv.add_vline(x=0.6, line_dash="dash", line_color="#f54242",
                         annotation_text="Volatile threshold")
        fig_cv.update_layout(
            paper_bgcolor="#0d0f0e", plot_bgcolor="#141614",
            font_color="#e8ede8", height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(title="Coefficient of Variation", gridcolor="#2a2e2a"),
            yaxis=dict(title="# SKUs", gridcolor="#2a2e2a"),
        )
        st.plotly_chart(fig_cv, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PRODUCTION PLAN
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏭 Production Plan":
    st.title("🏭 Production Plan")

    if not st.session_state.ran:
        st.warning("Run Analysis from the Overview page first.")
        st.stop()

    prod = st.session_state.prod_df
    wk_labels = week_display_labels(CURRENT_WEEK, CURRENT_YEAR)
    wk_keys   = week_labels(CURRENT_WEEK, CURRENT_YEAR)

    # ── KPIs ───────────────────────────────────────────────────────────────────
    needs_prod     = prod[prod["needs_production"] == True]
    total_lbs      = prod["total_prod_lbs"].sum()
    unassigned_ct  = (prod["line_name"] == "Unassigned").sum()
    n_runs_total   = prod["n_runs"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("SKUs Needing Production", f"{len(needs_prod)}")
    with c2: st.metric("Total Production (lbs)",  f"{int(total_lbs):,}")
    with c3: st.metric("Total Runs Scheduled",    f"{int(n_runs_total)}")
    with c4: st.metric("Unassigned to Line",       f"{unassigned_ct}")
    with c5: st.metric("SKUs Covered by Inv",
                       f"{len(prod) - len(needs_prod)}")

    if unassigned_ct > 0:
        st.warning(f"⚠️ {unassigned_ct} SKUs have no production line assigned. "
                   "Upload production_lines.csv in Settings.")

    st.markdown("---")

    # ── Filters ────────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns([2, 2, 2])
    with f1:
        lines = ["All"] + sorted(prod["line_name"].dropna().unique().tolist())
        line_filter = st.selectbox("Line", lines)
    with f2:
        cats = ["All"] + sorted(prod["cat_l1"].dropna().unique().tolist())
        cat_filter = st.selectbox("Category", cats, key="prod_cat")
    with f3:
        only_prod = st.checkbox("Show only SKUs needing production", value=True)

    display = prod.copy()
    if line_filter != "All":  display = display[display["line_name"] == line_filter]
    if cat_filter  != "All":  display = display[display["cat_l1"] == cat_filter]
    if only_prod:             display = display[display["needs_production"] == True]

    # ── Production table ───────────────────────────────────────────────────────
    rows = []
    for _, r in display.iterrows():
        prod_wk = r.get("prod_by_week_lbs", [])
        row = {
            "Variant":     r["variant_id"],
            "Product":     r["product_name"],
            "Line":        r["line_name"],
            "Allergens":   ", ".join(r["allergens"]) if r["allergens"] else "None",
            "On Hand":     int(r["on_hand_units"]),
            "# Runs":      int(r["n_runs"]),
            "Total lbs":   round(r["total_prod_lbs"], 0),
            "End Inv":     int(r["ending_inv_units"]),
        }
        for i, wl in enumerate(wk_labels):
            lbs = prod_wk[i] if isinstance(prod_wk, list) and i < len(prod_wk) else 0
            row[wl] = round(lbs, 0) if lbs > 0 else None
        rows.append(row)

    tbl = pd.DataFrame(rows)
    st.dataframe(tbl, use_container_width=True, height=480)

    # ── Run detail expander ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Production Run Detail")
    selected_sku = st.selectbox(
        "Select SKU for run detail",
        display["variant_id"].tolist()
    )
    if selected_sku:
        sku_row = display[display["variant_id"] == selected_sku].iloc[0]
        runs    = sku_row.get("runs", [])
        if runs:
            run_rows = []
            for run in runs:
                run_rows.append({
                    "Week":          wk_labels[run["week_idx"]] if run["week_idx"] < len(wk_labels) else f"W{run['week_idx']+1}",
                    "Run (lbs)":     run["run_lbs"],
                    "Run (units)":   run["run_units"],
                    "Covers (weeks)": run["covers_weeks"],
                    "Inv After (lbs)": run["inv_after_lbs"],
                    "Flag":          run.get("flag", "✓") or "✓",
                })
            st.dataframe(pd.DataFrame(run_rows), use_container_width=True)
        else:
            st.info("No production runs scheduled — inventory covers full 8-week horizon.")

    csv_prod = tbl.to_csv(index=False)
    st.download_button("⬇ Export Production Plan CSV", csv_prod,
                       file_name="production_plan_8wk.csv", mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PURCHASING PLAN
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🛒 Purchasing Plan":
    st.title("🛒 Purchasing Plan")

    if not st.session_state.ran:
        st.warning("Run Analysis from the Overview page first.")
        st.stop()

    agg  = st.session_state.purch_agg
    wk_labels = week_display_labels(CURRENT_WEEK, CURRENT_YEAR)

    if agg is None or agg.empty:
        st.info("No purchasing data — ensure BOM is loaded and production plan has been run.")
        st.stop()

    # ── KPIs ───────────────────────────────────────────────────────────────────
    raw   = agg[agg["ingredient_type"] == "raw_material"]
    pkg   = agg[agg["ingredient_type"] == "packaging"]
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Raw Material Lines",  f"{len(raw)}")
    with c2: st.metric("Packaging Lines",     f"{len(pkg)}")
    with c3: st.metric("Total Raw Mat (lbs)", f"{int(raw['total_need'].sum()):,}")

    st.markdown("---")

    # ── Filters ────────────────────────────────────────────────────────────────
    type_filter = st.radio("Type", ["All", "Raw Material", "Packaging"], horizontal=True)
    display = agg.copy()
    if type_filter == "Raw Material": display = display[display["ingredient_type"] == "raw_material"]
    if type_filter == "Packaging":    display = display[display["ingredient_type"] == "packaging"]

    # ── Table ──────────────────────────────────────────────────────────────────
    rows = []
    for _, r in display.iterrows():
        wn = r["weekly_need"]
        row = {
            "Ingredient ID":   r["ingredient_id"],
            "Ingredient":      r["ingredient_name"],
            "Type":            r["ingredient_type"].replace("_", " ").title(),
            "Unit":            r["unit"],
            "# Source SKUs":   r["n_source_skus"],
            f"Total ({r['unit']})": round(r["total_need"], 1),
        }
        for i, wl in enumerate(wk_labels):
            val = wn[i] if isinstance(wn, list) and i < len(wn) else 0
            row[wl] = round(val, 1) if val > 0 else None
        rows.append(row)

    tbl = pd.DataFrame(rows)
    st.dataframe(tbl, use_container_width=True, height=500)

    csv_purch = tbl.to_csv(index=False)
    st.download_button("⬇ Export Purchasing Plan CSV", csv_purch,
                       file_name="purchasing_plan_8wk.csv", mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DC NETWORK
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏪 DC Network":
    st.title("🏪 DC Network Inventory")

    if not st.session_state.ran:
        st.warning("Run Analysis from the Overview page first.")
        st.stop()

    dc_sum  = st.session_state.dc_summary
    if dc_sum is None or dc_sum.empty:
        st.info("No DC data available. Ensure inventory snapshot includes regional warehouses.")
        st.stop()

    net_sum = build_network_summary(dc_sum)

    # ── Network summary cards ──────────────────────────────────────────────────
    st.markdown("#### Network Summary")
    cols = st.columns(len(net_sum))
    status_colours = {
        "Out of Stock": "#f54242", "Critical": "#f54242",
        "Below Reorder": "#f5a842", "Low": "#f5a842",
        "Adequate": "#b8f542", "Healthy": "#42f5a8"
    }
    for col, (_, row) in zip(cols, net_sum.iterrows()):
        with col:
            oos  = int(row.get("out_of_stock", 0))
            crit = int(row.get("critical", 0))
            col_val = "#f54242" if (oos + crit) > 0 else "#b8f542"
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{row['warehouse_label']}</div>"
                f"<div class='metric-value' style='color:{col_val}'>{row['avg_weeks_cover']:.1f} wks</div>"
                f"<div class='metric-delta'>🔴 {oos} OOS &nbsp; 🟡 {int(row.get('below_reorder',0))} reorder</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── Filters ────────────────────────────────────────────────────────────────
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

    # ── Table ──────────────────────────────────────────────────────────────────
    rows = []
    for _, r in display.iterrows():
        rows.append({
            "Variant":      r["variant_id"],
            "Product":      r.get("product_name", ""),
            "DC":           r["warehouse_label"],
            "On Hand":      int(r["units_on_hand"]),
            "DC Avg/wk":    round(r["dc_avg_weekly"], 1),
            "8wk Demand":   round(r["dc_total_8wk"], 0),
            "Weeks Cover":  fmt_weeks(r["weeks_cover"]),
            "Target Units": int(r["target_units"]) if pd.notna(r.get("target_units")) else "—",
            "Replen Need":  int(r["replen_need"]) if pd.notna(r.get("replen_need")) else "—",
            "Status":       r["status"],
        })

    tbl = pd.DataFrame(rows)
    st.dataframe(tbl, use_container_width=True, height=500)

    csv_dc = tbl.to_csv(index=False)
    st.download_button("⬇ Export DC Report CSV", csv_dc,
                       file_name="dc_network_report.csv", mime="text/csv")

    # ── Coverage heatmap ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Coverage Heatmap (top 40 SKUs by volume)")
    from modules.dc_network import build_dc_heatmap
    if st.session_state.forecast is not None:
        top_skus = (
            st.session_state.forecast
            .nlargest(40, "total_8wk")["variant_id"]
            .tolist()
        )
        hmap_data = dc_sum[dc_sum["variant_id"].isin(top_skus)]
        pivot = hmap_data.pivot_table(
            index="variant_id", columns="warehouse_label",
            values="weeks_cover", aggfunc="first"
        ).fillna(0).clip(upper=20)

        if not pivot.empty:
            fig_hm = go.Figure(go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale=[
                    [0.0, "#f54242"], [0.1, "#f54242"],
                    [0.2, "#f5a842"], [0.4, "#f5d442"],
                    [0.6, "#b8f542"], [1.0, "#42f5a8"]
                ],
                zmin=0, zmax=20,
                text=pivot.values.round(1),
                texttemplate="%{text}",
                hovertemplate="SKU: %{y}<br>DC: %{x}<br>Weeks: %{z}<extra></extra>",
            ))
            fig_hm.update_layout(
                paper_bgcolor="#0d0f0e", plot_bgcolor="#141614",
                font_color="#e8ede8", height=600,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig_hm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ALERTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Alerts":
    st.title("🚨 Alerts & Flags")

    if not st.session_state.ran:
        st.warning("Run Analysis from the Overview page first.")
        st.stop()

    alerts = st.session_state.alerts
    if not alerts:
        st.success("No alerts — plan looks clean.")
        st.stop()

    crit  = [a for a in alerts if a["severity"] == "critical"]
    warns = [a for a in alerts if a["severity"] == "warning"]
    infos = [a for a in alerts if a["severity"] == "info"]

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("🔴 Critical", len(crit))
    with c2: st.metric("🟡 Warnings", len(warns))
    with c3: st.metric("🔵 Info",     len(infos))

    cat_filter = st.selectbox(
        "Filter by category",
        ["All"] + sorted({a["category"] for a in alerts})
    )

    for alert in alerts:
        if cat_filter != "All" and alert["category"] != cat_filter:
            continue
        sev  = alert["severity"]
        cls  = f"alert-{sev[:4]}"
        icon = "🔴" if sev == "critical" else "🟡" if sev == "warning" else "🔵"
        st.markdown(
            f'<div class="{cls}">'
            f'<div class="alert-title">{icon} [{alert["category"]}] {alert["title"]}</div>'
            f'<div class="alert-body">{alert["body"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.title("⚙️ Settings & Data Upload")

    # ── File uploads ───────────────────────────────────────────────────────────
    st.markdown("### Upload Data Files")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Sales Export** (order_nexus CSV) ✳️ Required")
        sales_file = st.file_uploader("Sales", type=["csv"],
                                       label_visibility="collapsed", key="up_sales")
        if sales_file:
            try:
                st.session_state.sales = load_sales(sales_file)
                st.success(f"✓ Loaded {len(st.session_state.sales):,} sales rows")
            except Exception as e:
                st.error(f"Error loading sales: {e}")

        st.markdown("**Inventory Snapshot** (CSV) ✳️ Required")
        inv_file = st.file_uploader("Inventory", type=["csv"],
                                     label_visibility="collapsed", key="up_inv")
        if inv_file:
            try:
                st.session_state.inventory = load_inventory(inv_file)
                st.success(f"✓ Loaded {len(st.session_state.inventory):,} inventory rows")
            except Exception as e:
                st.error(f"Error loading inventory: {e}")

    with col2:
        st.markdown("**SKU Master with BOM** (Excel .xlsx) ✳️ Required")
        bom_file = st.file_uploader("BOM Excel", type=["xlsx"],
                                     label_visibility="collapsed", key="up_bom")
        if bom_file:
            try:
                st.session_state.bom_data = load_bom_excel(bom_file)
                n_sku = len(st.session_state.bom_data["sku_master"])
                n_bom = len(st.session_state.bom_data["bom"])
                st.success(f"✓ Loaded {n_sku} SKUs, {n_bom} BOM lines")
            except Exception as e:
                st.error(f"Error loading BOM: {e}")

        st.markdown("**Production Lines** (CSV) — optional")
        lines_file = st.file_uploader("Production Lines", type=["csv"],
                                       label_visibility="collapsed", key="up_lines")
        if lines_file:
            try:
                st.session_state.prod_lines = load_production_lines(lines_file)
                st.success(f"✓ Loaded {len(st.session_state.prod_lines)} production lines")
            except Exception as e:
                st.error(f"Error loading production lines: {e}")

        st.markdown("**Marketing Calendar** (CSV) — optional")
        mktg_file = st.file_uploader("Marketing Calendar", type=["csv"],
                                      label_visibility="collapsed", key="up_mktg")
        if mktg_file:
            try:
                st.session_state.marketing_cal = load_marketing_calendar(mktg_file)
                st.success(f"✓ Loaded {len(st.session_state.marketing_cal)} marketing events")
            except Exception as e:
                st.error(f"Error loading marketing calendar: {e}")

    # ── Validation ─────────────────────────────────────────────────────────────
    if (st.session_state.sales is not None and
        st.session_state.inventory is not None and
        st.session_state.bom_data is not None):
        st.markdown("---")
        st.markdown("### Data Validation")
        issues = validate_data(
            st.session_state.sales,
            st.session_state.inventory,
            st.session_state.bom_data,
        )
        if not issues:
            st.success("All data files validated — no issues found.")
        for issue in issues:
            lvl = issue["level"]
            if lvl == "critical": st.error(f"🔴 {issue['message']}")
            elif lvl == "warning": st.warning(f"🟡 {issue['message']}")
            else: st.info(f"🔵 {issue['message']}")

    # ── Seasonal inventory settings ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Seasonal Inventory Targets")
    st.markdown("Set maximum weeks of inventory on hand by season. "
                "Used by the production run scheduler.")

    sc1, sc2, sc3, sc4 = st.columns(4)
    seasons = st.session_state.inv_max_seasons
    with sc1:
        seasons["Q1"] = st.number_input("Q1 Jan–Mar (wks)", 4, 52,
                                         int(seasons["Q1"]), key="s_q1")
    with sc2:
        seasons["Q2"] = st.number_input("Q2 Apr–Jun (wks)", 4, 52,
                                         int(seasons["Q2"]), key="s_q2")
    with sc3:
        seasons["Q3"] = st.number_input("Q3 Jul–Sep (wks)", 4, 52,
                                         int(seasons["Q3"]), key="s_q3")
    with sc4:
        seasons["Q4"] = st.number_input("Q4 Oct–Dec (wks)", 4, 52,
                                         int(seasons["Q4"]), key="s_q4")
    st.session_state.inv_max_seasons = seasons

    # ── Production line template ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Production Lines Template")
    st.markdown("Download this template, fill in your line names and constraints, then upload above.")
    template = pd.DataFrame({
        "line_name":    ["Line 1", "Line 2", "Line 3"],
        "min_run_lbs":  [200,       150,       300],
        "max_run_lbs":  [3000,      2000,      4000],
        "sku_ids":      ["1007;1015;1029", "4083;4921", "3803;8002"],
    })
    st.download_button(
        "⬇ Download production_lines_template.csv",
        template.to_csv(index=False),
        file_name="production_lines_template.csv",
        mime="text/csv"
    )

    # ── Marketing calendar template ────────────────────────────────────────────
    st.markdown("### Marketing Calendar Template")
    mktg_template = pd.DataFrame({
        "event_name":       ["Summer BBQ", "Holiday Gift Sets"],
        "event_type":       ["seasonal",   "seasonal"],
        "start_date":       ["2026-05-25", "2026-11-01"],
        "end_date":         ["2026-07-04", "2026-12-26"],
        "affected_sku_ids": ["1007;1015",  "4921;4083"],
        "uplift_percent":   [20,            50],
    })
    st.download_button(
        "⬇ Download marketing_calendar_template.csv",
        mktg_template.to_csv(index=False),
        file_name="marketing_calendar_template.csv",
        mime="text/csv"
    )


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;color:#2a2e2a;font-size:10px;margin-top:40px'>"
    "SupplyAI · F&B Demand & Supply Planner · Phase 1"
    "</div>",
    unsafe_allow_html=True
)
