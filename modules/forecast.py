"""
forecast.py
Demand forecasting engine.
Key changes from v1:
- YoY now compares trailing 4 weeks vs same 4-week period prior year
- Returns "New" tag when either period has zero sales
- Fixed inventory cover: uses primary facility (NJ+NJ2) only
"""

import pandas as pd
import numpy as np
import datetime
from utils.formatters import week_keys, week_short_labels
from utils.data_loader import PRIMARY_WAREHOUSES, REGIONAL_WAREHOUSES

CURRENT_WEEK   = 17
CURRENT_YEAR   = 2026
FORECAST_WEEKS = 8


def get_current_week_year():
    """Compute current 4-4-5 retail week based on today."""
    today = datetime.date.today()
    day_of_year = today.timetuple().tm_yday
    week = min(52, max(1, (day_of_year - 1) // 7 + 1))
    return week, today.year


def build_seasonal_index(sales: pd.DataFrame) -> dict:
    """Build week-of-year seasonal index from full sales history."""
    weekly = sales.groupby(["year", "week"])["pieces_sold"].sum().reset_index()
    by_week = weekly.groupby("week")["pieces_sold"].mean()
    overall = by_week.mean()
    if overall == 0:
        return {w: 1.0 for w in range(1, 53)}
    idx = {int(w): round(float(v / overall), 4) for w, v in by_week.items()}
    for w in range(1, 53):
        idx.setdefault(w, 1.0)
    return idx


def compute_yoy_trailing_4w(sales_for_variant: pd.DataFrame,
                             current_week: int, current_year: int):
    """
    Compare last 4 weeks of sales vs same 4 weeks prior year.
    Returns (yoy_decimal, status) where status is one of:
    'New' (either period had zero sales),
    'OK' (valid YoY computed)
    """
    # Recent 4 weeks: weeks (current_week - 4) through (current_week - 1) of current year
    recent_weeks = []
    w, y = current_week - 1, current_year
    for _ in range(4):
        if w < 1: w, y = 52, y - 1
        recent_weeks.append((y, w))
        w -= 1

    # Prior year same 4 weeks
    prior_weeks = [(y - 1, w) for (y, w) in recent_weeks]

    recent_sales = 0.0
    prior_sales  = 0.0
    for (yr, wk) in recent_weeks:
        m = sales_for_variant[(sales_for_variant["year"] == yr) &
                              (sales_for_variant["week"] == wk)]
        recent_sales += m["pieces_sold"].sum()
    for (yr, wk) in prior_weeks:
        m = sales_for_variant[(sales_for_variant["year"] == yr) &
                              (sales_for_variant["week"] == wk)]
        prior_sales += m["pieces_sold"].sum()

    if prior_sales == 0 or recent_sales == 0:
        return None, "New"

    yoy = (recent_sales - prior_sales) / prior_sales
    return float(yoy), "OK"


def build_variant_stats(sales: pd.DataFrame,
                         current_week: int = CURRENT_WEEK,
                         current_year: int = CURRENT_YEAR) -> pd.DataFrame:
    """Per-variant statistics for forecasting."""
    rows = []

    for vid, grp in sales.groupby("variant_id"):
        weekly = grp.groupby(["year", "week"])["pieces_sold"].sum().reset_index()
        vals   = weekly["pieces_sold"].values
        baseline = float(np.mean(vals)) if len(vals) > 0 else 0.0
        std_dev  = float(np.std(vals))  if len(vals) > 1 else 0.0
        cv       = (std_dev / baseline) if baseline > 0 else 0.0
        n_weeks  = len(weekly)

        # NEW: YoY using trailing 4 weeks
        yoy, yoy_status = compute_yoy_trailing_4w(grp, current_week, current_year)

        # Recent 4w avg (used as a more recent baseline anchor)
        recent_weeks = []
        w, y = current_week - 1, current_year
        for _ in range(4):
            if w < 1: w, y = 52, y - 1
            wkdata = grp[(grp["year"] == y) & (grp["week"] == w)]
            recent_weeks.append(wkdata["pieces_sold"].sum())
            w -= 1
        recent_4w_avg = float(np.mean(recent_weeks)) if recent_weeks else baseline

        if n_weeks >= 100: confidence = "High"
        elif n_weeks >= 20: confidence = "Medium"
        else: confidence = "Low"

        # DC split
        dc_split = (
            grp.groupby("warehouse")["pieces_sold"].sum() /
            max(grp["pieces_sold"].sum(), 1)
        ).to_dict()

        meta = grp.sort_values(["year", "week"]).iloc[-1]

        rows.append({
            "variant_id":     vid,
            "product_id":     meta.get("product_id"),
            "product_name":   meta.get("product_name", ""),
            "cat_l1":         meta.get("cat_l1", ""),
            "cat_l2":         meta.get("cat_l2", ""),
            "variant_name":   meta.get("variant_name", ""),
            "sku_weight_lbs": float(meta.get("sku_weight_lbs", 0) or 0),
            "baseline":       round(baseline, 2),
            "recent_4w_avg":  round(recent_4w_avg, 2),
            "std_dev":        round(std_dev, 2),
            "cv":             round(cv, 4),
            "yoy":            round(yoy, 4) if yoy is not None else None,
            "yoy_status":     yoy_status,
            "n_weeks":        n_weeks,
            "confidence":     confidence,
            "dc_split":       dc_split,
        })

    return pd.DataFrame(rows)


def build_marketing_uplift_map(marketing_cal, current_week, current_year, n_weeks):
    if marketing_cal is None or marketing_cal.empty:
        return {}
    uplift_map = {}
    forecast_week_keys = week_keys(current_week, current_year, n_weeks)

    for _, evt in marketing_cal.iterrows():
        if pd.isna(evt.get("start_date")) or pd.isna(evt.get("end_date")):
            continue
        multiplier = 1 + (float(evt.get("uplift_percent", 0)) / 100)
        sku_ids = str(evt.get("affected_sku_ids", "")).split(";")
        sku_ids = [s.strip() for s in sku_ids if s.strip()]

        for wi, yw in enumerate(forecast_week_keys):
            yr, wk = int(yw.split("-")[0]), int(yw.split("-")[1])
            jan1 = datetime.date(yr, 1, 1)
            week_start = jan1 + datetime.timedelta(weeks=wk - 1)
            week_end   = week_start + datetime.timedelta(days=6)
            ws = evt["start_date"].date() if hasattr(evt["start_date"], "date") else evt["start_date"]
            we = evt["end_date"].date()   if hasattr(evt["end_date"],   "date") else evt["end_date"]
            if week_start <= we and week_end >= ws:
                for sid in sku_ids:
                    key = (sid, wi)
                    uplift_map[key] = max(uplift_map.get(key, 1.0), multiplier)
    return uplift_map


def run_forecast(sales, sku_master, inventory,
                  marketing_cal=None,
                  current_week=CURRENT_WEEK,
                  current_year=CURRENT_YEAR,
                  n_weeks=FORECAST_WEEKS,
                  scenario_params=None):
    """
    Main forecast. scenario_params can override:
      - 'forecast_uplift_pct': global uplift on baseline (e.g. +10%)
      - 'season_multiplier_q1/q2/q3/q4': override seasonal indices
      - 'marketing_uplift_factor': scale all marketing uplifts (e.g. 0.5 = half)
    """
    if scenario_params is None:
        scenario_params = {}

    seasonal_idx  = build_seasonal_index(sales)
    variant_stats = build_variant_stats(sales, current_week, current_year)
    uplift_map    = build_marketing_uplift_map(
        marketing_cal, current_week, current_year, n_weeks)

    # Merge SKU master metadata
    sku_cols = ["variant_id", "ounces", "allergen_group", "packaging_format", "cat_l1"]
    merge_cols = [c for c in sku_cols if c in sku_master.columns]
    variant_stats = variant_stats.merge(
        sku_master[merge_cols].drop_duplicates("variant_id"),
        on="variant_id", how="left", suffixes=("", "_bom")
    )
    if "cat_l1_bom" in variant_stats.columns:
        variant_stats["cat_l1"] = variant_stats["cat_l1"].fillna(
            variant_stats["cat_l1_bom"])
        variant_stats.drop(columns=["cat_l1_bom"], inplace=True)

    # Scenario adjustments
    global_uplift  = 1 + (scenario_params.get("forecast_uplift_pct", 0) / 100)
    mktg_uplift_factor = scenario_params.get("marketing_uplift_factor", 1.0)

    rows = []
    for _, vs in variant_stats.iterrows():
        vid      = vs["variant_id"]
        # Use recent 4w avg as base (more responsive than full-history avg)
        base     = vs["recent_4w_avg"] if vs["recent_4w_avg"] > 0 else vs["baseline"]
        yoy      = vs["yoy"] if vs["yoy"] is not None else 0.0

        weekly = []
        promo_weeks = []
        for wi in range(n_weeks):
            wk = ((current_week - 1 + wi) % 52) + 1
            s_idx  = seasonal_idx.get(wk, 1.0)
            t_idx  = 1 + (yoy * (wi / 52))
            uplift_raw = uplift_map.get((str(vs.get("product_id", "")), wi), 1.0)
            uplift_adj = 1 + ((uplift_raw - 1) * mktg_uplift_factor)
            forecast = max(0.0, base * s_idx * t_idx * uplift_adj * global_uplift)
            weekly.append(round(forecast, 1))
            if uplift_raw > 1.0:
                promo_weeks.append(wi)

        total_8wk = round(sum(weekly), 1)
        avg_wk    = round(total_8wk / n_weeks, 1)

        # Inventory: PRIMARY facility only (where production replenishes from)
        primary_inv = inventory[
            (inventory["variant_id"] == vid) &
            (inventory["warehouse"].isin(PRIMARY_WAREHOUSES))
        ]["units_on_hand"].sum()
        all_inv = inventory[
            (inventory["variant_id"] == vid) &
            (~inventory["warehouse"].isin(["none"]))
        ]["units_on_hand"].sum()

        wks_cover = round(primary_inv / avg_wk, 1) if avg_wk > 0 else 99.0
        wks_cover = min(wks_cover, 99.0)

        rows.append({
            **vs.to_dict(),
            "weekly_forecast": weekly,
            "total_8wk":       total_8wk,
            "avg_weekly":      avg_wk,
            "primary_inv":     int(primary_inv),
            "network_inv":     int(all_inv),
            "weeks_cover":     wks_cover,
            "promo_weeks":     promo_weeks,
            "has_promo":       len(promo_weeks) > 0,
        })

    fc = pd.DataFrame(rows).sort_values("total_8wk", ascending=False).reset_index(drop=True)
    return fc


def build_dc_forecast(forecast, inventory, n_weeks=FORECAST_WEEKS):
    """Break out forecast by regional DC."""
    rows = []
    inv_by = (
        inventory[inventory["warehouse"] != "none"]
        .groupby(["variant_id", "warehouse"])["units_on_hand"].sum().reset_index()
    )

    for _, fc_row in forecast.iterrows():
        vid      = fc_row["variant_id"]
        dc_split = fc_row.get("dc_split", {})
        reg_split = {wh: v for wh, v in dc_split.items() if wh in REGIONAL_WAREHOUSES}
        total_reg = sum(reg_split.values())
        if total_reg == 0:
            continue
        reg_split = {wh: v / total_reg for wh, v in reg_split.items()}

        for wh, share in reg_split.items():
            wh_inv = inv_by[(inv_by["variant_id"] == vid) &
                            (inv_by["warehouse"]  == wh)]["units_on_hand"].sum()
            weekly_dc = [round(w * share, 1) for w in fc_row["weekly_forecast"]]
            avg_dc    = sum(weekly_dc) / n_weeks if n_weeks > 0 else 0
            wks_cover = round(wh_inv / avg_dc, 1) if avg_dc > 0 else 99.0
            rows.append({
                "variant_id":   vid,
                "product_name": fc_row.get("product_name", ""),
                "cat_l1":       fc_row.get("cat_l1", ""),
                "cat_l2":       fc_row.get("cat_l2", ""),
                "warehouse":    wh,
                "weekly_forecast": weekly_dc,
                "total_8wk":    round(sum(weekly_dc), 1),
                "avg_weekly":   round(avg_dc, 1),
                "inv_units":    int(wh_inv),
                "weeks_cover":  min(wks_cover, 99.0),
            })
    return pd.DataFrame(rows)
