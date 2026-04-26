"""
forecast.py
Demand forecasting engine.
- Builds seasonal index from real 4-4-5 sales history
- Applies YoY trend per variant
- Calculates demand variability (CV) per variant
- Applies marketing calendar uplifts when available
- Returns 8-week forecast by variant and by DC
"""

import pandas as pd
import numpy as np
from utils.formatters import week_labels, week_display_labels

CURRENT_WEEK = 16   # 4-4-5 week of Apr 24 2026
CURRENT_YEAR = 2026
FORECAST_WEEKS = 8


def build_seasonal_index(sales: pd.DataFrame) -> dict:
    """
    Build week-of-year seasonal index from full sales history.
    Returns dict: {week_num (int): multiplier (float)}
    """
    weekly = (
        sales.groupby(["year", "week"])["pieces_sold"]
        .sum()
        .reset_index()
    )
    # Average pieces per week number across all years
    by_week = weekly.groupby("week")["pieces_sold"].mean()
    overall = by_week.mean()
    if overall == 0:
        return {w: 1.0 for w in range(1, 53)}
    index = {int(w): round(float(v / overall), 4) for w, v in by_week.items()}
    # Fill any missing weeks with 1.0
    for w in range(1, 53):
        index.setdefault(w, 1.0)
    return index


def build_variant_stats(sales: pd.DataFrame) -> pd.DataFrame:
    """
    Per-variant statistics used in forecasting.
    Returns DataFrame with one row per variant.
    """
    rows = []

    for vid, grp in sales.groupby("variant_id"):
        weekly = grp.groupby(["year", "week"])["pieces_sold"].sum().reset_index()
        vals   = weekly["pieces_sold"].values

        baseline = float(np.mean(vals)) if len(vals) > 0 else 0.0
        std_dev  = float(np.std(vals))  if len(vals) > 1 else 0.0
        cv       = (std_dev / baseline) if baseline > 0 else 0.0
        n_weeks  = len(weekly)

        # YoY trend: most recent complete year vs prior year
        max_year = int(weekly["year"].max())
        y_curr   = weekly[weekly["year"] == max_year]["pieces_sold"].mean()
        y_prior  = weekly[weekly["year"] == max_year - 1]["pieces_sold"].mean()
        if pd.notna(y_curr) and pd.notna(y_prior) and y_prior > 0:
            yoy = float((y_curr - y_prior) / y_prior)
        else:
            yoy = 0.0

        # Confidence tier
        if n_weeks >= 100:
            confidence = "High"
        elif n_weeks >= 20:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Per-DC demand split (% of total pieces from each warehouse)
        dc_split = (
            grp.groupby("warehouse")["pieces_sold"].sum() /
            grp["pieces_sold"].sum()
        ).to_dict()

        # Pull through metadata from most recent row
        meta = grp.sort_values(["year", "week"]).iloc[-1]

        rows.append({
            "variant_id":   vid,
            "product_id":   meta.get("product_id", None),
            "product_name": meta.get("product_name", ""),
            "cat_l1":       meta.get("cat_l1", ""),
            "cat_l2":       meta.get("cat_l2", ""),
            "variant_name": meta.get("variant_name", ""),
            "sku_weight_lbs": float(meta.get("sku_weight_lbs", 0) or 0),
            "baseline":     round(baseline, 2),
            "std_dev":      round(std_dev, 2),
            "cv":           round(cv, 4),
            "yoy":          round(yoy, 4),
            "n_weeks":      n_weeks,
            "confidence":   confidence,
            "dc_split":     dc_split,
        })

    return pd.DataFrame(rows)


def build_marketing_uplift_map(
    marketing_cal: pd.DataFrame,
    current_week: int,
    current_year: int,
    n_weeks: int = FORECAST_WEEKS
) -> dict:
    """
    Build {(product_id, week_idx): uplift_multiplier} from marketing calendar.
    week_idx is 0-based index into the forecast window.
    """
    if marketing_cal is None or marketing_cal.empty:
        return {}

    uplift_map = {}
    forecast_week_keys = week_labels(current_week, current_year, n_weeks)

    for _, evt in marketing_cal.iterrows():
        if pd.isna(evt.get("start_date")) or pd.isna(evt.get("end_date")):
            continue
        multiplier = 1 + (float(evt.get("uplift_percent", 0)) / 100)
        sku_ids    = str(evt.get("affected_sku_ids", "")).split(";")
        sku_ids    = [s.strip() for s in sku_ids if s.strip()]

        for wi, yw in enumerate(forecast_week_keys):
            yr, wk = int(yw.split("-")[0]), int(yw.split("-")[1])
            # Build approximate date range for this week
            import datetime
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


def run_forecast(
    sales:          pd.DataFrame,
    sku_master:     pd.DataFrame,
    inventory:      pd.DataFrame,
    marketing_cal:  pd.DataFrame  = None,
    current_week:   int           = CURRENT_WEEK,
    current_year:   int           = CURRENT_YEAR,
    n_weeks:        int           = FORECAST_WEEKS,
    inv_max_weeks_by_season: dict = None,
) -> pd.DataFrame:
    """
    Main forecast function. Returns DataFrame with one row per variant,
    columns: variant metadata + weekly_forecast (list) + aggregates.
    """
    if inv_max_weeks_by_season is None:
        inv_max_weeks_by_season = {
            "Q1": 12, "Q2": 12, "Q3": 14, "Q4": 20
        }

    seasonal_idx  = build_seasonal_index(sales)
    variant_stats = build_variant_stats(sales)
    uplift_map    = build_marketing_uplift_map(
        marketing_cal, current_week, current_year, n_weeks
    )

    # Merge BOM SKU master to pull in ounces and allergen group
    sku_cols = ["variant_id", "ounces", "allergen_group", "packaging_format", "cat_l1"]
    merge_cols = [c for c in sku_cols if c in sku_master.columns]
    variant_stats = variant_stats.merge(
        sku_master[merge_cols].drop_duplicates("variant_id"),
        on="variant_id", how="left", suffixes=("", "_bom")
    )
    # Prefer BOM cat_l1 if missing from sales
    if "cat_l1_bom" in variant_stats.columns:
        variant_stats["cat_l1"] = variant_stats["cat_l1"].fillna(
            variant_stats["cat_l1_bom"])
        variant_stats.drop(columns=["cat_l1_bom"], inplace=True)

    forecast_rows = []

    for _, vs in variant_stats.iterrows():
        vid      = vs["variant_id"]
        baseline = vs["baseline"]
        yoy      = vs["yoy"]

        weekly = []
        promo_weeks = []

        for wi in range(n_weeks):
            wk = ((current_week - 1 + wi) % 52) + 1
            s_idx    = seasonal_idx.get(wk, 1.0)
            t_idx    = 1 + yoy * (wi / 52)         # gradual trend
            uplift   = uplift_map.get((str(vs.get("product_id", "")), wi), 1.0)
            forecast = max(0.0, baseline * s_idx * t_idx * uplift)
            weekly.append(round(forecast, 1))
            if uplift > 1.0:
                promo_weeks.append(wi)

        total_8wk = round(sum(weekly), 1)
        avg_wk    = round(total_8wk / n_weeks, 1)

        # Current inventory for this SKU across all warehouses
        inv_units = inventory[
            (inventory["variant_id"] == vid) &
            (~inventory["warehouse"].isin(["none"]))
        ]["units_on_hand"].sum()

        wks_cover = round(inv_units / avg_wk, 1) if avg_wk > 0 else 99.0
        wks_cover = min(wks_cover, 99.0)

        forecast_rows.append({
            **vs.to_dict(),
            "weekly_forecast":  weekly,
            "total_8wk":        total_8wk,
            "avg_weekly":       avg_wk,
            "inv_units":        int(inv_units),
            "weeks_cover":      wks_cover,
            "promo_weeks":      promo_weeks,
            "has_promo":        len(promo_weeks) > 0,
        })

    fc = pd.DataFrame(forecast_rows)

    # Sort by total_8wk descending
    fc = fc.sort_values("total_8wk", ascending=False).reset_index(drop=True)

    return fc


def build_dc_forecast(
    forecast:     pd.DataFrame,
    inventory:    pd.DataFrame,
    n_weeks:      int = FORECAST_WEEKS,
) -> pd.DataFrame:
    """
    Break out forecast by DC using historical demand splits.
    Returns long DataFrame: variant_id × warehouse × week → units.
    """
    from utils.data_loader import REGIONAL_WAREHOUSES

    rows = []
    inv_by_sku_wh = (
        inventory[inventory["warehouse"] != "none"]
        .groupby(["variant_id", "warehouse"])["units_on_hand"]
        .sum()
        .reset_index()
    )

    for _, fc_row in forecast.iterrows():
        vid      = fc_row["variant_id"]
        dc_split = fc_row.get("dc_split", {})

        # Only regional DCs
        reg_split = {
            wh: v for wh, v in dc_split.items()
            if wh in REGIONAL_WAREHOUSES
        }
        total_reg = sum(reg_split.values())
        if total_reg == 0:
            continue

        # Normalise to sum=1 over regional DCs
        reg_split = {wh: v / total_reg for wh, v in reg_split.items()}

        for wh, share in reg_split.items():
            wh_inv = inv_by_sku_wh[
                (inv_by_sku_wh["variant_id"] == vid) &
                (inv_by_sku_wh["warehouse"]  == wh)
            ]["units_on_hand"].sum()

            weekly_dc = [round(w * share, 1) for w in fc_row["weekly_forecast"]]
            avg_dc    = sum(weekly_dc) / n_weeks if n_weeks > 0 else 0
            wks_cover = round(wh_inv / avg_dc, 1) if avg_dc > 0 else 99.0

            rows.append({
                "variant_id":   vid,
                "product_name": fc_row.get("product_name", ""),
                "cat_l1":       fc_row.get("cat_l1", ""),
                "warehouse":    wh,
                "weekly_forecast": weekly_dc,
                "total_8wk":    round(sum(weekly_dc), 1),
                "avg_weekly":   round(avg_dc, 1),
                "inv_units":    int(wh_inv),
                "weeks_cover":  min(wks_cover, 99.0),
            })

    return pd.DataFrame(rows)
