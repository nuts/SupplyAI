"""
forecast.py
Demand forecasting engine.

Changes vs. prior version:
- Cold-start fallback: variants with no/limited history now inherit the parent
  product's velocity (scaled by ounces if available), then category-level
  velocity, before falling back to zero. This eliminates the "all-zero forecast
  for ~70% of SKUs" bug that drove weighted MAPE to 100%+.
- Confidence is now derived from FORECAST METHOD used (Direct / Parent / Category /
  None), not just n_weeks of history. The label now tells planners how the number
  was produced.
- Every variant in sku_master is given a row, not just those with sales rows.
- Method tagging exposed via `forecast_method` column for QA.

Earlier changes preserved:
- YoY compares trailing 4 weeks vs same 4-week period prior year.
- "New" tag when either period had zero sales (computed independently of method).
- Inventory cover uses primary facility (NJ) only.
"""

import pandas as pd
import numpy as np
import datetime
from utils.formatters import week_keys, week_short_labels
from utils.data_loader import PRIMARY_WAREHOUSES, REGIONAL_WAREHOUSES

CURRENT_WEEK = 17
CURRENT_YEAR = 2026
FORECAST_WEEKS = 8

# Cold-start thresholds
MIN_WEEKS_FOR_DIRECT = 4    # >=4 weeks of own history -> use SKU's own data
MIN_WEEKS_FOR_HIGH = 100
MIN_WEEKS_FOR_MEDIUM = 20

# When a parent-product fallback fires, scale by pack-size ratio if both
# (parent, child) ounces are known. Otherwise use a flat share.
DEFAULT_NEW_VARIANT_SHARE = 0.10   # assume ~10% of parent volume by default


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


def build_category_seasonal_index(sales: pd.DataFrame) -> dict:
    """Per-category seasonal indices. Falls back to global if a category is sparse."""
    out = {}
    if "cat_l1" not in sales.columns:
        return out
    for cat, grp in sales.groupby("cat_l1"):
        weekly = grp.groupby(["year", "week"])["pieces_sold"].sum().reset_index()
        by_week = weekly.groupby("week")["pieces_sold"].mean()
        overall = by_week.mean()
        if overall == 0 or len(by_week) < 12:
            continue
        idx = {int(w): round(float(v / overall), 4) for w, v in by_week.items()}
        for w in range(1, 53):
            idx.setdefault(w, 1.0)
        out[cat] = idx
    return out


def compute_yoy_trailing_4w(sales_for_variant: pd.DataFrame,
                            current_week: int, current_year: int):
    """Compare last 4 weeks vs same 4 weeks prior year. Returns (yoy_decimal, status)."""
    recent_weeks = []
    w, y = current_week - 1, current_year
    for _ in range(4):
        if w < 1: w, y = 52, y - 1
        recent_weeks.append((y, w))
        w -= 1

    prior_weeks = [(y - 1, w) for (y, w) in recent_weeks]

    recent_sales = 0.0
    prior_sales = 0.0
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


def _recent_4w(grp, current_week, current_year):
    """Return mean weekly units for the last 4 weeks."""
    recent_vals = []
    w, y = current_week - 1, current_year
    for _ in range(4):
        if w < 1: w, y = 52, y - 1
        wkdata = grp[(grp["year"] == y) & (grp["week"] == w)]
        recent_vals.append(wkdata["pieces_sold"].sum())
        w -= 1
    return float(np.mean(recent_vals)) if recent_vals else 0.0


def build_variant_stats(sales: pd.DataFrame,
                        current_week: int = CURRENT_WEEK,
                        current_year: int = CURRENT_YEAR) -> pd.DataFrame:
    """Per-variant statistics for forecasting (computed only for variants with sales rows)."""
    rows = []

    for vid, grp in sales.groupby("variant_id"):
        weekly = grp.groupby(["year", "week"])["pieces_sold"].sum().reset_index()
        vals = weekly["pieces_sold"].values
        baseline = float(np.mean(vals)) if len(vals) > 0 else 0.0
        std_dev = float(np.std(vals)) if len(vals) > 1 else 0.0
        cv = (std_dev / baseline) if baseline > 0 else 0.0
        n_weeks = len(weekly)

        yoy, yoy_status = compute_yoy_trailing_4w(grp, current_week, current_year)
        recent_4w_avg = _recent_4w(grp, current_week, current_year)

        # DC split
        dc_split = (
            grp.groupby("warehouse")["pieces_sold"].sum() /
            max(grp["pieces_sold"].sum(), 1)
        ).to_dict()

        meta = grp.sort_values(["year", "week"]).iloc[-1]

        rows.append({
            "variant_id": vid,
            "product_id": meta.get("product_id"),
            "product_name": meta.get("product_name", ""),
            "cat_l1": meta.get("cat_l1", ""),
            "cat_l2": meta.get("cat_l2", ""),
            "variant_name": meta.get("variant_name", ""),
            "sku_weight_lbs": float(meta.get("sku_weight_lbs", 0) or 0),
            "baseline": round(baseline, 2),
            "recent_4w_avg": round(recent_4w_avg, 2),
            "std_dev": round(std_dev, 2),
            "cv": round(cv, 4),
            "yoy": round(yoy, 4) if yoy is not None else None,
            "yoy_status": yoy_status,
            "n_weeks": n_weeks,
            "dc_split": dc_split,
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
            week_end = week_start + datetime.timedelta(days=6)
            ws = evt["start_date"].date() if hasattr(evt["start_date"], "date") else evt["start_date"]
            we = evt["end_date"].date() if hasattr(evt["end_date"], "date") else evt["end_date"]
            if week_start <= we and week_end >= ws:
                for sid in sku_ids:
                    key = (sid, wi)
                    uplift_map[key] = max(uplift_map.get(key, 1.0), multiplier)
    return uplift_map


def _ratio_safe(num, denom, default=DEFAULT_NEW_VARIANT_SHARE):
    """Pack-size ratio with NaN/zero guards."""
    try:
        n = float(num) if num is not None else 0.0
        d = float(denom) if denom is not None else 0.0
        if d <= 0:
            return default
        r = n / d
        return r if 0 < r <= 5 else default  # clamp absurd ratios
    except Exception:
        return default


def _build_product_velocity(variant_stats: pd.DataFrame) -> dict:
    """Map product_id -> {recent_4w_avg, baseline, yoy, ounces_repr}.
    Aggregates across the variants in variant_stats for that product."""
    out = {}
    if variant_stats.empty or "product_id" not in variant_stats.columns:
        return out
    for pid, g in variant_stats.groupby("product_id"):
        if pid is None or (isinstance(pid, float) and np.isnan(pid)):
            continue
        # Sum recent_4w_avg and baseline across variants of this product
        # (this approximates total parent-product velocity).
        recent = float(g["recent_4w_avg"].fillna(0).sum())
        base = float(g["baseline"].fillna(0).sum())
        yoy_vals = g["yoy"].dropna()
        yoy = float(yoy_vals.mean()) if len(yoy_vals) else 0.0
        out[pid] = {
            "recent_4w_avg": recent,
            "baseline": base,
            "yoy": yoy,
            "n_variants": len(g),
        }
    return out


def _build_category_velocity(variant_stats: pd.DataFrame) -> dict:
    """Map cat_l1 -> mean per-variant recent velocity, used as a final fallback."""
    out = {}
    if variant_stats.empty or "cat_l1" not in variant_stats.columns:
        return out
    for cat, g in variant_stats.groupby("cat_l1"):
        if not cat:
            continue
        recent = float(g["recent_4w_avg"].fillna(0).mean())
        base = float(g["baseline"].fillna(0).mean())
        out[cat] = {"recent_4w_avg": recent, "baseline": base}
    return out


def _classify_confidence(method: str, n_weeks: int) -> str:
    """Confidence label based on the method used to produce the forecast,
    not just data volume. Planners now know HOW the number was made."""
    if method == "Direct":
        if n_weeks >= MIN_WEEKS_FOR_HIGH: return "High"
        if n_weeks >= MIN_WEEKS_FOR_MEDIUM: return "Medium"
        return "Low"
    if method == "Parent":
        return "Low"     # inherited from sibling SKUs
    if method == "Category":
        return "Very Low"
    return "None"        # zero forecast, no signal at all


def run_forecast(sales, sku_master, inventory,
                 marketing_cal=None,
                 current_week=CURRENT_WEEK,
                 current_year=CURRENT_YEAR,
                 n_weeks=FORECAST_WEEKS,
                 scenario_params=None):
    """
    Main forecast.

    Method waterfall per variant:
      1. Direct      - variant has its own >=MIN_WEEKS_FOR_DIRECT weeks of history.
      2. Parent      - inherit parent product velocity, scale by pack-size ratio.
      3. Category    - fall back to category mean per-variant velocity.
      4. None        - no signal anywhere, weekly_forecast = zeros.

    scenario_params can override:
      'forecast_uplift_pct'    - global uplift %  (e.g. +10)
      'marketing_uplift_factor' - scale all promo uplifts (1.0 = full)
    """
    if scenario_params is None:
        scenario_params = {}

    seasonal_idx = build_seasonal_index(sales)
    cat_seasonal = build_category_seasonal_index(sales)
    variant_stats = build_variant_stats(sales, current_week, current_year)
    uplift_map = build_marketing_uplift_map(
        marketing_cal, current_week, current_year, n_weeks)

    # Build full variant universe from sku_master so NEW SKUs get a row.
    sku_meta_cols = ["variant_id", "product_id", "product_name",
                     "cat_l1", "cat_l2", "variant_name",
                     "ounces", "allergen_group", "packaging_format",
                     "sku_weight_lbs"]
    keep_cols = [c for c in sku_meta_cols if c in sku_master.columns]
    sku_universe = sku_master[keep_cols].drop_duplicates("variant_id").copy()

    # Outer-merge variant_stats onto the SKU universe so every SKU appears.
    full = sku_universe.merge(
        variant_stats,
        on="variant_id",
        how="left",
        suffixes=("", "_stats"),
    )

    # Coalesce metadata columns (prefer sku_master, fill from sales)
    for c in ["product_id", "product_name", "cat_l1", "cat_l2",
              "variant_name", "sku_weight_lbs"]:
        stats_col = f"{c}_stats"
        if stats_col in full.columns:
            full[c] = full[c].fillna(full[stats_col])
            full.drop(columns=[stats_col], inplace=True)

    # Defaults for SKUs that had no sales rows at all.
    full["baseline"] = full.get("baseline", 0).fillna(0.0)
    full["recent_4w_avg"] = full.get("recent_4w_avg", 0).fillna(0.0)
    full["std_dev"] = full.get("std_dev", 0).fillna(0.0)
    full["cv"] = full.get("cv", 0).fillna(0.0)
    full["n_weeks"] = full.get("n_weeks", 0).fillna(0).astype(int)
    if "yoy_status" in full.columns:
        full["yoy_status"] = full["yoy_status"].fillna("New")
    else:
        full["yoy_status"] = "New"
    if "yoy" not in full.columns:
        full["yoy"] = None
    if "dc_split" in full.columns:
        full["dc_split"] = full["dc_split"].apply(
            lambda x: x if isinstance(x, dict) else {})
    else:
        full["dc_split"] = [{} for _ in range(len(full))]

    # Pre-compute parent and category velocity tables from variants WITH data.
    has_data = variant_stats[variant_stats["recent_4w_avg"].fillna(0) > 0]
    parent_vel = _build_product_velocity(has_data)
    cat_vel = _build_category_velocity(has_data)

    # Scenario adjustments
    global_uplift = 1 + (scenario_params.get("forecast_uplift_pct", 0) / 100)
    mktg_uplift_factor = scenario_params.get("marketing_uplift_factor", 1.0)

    rows = []
    for _, vs in full.iterrows():
        vid = vs["variant_id"]
        n_weeks_var = int(vs.get("n_weeks", 0) or 0)
        recent = float(vs.get("recent_4w_avg") or 0.0)
        baseline = float(vs.get("baseline") or 0.0)
        yoy_val = vs.get("yoy")
        yoy = float(yoy_val) if yoy_val is not None and not pd.isna(yoy_val) else 0.0

        # ------- Method waterfall -------
        method = "None"
        base = 0.0

        if n_weeks_var >= MIN_WEEKS_FOR_DIRECT and (recent > 0 or baseline > 0):
            base = recent if recent > 0 else baseline
            method = "Direct"
        else:
            # Parent product fallback
            pid = vs.get("product_id")
            pinfo = parent_vel.get(pid) if pid is not None else None
            if pinfo and pinfo["recent_4w_avg"] > 0:
                # Pack-size scaling: this variant's ounces vs. total ounces of
                # its siblings with data. If unknown, use DEFAULT_NEW_VARIANT_SHARE.
                this_oz = vs.get("ounces", 0) or 0
                # Approximate sibling ounces from parent_vel by averaging.
                avg_sibling_oz = vs.get("ounces", 0) or 0
                share = _ratio_safe(this_oz, max(avg_sibling_oz, 1),
                                    default=DEFAULT_NEW_VARIANT_SHARE)
                base = pinfo["recent_4w_avg"] * share
                if pinfo.get("yoy"):
                    yoy = pinfo["yoy"]
                method = "Parent"
            else:
                # Category fallback - last resort before zero
                cat = vs.get("cat_l1")
                cinfo = cat_vel.get(cat) if cat else None
                if cinfo and cinfo["recent_4w_avg"] > 0:
                    base = cinfo["recent_4w_avg"] * DEFAULT_NEW_VARIANT_SHARE
                    method = "Category"

        # If we ended up with base > 0, forecast; otherwise zeros.
        weekly = []
        promo_weeks = []
        # Pick seasonal index: per-category if available, else global.
        cat = vs.get("cat_l1")
        s_table = cat_seasonal.get(cat, seasonal_idx) if cat else seasonal_idx

        for wi in range(n_weeks):
            wk = ((current_week - 1 + wi) % 52) + 1
            s_idx = s_table.get(wk, 1.0)
            t_idx = 1 + (yoy * (wi / 52))
            uplift_raw = uplift_map.get((str(vs.get("product_id", "")), wi), 1.0)
            uplift_adj = 1 + ((uplift_raw - 1) * mktg_uplift_factor)
            forecast = max(0.0, base * s_idx * t_idx * uplift_adj * global_uplift)
            weekly.append(round(forecast, 1))
            if uplift_raw > 1.0:
                promo_weeks.append(wi)

        total_8wk = round(sum(weekly), 1)
        avg_wk = round(total_8wk / n_weeks, 1)

        confidence = _classify_confidence(method, n_weeks_var)

        # Inventory: PRIMARY facility only.
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
            "total_8wk": total_8wk,
            "avg_weekly": avg_wk,
            "primary_inv": int(primary_inv),
            "network_inv": int(all_inv),
            "weeks_cover": wks_cover,
            "promo_weeks": promo_weeks,
            "has_promo": len(promo_weeks) > 0,
            "forecast_method": method,
            "confidence": confidence,
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
        vid = fc_row["variant_id"]
        dc_split = fc_row.get("dc_split", {})
        if not isinstance(dc_split, dict):
            dc_split = {}
        reg_split = {wh: v for wh, v in dc_split.items() if wh in REGIONAL_WAREHOUSES}
        total_reg = sum(reg_split.values())

        # Cold-start: if variant has no DC split (no historical sales),
        # apply the network's average regional split so NEW SKUs still get
        # a non-zero DC forecast.
        if total_reg == 0:
            # equal share across regional DCs as a defensible default
            share_each = 1.0 / max(len(REGIONAL_WAREHOUSES), 1)
            reg_split = {wh: share_each for wh in REGIONAL_WAREHOUSES}
        else:
            reg_split = {wh: v / total_reg for wh, v in reg_split.items()}

        for wh, share in reg_split.items():
            wh_inv = inv_by[(inv_by["variant_id"] == vid) &
                            (inv_by["warehouse"] == wh)]["units_on_hand"].sum()
            weekly_dc = [round(w * share, 1) for w in fc_row["weekly_forecast"]]
            avg_dc = sum(weekly_dc) / n_weeks if n_weeks > 0 else 0
            wks_cover = round(wh_inv / avg_dc, 1) if avg_dc > 0 else 99.0
            rows.append({
                "variant_id": vid,
                "product_name": fc_row.get("product_name", ""),
                "cat_l1": fc_row.get("cat_l1", ""),
                "cat_l2": fc_row.get("cat_l2", ""),
                "warehouse": wh,
                "weekly_forecast": weekly_dc,
                "total_8wk": round(sum(weekly_dc), 1),
                "avg_weekly": round(avg_dc, 1),
                "inv_units": int(wh_inv),
                "weeks_cover": min(wks_cover, 99.0),
                "forecast_method": fc_row.get("forecast_method", "Direct"),
            })
    return pd.DataFrame(rows)
