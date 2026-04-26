"""
purchasing.py
Purchasing plan engine.
- Explodes production schedule through BOM
- Calculates raw material and packaging requirements by week
- Groups by ingredient/supplier
- Flags missing BOM coverage
"""

import pandas as pd
import numpy as np

FORECAST_WEEKS = 8


def explode_bom(
    prod_df:  pd.DataFrame,
    bom_data: dict,
    n_weeks:  int = FORECAST_WEEKS,
) -> pd.DataFrame:
    """
    Explode production plan through BOM to get ingredient requirements.
    Returns DataFrame with one row per ingredient per week.
    """
    bom   = bom_data["bom"]
    items = bom_data.get("items", pd.DataFrame())

    # Separate raw materials (lbs) from packaging (eaches)
    raw_bom = bom[bom["input_unit_of_measure"] == "lbs"].copy()
    pkg_bom = bom[bom["input_unit_of_measure"] == "eaches"].copy()

    rows = []

    for _, prod_row in prod_df.iterrows():
        vid          = prod_row["variant_id"]
        prod_lbs_wk  = prod_row.get("prod_by_week_lbs", [0.0] * n_weeks)
        weight_lbs   = prod_row.get("sku_weight_lbs", 0) or 1.0

        # ── Raw materials ──────────────────────────────────────────────────────
        raw_lines = raw_bom[raw_bom["output_sku"] == vid]
        for _, bom_line in raw_lines.iterrows():
            # input_measurement is lbs of ingredient per finished unit weight
            # ratio = ingredient_lbs / finished_lbs
            qty_per_finished_lb = float(bom_line["input_measurement"]) / weight_lbs \
                if weight_lbs > 0 else 0

            weekly_need = [
                round(p * qty_per_finished_lb, 2)
                for p in prod_lbs_wk
            ]

            rows.append({
                "ingredient_id":   int(bom_line["input_sku"]),
                "ingredient_name": str(bom_line["input_sku_name"]),
                "ingredient_type": "raw_material",
                "unit":            "lbs",
                "source_sku":      vid,
                "source_product":  prod_row.get("product_name", ""),
                "weekly_need":     weekly_need,
                "total_need_lbs":  round(sum(weekly_need), 2),
            })

        # ── Packaging components ───────────────────────────────────────────────
        pkg_lines = pkg_bom[pkg_bom["output_sku"] == vid]
        for _, bom_line in pkg_lines.iterrows():
            # 1 packaging unit per finished unit of product
            # prod_lbs / weight_lbs = finished units
            weekly_units = [
                round(p / weight_lbs) if weight_lbs > 0 else 0
                for p in prod_lbs_wk
            ]

            rows.append({
                "ingredient_id":   int(bom_line["input_sku"]),
                "ingredient_name": str(bom_line["input_sku_name"]),
                "ingredient_type": "packaging",
                "unit":            "eaches",
                "source_sku":      vid,
                "source_product":  prod_row.get("product_name", ""),
                "weekly_need":     weekly_units,
                "total_need_lbs":  0,   # not applicable for packaging
            })

    if not rows:
        return pd.DataFrame()

    detail_df = pd.DataFrame(rows)
    return detail_df


def aggregate_purchasing(detail_df: pd.DataFrame, n_weeks: int = FORECAST_WEEKS) -> pd.DataFrame:
    """
    Aggregate ingredient requirements across all source SKUs.
    Returns one row per ingredient with weekly totals.
    """
    if detail_df.empty:
        return pd.DataFrame()

    agg_rows = []

    for (ing_id, ing_name, ing_type, unit), grp in detail_df.groupby(
        ["ingredient_id", "ingredient_name", "ingredient_type", "unit"]
    ):
        # Sum weekly needs across all source SKUs
        weekly_totals = [0.0] * n_weeks
        for _, row in grp.iterrows():
            wn = row["weekly_need"]
            for wi in range(min(len(wn), n_weeks)):
                weekly_totals[wi] += wn[wi]

        weekly_totals = [round(w, 1) for w in weekly_totals]
        total = round(sum(weekly_totals), 1)

        # Count source SKUs
        source_skus = grp["source_sku"].unique().tolist()

        agg_rows.append({
            "ingredient_id":    ing_id,
            "ingredient_name":  ing_name,
            "ingredient_type":  ing_type,
            "unit":             unit,
            "source_skus":      source_skus,
            "n_source_skus":    len(source_skus),
            "weekly_need":      weekly_totals,
            "total_need":       total,
        })

    agg_df = pd.DataFrame(agg_rows)

    # Sort: raw materials first, then by total volume desc
    type_order = {"raw_material": 0, "packaging": 1}
    agg_df["_type_order"] = agg_df["ingredient_type"].map(type_order).fillna(2)
    agg_df = agg_df.sort_values(
        ["_type_order", "total_need"], ascending=[True, False]
    ).drop(columns=["_type_order"]).reset_index(drop=True)

    return agg_df


def build_purchasing_plan(
    prod_df:  pd.DataFrame,
    bom_data: dict,
    n_weeks:  int = FORECAST_WEEKS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full purchasing plan build.
    Returns (detail_df, aggregate_df)
    """
    detail_df = explode_bom(prod_df, bom_data, n_weeks)
    if detail_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    agg_df = aggregate_purchasing(detail_df, n_weeks)
    return detail_df, agg_df


def flag_bom_gaps(
    forecast:  pd.DataFrame,
    bom_data:  dict,
) -> list[dict]:
    """
    Identify SKUs with demand but incomplete or missing BOM.
    Returns list of gap dicts.
    """
    bom        = bom_data["bom"]
    sku_master = bom_data["sku_master"]
    gaps       = []

    bom_skus     = set(bom["output_sku"].unique())
    forecast_skus = set(forecast["variant_id"].unique())

    # SKUs with demand but no BOM at all
    no_bom = forecast_skus - bom_skus
    for vid in no_bom:
        row = forecast[forecast["variant_id"] == vid]
        avg = row["avg_weekly"].values[0] if len(row) > 0 else 0
        if avg > 0:
            gaps.append({
                "variant_id": vid,
                "issue":      "No BOM found",
                "severity":   "critical" if avg > 10 else "warning",
            })

    # SKUs with BOM but no raw material (lbs) component
    raw_skus = set(bom[bom["input_unit_of_measure"] == "lbs"]["output_sku"])
    no_raw   = bom_skus & forecast_skus - raw_skus
    for vid in no_raw:
        gaps.append({
            "variant_id": vid,
            "issue":      "BOM has no raw material (lbs) component",
            "severity":   "warning",
        })

    return gaps
