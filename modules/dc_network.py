"""
dc_network.py
DC network replenishment engine.
- Calculates weeks of cover per variant per DC
- Compares against stock targets from SKUs sheet
- Identifies replenishment needs
- Allocates production output to DCs
"""

import pandas as pd
import numpy as np
from utils.data_loader import REGIONAL_WAREHOUSES, WAREHOUSE_LABELS

FORECAST_WEEKS = 8


def build_dc_inventory_summary(
    inventory:   pd.DataFrame,
    forecast:    pd.DataFrame,
    bom_data:    dict,
    n_weeks:     int = FORECAST_WEEKS,
) -> pd.DataFrame:
    """
    Build per-DC per-SKU inventory position vs demand.
    Returns DataFrame with coverage and replenishment need.
    """
    skus_df    = bom_data.get("skus", pd.DataFrame())
    sku_master = bom_data["sku_master"]

    # Filter to regional DCs only
    dc_inv = inventory[
        inventory["warehouse"].isin(REGIONAL_WAREHOUSES)
    ].copy()

    # Build demand split per DC per SKU from forecast
    dc_demand_rows = []
    for _, fc_row in forecast.iterrows():
        vid      = fc_row["variant_id"]
        dc_split = fc_row.get("dc_split", {})
        avg_wk   = fc_row.get("avg_weekly", 0)

        for wh, share in dc_split.items():
            if wh not in REGIONAL_WAREHOUSES:
                continue
            dc_demand_rows.append({
                "variant_id":    vid,
                "warehouse":     wh,
                "dc_avg_weekly": round(avg_wk * share, 2),
                "dc_total_8wk":  round(fc_row["total_8wk"] * share, 1),
                "weekly_forecast": [round(w * share, 1)
                                    for w in fc_row["weekly_forecast"]],
            })

    dc_demand = pd.DataFrame(dc_demand_rows)

    if dc_demand.empty:
        return pd.DataFrame()

    # Merge with inventory
    merged = dc_demand.merge(
        dc_inv[["variant_id", "warehouse", "units_on_hand"]],
        on=["variant_id", "warehouse"],
        how="left"
    )
    merged["units_on_hand"] = merged["units_on_hand"].fillna(0)

    # Merge with stock targets from SKUs sheet
    if not skus_df.empty and "target_units" in skus_df.columns:
        targets = (
            skus_df[["variant_id", "produce_for_warehouse", "target_units", "reorder_point"]]
            .dropna(subset=["variant_id"])
            .rename(columns={"produce_for_warehouse": "warehouse"})
        )
        merged = merged.merge(targets, on=["variant_id", "warehouse"], how="left")
    else:
        merged["target_units"]  = np.nan
        merged["reorder_point"] = np.nan

    # Merge in product metadata
    meta_cols = ["variant_id", "product_name", "cat_l1", "cat_l2",
                 "variant_name", "allergen_group"]
    meta_cols = [c for c in meta_cols if c in forecast.columns]
    merged = merged.merge(
        forecast[meta_cols].drop_duplicates("variant_id"),
        on="variant_id", how="left"
    )

    # Calculate coverage
    merged["weeks_cover"] = merged.apply(
        lambda r: round(r["units_on_hand"] / r["dc_avg_weekly"], 1)
        if r["dc_avg_weekly"] > 0 else 99.0,
        axis=1
    )
    merged["weeks_cover"] = merged["weeks_cover"].clip(upper=99.0)

    # Replenishment need = target - on_hand (if target set)
    merged["replen_need"] = merged.apply(
        lambda r: max(0, r["target_units"] - r["units_on_hand"])
        if pd.notna(r.get("target_units")) else np.nan,
        axis=1
    )

    # Below reorder point flag
    merged["below_reorder"] = merged.apply(
        lambda r: r["units_on_hand"] <= r["reorder_point"]
        if pd.notna(r.get("reorder_point")) and r["reorder_point"] > 0
        else False,
        axis=1
    )

    # Status label
    def _status(row):
        wc = row["weeks_cover"]
        if wc <= 0:
            return "Out of Stock"
        elif row.get("below_reorder", False):
            return "Below Reorder"
        elif wc < 2:
            return "Critical"
        elif wc < 4:
            return "Low"
        elif wc < 8:
            return "Adequate"
        else:
            return "Healthy"

    merged["status"] = merged.apply(_status, axis=1)

    # Warehouse display label
    merged["warehouse_label"] = merged["warehouse"].map(WAREHOUSE_LABELS).fillna(
        merged["warehouse"]
    )

    return merged.sort_values(
        ["warehouse", "weeks_cover"]
    ).reset_index(drop=True)


def build_network_summary(dc_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate DC summary to network-level view by warehouse.
    """
    if dc_summary.empty:
        return pd.DataFrame()

    agg = dc_summary.groupby("warehouse").agg(
        warehouse_label  = ("warehouse_label", "first"),
        total_skus       = ("variant_id", "nunique"),
        total_units      = ("units_on_hand", "sum"),
        out_of_stock     = ("status", lambda x: (x == "Out of Stock").sum()),
        critical         = ("status", lambda x: (x == "Critical").sum()),
        below_reorder    = ("below_reorder", "sum"),
        avg_weeks_cover  = ("weeks_cover", lambda x: round(x[x < 99].mean(), 1)),
        total_replen_need = ("replen_need", lambda x: x.fillna(0).sum()),
    ).reset_index()

    return agg


def build_dc_heatmap(dc_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Build pivot table: SKU × warehouse → weeks_cover.
    Used for heatmap visualisation.
    """
    if dc_summary.empty:
        return pd.DataFrame()

    pivot = dc_summary.pivot_table(
        index="variant_id",
        columns="warehouse_label",
        values="weeks_cover",
        aggfunc="first"
    ).fillna(0)

    return pivot
