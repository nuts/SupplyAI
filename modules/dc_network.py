"""
dc_network.py
DC inventory analysis + replenishment transfer plan.
"""

import pandas as pd
import numpy as np
from utils.data_loader import (
    REGIONAL_WAREHOUSES, PRIMARY_WAREHOUSES, WAREHOUSE_LABELS, DC_SHIP_CADENCE
)

FORECAST_WEEKS = 8


def build_dc_inventory_summary(inventory, forecast, bom_data, n_weeks=FORECAST_WEEKS):
    """Per-DC per-SKU inventory position vs demand."""
    skus_df = bom_data.get("skus", pd.DataFrame())

    dc_inv = inventory[inventory["warehouse"].isin(REGIONAL_WAREHOUSES)].copy()

    rows = []
    for _, fc_row in forecast.iterrows():
        vid      = fc_row["variant_id"]
        dc_split = fc_row.get("dc_split", {})
        avg_wk   = fc_row.get("avg_weekly", 0)

        for wh, share in dc_split.items():
            if wh not in REGIONAL_WAREHOUSES: continue
            rows.append({
                "variant_id":      vid,
                "warehouse":       wh,
                "dc_avg_weekly":   round(avg_wk * share, 2),
                "dc_total_8wk":    round(fc_row["total_8wk"] * share, 1),
                "weekly_forecast": [round(w * share, 1) for w in fc_row["weekly_forecast"]],
            })

    dc_demand = pd.DataFrame(rows)
    if dc_demand.empty: return pd.DataFrame()

    merged = dc_demand.merge(
        dc_inv[["variant_id", "warehouse", "units_on_hand"]],
        on=["variant_id", "warehouse"], how="left"
    )
    merged["units_on_hand"] = merged["units_on_hand"].fillna(0)

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

    meta_cols = ["variant_id", "product_name", "cat_l1", "cat_l2",
                 "variant_name", "allergen_group"]
    meta_cols = [c for c in meta_cols if c in forecast.columns]
    merged = merged.merge(forecast[meta_cols].drop_duplicates("variant_id"),
                            on="variant_id", how="left")

    merged["weeks_cover"] = merged.apply(
        lambda r: round(r["units_on_hand"] / r["dc_avg_weekly"], 1)
        if r["dc_avg_weekly"] > 0 else 99.0, axis=1
    )
    merged["weeks_cover"] = merged["weeks_cover"].clip(upper=99.0)

    merged["replen_need"] = merged.apply(
        lambda r: max(0, r["target_units"] - r["units_on_hand"])
        if pd.notna(r.get("target_units")) else np.nan, axis=1
    )
    merged["below_reorder"] = merged.apply(
        lambda r: r["units_on_hand"] <= r["reorder_point"]
        if pd.notna(r.get("reorder_point")) and r["reorder_point"] > 0 else False, axis=1
    )

    def _status(row):
        wc = row["weeks_cover"]
        if wc <= 0: return "Out of Stock"
        if row.get("below_reorder", False): return "Below Reorder"
        if wc < 2: return "Critical"
        if wc < 4: return "Low"
        if wc < 8: return "Adequate"
        return "Healthy"

    merged["status"]          = merged.apply(_status, axis=1)
    merged["warehouse_label"] = merged["warehouse"].map(WAREHOUSE_LABELS).fillna(merged["warehouse"])

    return merged.sort_values(["warehouse", "weeks_cover"]).reset_index(drop=True)


def build_network_summary(dc_summary):
    if dc_summary.empty: return pd.DataFrame()
    return dc_summary.groupby("warehouse").agg(
        warehouse_label   = ("warehouse_label", "first"),
        total_skus        = ("variant_id", "nunique"),
        total_units       = ("units_on_hand", "sum"),
        out_of_stock      = ("status", lambda x: (x == "Out of Stock").sum()),
        critical          = ("status", lambda x: (x == "Critical").sum()),
        below_reorder     = ("below_reorder", "sum"),
        avg_weeks_cover   = ("weeks_cover", lambda x: round(x[x < 99].mean() if any(x < 99) else 0, 1)),
        total_replen_need = ("replen_need", lambda x: x.fillna(0).sum()),
    ).reset_index()


def build_replenishment_transfers(dc_summary, inventory, forecast):
    """
    Build transfer plan from primary facility (NJ) to regional DCs.

    For each DC × SKU:
      transfer_need = max(0, target_units - on_hand)
      OR
      transfer_need = max(0, (4 weeks of cover target) - on_hand) if no target

    Source = primary inventory (nj + nj2)
    Cadence: NV/IN twice per week, FL/TX once per week
    """
    if dc_summary.empty:
        return pd.DataFrame(), pd.DataFrame()

    primary_inv = (
        inventory[inventory["warehouse"].isin(PRIMARY_WAREHOUSES)]
        .groupby("variant_id")["units_on_hand"].sum().to_dict()
    )

    rows = []
    for _, r in dc_summary.iterrows():
        vid       = r["variant_id"]
        wh        = r["warehouse"]
        on_hand   = r["units_on_hand"]
        target    = r.get("target_units")
        avg_wk    = r["dc_avg_weekly"]

        # Calculate target if not set: aim for 4 weeks of cover at this DC
        if pd.isna(target) or target == 0:
            target = avg_wk * 4

        transfer_need = max(0, target - on_hand)
        if transfer_need < 1:
            continue

        primary_avail = primary_inv.get(vid, 0)
        can_fulfill   = min(transfer_need, primary_avail)
        shortage      = transfer_need - can_fulfill

        cadence = DC_SHIP_CADENCE.get(wh, 1)
        # Split transfer across cadence shipments per week
        per_shipment = round(can_fulfill / cadence) if cadence > 0 else can_fulfill

        # Priority based on weeks of cover
        wks_cover = r["weeks_cover"]
        if wks_cover < 1:    priority = "Urgent"
        elif wks_cover < 2:  priority = "High"
        elif wks_cover < 4:  priority = "Medium"
        else:                priority = "Low"

        rows.append({
            "variant_id":       vid,
            "product_name":     r.get("product_name", ""),
            "cat_l1":           r.get("cat_l1", ""),
            "destination":      wh,
            "destination_label": r["warehouse_label"],
            "current_on_hand":  int(on_hand),
            "target_units":     int(target),
            "transfer_need":    int(transfer_need),
            "primary_avail":    int(primary_avail),
            "can_fulfill":      int(can_fulfill),
            "shortage":         int(shortage),
            "shipments_per_wk": cadence,
            "per_shipment":     int(per_shipment),
            "weeks_cover":      wks_cover,
            "priority":         priority,
            "status":           r["status"],
        })

    transfers = pd.DataFrame(rows)
    if transfers.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Summary by destination
    summary = transfers.groupby(["destination", "destination_label"]).agg(
        total_skus      = ("variant_id", "nunique"),
        total_units     = ("can_fulfill", "sum"),
        urgent_count    = ("priority", lambda x: (x == "Urgent").sum()),
        high_count      = ("priority", lambda x: (x == "High").sum()),
        shortage_skus   = ("shortage", lambda x: (x > 0).sum()),
        shipments_per_wk = ("shipments_per_wk", "first"),
    ).reset_index()

    return transfers.sort_values(["destination", "priority", "weeks_cover"]), summary
