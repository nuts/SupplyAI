"""
production.py
Production scheduling engine — system-optimized run grouping.
"""

import pandas as pd
import numpy as np
from utils.data_loader import decode_allergen_group

FORECAST_WEEKS = 8


def get_current_season(week):
    if week <= 13: return "Q1"
    if week <= 26: return "Q2"
    if week <= 39: return "Q3"
    return "Q4"


def compute_run_schedule(weekly_demand, sku_weight_lbs, on_hand_units,
                          min_run_lbs, max_run_lbs, max_inv_weeks):
    """Build run schedule for one SKU."""
    n = len(weekly_demand)
    if n == 0 or sku_weight_lbs <= 0:
        return []

    demand_lbs    = [d * sku_weight_lbs for d in weekly_demand]
    avg_demand_lbs = np.mean(demand_lbs) if demand_lbs else 0
    if avg_demand_lbs <= 0:
        return []

    inv_ceiling_lbs = max_inv_weeks * avg_demand_lbs
    effective_max   = min(max_run_lbs, inv_ceiling_lbs)
    optimal_interval = max(1, min(int(effective_max / avg_demand_lbs), n)) \
                       if avg_demand_lbs > 0 else n

    inv_lbs     = on_hand_units * sku_weight_lbs
    runs        = []
    next_run_at = 0

    for wi in range(n):
        inv_lbs = max(0.0, inv_lbs - demand_lbs[wi])
        if wi >= next_run_at:
            lookahead_lbs = sum(demand_lbs[wi: wi + optimal_interval])
            needed_lbs = max(0.0, lookahead_lbs - inv_lbs)
            if needed_lbs >= min_run_lbs or inv_lbs <= 0:
                run_lbs = max(needed_lbs, min_run_lbs)
                run_lbs = round(min(run_lbs, effective_max), 1)
                run_units = round(run_lbs / sku_weight_lbs) if sku_weight_lbs > 0 else 0
                inv_lbs += run_lbs
                covers = 0
                tmp    = inv_lbs
                for fwi in range(wi, n):
                    if tmp <= 0: break
                    tmp -= demand_lbs[fwi]
                    covers += 1
                flag = None
                if run_lbs < min_run_lbs:    flag = "below_min"
                elif run_lbs >= effective_max * 0.98: flag = "at_capacity"
                runs.append({
                    "week_idx":      wi,
                    "run_lbs":       run_lbs,
                    "run_units":     run_units,
                    "covers_weeks":  covers,
                    "inv_after_lbs": round(inv_lbs, 1),
                    "flag":          flag,
                })
                next_run_at = wi + covers
    return runs


def build_production_schedule(forecast, inventory, production_lines, bom_data,
                                inv_max_by_season=None,
                                current_week=17,
                                n_weeks=FORECAST_WEEKS):
    if inv_max_by_season is None:
        inv_max_by_season = {"Q1": 12, "Q2": 12, "Q3": 14, "Q4": 20}
    season       = get_current_season(current_week)
    max_inv_wks  = inv_max_by_season.get(season, 14)

    sku_master = bom_data["sku_master"]
    inv_by_sku = (
        inventory[inventory["warehouse"] != "none"]
        .groupby("variant_id")["units_on_hand"].sum().to_dict()
    )

    line_lookup = {}
    if production_lines is not None and not production_lines.empty:
        for _, line in production_lines.iterrows():
            for vid in line.get("sku_list", []):
                line_lookup[vid] = line

    rows = []
    for _, fc_row in forecast.iterrows():
        vid      = fc_row["variant_id"]
        weekly_demand = fc_row.get("weekly_forecast", [])
        weight_lbs = fc_row.get("sku_weight_lbs", 0) or 0
        on_hand    = int(inv_by_sku.get(vid, 0))

        sku_row = sku_master[sku_master["variant_id"] == vid]
        allergen_grp = sku_row.iloc[0].get("allergen_group", None) if not sku_row.empty else None
        allergens    = decode_allergen_group(allergen_grp) if allergen_grp else []

        line = line_lookup.get(vid)
        if line is not None:
            min_run, max_run = float(line["min_run_lbs"]), float(line["max_run_lbs"])
            line_name = str(line["line_name"])
        else:
            min_run, max_run, line_name = 100.0, 5000.0, "Unassigned"

        runs = compute_run_schedule(weekly_demand, weight_lbs, on_hand,
                                       min_run, max_run, max_inv_wks)

        prod_by_week = [0.0] * n_weeks
        for run in runs:
            wi = run["week_idx"]
            if wi < n_weeks: prod_by_week[wi] = run["run_lbs"]

        total_prod_lbs   = round(sum(prod_by_week), 1)
        total_prod_units = round(total_prod_lbs / weight_lbs) if weight_lbs > 0 else 0
        total_demand_lbs = sum(d * weight_lbs for d in weekly_demand)
        ending_inv_lbs   = (on_hand * weight_lbs) + total_prod_lbs - total_demand_lbs
        ending_inv_units = round(ending_inv_lbs / weight_lbs) if weight_lbs > 0 else 0

        rows.append({
            "variant_id":       vid,
            "product_name":     fc_row.get("product_name", ""),
            "cat_l1":           fc_row.get("cat_l1", ""),
            "cat_l2":           fc_row.get("cat_l2", ""),
            "variant_name":     fc_row.get("variant_name", ""),
            "sku_weight_lbs":   weight_lbs,
            "line_name":        line_name,
            "allergens":        allergens,
            "allergen_group":   allergen_grp or "",
            "on_hand_units":    on_hand,
            "weekly_demand":    weekly_demand,
            "prod_by_week_lbs": prod_by_week,
            "runs":             runs,
            "n_runs":           len(runs),
            "total_prod_lbs":   total_prod_lbs,
            "total_prod_units": total_prod_units,
            "ending_inv_units": max(0, ending_inv_units),
            "min_run_lbs":      min_run,
            "max_run_lbs":      max_run,
            "max_inv_weeks":    max_inv_wks,
            "needs_production": total_prod_lbs > 0,
        })

    return pd.DataFrame(rows).sort_values(
        ["line_name", "allergen_group", "product_name"]).reset_index(drop=True)


def detect_changeover_conflicts(prod_df, n_weeks=FORECAST_WEEKS):
    conflicts = []
    for line_name, line_grp in prod_df.groupby("line_name"):
        if line_name == "Unassigned": continue
        for wi in range(n_weeks):
            active = line_grp[line_grp["prod_by_week_lbs"].apply(
                lambda x: x[wi] > 0 if isinstance(x, list) and len(x) > wi else False)]
            if len(active) <= 1: continue
            for i, row_a in active.iterrows():
                for j, row_b in active.iterrows():
                    if i >= j: continue
                    a_set = set(row_a["allergens"])
                    b_set = set(row_b["allergens"])
                    if a_set != b_set:
                        conflicts.append({
                            "week_idx": wi,
                            "line":     line_name,
                            "sku_a":    row_a["variant_id"],
                            "sku_b":    row_b["variant_id"],
                            "allergens_a": list(a_set),
                            "allergens_b": list(b_set),
                        })
    return conflicts


def build_capacity_utilization(prod_df, production_lines, n_weeks=FORECAST_WEEKS):
    """
    Calculate per-line per-week utilization %.
    Returns DataFrame: lines × weeks → utilization %.
    """
    if production_lines is None or production_lines.empty:
        return pd.DataFrame()

    rows = []
    for _, line in production_lines.iterrows():
        line_name = line["line_name"]
        max_per_week = float(line["max_run_lbs"])
        # Sum production scheduled for SKUs on this line per week
        line_skus = prod_df[prod_df["line_name"] == line_name]
        weekly_lbs = [0.0] * n_weeks
        for _, sku in line_skus.iterrows():
            for wi, lbs in enumerate(sku.get("prod_by_week_lbs", [])):
                if wi < n_weeks: weekly_lbs[wi] += lbs

        utilization = [
            round(w / max_per_week * 100, 1) if max_per_week > 0 else 0
            for w in weekly_lbs
        ]

        rows.append({
            "line_name":      line_name,
            "max_run_lbs":    max_per_week,
            "weekly_lbs":     [round(w, 1) for w in weekly_lbs],
            "utilization_pct": utilization,
            "avg_util":       round(np.mean(utilization), 1),
            "peak_util":      round(max(utilization) if utilization else 0, 1),
            "n_skus":         len(line_skus),
        })
    return pd.DataFrame(rows)
