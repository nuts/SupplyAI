"""
production.py
Production scheduling engine.
- Groups weeks of demand into optimal production runs
- Respects min/max run size per line
- Respects seasonal max-inventory ceiling (dynamic)
- Sequences runs to minimise allergen changeovers
- Returns weekly production schedule with named runs
"""

import pandas as pd
import numpy as np
from utils.data_loader import decode_allergen_group

FORECAST_WEEKS = 8


def get_current_season(week: int) -> str:
    """Map 4-4-5 week number to season label."""
    if week <= 13:
        return "Q1"
    elif week <= 26:
        return "Q2"
    elif week <= 39:
        return "Q3"
    else:
        return "Q4"


def compute_run_schedule(
    variant_id:     str,
    weekly_demand:  list[float],
    sku_weight_lbs: float,
    on_hand_units:  int,
    min_run_lbs:    float,
    max_run_lbs:    float,
    max_inv_weeks:  float,
    current_week:   int,
) -> list[dict]:
    """
    For a single SKU, compute optimal production runs over the forecast horizon.

    Strategy:
    1. Convert demand to lbs
    2. Find the run interval that maximises batch size within constraints
    3. Schedule runs at that interval, each as large as possible up to:
       - max_run_lbs (line capacity)
       - max_inv_weeks × avg_weekly_demand (inventory ceiling)
    4. Minimum run = min_run_lbs (if weekly demand never reaches min, flag it)

    Returns list of run dicts:
    {week_idx, run_lbs, units, covers_weeks, on_hand_after, flag}
    """
    n = len(weekly_demand)
    if n == 0:
        return []

    # Convert to lbs
    demand_lbs = [d * sku_weight_lbs for d in weekly_demand]
    avg_demand_lbs = np.mean(demand_lbs) if demand_lbs else 0

    if avg_demand_lbs <= 0:
        return []  # No demand — no production needed

    # Max run size = min(line max, inventory ceiling)
    inv_ceiling_lbs = max_inv_weeks * avg_demand_lbs
    effective_max   = min(max_run_lbs, inv_ceiling_lbs)

    # Optimal run interval = how many weeks one max run covers
    if avg_demand_lbs > 0:
        optimal_interval = int(effective_max / avg_demand_lbs)
        optimal_interval = max(1, min(optimal_interval, n))
    else:
        optimal_interval = n

    # Rolling inventory simulation
    inv_lbs     = on_hand_units * sku_weight_lbs
    runs        = []
    week_inv    = [inv_lbs]
    next_run_at = 0

    for wi in range(n):
        # Consume demand
        inv_lbs = max(0.0, inv_lbs - demand_lbs[wi])

        # Is a run scheduled this week?
        if wi >= next_run_at:
            # How much to produce?
            # Look ahead optimal_interval weeks of demand
            lookahead_lbs = sum(demand_lbs[wi: wi + optimal_interval])
            # Target to cover lookahead minus what we already have
            needed_lbs    = max(0.0, lookahead_lbs - inv_lbs)

            if needed_lbs >= min_run_lbs or inv_lbs <= 0:
                # Round up to min run if needed
                run_lbs = max(needed_lbs, min_run_lbs)
                # Cap at effective max
                run_lbs = min(run_lbs, effective_max)
                run_lbs = round(run_lbs, 1)

                run_units   = round(run_lbs / sku_weight_lbs) if sku_weight_lbs > 0 else 0
                inv_lbs    += run_lbs

                # Estimate how many weeks this run covers
                covers = 0
                tmp    = inv_lbs
                for fwi in range(wi, n):
                    if tmp <= 0:
                        break
                    tmp -= demand_lbs[fwi]
                    covers += 1

                flag = None
                if run_lbs < min_run_lbs:
                    flag = "below_min"
                elif run_lbs >= effective_max * 0.98:
                    flag = "at_capacity"

                runs.append({
                    "week_idx":       wi,
                    "run_lbs":        run_lbs,
                    "run_units":      run_units,
                    "covers_weeks":   covers,
                    "inv_after_lbs":  round(inv_lbs, 1),
                    "flag":           flag,
                })

                next_run_at = wi + covers

        week_inv.append(round(inv_lbs, 1))

    return runs


def build_production_schedule(
    forecast:        pd.DataFrame,
    inventory:       pd.DataFrame,
    production_lines: pd.DataFrame,
    bom_data:        dict,
    inv_max_by_season: dict  = None,
    current_week:    int     = 16,
    n_weeks:         int     = FORECAST_WEEKS,
) -> pd.DataFrame:
    """
    Build full production schedule across all SKUs and lines.
    Returns DataFrame with one row per SKU, with run schedule embedded.
    """
    if inv_max_by_season is None:
        inv_max_by_season = {"Q1": 12, "Q2": 12, "Q3": 14, "Q4": 20}

    season       = get_current_season(current_week)
    max_inv_wks  = inv_max_by_season.get(season, 14)

    sku_master   = bom_data["sku_master"]
    inv_by_sku   = (
        inventory[inventory["warehouse"] != "none"]
        .groupby("variant_id")["units_on_hand"]
        .sum()
        .to_dict()
    )

    # Build line lookup: variant_id → line config
    line_lookup = {}
    if production_lines is not None and not production_lines.empty:
        for _, line in production_lines.iterrows():
            for vid in line.get("sku_list", []):
                line_lookup[vid] = line
    else:
        # Default line if none configured — use production_path from inventory
        pass

    prod_rows = []

    for _, fc_row in forecast.iterrows():
        vid            = fc_row["variant_id"]
        weekly_demand  = fc_row.get("weekly_forecast", [])
        weight_lbs     = fc_row.get("sku_weight_lbs", 0) or 0
        on_hand        = int(inv_by_sku.get(vid, 0))
        allergen_grp   = None

        # Get allergens from SKU master
        sku_row = sku_master[sku_master["variant_id"] == vid]
        if not sku_row.empty:
            allergen_grp = sku_row.iloc[0].get("allergen_group", None)

        allergens = decode_allergen_group(allergen_grp) if allergen_grp else []

        # Get line config
        line = line_lookup.get(vid)
        if line is not None:
            min_run  = float(line["min_run_lbs"])
            max_run  = float(line["max_run_lbs"])
            line_name = str(line["line_name"])
        else:
            # No line assigned yet — use sensible defaults
            min_run   = 100.0
            max_run   = 5000.0
            line_name = "Unassigned"

        runs = compute_run_schedule(
            variant_id     = vid,
            weekly_demand  = weekly_demand,
            sku_weight_lbs = weight_lbs,
            on_hand_units  = on_hand,
            min_run_lbs    = min_run,
            max_run_lbs    = max_run,
            max_inv_weeks  = max_inv_wks,
            current_week   = current_week,
        )

        # Build weekly production array (lbs per week)
        prod_by_week = [0.0] * n_weeks
        for run in runs:
            wi = run["week_idx"]
            if wi < n_weeks:
                prod_by_week[wi] = run["run_lbs"]

        total_prod_lbs  = round(sum(prod_by_week), 1)
        total_prod_units = round(total_prod_lbs / weight_lbs) if weight_lbs > 0 else 0
        n_runs           = len(runs)

        # Weeks of coverage after production
        total_demand_lbs = sum(d * weight_lbs for d in weekly_demand)
        ending_inv_lbs   = (on_hand * weight_lbs) + total_prod_lbs - total_demand_lbs
        ending_inv_units = round(ending_inv_lbs / weight_lbs) if weight_lbs > 0 else 0

        prod_rows.append({
            "variant_id":        vid,
            "product_name":      fc_row.get("product_name", ""),
            "cat_l1":            fc_row.get("cat_l1", ""),
            "cat_l2":            fc_row.get("cat_l2", ""),
            "variant_name":      fc_row.get("variant_name", ""),
            "sku_weight_lbs":    weight_lbs,
            "line_name":         line_name,
            "allergens":         allergens,
            "allergen_group":    allergen_grp or "",
            "on_hand_units":     on_hand,
            "weekly_demand":     weekly_demand,
            "prod_by_week_lbs":  prod_by_week,
            "runs":              runs,
            "n_runs":            n_runs,
            "total_prod_lbs":    total_prod_lbs,
            "total_prod_units":  total_prod_units,
            "ending_inv_units":  max(0, ending_inv_units),
            "min_run_lbs":       min_run,
            "max_run_lbs":       max_run,
            "max_inv_weeks":     max_inv_wks,
            "needs_production":  total_prod_lbs > 0,
        })

    prod_df = pd.DataFrame(prod_rows)
    prod_df = prod_df.sort_values(
        ["line_name", "allergen_group", "product_name"]
    ).reset_index(drop=True)

    return prod_df


def detect_changeover_conflicts(prod_df: pd.DataFrame, n_weeks: int = FORECAST_WEEKS) -> list[dict]:
    """
    Identify weeks where allergen changeovers are required on the same line.
    Returns list of conflict dicts.
    """
    conflicts = []

    for line_name, line_grp in prod_df.groupby("line_name"):
        for wi in range(n_weeks):
            # SKUs with production in this week on this line
            active = line_grp[line_grp["prod_by_week_lbs"].apply(
                lambda x: x[wi] > 0 if isinstance(x, list) and len(x) > wi else False
            )]
            if len(active) <= 1:
                continue

            # Check for allergen conflicts
            allergen_sets = active["allergens"].tolist()
            all_allergens = set()
            for a_list in allergen_sets:
                all_allergens.update(a_list)

            for i, row_a in active.iterrows():
                for j, row_b in active.iterrows():
                    if i >= j:
                        continue
                    a_allergens = set(row_a["allergens"])
                    b_allergens = set(row_b["allergens"])
                    # Conflict if one has allergen other doesn't
                    if a_allergens != b_allergens:
                        conflicts.append({
                            "week_idx":   wi,
                            "line":       line_name,
                            "sku_a":      row_a["variant_id"],
                            "sku_b":      row_b["variant_id"],
                            "allergens_a": list(a_allergens),
                            "allergens_b": list(b_allergens),
                        })

    return conflicts
