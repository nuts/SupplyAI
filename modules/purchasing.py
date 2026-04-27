"""
purchasing.py
Purchasing plan engine with MULTI-LEVEL BOM EXPLOSION.

Major change: recursively explodes BOM down to lowest level.
- "Make" items (those that appear as output_sku) are exploded into their components
- "Buy" items (those that only appear as input_sku) are treated as raw materials
- Cycles are detected and broken safely
"""

import pandas as pd
import numpy as np

FORECAST_WEEKS = 8
MAX_RECURSION_DEPTH = 10


def classify_make_buy(bom_df: pd.DataFrame) -> dict:
    """
    Classify every item in the BOM as Make or Buy.
    Returns dict: {item_id_string: 'make' | 'buy'}

    Detection logic:
    - 'make' = item is itself an output_sku (has its own BOM)
    - 'buy'  = no BOM exists for this item; treat as purchased raw material

    NOTE on data structure:
      output_sku is a variant_id like '3333-012R' (packaged form)
      input_sku is a product_id integer like 3333 (bulk form)
      In the current BOM, only the packaging step is modeled — bulk products
      like '3333' (Honey & Heat Snack Mix) appear as inputs but have no BOM
      describing how the bulk is made from raw ingredients.

      So this function flags items as "make" only when an explicit BOM exists
      for that exact ID. For bulk product_ids that are conceptually made on-site
      but have no bulk-level BOM, they will be treated as Buy at the leaf level
      (with a note in the alerts).
    """
    output_skus = set(bom_df["output_sku"].dropna().astype(str).unique())
    classification = {}
    input_items = bom_df["input_sku"].dropna().unique()
    for item in input_items:
        item_str = str(int(item)) if isinstance(item, (int, float)) else str(item)
        if item_str in output_skus:
            classification[item_str] = "make"
        else:
            classification[item_str] = "buy"
    for output in output_skus:
        classification[str(output)] = "make"
    return classification


def build_bom_index(bom_df: pd.DataFrame) -> dict:
    """
    Build a fast lookup: {output_sku: [list of bom rows as dicts]}
    """
    idx = {}
    for _, row in bom_df.iterrows():
        out = str(row["output_sku"])
        if pd.isna(row["output_sku"]): continue
        idx.setdefault(out, []).append({
            "input_sku":            str(row["input_sku"]) if pd.notna(row["input_sku"]) else None,
            "input_sku_int":        int(row["input_sku"]) if pd.notna(row["input_sku"]) else None,
            "input_sku_name":       row["input_sku_name"],
            "input_measurement":    float(row["input_measurement"]) if pd.notna(row["input_measurement"]) else 0,
            "input_unit_of_measure": row["input_unit_of_measure"],
        })
    return idx


def explode_recursive(output_sku: str,
                       finished_lbs: float,
                       finished_weight_lbs: float,
                       bom_index: dict,
                       depth: int = 0,
                       visited: set = None) -> list:
    """
    Recursively explode a finished good's BOM down to lowest-level (Buy) ingredients.

    finished_lbs:        how many lbs of the finished good we need to make
    finished_weight_lbs: weight per unit of the finished good (for unit conversion)

    Returns a flat list of leaf-level ingredient requirements:
    [{ingredient_id, ingredient_name, qty, unit, type, depth}, ...]
    """
    if visited is None: visited = set()
    if depth > MAX_RECURSION_DEPTH or output_sku in visited:
        # Treat as a Buy at this level (cycle break or too deep)
        return [{
            "ingredient_id":   output_sku,
            "ingredient_name": f"[CYCLE/DEPTH] {output_sku}",
            "qty":             finished_lbs,
            "unit":            "lbs",
            "type":            "raw_material",
            "depth":           depth,
        }]

    visited = visited | {output_sku}
    bom_lines = bom_index.get(output_sku, [])

    # If no BOM exists for this output_sku, treat as a Buy (lowest level)
    if not bom_lines:
        return [{
            "ingredient_id":   output_sku,
            "ingredient_name": output_sku,
            "qty":             finished_lbs,
            "unit":            "lbs",
            "type":            "raw_material",
            "depth":           depth,
        }]

    leaves = []

    for line in bom_lines:
        in_sku  = line["input_sku"]
        in_int  = line["input_sku_int"]
        in_name = line["input_sku_name"]
        meas    = line["input_measurement"]
        unit    = line["input_unit_of_measure"]

        if in_sku is None or in_int is None: continue

        # Both string-int and the original IDs are checked in BOM index
        in_key_int = str(in_int)

        if unit == "lbs":
            # Raw material in lbs — quantity is meas (lbs of input per finished lbs)
            qty_needed = (meas / finished_weight_lbs) * finished_lbs if finished_weight_lbs > 0 else 0

            # Recurse if this input is itself a Make item (has its own BOM)
            if in_key_int in bom_index:
                # This is a Make item — recurse
                # qty_needed is the lbs of the Make item we need
                # We need to know the weight of one unit of that Make item too
                # For sub-assembly raw materials (like a mix), the "weight" concept
                # is just lbs to lbs, so weight_lbs = 1 effectively
                sub_leaves = explode_recursive(
                    output_sku          = in_key_int,
                    finished_lbs        = qty_needed,
                    finished_weight_lbs = 1.0,
                    bom_index           = bom_index,
                    depth               = depth + 1,
                    visited             = visited,
                )
                leaves.extend(sub_leaves)
            else:
                # Pure raw material — leaf
                leaves.append({
                    "ingredient_id":   in_key_int,
                    "ingredient_name": in_name,
                    "qty":             round(qty_needed, 3),
                    "unit":            "lbs",
                    "type":            "raw_material",
                    "depth":           depth,
                })
        elif unit == "eaches":
            # Packaging — 1 packaging unit per finished unit
            # finished_lbs / finished_weight_lbs = number of finished units
            n_units = finished_lbs / finished_weight_lbs if finished_weight_lbs > 0 else 0
            qty_needed = round(n_units * meas, 0)
            leaves.append({
                "ingredient_id":   in_key_int,
                "ingredient_name": in_name,
                "qty":             qty_needed,
                "unit":            "eaches",
                "type":            "packaging",
                "depth":           depth,
            })

    return leaves


def explode_bom_full(prod_df: pd.DataFrame, bom_data: dict,
                       n_weeks: int = FORECAST_WEEKS) -> pd.DataFrame:
    """
    Multi-level BOM explosion.
    For each finished good in the production plan, recursively explode through
    the BOM until every requirement is a true raw material (Buy item).
    Returns one row per (week, leaf-ingredient, source-sku).
    """
    bom = bom_data["bom"]
    bom_index = build_bom_index(bom)

    rows = []

    for _, prod_row in prod_df.iterrows():
        vid          = prod_row["variant_id"]
        prod_lbs_wk  = prod_row.get("prod_by_week_lbs", [0.0] * n_weeks)
        weight_lbs   = prod_row.get("sku_weight_lbs", 0) or 1.0

        for wi, finished_lbs in enumerate(prod_lbs_wk):
            if finished_lbs <= 0: continue
            leaves = explode_recursive(
                output_sku         = str(vid),
                finished_lbs       = finished_lbs,
                finished_weight_lbs = weight_lbs,
                bom_index          = bom_index,
            )
            for leaf in leaves:
                rows.append({
                    "week_idx":        wi,
                    "ingredient_id":   leaf["ingredient_id"],
                    "ingredient_name": leaf["ingredient_name"],
                    "ingredient_type": leaf["type"],
                    "unit":            leaf["unit"],
                    "depth":           leaf["depth"],
                    "qty":             leaf["qty"],
                    "source_sku":      vid,
                    "source_product":  prod_row.get("product_name", ""),
                })

    return pd.DataFrame(rows)


def aggregate_purchasing(detail_df: pd.DataFrame, n_weeks: int = FORECAST_WEEKS) -> pd.DataFrame:
    """
    Aggregate ingredient requirements across all source SKUs and weeks.
    Returns one row per ingredient with weekly totals.
    """
    if detail_df.empty:
        return pd.DataFrame()

    rows = []
    for (ing_id, ing_name, ing_type, unit), grp in detail_df.groupby(
        ["ingredient_id", "ingredient_name", "ingredient_type", "unit"]
    ):
        weekly_totals = [0.0] * n_weeks
        for _, r in grp.iterrows():
            wi = int(r["week_idx"])
            if wi < n_weeks:
                weekly_totals[wi] += float(r["qty"])

        weekly_totals = [round(w, 1) for w in weekly_totals]
        total = round(sum(weekly_totals), 1)
        source_skus = grp["source_sku"].unique().tolist()

        rows.append({
            "ingredient_id":   ing_id,
            "ingredient_name": ing_name,
            "ingredient_type": ing_type,
            "unit":            unit,
            "source_skus":     source_skus,
            "n_source_skus":   len(source_skus),
            "weekly_need":     weekly_totals,
            "total_need":      total,
        })

    agg = pd.DataFrame(rows)
    type_order = {"raw_material": 0, "packaging": 1}
    agg["_type_order"] = agg["ingredient_type"].map(type_order).fillna(2)
    agg = agg.sort_values(["_type_order", "total_need"], ascending=[True, False]) \
              .drop(columns=["_type_order"]).reset_index(drop=True)
    return agg


def build_purchasing_plan(prod_df, bom_data, n_weeks=FORECAST_WEEKS):
    detail_df = explode_bom_full(prod_df, bom_data, n_weeks)
    if detail_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    agg_df = aggregate_purchasing(detail_df, n_weeks)
    return detail_df, agg_df


def flag_bom_gaps(forecast, bom_data):
    """
    Flag two types of BOM issues:
    1. Sales SKUs with no BOM at all
    2. "Bulk" intermediate items that are inputs to other BOMs but have no
       BOM of their own — these are likely made on-site but the recipe isn't
       captured in the BOM file.
    """
    bom = bom_data["bom"]
    sku_master = bom_data["sku_master"]
    bom_output_skus = set(bom["output_sku"].astype(str).unique())
    forecast_skus = set(forecast["variant_id"].unique())
    gaps = []

    # Type 1: Sales SKUs with no BOM
    no_bom = forecast_skus - bom_output_skus
    for vid in no_bom:
        row = forecast[forecast["variant_id"] == vid]
        avg = row["avg_weekly"].values[0] if len(row) > 0 else 0
        if avg > 0:
            gaps.append({
                "variant_id": vid,
                "issue":      "No BOM found",
                "severity":   "critical" if avg > 10 else "warning",
            })

    # Type 2: "Bulk" intermediate items used as inputs but have no own BOM
    # Identify product_ids that appear as input_sku (bulk forms)
    input_pids = set()
    for v in bom["input_sku"].dropna().unique():
        input_pids.add(str(int(v)) if isinstance(v, (int, float)) else str(v))

    # Get product_ids that have variants as output_skus (they're "Make" at variant level)
    make_pids = set()
    for vid in bom_output_skus:
        prefix = ""
        for ch in str(vid):
            if ch.isdigit(): prefix += ch
            else: break
        if prefix: make_pids.add(prefix)

    # Bulk items: input_pids that match make_pids = "we make these but no bulk BOM"
    bulk_no_recipe = input_pids & make_pids

    # Filter to only ones used in lbs (not packaging eaches)
    for pid in bulk_no_recipe:
        # Check if used as lbs input
        usage = bom[(bom["input_sku"].astype(str) == pid) &
                     (bom["input_unit_of_measure"] == "lbs")]
        if len(usage) > 0:
            name = usage.iloc[0]["input_sku_name"] if not usage.empty else pid
            gaps.append({
                "variant_id": pid,
                "issue":      f"Bulk recipe missing for {name} (treated as raw material)",
                "severity":   "info",
            })

    return gaps


# ── Supplier PO export ──────────────────────────────────────────────────────────
def build_supplier_pos(agg_df: pd.DataFrame, items_df: pd.DataFrame = None,
                          week_keys: list = None) -> dict:
    """
    Group purchasing aggregate by supplier (placeholder — uses ingredient_id mod 5
    to assign synthetic supplier IDs since we don't have a supplier field yet).
    Returns dict: {supplier_id: DataFrame of POs}
    """
    if agg_df.empty:
        return {}

    # Synthesize supplier from ingredient_id (placeholder for real supplier mapping)
    agg = agg_df.copy()
    agg["supplier_id"] = agg["ingredient_id"].astype(str).apply(
        lambda x: f"SUP-{(int(x) % 6) + 1:03d}" if x.replace("-","").isdigit() else "SUP-099"
    )

    pos = {}
    for sup, grp in agg.groupby("supplier_id"):
        po_rows = []
        for _, r in grp.iterrows():
            wn = r["weekly_need"]
            for wi, need in enumerate(wn):
                if need > 0 and week_keys and wi < len(week_keys):
                    po_rows.append({
                        "po_week":         week_keys[wi],
                        "ingredient_id":   r["ingredient_id"],
                        "ingredient_name": r["ingredient_name"],
                        "type":            r["ingredient_type"],
                        "qty":             need,
                        "unit":            r["unit"],
                    })
        if po_rows:
            pos[sup] = pd.DataFrame(po_rows)
    return pos
