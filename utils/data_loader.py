"""
data_loader.py
Parses, cleans, and validates all input files.
Integrates with persistence layer for cache support.
"""

import pandas as pd
import numpy as np
import io
from pathlib import Path

# ── Warehouse mapping ──────────────────────────────────────────────────────────
WAREHOUSE_MAP = {
    "New Jersey": "nj",
    "Nevada":     "nv",
    "Indiana":    "in",
    "Texas":      "tx",
    "Florida":    "fl",
}

WAREHOUSE_LABELS = {
    "nj":  "NJ Primary",
    "nj2": "NJ Primary 2",
    "nv":  "Nevada DC",
    "in":  "Indiana DC",
    "tx":  "Texas DC",
    "fl":  "Florida DC",
}

PRIMARY_WAREHOUSES  = {"nj", "nj2"}
REGIONAL_WAREHOUSES = {"nv", "in", "tx", "fl"}

# Ship cadence per regional DC (per week)
DC_SHIP_CADENCE = {
    "nv": 2,   # Nevada — twice per week
    "in": 2,   # Indiana — twice per week
    "tx": 1,   # Texas — once per week
    "fl": 1,   # Florida — once per week
}

# ── Allergen group decoder ─────────────────────────────────────────────────────
ALLERGEN_CODE_MAP = {
    "P":  "Peanut",      "T":  "Tree Nut",   "W":  "Wheat",
    "M":  "Milk",        "S":  "Soy",        "SE": "Sesame",
    "SO": "Sulfites",    "SH": "Shellfish",  "F":  "Fish",
    "E":  "Egg",
}


def _clean_numeric(series):
    return pd.to_numeric(
        series.astype(str).str.replace(r"[\$,\s]", "", regex=True),
        errors="coerce"
    ).fillna(0)


# ── Sales ──────────────────────────────────────────────────────────────────────
def load_sales(source) -> pd.DataFrame:
    df = _read_csv(source)
    rename = {
        "Calendar (Dynamic) Retail Year Week Concat": "year_week",
        "Order SKUs SKU ID":                          "variant_id",
        "Shipments Warehouse":                        "warehouse_name",
        "Product Hierarchy Product ID":               "product_id",
        "Product Hierarchy Product Name":             "product_name",
        "Product Hierarchy Reporting Category L1":    "cat_l1",
        "Product Hierarchy Reporting Category L2":    "cat_l2",
        "Product Hierarchy Variant Name":             "variant_name",
        "Product Hierarchy SKU Weight":               "sku_weight_lbs",
        "Order SKUs Gross Pieces Sold":               "pieces_sold",
        "Order SKUs Gross Weight Sold":               "weight_sold_lbs",
    }
    df = df.rename(columns=rename)
    df["pieces_sold"]     = _clean_numeric(df["pieces_sold"])
    df["weight_sold_lbs"] = _clean_numeric(df["weight_sold_lbs"])
    df["sku_weight_lbs"]  = pd.to_numeric(df["sku_weight_lbs"], errors="coerce").fillna(0)
    df[["year", "week"]]  = df["year_week"].str.split("-", expand=True).astype(int)
    df["warehouse"]       = df["warehouse_name"].map(WAREHOUSE_MAP).fillna(df["warehouse_name"])
    df = df.dropna(subset=["variant_id", "year_week"])
    return df.reset_index(drop=True)


# ── Inventory ──────────────────────────────────────────────────────────────────
def load_inventory(source) -> pd.DataFrame:
    df = _read_csv(source)
    rename = {
        "Warehouse":           "warehouse",
        "SKU":                 "variant_id",
        "Product Name":        "product_name",
        "Production Category": "production_path",
        "Count":               "units_on_hand",
        "Product ID":          "product_id",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["units_on_hand"] = _clean_numeric(df["units_on_hand"])
    df["warehouse"]     = df["warehouse"].fillna("none")
    return df.reset_index(drop=True)


# ── BOM Excel ──────────────────────────────────────────────────────────────────
def load_bom_excel(source) -> dict:
    raw = _read_bytes(source)
    bom_df = pd.read_excel(raw, sheet_name="sku_master w BOM", engine="openpyxl")
    bom_df.columns = [
        "product_id", "product_name", "cat_l1", "variant_id",
        "ounces", "packaging_format", "allergen_group",
        "output_sku", "input_sku", "input_sku_name",
        "input_measurement", "input_unit_of_measure"
    ]
    bom_df["product_id"]        = pd.to_numeric(bom_df["product_id"],        errors="coerce")
    bom_df["input_sku"]         = pd.to_numeric(bom_df["input_sku"],         errors="coerce")
    bom_df["ounces"]             = pd.to_numeric(bom_df["ounces"],            errors="coerce").fillna(0)
    bom_df["input_measurement"] = pd.to_numeric(bom_df["input_measurement"], errors="coerce").fillna(0)

    sku_master = (
        bom_df[["product_id", "product_name", "cat_l1", "variant_id",
                "ounces", "packaging_format", "allergen_group"]]
        .drop_duplicates(subset=["variant_id"])
        .reset_index(drop=True)
    )

    # ── items sheet (allergens, flags) ────────────────────────────────────────
    try:
        items_df = pd.read_excel(raw, sheet_name="items", engine="openpyxl")
        items_df = items_df.rename(columns={"external_id": "product_id"})
        items_df["product_id"] = pd.to_numeric(items_df["product_id"], errors="coerce")
    except Exception:
        items_df = pd.DataFrame()

    # ── SKUs sheet (replenishment targets) ────────────────────────────────────
    try:
        skus_df = pd.read_excel(raw, sheet_name="SKUs", engine="openpyxl")
        skus_df = skus_df.rename(columns={
            "external_id":             "variant_id",
            "item.name":               "product_name",
            "net_weight":              "net_weight_lbs",
            "target_pieces_on_hand":   "target_units",
            "replenishment_threshold": "reorder_point",
            "produce_for":             "produce_for_warehouse",
        })
        skus_df["target_units"]  = pd.to_numeric(skus_df["target_units"],  errors="coerce")
        skus_df["reorder_point"] = pd.to_numeric(skus_df["reorder_point"], errors="coerce")
    except Exception:
        skus_df = pd.DataFrame()

    return {
        "sku_master": sku_master,
        "bom":        bom_df,
        "items":      items_df,
        "skus":       skus_df,
    }


# ── Production lines (with placeholder generator) ─────────────────────────────
def load_production_lines(source) -> pd.DataFrame:
    df = _read_csv(source)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    required = {"line_name", "min_run_lbs", "max_run_lbs"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"production_lines missing: {missing}")
    df["min_run_lbs"] = pd.to_numeric(df["min_run_lbs"], errors="coerce").fillna(0)
    df["max_run_lbs"] = pd.to_numeric(df["max_run_lbs"], errors="coerce").fillna(9999)
    if "sku_ids" in df.columns:
        df["sku_list"] = df["sku_ids"].astype(str).str.split(";").apply(
            lambda x: [s.strip() for s in x if s.strip()]
        )
    else:
        df["sku_list"] = [[] for _ in range(len(df))]
    return df.reset_index(drop=True)


def make_placeholder_production_lines(sku_master: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-generate placeholder production lines based on category L1.
    Used when no production_lines.csv has been uploaded.
    """
    if sku_master is None or sku_master.empty:
        return pd.DataFrame()

    cats = sku_master["cat_l1"].dropna().unique().tolist()
    rows = []
    for i, cat in enumerate(cats, start=1):
        sku_list = sku_master[sku_master["cat_l1"] == cat]["variant_id"].tolist()
        rows.append({
            "line_name":   f"Line {i} ({cat})",
            "min_run_lbs": 200.0,
            "max_run_lbs": 4000.0,
            "sku_ids":     ";".join(sku_list),
            "sku_list":    sku_list,
        })
    return pd.DataFrame(rows)


# ── Marketing calendar ─────────────────────────────────────────────────────────
def load_marketing_calendar(source) -> pd.DataFrame:
    df = _read_csv(source)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["start_date"]    = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"]      = pd.to_datetime(df["end_date"],   errors="coerce")
    df["uplift_percent"] = pd.to_numeric(df["uplift_percent"], errors="coerce").fillna(0)
    return df.reset_index(drop=True)


def make_placeholder_marketing_calendar() -> pd.DataFrame:
    """Empty placeholder."""
    return pd.DataFrame(columns=[
        "event_name", "event_type", "start_date", "end_date",
        "affected_sku_ids", "uplift_percent"
    ])


# ── Helpers ────────────────────────────────────────────────────────────────────
def _read_csv(source):
    if isinstance(source, (str, Path)):
        return pd.read_csv(source)
    if isinstance(source, bytes):
        return pd.read_csv(io.BytesIO(source))
    return pd.read_csv(io.BytesIO(source.read()))


def _read_bytes(source):
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            return io.BytesIO(f.read())
    if isinstance(source, bytes):
        return io.BytesIO(source)
    return io.BytesIO(source.read())


# ── File-bytes capture (for caching) ───────────────────────────────────────────
def capture_bytes(source) -> bytes:
    """Capture raw file bytes from upload for caching."""
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            return f.read()
    if isinstance(source, bytes):
        return source
    source.seek(0)
    return source.read()


# ── Validation ─────────────────────────────────────────────────────────────────
def validate_data(sales, inventory, bom_data) -> list:
    issues = []
    sku_master    = bom_data["sku_master"]
    sales_skus    = set(sales["variant_id"].unique())
    inv_skus      = set(inventory["variant_id"].unique())
    bom_skus      = set(sku_master["variant_id"].unique())

    missing_bom   = sales_skus - bom_skus
    if missing_bom:
        issues.append({
            "level":   "warning",
            "message": f"{len(missing_bom)} sales SKUs not found in BOM/SKU master"
        })

    missing_inv   = sales_skus - inv_skus
    if missing_inv:
        issues.append({
            "level":   "info",
            "message": f"{len(missing_inv)} sales SKUs have no inventory record"
        })

    # Inventory sanity check — flag SKUs with extreme weeks-of-cover
    if not sales.empty:
        avg_weekly = sales.groupby("variant_id")["pieces_sold"].sum() / max(
            sales["year_week"].nunique(), 1)
        inv_total  = inventory[inventory["warehouse"] != "none"].groupby(
            "variant_id")["units_on_hand"].sum()
        cover = (inv_total / avg_weekly).replace([np.inf, -np.inf], np.nan).dropna()
        excessive = cover[cover > 52]
        if len(excessive) > 0:
            issues.append({
                "level":   "warning",
                "message": f"{len(excessive)} SKUs have >52 weeks of inventory cover. "
                           "May indicate inventory data quality issue (e.g. duplicated warehouse rows)."
            })

    return issues


# ── Allergen utilities ─────────────────────────────────────────────────────────
def decode_allergen_group(code) -> list:
    if not code or pd.isna(code):
        return []
    parts = [p.strip() for p in str(code).split(",")]
    return [ALLERGEN_CODE_MAP.get(p, p) for p in parts if p]
