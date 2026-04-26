"""
data_loader.py
Handles parsing, cleaning, and validation of all input files.
Supports both uploaded files (Streamlit UploadedFile) and local paths.
"""

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import io

# ── Warehouse mapping ──────────────────────────────────────────────────────────
WAREHOUSE_MAP = {
    "New Jersey": "nj",
    "Nevada":     "nv",
    "Indiana":    "in",
    "Texas":      "tx",
    "Florida":    "fl",
}

WAREHOUSE_LABELS = {
    "nj":  "New Jersey (Primary)",
    "nj2": "New Jersey 2 (Primary)",
    "nv":  "Nevada DC",
    "in":  "Indiana DC",
    "tx":  "Texas DC",
    "fl":  "Florida DC",
}

PRIMARY_WAREHOUSES  = {"nj", "nj2"}
REGIONAL_WAREHOUSES = {"nv", "in", "tx", "fl"}

# ── Allergen group decoder ─────────────────────────────────────────────────────
ALLERGEN_CODE_MAP = {
    "P":  "Peanut",
    "T":  "Tree Nut",
    "W":  "Wheat",
    "M":  "Milk",
    "S":  "Soy",
    "SE": "Sesame",
    "SO": "Sulfites",
    "SH": "Shellfish",
    "F":  "Fish",
    "E":  "Egg",
}


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Strip $, commas, whitespace and coerce to float."""
    return pd.to_numeric(
        series.astype(str)
              .str.replace(r"[\$,\s]", "", regex=True),
        errors="coerce"
    ).fillna(0)


# ── Sales ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_sales(source) -> pd.DataFrame:
    """
    Load order_nexus sales export.
    Returns clean DataFrame with standardised column names.
    """
    df = _read_source(source)

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

    # Clean numerics
    df["pieces_sold"]     = _clean_numeric(df["pieces_sold"])
    df["weight_sold_lbs"] = _clean_numeric(df["weight_sold_lbs"])
    df["sku_weight_lbs"]  = pd.to_numeric(df["sku_weight_lbs"], errors="coerce").fillna(0)

    # Parse year_week → year (int) and week (int)
    df[["year", "week"]] = df["year_week"].str.split("-", expand=True).astype(int)

    # Map warehouse name → code
    df["warehouse"] = df["warehouse_name"].map(WAREHOUSE_MAP).fillna(df["warehouse_name"])

    # Drop nulls on key fields
    df = df.dropna(subset=["variant_id", "year_week"])

    return df.reset_index(drop=True)


# ── Inventory ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_inventory(source) -> pd.DataFrame:
    """
    Load inventory snapshot (all warehouses).
    Returns one row per SKU per warehouse.
    """
    df = _read_source(source)

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

    # Rows with null warehouse are zero-stock catalog entries — keep but flag
    df["warehouse"] = df["warehouse"].fillna("none")

    return df.reset_index(drop=True)


# ── BOM + SKU master (Excel) ───────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_bom_excel(source) -> dict:
    """
    Load sku_master_w_BOMs.xlsx.
    Returns dict with keys: sku_master, bom, items, skus
    """
    raw = _read_bytes(source)

    # Sheet: sku_master w BOM
    bom_df = pd.read_excel(raw, sheet_name="sku_master w BOM", engine="openpyxl")
    bom_df.columns = [
        "product_id", "product_name", "cat_l1", "variant_id",
        "ounces", "packaging_format", "allergen_group",
        "output_sku", "input_sku", "input_sku_name",
        "input_measurement", "input_unit_of_measure"
    ]
    bom_df["product_id"]  = pd.to_numeric(bom_df["product_id"],  errors="coerce")
    bom_df["input_sku"]   = pd.to_numeric(bom_df["input_sku"],   errors="coerce")
    bom_df["ounces"]      = pd.to_numeric(bom_df["ounces"],       errors="coerce").fillna(0)
    bom_df["input_measurement"] = pd.to_numeric(
        bom_df["input_measurement"], errors="coerce").fillna(0)

    # Build SKU master from BOM (one row per variant)
    sku_master = (
        bom_df[["product_id", "product_name", "cat_l1", "variant_id",
                "ounces", "packaging_format", "allergen_group"]]
        .drop_duplicates(subset=["variant_id"])
        .reset_index(drop=True)
    )

    # Sheet: items (allergen detail)
    items_df = pd.read_excel(raw, sheet_name="items", engine="openpyxl")
    items_df = items_df.rename(columns={"external_id": "product_id"})
    items_df["product_id"] = pd.to_numeric(items_df["product_id"], errors="coerce")

    allergen_bool_cols = [
        "peanut", "tree_nut", "wheat", "milk", "soy",
        "shellfish", "fish", "egg", "sesame", "mustard", "sulfites"
    ]
    for col in allergen_bool_cols:
        if col in items_df.columns:
            items_df[col] = items_df[col].astype(bool)

    # Sheet: SKUs (replenishment targets)
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

    return {
        "sku_master": sku_master,
        "bom":        bom_df,
        "items":      items_df,
        "skus":       skus_df,
    }


# ── Production lines ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_production_lines(source) -> pd.DataFrame:
    """
    Load production_lines.csv
    Expected columns: line_name, min_run_lbs, max_run_lbs, sku_ids (semicolon-separated)
    """
    df = _read_source(source)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = {"line_name", "min_run_lbs", "max_run_lbs"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"production_lines.csv missing columns: {missing}")

    df["min_run_lbs"] = pd.to_numeric(df["min_run_lbs"], errors="coerce").fillna(0)
    df["max_run_lbs"] = pd.to_numeric(df["max_run_lbs"], errors="coerce").fillna(9999)

    # Parse SKU assignments if present
    if "sku_ids" in df.columns:
        df["sku_list"] = df["sku_ids"].astype(str).str.split(";").apply(
            lambda x: [s.strip() for s in x if s.strip()]
        )
    else:
        df["sku_list"] = [[] for _ in range(len(df))]

    return df.reset_index(drop=True)


# ── Marketing calendar ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_marketing_calendar(source) -> pd.DataFrame:
    """
    Load marketing_calendar.csv
    Expected columns: event_name, event_type, start_date, end_date,
                      affected_sku_ids, uplift_percent
    """
    df = _read_source(source)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["start_date"]    = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"]      = pd.to_datetime(df["end_date"],   errors="coerce")
    df["uplift_percent"] = pd.to_numeric(df["uplift_percent"], errors="coerce").fillna(0)
    return df.reset_index(drop=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def _read_source(source):
    """Accept file path string, Path, or Streamlit UploadedFile."""
    if isinstance(source, (str, Path)):
        return pd.read_csv(source)
    else:
        # Streamlit UploadedFile
        return pd.read_csv(io.BytesIO(source.read()))


def _read_bytes(source):
    """Return BytesIO for Excel reading."""
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            return io.BytesIO(f.read())
    else:
        return io.BytesIO(source.read())


# ── Validation ─────────────────────────────────────────────────────────────────
def validate_data(sales: pd.DataFrame, inventory: pd.DataFrame,
                  bom_data: dict) -> list[dict]:
    """
    Cross-validate loaded datasets.
    Returns list of {level, message} dicts.
    """
    issues = []
    sku_master = bom_data["sku_master"]

    sales_skus = set(sales["variant_id"].unique())
    inv_skus   = set(inventory["variant_id"].unique())
    bom_skus   = set(sku_master["variant_id"].unique())

    # SKUs in sales but not BOM
    missing_bom = sales_skus - bom_skus
    if missing_bom:
        issues.append({
            "level":   "warning",
            "message": f"{len(missing_bom)} sales SKUs not found in BOM/SKU master"
        })

    # SKUs in sales but not inventory
    missing_inv = sales_skus - inv_skus
    if missing_inv:
        issues.append({
            "level":   "info",
            "message": f"{len(missing_inv)} sales SKUs have no inventory record "
                       f"(may be dropship or inactive)"
        })

    # Zero-stock items with recent demand
    recent_sales = sales[sales["year"] >= sales["year"].max() - 1]
    recent_skus  = set(recent_sales["variant_id"].unique())
    inv_by_sku   = inventory.groupby("variant_id")["units_on_hand"].sum()
    zero_stock   = [v for v in recent_skus
                    if inv_by_sku.get(v, 0) == 0]
    if zero_stock:
        issues.append({
            "level":   "critical",
            "message": f"{len(zero_stock)} active SKUs have zero inventory across all locations"
        })

    # BOM completeness
    bom_df      = bom_data["bom"]
    lbs_skus    = set(bom_df[bom_df["input_unit_of_measure"] == "lbs"]["output_sku"])
    no_raw_mat  = bom_skus - lbs_skus
    if no_raw_mat:
        issues.append({
            "level":   "warning",
            "message": f"{len(no_raw_mat)} SKUs in BOM have no raw material (lbs) component"
        })

    return issues


# ── Allergen utilities ─────────────────────────────────────────────────────────
def decode_allergen_group(code: str) -> list[str]:
    """'T,S,M' → ['Tree Nut', 'Soy', 'Milk']"""
    if not code or pd.isna(code):
        return []
    parts = [p.strip() for p in str(code).split(",")]
    return [ALLERGEN_CODE_MAP.get(p, p) for p in parts if p]


def allergens_for_sku(variant_id: str, bom_data: dict) -> list[str]:
    """Return decoded allergen list for a variant."""
    row = bom_data["sku_master"][
        bom_data["sku_master"]["variant_id"] == variant_id
    ]
    if row.empty:
        return []
    return decode_allergen_group(row.iloc[0]["allergen_group"])
