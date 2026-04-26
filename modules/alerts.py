"""
alerts.py
Alert generation engine.
Scans forecast, production, purchasing, and DC data
and returns a structured list of alerts by severity.
"""

import pandas as pd
import numpy as np

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def generate_alerts(
    forecast:        pd.DataFrame,
    prod_df:         pd.DataFrame,
    dc_summary:      pd.DataFrame,
    bom_gaps:        list,
    changeover_conflicts: list,
    marketing_cal:   pd.DataFrame = None,
) -> list[dict]:
    """
    Generate all alerts. Returns list sorted by severity.
    Each alert: {severity, category, title, body, variant_ids}
    """
    alerts = []

    # ── Forecast alerts ────────────────────────────────────────────────────────
    alerts += _forecast_alerts(forecast)

    # ── Production alerts ──────────────────────────────────────────────────────
    alerts += _production_alerts(prod_df)

    # ── Changeover conflicts ───────────────────────────────────────────────────
    alerts += _changeover_alerts(changeover_conflicts)

    # ── DC / inventory alerts ──────────────────────────────────────────────────
    if not dc_summary.empty:
        alerts += _dc_alerts(dc_summary)

    # ── BOM gap alerts ─────────────────────────────────────────────────────────
    alerts += _bom_gap_alerts(bom_gaps)

    # ── Missing data alerts ────────────────────────────────────────────────────
    if marketing_cal is None:
        alerts.append({
            "severity":    "info",
            "category":    "Data",
            "title":       "Marketing Calendar Not Loaded",
            "body":        "Promotional uplifts have not been applied to the forecast. "
                           "Upload the marketing calendar in Settings to factor in "
                           "campaigns and seasonal promotions.",
            "variant_ids": [],
        })

    # Sort by severity
    alerts.sort(key=lambda a: SEVERITY_ORDER.get(a["severity"], 99))
    return alerts


def _forecast_alerts(forecast: pd.DataFrame) -> list[dict]:
    alerts = []

    # Zero inventory with positive demand
    zero_inv = forecast[
        (forecast["inv_units"] == 0) &
        (forecast["avg_weekly"] > 5)
    ]
    if not zero_inv.empty:
        alerts.append({
            "severity":    "critical",
            "category":    "Inventory",
            "title":       f"{len(zero_inv)} Active SKUs Have Zero Inventory",
            "body":        "These SKUs have forecasted demand but no stock on hand: " +
                           ", ".join(zero_inv["variant_id"].tolist()[:5]) +
                           (f" and {len(zero_inv)-5} more." if len(zero_inv) > 5 else "."),
            "variant_ids": zero_inv["variant_id"].tolist(),
        })

    # W1 shortfall: on-hand < W1 demand
    if "weekly_forecast" in forecast.columns:
        w1 = forecast.copy()
        w1["w1_demand"] = w1["weekly_forecast"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 0
        )
        shortfall = w1[
            (w1["inv_units"] < w1["w1_demand"]) &
            (w1["w1_demand"] > 0)
        ]
        if not shortfall.empty:
            alerts.append({
                "severity":    "critical",
                "category":    "Inventory",
                "title":       f"{len(shortfall)} SKUs Have W1 Demand Exceeding On-Hand Stock",
                "body":        "Immediate replenishment or production required for: " +
                               ", ".join(shortfall["variant_id"].tolist()[:5]) +
                               (f" and {len(shortfall)-5} more." if len(shortfall) > 5 else "."),
                "variant_ids": shortfall["variant_id"].tolist(),
            })

    # Low confidence SKUs with significant demand
    low_conf = forecast[
        (forecast["confidence"] == "Low") &
        (forecast["avg_weekly"] > 10)
    ]
    if not low_conf.empty:
        alerts.append({
            "severity":    "warning",
            "category":    "Forecast",
            "title":       f"{len(low_conf)} High-Volume SKUs Have Low Forecast Confidence",
            "body":        "Less than 20 weeks of history for: " +
                           ", ".join(low_conf["product_name"].tolist()[:4]) +
                           (f" and {len(low_conf)-4} more." if len(low_conf) > 4 else ".") +
                           " Treat these forecasts conservatively.",
            "variant_ids": low_conf["variant_id"].tolist(),
        })

    # Significant YoY decline
    declining = forecast[
        (forecast["yoy"] < -0.20) &
        (forecast["confidence"].isin(["High", "Medium"]))
    ]
    if not declining.empty:
        alerts.append({
            "severity":    "warning",
            "category":    "Forecast",
            "title":       f"{len(declining)} SKUs Showing >20% YoY Decline",
            "body":        "Review for possible discontinuation or promotional support: " +
                           ", ".join(declining["product_name"].tolist()[:4]) +
                           (f" and {len(declining)-4} more." if len(declining) > 4 else "."),
            "variant_ids": declining["variant_id"].tolist(),
        })

    # Significant YoY growth
    growing = forecast[
        (forecast["yoy"] > 0.25) &
        (forecast["confidence"].isin(["High", "Medium"]))
    ]
    if not growing.empty:
        alerts.append({
            "severity":    "info",
            "category":    "Forecast",
            "title":       f"{len(growing)} SKUs Showing >25% YoY Growth",
            "body":        "Verify production capacity is sufficient: " +
                           ", ".join(growing["product_name"].tolist()[:4]) +
                           (f" and {len(growing)-4} more." if len(growing) > 4 else "."),
            "variant_ids": growing["variant_id"].tolist(),
        })

    # High volatility
    volatile = forecast[forecast["cv"] > 0.6] if "cv" in forecast.columns else pd.DataFrame()
    if not volatile.empty:
        alerts.append({
            "severity":    "info",
            "category":    "Forecast",
            "title":       f"{len(volatile)} SKUs Have High Demand Volatility (CV > 0.6)",
            "body":        "These items have erratic weekly sales patterns. "
                           "Consider higher safety stock or more frequent replenishment.",
            "variant_ids": volatile["variant_id"].tolist(),
        })

    return alerts


def _production_alerts(prod_df: pd.DataFrame) -> list[dict]:
    alerts = []
    if prod_df.empty:
        return alerts

    # SKUs needing production with no line assigned
    unassigned = prod_df[
        (prod_df["line_name"] == "Unassigned") &
        (prod_df["needs_production"] == True)
    ]
    if not unassigned.empty:
        alerts.append({
            "severity":    "warning",
            "category":    "Production",
            "title":       f"{len(unassigned)} SKUs Needing Production Have No Line Assigned",
            "body":        "Upload production_lines.csv in Settings to assign SKUs to lines "
                           "and enable constraint-based run scheduling.",
            "variant_ids": unassigned["variant_id"].tolist(),
        })

    # Runs flagged as below minimum
    below_min_skus = []
    for _, row in prod_df.iterrows():
        for run in row.get("runs", []):
            if run.get("flag") == "below_min":
                below_min_skus.append(row["variant_id"])
                break
    if below_min_skus:
        alerts.append({
            "severity":    "warning",
            "category":    "Production",
            "title":       f"{len(below_min_skus)} SKUs Have Runs Below Minimum Run Size",
            "body":        "Demand is too low to meet minimum run sizes. "
                           "Consider consolidating with similar SKUs or adjusting minimums.",
            "variant_ids": below_min_skus,
        })

    return alerts


def _changeover_alerts(conflicts: list) -> list[dict]:
    alerts = []
    if not conflicts:
        return alerts

    by_week = {}
    for c in conflicts:
        w = c["week_idx"] + 1
        by_week.setdefault(w, []).append(c)

    for week, week_conflicts in by_week.items():
        lines = list({c["line"] for c in week_conflicts})
        alerts.append({
            "severity":    "warning",
            "category":    "Production",
            "title":       f"W{week}: Allergen Changeover Required on {', '.join(lines)}",
            "body":        f"{len(week_conflicts)} allergen conflict(s) detected. "
                           "Ensure full line cleaning between runs. "
                           "Consider resequencing to minimise changeover frequency.",
            "variant_ids": list({c["sku_a"] for c in week_conflicts} |
                                {c["sku_b"] for c in week_conflicts}),
        })

    return alerts


def _dc_alerts(dc_summary: pd.DataFrame) -> list[dict]:
    alerts = []

    out_of_stock = dc_summary[dc_summary["status"] == "Out of Stock"]
    if not out_of_stock.empty:
        by_wh = out_of_stock.groupby("warehouse_label")["variant_id"].count()
        alerts.append({
            "severity":    "critical",
            "category":    "DC Network",
            "title":       f"{len(out_of_stock)} SKU-DC Combinations Are Out of Stock",
            "body":        "Out-of-stock by DC: " +
                           ", ".join([f"{wh}: {n}" for wh, n in by_wh.items()]) + ".",
            "variant_ids": out_of_stock["variant_id"].unique().tolist(),
        })

    critical = dc_summary[dc_summary["status"] == "Critical"]
    if not critical.empty:
        alerts.append({
            "severity":    "critical",
            "category":    "DC Network",
            "title":       f"{len(critical)} SKU-DC Combinations Are Critically Low (<2 weeks)",
            "body":        "Replenishment transfers needed immediately to prevent stockouts.",
            "variant_ids": critical["variant_id"].unique().tolist(),
        })

    below_reorder = dc_summary[dc_summary["below_reorder"] == True]
    if not below_reorder.empty:
        alerts.append({
            "severity":    "warning",
            "category":    "DC Network",
            "title":       f"{len(below_reorder)} SKU-DC Combinations Are Below Reorder Point",
            "body":        "These locations have fallen below their configured reorder threshold.",
            "variant_ids": below_reorder["variant_id"].unique().tolist(),
        })

    return alerts


def _bom_gap_alerts(bom_gaps: list) -> list[dict]:
    alerts = []
    if not bom_gaps:
        return alerts

    critical_gaps = [g for g in bom_gaps if g.get("severity") == "critical"]
    warning_gaps  = [g for g in bom_gaps if g.get("severity") == "warning"]

    if critical_gaps:
        alerts.append({
            "severity":    "critical",
            "category":    "BOM",
            "title":       f"{len(critical_gaps)} High-Demand SKUs Have No BOM",
            "body":        "Purchasing plan is incomplete for: " +
                           ", ".join([g["variant_id"] for g in critical_gaps[:5]]) +
                           (f" and {len(critical_gaps)-5} more." if len(critical_gaps) > 5 else "."),
            "variant_ids": [g["variant_id"] for g in critical_gaps],
        })

    if warning_gaps:
        alerts.append({
            "severity":    "warning",
            "category":    "BOM",
            "title":       f"{len(warning_gaps)} SKUs Have Incomplete BOM Data",
            "body":        "Missing raw material components for purchasing calculation.",
            "variant_ids": [g["variant_id"] for g in warning_gaps],
        })

    return alerts
