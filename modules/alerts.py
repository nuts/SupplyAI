"""
alerts.py
Alert generation across all modules.
"""

import pandas as pd

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def generate_alerts(forecast, prod_df, dc_summary, transfers,
                       bom_gaps, changeover_conflicts,
                       capacity_df=None, marketing_cal=None):
    alerts = []
    alerts += _forecast_alerts(forecast)
    alerts += _production_alerts(prod_df)
    alerts += _changeover_alerts(changeover_conflicts)
    if dc_summary is not None and not dc_summary.empty:
        alerts += _dc_alerts(dc_summary)
    if transfers is not None and not transfers.empty:
        alerts += _transfer_alerts(transfers)
    alerts += _bom_gap_alerts(bom_gaps)
    if capacity_df is not None and not capacity_df.empty:
        alerts += _capacity_alerts(capacity_df)
    if marketing_cal is None or marketing_cal.empty:
        alerts.append({
            "severity":    "info",
            "category":    "Data",
            "title":       "Marketing Calendar Not Loaded",
            "body":        "No promotional uplifts applied. Upload the marketing "
                           "calendar in Settings to factor in campaigns.",
            "variant_ids": [],
        })
    alerts.sort(key=lambda a: SEVERITY_ORDER.get(a["severity"], 99))
    return alerts


def _forecast_alerts(forecast):
    alerts = []
    zero_inv = forecast[(forecast["primary_inv"] == 0) & (forecast["avg_weekly"] > 5)]
    if not zero_inv.empty:
        alerts.append({
            "severity": "critical", "category": "Inventory",
            "title": f"{len(zero_inv)} Active SKUs Have Zero Primary Inventory",
            "body": "SKUs with demand but no NJ stock: " +
                    ", ".join(zero_inv["variant_id"].tolist()[:5]) +
                    (f" and {len(zero_inv)-5} more." if len(zero_inv)>5 else "."),
            "variant_ids": zero_inv["variant_id"].tolist(),
        })

    if "weekly_forecast" in forecast.columns:
        w1 = forecast.copy()
        w1["w1_demand"] = w1["weekly_forecast"].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 0)
        shortfall = w1[(w1["primary_inv"] < w1["w1_demand"]) & (w1["w1_demand"] > 0)]
        if not shortfall.empty:
            alerts.append({
                "severity": "critical", "category": "Inventory",
                "title": f"{len(shortfall)} SKUs Have W1 Demand > NJ Inventory",
                "body": "Production needed before week starts.",
                "variant_ids": shortfall["variant_id"].tolist(),
            })

    declining = forecast[(forecast["yoy"].fillna(0) < -0.20) &
                          (forecast["confidence"].isin(["High", "Medium"]))]
    if not declining.empty:
        alerts.append({
            "severity": "warning", "category": "Forecast",
            "title": f"{len(declining)} SKUs Showing >20% YoY Decline",
            "body": "Top decliners: " +
                    ", ".join(declining["product_name"].tolist()[:4]) +
                    (f" and {len(declining)-4} more." if len(declining)>4 else "."),
            "variant_ids": declining["variant_id"].tolist(),
        })

    growing = forecast[(forecast["yoy"].fillna(0) > 0.25) &
                        (forecast["confidence"].isin(["High", "Medium"]))]
    if not growing.empty:
        alerts.append({
            "severity": "info", "category": "Forecast",
            "title": f"{len(growing)} SKUs Showing >25% YoY Growth",
            "body": "Top growers: " +
                    ", ".join(growing["product_name"].tolist()[:4]) + ".",
            "variant_ids": growing["variant_id"].tolist(),
        })

    new_skus = forecast[forecast["yoy_status"] == "New"]
    if not new_skus.empty:
        alerts.append({
            "severity": "info", "category": "Forecast",
            "title": f"{len(new_skus)} SKUs Marked as New",
            "body": "These items have zero sales in either current or prior 4-week period. "
                    "YoY comparisons will activate once both periods have sales history.",
            "variant_ids": new_skus["variant_id"].tolist(),
        })
    return alerts


def _production_alerts(prod_df):
    alerts = []
    if prod_df.empty: return alerts
    unassigned = prod_df[(prod_df["line_name"] == "Unassigned") &
                          (prod_df["needs_production"] == True)]
    if not unassigned.empty:
        alerts.append({
            "severity": "warning", "category": "Production",
            "title": f"{len(unassigned)} SKUs Need Production But Have No Line",
            "body": "Upload production_lines.csv to assign SKUs to lines.",
            "variant_ids": unassigned["variant_id"].tolist(),
        })
    below_min = []
    for _, r in prod_df.iterrows():
        for run in r.get("runs", []):
            if run.get("flag") == "below_min":
                below_min.append(r["variant_id"]); break
    if below_min:
        alerts.append({
            "severity": "warning", "category": "Production",
            "title": f"{len(below_min)} SKUs Have Runs Below Minimum",
            "body": "Demand too low to hit min run size.",
            "variant_ids": below_min,
        })
    return alerts


def _changeover_alerts(conflicts):
    if not conflicts: return []
    by_week = {}
    for c in conflicts:
        by_week.setdefault(c["week_idx"] + 1, []).append(c)
    alerts = []
    for week, week_conflicts in by_week.items():
        lines = list({c["line"] for c in week_conflicts})
        alerts.append({
            "severity": "warning", "category": "Production",
            "title": f"W{week}: Allergen Changeover Required on {', '.join(lines)}",
            "body": f"{len(week_conflicts)} allergen conflict(s). Resequence to minimize.",
            "variant_ids": list({c["sku_a"] for c in week_conflicts} |
                                  {c["sku_b"] for c in week_conflicts}),
        })
    return alerts


def _dc_alerts(dc_summary):
    alerts = []
    oos = dc_summary[dc_summary["status"] == "Out of Stock"]
    if not oos.empty:
        by_wh = oos.groupby("warehouse_label")["variant_id"].count()
        alerts.append({
            "severity": "critical", "category": "DC Network",
            "title": f"{len(oos)} SKU-DC Combos Out of Stock",
            "body": "By DC: " + ", ".join([f"{wh}: {n}" for wh, n in by_wh.items()]),
            "variant_ids": oos["variant_id"].unique().tolist(),
        })
    crit = dc_summary[dc_summary["status"] == "Critical"]
    if not crit.empty:
        alerts.append({
            "severity": "critical", "category": "DC Network",
            "title": f"{len(crit)} SKU-DC Combos Critically Low (<2 wks)",
            "body": "Replenishment transfers needed immediately.",
            "variant_ids": crit["variant_id"].unique().tolist(),
        })
    return alerts


def _transfer_alerts(transfers):
    alerts = []
    shortages = transfers[transfers["shortage"] > 0]
    if not shortages.empty:
        alerts.append({
            "severity": "critical", "category": "Transfers",
            "title": f"{len(shortages)} Transfers Have Shortage at Primary",
            "body": "Primary facility cannot fulfill all DC transfer needs. "
                    "Production required to cover gap.",
            "variant_ids": shortages["variant_id"].unique().tolist(),
        })
    urgent = transfers[transfers["priority"] == "Urgent"]
    if not urgent.empty:
        alerts.append({
            "severity": "warning", "category": "Transfers",
            "title": f"{len(urgent)} Urgent Transfers Required",
            "body": "DCs at <1 week of cover need immediate replenishment.",
            "variant_ids": urgent["variant_id"].unique().tolist(),
        })
    return alerts


def _bom_gap_alerts(bom_gaps):
    if not bom_gaps: return []
    alerts = []
    crit = [g for g in bom_gaps if g.get("severity") == "critical"]
    warn = [g for g in bom_gaps if g.get("severity") == "warning"]
    if crit:
        alerts.append({
            "severity": "critical", "category": "BOM",
            "title": f"{len(crit)} High-Demand SKUs Have No BOM",
            "body": "Purchasing plan incomplete: " +
                    ", ".join([g["variant_id"] for g in crit[:5]]),
            "variant_ids": [g["variant_id"] for g in crit],
        })
    if warn:
        alerts.append({
            "severity": "warning", "category": "BOM",
            "title": f"{len(warn)} SKUs Have Incomplete BOM",
            "body": "Missing components for purchasing calculation.",
            "variant_ids": [g["variant_id"] for g in warn],
        })
    return alerts


def _capacity_alerts(capacity_df):
    alerts = []
    over = capacity_df[capacity_df["peak_util"] > 100]
    if not over.empty:
        alerts.append({
            "severity": "critical", "category": "Capacity",
            "title": f"{len(over)} Lines Have Weeks Over 100% Capacity",
            "body": "Lines: " + ", ".join(over["line_name"].tolist()) +
                    ". Production cannot fit — reschedule or expand capacity.",
            "variant_ids": [],
        })
    high = capacity_df[(capacity_df["peak_util"] >= 85) & (capacity_df["peak_util"] <= 100)]
    if not high.empty:
        alerts.append({
            "severity": "warning", "category": "Capacity",
            "title": f"{len(high)} Lines Have Peak Utilization Above 85%",
            "body": "Limited buffer for unexpected demand or downtime.",
            "variant_ids": [],
        })
    return alerts
