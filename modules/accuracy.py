"""
accuracy.py
Forecast vs Actual tracking.
Compares locked snapshots against actual sales to compute MAPE and bias.
"""

import pandas as pd
import numpy as np


def compute_accuracy(snapshot_forecast: pd.DataFrame,
                       actual_sales:       pd.DataFrame) -> pd.DataFrame:
    """
    Compare a locked forecast snapshot to actual sales for the forecasted period.

    snapshot_forecast: forecast df with weekly_forecast list and current_week metadata
    actual_sales: full sales df

    Returns variant-level accuracy metrics.
    """
    if snapshot_forecast is None or snapshot_forecast.empty:
        return pd.DataFrame()

    # We need to know which weeks were forecasted. The snapshot uses CURRENT_WEEK at
    # save time — we'll embed this as a 'snapshot_week' column when saving. For now,
    # use the first week label if present.
    rows = []
    for _, fc_row in snapshot_forecast.iterrows():
        vid    = fc_row["variant_id"]
        weekly_fc = fc_row.get("weekly_forecast", [])
        snap_week  = fc_row.get("snapshot_week", None)
        snap_year  = fc_row.get("snapshot_year", None)

        if snap_week is None or snap_year is None:
            continue

        actuals = []
        forecasts = []
        for wi, fc_val in enumerate(weekly_fc):
            target_week = ((snap_week - 1 + wi) % 52) + 1
            target_year = snap_year + ((snap_week - 1 + wi) // 52)
            actual_data = actual_sales[
                (actual_sales["variant_id"] == vid) &
                (actual_sales["year"] == target_year) &
                (actual_sales["week"] == target_week)
            ]
            actual_val = actual_data["pieces_sold"].sum() if not actual_data.empty else 0
            actuals.append(actual_val)
            forecasts.append(fc_val)

        # Only include weeks where we have actual data
        valid = [(f, a) for f, a in zip(forecasts, actuals) if a > 0]
        if not valid:
            continue

        n      = len(valid)
        forecasts_v = [v[0] for v in valid]
        actuals_v   = [v[1] for v in valid]

        # MAPE: mean absolute percentage error
        mape = np.mean([abs(f - a) / a * 100 for f, a in valid])
        # Bias: signed % error (positive = forecast too high)
        bias = np.mean([(f - a) / a * 100 for f, a in valid])
        # Total volume comparison
        total_f = sum(forecasts_v)
        total_a = sum(actuals_v)

        rows.append({
            "variant_id":    vid,
            "product_name":  fc_row.get("product_name", ""),
            "cat_l1":        fc_row.get("cat_l1", ""),
            "n_weeks":       n,
            "total_forecast": round(total_f, 0),
            "total_actual":   round(total_a, 0),
            "mape":          round(mape, 1),
            "bias":          round(bias, 1),
        })

    return pd.DataFrame(rows)


def aggregate_accuracy_by_category(acc_df: pd.DataFrame) -> pd.DataFrame:
    if acc_df.empty: return pd.DataFrame()
    return acc_df.groupby("cat_l1").agg(
        n_skus         = ("variant_id", "nunique"),
        avg_mape       = ("mape", "mean"),
        avg_bias       = ("bias", "mean"),
        total_forecast = ("total_forecast", "sum"),
        total_actual   = ("total_actual", "sum"),
    ).round(1).reset_index()


def overall_accuracy_metrics(acc_df: pd.DataFrame) -> dict:
    if acc_df.empty:
        return {
            "n_variants": 0,
            "weighted_mape": None,
            "weighted_bias": None,
            "median_mape": None,
        }

    # Volume-weighted MAPE
    total_actual = acc_df["total_actual"].sum()
    if total_actual > 0:
        weighted_mape = (acc_df["mape"] * acc_df["total_actual"]).sum() / total_actual
        weighted_bias = (acc_df["bias"] * acc_df["total_actual"]).sum() / total_actual
    else:
        weighted_mape = None
        weighted_bias = None

    return {
        "n_variants":     len(acc_df),
        "weighted_mape":  round(weighted_mape, 1) if weighted_mape is not None else None,
        "weighted_bias":  round(weighted_bias, 1) if weighted_bias is not None else None,
        "median_mape":    round(acc_df["mape"].median(), 1),
    }
