# SupplyAI — F&B Demand & Supply Planner

A Streamlit-based supply planning tool for F&B manufacturers. Built for Nuts.com's NJ facility.

## Features

- **Demand Forecast** — 8-week rolling forecast with seasonality, trailing 4-week YoY, and marketing uplifts
- **Production Plan** — System-optimized run grouping with seasonal inventory caps
- **Capacity Utilization** — Per-line, per-week heatmap showing % capacity used
- **Purchasing Plan** — Multi-level recursive BOM explosion to lowest-level raw materials
- **DC Network** — Per-DC inventory analysis with status flags
- **DC Transfers** — Replenishment plan with shipping cadence (FL/TX 1x/wk, NV/IN 2x/wk)
- **Forecast Accuracy** — Lock forecasts as snapshots; track MAPE and bias over time
- **Scenario Modeling** — Adjust uplift % and marketing factors; compare side-by-side
- **Alerts** — Cross-module flagging of stockouts, capacity, BOM gaps, and more
- **PO Export** — One-click ZIP of supplier-grouped purchase orders

## Data Persistence

Uploaded files are cached to disk in `cache/` and auto-load on session start. You don't need to re-upload between sessions.

## Required Files

1. **Sales** (CSV) — order_nexus fact table
2. **Inventory** (CSV) — warehouse-level snapshot
3. **BOM** (xlsx) — `sku_master_w_BOMs.xlsx` with sheets: `sku_master w BOM`, `items`, `SKUs`

## Optional Files (placeholders auto-generated if missing)

4. **Production Lines** (CSV) — line names, min/max run lbs, SKU assignments
5. **Marketing Calendar** (CSV) — promotional events with uplift %

Templates are downloadable from the Settings page.

## Run

Deployed to Streamlit Cloud at `nuts-supplyplanner.streamlit.app`.

Local development:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Stack

- Streamlit · pandas · numpy · plotly · openpyxl · scipy
