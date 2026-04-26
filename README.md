# SupplyAI — F&B Demand & Supply Planner

A Streamlit-based demand and supply planning tool for food & beverage operations.

## Features

- **Demand Forecast** — 8-week forecast using 4-4-5 retail calendar, seasonal index, YoY trend, and demand variability (CV)
- **Production Plan** — System-optimised run scheduling with min/max constraints and seasonal inventory targets
- **Purchasing Plan** — BOM explosion to raw material and packaging requirements by week
- **DC Network** — Regional inventory coverage vs stock targets across all warehouse locations
- **Alerts** — Stockout risks, allergen changeover conflicts, BOM gaps, and trend flags

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/supplyai-planner.git
cd supplyai-planner

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## Deploying to Streamlit Cloud

1. Push this repo to GitHub (must be public, or private with Streamlit Cloud Pro)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Select your repo, branch (`main`), and set **Main file path** to `app.py`
5. Click **Deploy**

Your app will be live at `https://YOUR_APP_NAME.streamlit.app`

### Sharing with your team
Once deployed, share the URL. Streamlit Cloud apps are accessible to anyone with the link by default. For access control, use Streamlit Cloud's built-in viewer authentication (available on Team plan).

---

## Data Files

Upload all files via the **⚙️ Settings** page in the app.

| File | Format | Required |
|---|---|---|
| Sales export (`order_nexus`) | CSV | ✅ Yes |
| Inventory snapshot | CSV | ✅ Yes |
| SKU master with BOM | Excel (.xlsx) | ✅ Yes |
| Production lines | CSV | Optional |
| Marketing calendar | CSV | Optional |

### Column requirements

**Sales CSV** (order_nexus export — use as-is):
- `Calendar (Dynamic) Retail Year Week Concat`
- `Order SKUs SKU ID`
- `Shipments Warehouse`
- `Product Hierarchy Product ID`
- `Product Hierarchy Product Name`
- `Product Hierarchy Reporting Category L1`
- `Product Hierarchy Reporting Category L2`
- `Product Hierarchy Variant Name`
- `Product Hierarchy SKU Weight`
- `Order SKUs Gross Pieces Sold`
- `Order SKUs Gross Weight Sold`

**Inventory CSV** (warehouse snapshot — use as-is):
- `Warehouse`
- `SKU`
- `Product Name`
- `Production Category`
- `Count`

**Production Lines CSV** (download template from Settings):
```
line_name, min_run_lbs, max_run_lbs, sku_ids
Line 1, 200, 3000, 1007;1015;1029
```

**Marketing Calendar CSV** (download template from Settings):
```
event_name, event_type, start_date, end_date, affected_sku_ids, uplift_percent
Summer BBQ, seasonal, 2026-05-25, 2026-07-04, 1007;1015, 20
```

---

## Project Structure

```
supplyai-planner/
├── app.py                    ← Streamlit entry point (7 pages)
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml           ← Theme and server settings
├── modules/
│   ├── forecast.py           ← Demand forecasting engine
│   ├── production.py         ← Run scheduling + constraint engine
│   ├── purchasing.py         ← BOM explosion + purchasing plan
│   ├── dc_network.py         ← DC replenishment + coverage analysis
│   └── alerts.py             ← Alert generation across all modules
└── utils/
    ├── data_loader.py        ← File parsing, cleaning, validation
    └── formatters.py         ← Display helpers, colour coding
```

---

## Production Run Logic

The scheduler maximises batch sizes within four constraints (in priority order):

1. **Min run size** — floor set per line (e.g. 200 lbs)
2. **Max run size** — line capacity ceiling (e.g. 3,000 lbs)
3. **Max inventory** — seasonal weeks-on-hand ceiling (configurable in Settings)
4. **Line capacity** — no two runs on the same line in the same week

The result is the least-frequent run cadence possible — largest batches, fewest changeovers.

---

## Roadmap (Phase 2)

- [ ] Cowork folder integration for live data refresh
- [ ] Forecast bias tracking (forecast vs actual, rolling 4wk and 13wk)
- [ ] Marketing calendar uplift engine
- [ ] Supplier lead time back-calculation for order placement dates
- [ ] Full 2,000 SKU / 5,000 variant scale with batch processing
- [ ] User authentication for team access control
