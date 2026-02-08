# AI-Driven Marketing Measurement & Optimization (Anakin Adventures — Synthetic)

This project is a travel agency marketing measurement workflow using realistic synthetic data modeled after a multi-channel funnel (sessions, leads, bookings, revenue). It demonstrates incrementality measurement using geo-based Difference-in-Differences (DiD), plus diagnostics (parallel trends + placebo tests).

## Data (Synthetic, Travel Agency–Style)
- `data/marketing_spend_daily.csv`: daily marketing spend by geo × channel (plus impressions/clicks)
- `data/web_funnel_daily.csv`: daily funnel outcomes by geo (sessions, leads, bookings, revenue)
- `data/bookings.csv`: booking-level table (destination, trip type, lead time, cancellation, margin)
- `data/customer_features.csv`: customer-level features for segmentation/propensity modeling


## Phase 2 — Incrementality (Difference-in-Differences)
Method:
- Treatment window: **2024-08-01 → 2024-10-15**
- Pre window (matched length): **2024-05-17 → 2024-07-31**
- Groups: **4 treated geos vs 8 control geos**
- Outputs: incremental bookings, incremental revenue, incremental spend, iROAS, cost per incremental booking
- Uncertainty: bootstrap confidence intervals (resampling geos)

Outputs:
- `outputs/did_outputs/did_topline_kpis.csv`
- `outputs/did_outputs/did_bootstrap_ci.csv`
- `outputs/did_outputs/geo_period_totals.csv`
- `outputs/did_outputs/geo_changes_pre_post.csv`

## Phase 3 — Diagnostics
- Parallel trends plots (indexed):
  - `outputs/did_outputs/parallel_trends_bookings.png`
  - `outputs/did_outputs/parallel_trends_revenue.png`
- Placebo DiD (pre-period):
  - Bookings per geo: **3.0**
  - Revenue per geo: **462.2**
  - Placebo window: **2024-06-06 → 2024-06-26**

Interpretation:
- Pre-trends track closely between treated and control groups.
- Placebo estimates are economically insignificant relative to observed treatment effects, supporting the parallel trends assumption.

## How to run
```bash
python src/01_did_incrementality.py
python src/02_parallel_trends_placebo.py
