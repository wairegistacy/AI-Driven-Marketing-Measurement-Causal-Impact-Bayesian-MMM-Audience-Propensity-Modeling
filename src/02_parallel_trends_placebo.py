import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "C:/Users/stacy/Downloads/Projects/marketing measurement ai/data"
OUT_DIR = "did_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

EXP_START = pd.to_datetime("2024-08-01")
EXP_END = pd.to_datetime("2024-10-15")

# Match your DiD pre-window length
window_len = (EXP_END - EXP_START).days + 1
pre_end = EXP_START - pd.Timedelta(days=1)
pre_start = pre_end - pd.Timedelta(days=window_len - 1)

# Load outcomes
funnel = pd.read_csv(os.path.join(DATA_DIR, "web_funnel_daily.csv"))
funnel["date"] = pd.to_datetime(funnel["date"])

# Focus on PRE period only for parallel trends check
pre_df = funnel[(funnel["date"] >= pre_start) & (funnel["date"]<= pre_end)].copy()

# Aggregate daily by treatment group
daily = (
    pre_df.groupby(["date", "is_treatment_geo"], as_index=False)
          .agg(bookings=("bookings", "sum"),
               revenue=("revenue", "sum"),
               sessions=("sessions", "sum"))
)

# Add rates (optional)
daily["cvr_booking_per_session"] = daily["bookings"] / daily["sessions"]

treated = daily[daily["is_treatment_geo"] == 1].set_index("date")
control = daily[daily["is_treatment_geo"] == 0].set_index("date")

# Align dates
idx = treated.index.intersection(control.index)
treated = treated.loc[idx]
control = control.loc[idx]

# Compute indexed trends (normalize to 1.0 at start of pre-period)
def index_series(s: pd.Series) -> pd.Series:
    return s / s.iloc[0] if s.iloc[0] != 0 else s

trend = pd.DataFrame({
    "treated_bookings_idx": index_series(treated["bookings"]),
    "control_bookings_idx": index_series(control["bookings"]),
    "treated_revenue_idx": index_series(treated["revenue"]),
    "control_revenue_idx": index_series(control["revenue"]),
})

# ---- Plot: bookings & revenue indexed trends ----
plt.figure()
plt.plot(trend.index, trend["treated_bookings_idx"], label="Treated (indexed)")
plt.plot(trend.index, trend["control_bookings_idx"], label="Control (indexed)")
plt.title("Parallel Trends Check (Pre-Period) — Bookings (Indexed)")
plt.xlabel("Date")
plt.ylabel("Indexed Bookings (start=1.0)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "parallel_trends_bookings.png"), dpi=200)
plt.close()

plt.figure()
plt.plot(trend.index, trend["treated_revenue_idx"], label="Treated (indexed)")
plt.plot(trend.index, trend["control_revenue_idx"], label="Control (indexed)")
plt.title("Parallel Trends Check (Pre-Period) — Revenue (Indexed)")
plt.xlabel("Date")
plt.ylabel("Indexed Revenue (start=1.0)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "parallel_trends_revenue.png"), dpi=200)
plt.close()

print("✅ Saved parallel trends plots to:", OUT_DIR)

# ---- Placebo Test ----
# Pick a fake intervention date in the middle of PRE and rerun DiD logic.
placebo_start = pre_start + pd.Timedelta(days=20)
placebo_end = placebo_start + pd.Timedelta(days=20)

def label_period(d):
    if (d >= placebo_start) and (d <= placebo_end):
        return "post"
    if (d >= pre_start) and (d < placebo_start):
        return "pre"
    return "other"

pl = funnel[(funnel["date"] >= pre_start) & (funnel["date"] <= placebo_end)].copy()
pl["period"] = pl["date"].apply(label_period)
pl = pl[pl["period"].isin(["pre", "post"])]

# Aggregate by geo × period
gp = (
    pl.groupby(["geo", "is_treatment_geo", "period"], as_index=False)
      .agg(bookings=("bookings", "sum"),
           revenue=("revenue", "sum"))
)

pre = gp[gp["period"]=="pre"].set_index(["geo","is_treatment_geo"])
post = gp[gp["period"]=="post"].set_index(["geo","is_treatment_geo"])

geo_change = (post[["bookings","revenue"]] - pre[["bookings","revenue"]]).reset_index()
treated = geo_change[geo_change["is_treatment_geo"]==1]
control = geo_change[geo_change["is_treatment_geo"]==0]

did_bookings_placebo = treated["bookings"].mean() - control["bookings"].mean()
did_revenue_placebo  = treated["revenue"].mean() - control["revenue"].mean()

print("\n✅ Placebo DiD (should be near 0 if parallel trends holds):")
print("Placebo DiD bookings per geo:", round(did_bookings_placebo, 3))
print("Placebo DiD revenue per geo :", round(did_revenue_placebo, 2))
print("Placebo window:", placebo_start.date(), "→", placebo_end.date())
