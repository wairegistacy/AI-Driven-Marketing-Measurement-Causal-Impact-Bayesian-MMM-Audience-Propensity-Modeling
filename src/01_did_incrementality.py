import os
import numpy as np
import pandas as pd

# Step 1: Config
DATA_DIR = "C:/Users/stacy/Downloads/Projects/marketing measurement ai/data"
OUT_DIR = "did_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
EXP_START = "2024-08-01"
EXP_END = "2024-10-15"
BOOTSTRAP_SAMPLES = 2000
SEED = 0

# Step 2: Load data + basic checks
marketing = pd.read_csv(os.path.join(DATA_DIR, "marketing_spend_daily.csv"))
funnel = pd.read_csv(os.path.join(DATA_DIR,"web_funnel_daily.csv"))

marketing["date"] = pd.to_datetime(marketing["date"])
funnel["date"] = pd.to_datetime(funnel["date"])

print("Step 1: Loaded data")
print("marketing rows", len(marketing), "funnel rows:", len(funnel))
print("date range funnel:", funnel["date"].min().date(), "->", funnel["date"].max().date())
print("treated geos:", funnel.query("is_treatment_geo==1")["geo"].nunique())
print("control geos:", funnel.query("is_treatment_geo==0")["geo"].nunique())

# Step 2: Define pre/post windows
# -post = experiement window
# -pre = same length immediately before post

exp_start = pd.to_datetime(EXP_START)
exp_end = pd.to_datetime(EXP_END)

window_len = (exp_end - exp_start).days + 1
pre_end = exp_start - pd.Timedelta(days=1)
pre_start = pre_end - pd.Timedelta(days=window_len - 1)

print("POST:", exp_start.date(), "->", exp_end.date(), "days:", window_len)
print("POST:", pre_start.date(), "->", pre_end.date(), "days:", window_len)

# STEP 3: Label each row as pre/post/other
def label_period(d):
    if pre_start <= d <= pre_end:
        return "pre"
    if exp_start <= d <= exp_end:
        return "post"
    return "other"

funnel["period"] = funnel["date"].apply(label_period)
marketing["period"] = marketing["date"].apply(label_period)

#Keep only pre/post
f = funnel[funnel["period"].isin(["pre", "post"])].copy()
m = marketing[marketing["period"].isin(["pre", "post"])].copy()

print("Step 3: Labeled periods")
print("funnel pre/post rows:", len(f), "Marketing pre/post rows:", len(m))

# Step 4: Aggregate outcomes and spend by ego * period
# Outcomes: sessions -> leads -> bookings -> revenue
agg_outcomes = (
    f.groupby(["geo", "is_treatment_geo", "period"], as_index=False)
    .agg(
        sessions=("sessions", "sum"),
        leads=("leads", "sum"),
        bookings=("bookings", "sum"),
        revenue=("revenue", "sum"),
    )
)

print(agg_outcomes.head())
print("rows:", len(agg_outcomes))

# Spend: sum across channels
agg_spend = (
    m.groupby(["geo", "is_treatment_geo", "period"], as_index=False)
    .agg(
        spend=("spend", "sum"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
    )
)

print(agg_spend.head())

# Step 5: Merge outcomes + spend into one table
gp = agg_outcomes.merge(
    agg_spend,
    on=["geo", "is_treatment_geo", "period"],
    how="left"
)

gp["lead_rate"] = gp["leads"] / gp["sessions"]
gp["booking_rate_from_lead"] = gp["bookings"] / gp["leads"].replace(0, np.nan)
gp["booking_rate_from_sessions"] = gp["bookings"] / gp["sessions"]

print(gp.head())

# Step 6: Compute each geo's change (post - pre)
pre = gp[gp["period"]=="pre"].set_index(["geo", "is_treatment_geo"])
post = gp[gp["period"]=="post"].set_index(["geo", "is_treatment_geo"])

metrics = ["bookings", "revenue", "spend", "lead_rate", "booking_rate_from_sessions"]

geo_change = (post[metrics] - pre[metrics]).reset_index()
geo_change = geo_change.rename(columns={c: f"delta_{c}" for c in metrics})

print(geo_change.head())

# Step 7: Compute DiD = (avg treated change) - (avg control change)
treated = geo_change[geo_change["is_treatment_geo"]==1]
control = geo_change[geo_change["is_treatment_geo"]==0]

did_bookings = treated["delta_bookings"].mean() - control["delta_bookings"].mean()
did_revenue = treated["delta_revenue"].mean() - control["delta_revenue"].mean()
did_spend = treated["delta_spend"].mean() - control["delta_spend"].mean()

n_treated = treated["geo"].nunique()

# total incrementality across treated geos
inc_bookings = did_bookings * n_treated
inc_revenue = did_revenue * n_treated
inc_spend = did_spend * n_treated

iroas = inc_revenue / inc_spend if inc_spend != 0  else np.nan
cpib = inc_spend / inc_bookings if inc_bookings != 0  else np.nan

print("DiD per-geo bookings:", did_bookings)
print("DiD per-geo revenue:", did_revenue)
print("DiD per-geo spend:", did_spend)

print("\nTOTAL incrementality (treated geos):")
print("Incremental bookings:", inc_bookings)
print("Incremental revenue:", inc_revenue)
print("Incremental spend:", inc_spend)
print("iROAS:", iroas)
print("Cost per incremental booking:", cpib)

# Step 8: Bootstrap confidence intervals
rng = np.random.default_rng(0)
B = 2000

t_geos = treated["geo"].unique()
c_geos = control["geo"].unique()

treated_map = treated.set_index("geo")
control_map = control.set_index("geo")

boot = []
for _ in range(B):
    t_sample = rng.choice(t_geos, size=len(t_geos), replace=True)
    c_sample = rng.choice(c_geos, size=len(c_geos), replace=True)

    did_rev_b = treated_map.loc[t_sample]["delta_revenue"].mean() - control_map.loc[c_sample]["delta_revenue"].mean()
    did_book_b = treated_map.loc[t_sample]["delta_bookings"].mean() - control_map.loc[c_sample]["delta_bookings"].mean()
    did_sp_b = treated_map.loc[t_sample]["delta_spend"].mean() - control_map.loc[c_sample]["delta_spend"].mean()

    inc_rev_b = did_rev_b * len(t_geos)
    inc_book_b = did_book_b * len(t_geos)
    inc_spend_b = did_sp_b * len(t_geos)

    iroas_b = inc_rev_b / inc_spend_b if inc_spend_b != 0 else np.nan
    cpib_b = inc_spend_b / inc_book_b if inc_book_b != 0 else np.nan

    boot.append([inc_rev_b, inc_book_b, inc_spend_b, iroas_b, cpib_b])

boot = pd.DataFrame(boot, columns=["inc_rev", "inc_book", "inc_spend", "iroas", "cpib"])

ci = boot.quantile([0.025, 0.975]).T
ci.columns = ["ci_2_5", "ci_97_5"]

print(ci)

# Save geo-level totals (pre vs post)
gp.to_csv(
    os.path.join(OUT_DIR, "geo_period_totals.csv"),
    index=False
)
print("Saved: geo_period_totals.csv")

# Save geo-level changes (post - pre)
geo_change.to_csv(
    os.path.join(OUT_DIR, "geo_changes_pre_post.csv"),
    index=False
)

print("Saved: geo_changes_pre_post.csv")

# Save topline DiD KPIs (executive-ready)
topline = pd.DataFrame([{
    "pre_start": pre_start.date().isoformat(),
    "pre_end": pre_end.date().isoformat(),
    "post_start": exp_start.date().isoformat(),
    "post_end": exp_end.date().isoformat(),
    "treated_geos": int(n_treated),
    "contol_geos": int(control["geo"].nunique()),
    "did_bookings_per_geo": float(did_bookings),
    "did_revenue_per_geo": float(did_revenue),
    "did_spend_per_geo": float(did_spend),
    "incremental_bookings_est": float(inc_bookings),
    "incremental_revenue_est": float(inc_revenue),
    "incremental_spend_est": float(inc_spend),
    "iROAS_est": float(iroas),
    "cost_per_incremental_booking_est": float(cpib),
}])

topline.to_csv(
    os.path.join(OUT_DIR, "did_topline_kpis.csv"),
    index=False
)

print("Saved: did_topline_kpis.csv")

ci.to_csv(
    os.path.join(OUT_DIR, "did_bootstrap_ci.csv"),
    index=False
)

print("Saved: did_bootstrap_ci.csv")
