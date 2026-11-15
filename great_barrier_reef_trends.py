#!/usr/bin/env python3
"""
Great Barrier Reef trend detective
==================================
Usage (Windows or Linux):

    python great_barrier_reef_trends.py  [folder_with_giovanni_csvs]  [output_folder]
    python great_barrier_reef_trends.py  data/great_barrier_reef  reef_outputs

Expected box (Giovanni, West,South,East,North): 142,-24,154,-10  (the reef and its lagoon).

For every variable it:
  1. loads the Giovanni CSV, masks fill values, converts units, builds a complete
     monthly calendar (daily data -> monthly means, mid-month time stamps are fine);
  2. checks the bounding box and how much of it the product really covers;
  3. tests trends in a way that respects the seasons:
       - Seasonal Mann-Kendall (Hirsch-Slack, includes the correlation between months)
       - annual mean, the four seasons, the BLEACHING SEASON (Jan-Mar) mean and the
         HOTTEST MONTH of each year, each with Hamed-Rao autocorrelation correction
       - month-by-month trend heatmap
  4. Sen's slope per decade with 95% CI;
  5. Pettitt step-change test;
  6. "drop the most extreme year" check (does the trend survive without one
     record year such as a mass-bleaching summer?);
  7. for sea surface temperature only: a simple HEAT-STRESS PROXY in the spirit of
     NOAA Coral Reef Watch. MMM = the warmest of the 12 climatological monthly means.
     Each summer gets "heat stress" = sum of (SST - MMM) over months warmer than MMM
     (in degC-months) and a count of such months. Both are trend-tested.
     NOTE: NOAA uses daily 5 km SST and degree heating WEEKS. This monthly,
     box-averaged version is only an indicator, not a bleaching alert;
  8. Benjamini-Hochberg FDR correction across ALL tests.

Southern hemisphere: summer is DJF. December is counted in the FOLLOWING year's
summer, so "summer 2016" = Dec 2015 + Jan-Feb 2016.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- settings --
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "reef_outputs"
ALPHA = 0.05
MIN_YEARS = 10
HOT_MONTHS = (1, 2, 3)                   # bleaching season on the GBR (peak SST Jan-Mar)
SEASONS = {"DJF (summer)": (12, 1, 2), "MAM (autumn)": (3, 4, 5),
           "JJA (winter)": (6, 7, 8), "SON (spring)": (9, 10, 11),
           "Bleaching season (JFM)": HOT_MONTHS}
PEAK = "Hottest / highest month"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
EXPECTED_BOX = (142.0, -24.0, 154.0, -10.0)    # W, S, E, N

# key found in the Giovanni column name ->
#   (label, unit, scale, offset, percent_makes_sense)
# value_shown = raw * scale + offset. Checked against the files: T2M is Kelvin, cloud
# fraction is 0-1, IMERG was downloaded with units=mm/day, SST is already degC.
VARIABLES = {
    "_sst":           ("Sea surface temperature (MODIS-Aqua)", "degC", 1.0, 0.0, False),
    "chlor_a":        ("Chlorophyll-a (MODIS-Aqua)", "mg/m3", 1.0, 0.0, True),
    "_T2M":           ("2 m air temperature (MERRA-2)", "degC", 1.0, -273.15, False),
    "SPEED":          ("Surface wind speed (MERRA-2)", "m/s", 1.0, 0.0, True),
    "Cloud_Fraction": ("Cloud fraction (MODIS-Aqua)", "%", 100.0, 0.0, True),
    "precipitation":  ("Precipitation (GPM IMERG)", "mm/day", 1.0, 0.0, True),
}


# ------------------------------------------------------------------ loading --
def parse_box(txt):
    try:
        w, s, e, n = [float(x) for x in str(txt).split(",")]
        return w, s, e, n
    except Exception:                                   # noqa
        return None


def load_giovanni(path):
    """Read one Giovanni area-averaged CSV -> (monthly Series, info dict)."""
    with open(path, encoding="utf-8-sig") as fh:
        lines = fh.read().splitlines()
    hdr = next(i for i, ln in enumerate(lines) if ln.lower().startswith("time"))
    meta = {}
    for ln in lines[:hdr]:
        key, _, val = ln.partition(",")
        meta[key.strip().rstrip(":")] = val.strip().strip('"').strip()

    df = pd.read_csv(path, skiprows=hdr, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    col = [c for c in df.columns if c != "time"][0]

    v = pd.to_numeric(df[col], errors="coerce")
    fill = next((float(val) for k, val in meta.items() if k.startswith("Fill Value")), None)
    if fill is not None:
        v = v.mask(np.isclose(v, fill))
    v = v.mask(v.abs() > 1e14)

    s = pd.Series(v.values, index=pd.to_datetime(df["time"]))
    step = s.index.to_series().diff().median()
    daily = step <= pd.Timedelta(days=2)
    if daily:
        g = s.resample("MS").agg(["mean", "count"])
        monthly = g["mean"].where(g["count"] >= 0.8 * g.index.days_in_month)
    else:                                       # odd stamps (2002-08-06 15:17) -> that month
        s.index = s.index.to_period("M").to_timestamp()
        monthly = s.groupby(level=0).mean().asfreq("MS")

    key = next((k for k in VARIABLES if k in col), None)
    if key is None:
        short = col.replace("mean_", "")
        label, unit, scale, offset, pct_ok = (short, "raw units", 1.0, 0.0, True)
        print(f"   note: {short} is not in VARIABLES; using raw units (check them in Giovanni)")
    else:
        label, unit, scale, offset, pct_ok = VARIABLES[key]
    monthly = monthly * scale + offset
    info = dict(file=os.path.basename(path), column=col, key=key, label=label, unit=unit,
                pct_ok=pct_ok, user_box=parse_box(meta.get("User Bounding Box")),
                data_box=parse_box(meta.get("Data Bounding Box")),
                cadence="daily->monthly" if daily else "monthly")
    return monthly, info


def box_warnings(info):
    out, ub, db = [], info["user_box"], info["data_box"]
    if not ub:
        return out
    W, S, E, N = ub
    exp = EXPECTED_BOX
    if not (W >= exp[0] - 1 and S >= exp[1] - 1 and E <= exp[2] + 1 and N <= exp[3] + 1):
        out.append(f"box {ub} is not inside the suggested reef box {exp}")
    if db:
        ua, da = (E - W) * (N - S), (db[2] - db[0]) * (db[3] - db[1])
        if ua > 0 and da / ua < 0.6:
            out.append(f"product only covers {100 * da / ua:.0f}% of the requested box (data box {db})")
    return out


# ---------------------------------------------------------------- calendar ---
def deseasonalise(m):
    return m - m.groupby(m.index.month).transform("mean")


def decimal_year(idx):
    return idx.year + (idx.month - 0.5) / 12.0


def season_series(m):
    """Annual, seasonal and bleaching-season means (+ hottest month); complete only.
    December belongs to the following year's DJF summer."""
    d = m.dropna().to_frame("v")
    d["mo"], d["yr"] = d.index.month, d.index.year
    out = {}
    g = d.groupby("yr").v.agg(["mean", "count"])
    out["Annual"] = g.loc[g["count"] == 12, "mean"]
    for name, months in SEASONS.items():
        x = d[d.mo.isin(months)].copy()
        x["sy"] = x.yr + ((x.mo == 12) & (1 in months)).astype(int)
        g = x.groupby("sy").v.agg(["mean", "count"])
        out[name] = g.loc[g["count"] == len(months), "mean"]
    g = d.groupby("yr").v.agg(["max", "count"])
    out[PEAK] = g.loc[g["count"] == 12, "max"]
    return out


def heat_stress(sst):
    """Monthly heat-stress proxy. Returns (MMM, DataFrame per summer-year)."""
    m = sst.dropna()
    clim = m.groupby(m.index.month).mean()
    mmm = float(clim.max())
    d = m.to_frame("v")
    d["sy"] = d.index.year + (d.index.month == 12).astype(int)     # Dec -> next summer
    d = d[d.index.month.isin((12, 1, 2, 3, 4))]                     # warm half of the year
    d["hot"] = (d.v - mmm).clip(lower=0)
    g = d.groupby("sy").agg(stress=("hot", "sum"), hot_months=("hot", lambda x: int((x > 0).sum())),
                            n=("v", "count"))
    return mmm, g[g.n == 5].drop(columns="n")


# --------------------------------------------------------------- statistics --
def _sen_ci(slopes, varS):
    N = len(slopes)
    C = stats.norm.ppf(1 - ALPHA / 2) * np.sqrt(varS)
    srt = np.sort(slopes)
    lo = max(int(np.floor((N - C) / 2)) - 1, 0)
    hi = min(int(np.ceil((N + C) / 2)), N - 1)
    return srt[lo], srt[hi]


def _z_from_S(S, varS):
    if varS <= 0 or S == 0:
        return 0.0
    return (S - np.sign(S)) / np.sqrt(varS)


def hamed_rao_factor(x, t, sen):
    """Variance inflation for autocorrelation (Hamed & Rao 1998), clipped at 1."""
    n = len(x)
    r = stats.rankdata(x - sen * t)
    r = r - r.mean()
    denom = (r ** 2).sum()
    if denom == 0:
        return 1.0
    k = np.arange(1, n)
    acf = np.array([(r[:-kk] * r[kk:]).sum() / denom for kk in k])
    sig = np.abs(acf) > stats.norm.ppf(1 - ALPHA / 2) / np.sqrt(n)
    w = (n - k) * (n - k - 1) * (n - k - 2)
    f = 1 + 2.0 / (n * (n - 1) * (n - 2)) * np.sum(w[sig] * acf[sig])
    return max(f, 1.0)


def mk_test(x, t, autocorr=True):
    """Mann-Kendall + Sen slope (per unit of t). Tie-corrected; optional Hamed-Rao."""
    x, t = np.asarray(x, float), np.asarray(t, float)
    n = len(x)
    iu = np.triu_indices(n, 1)
    diff = x[None, :] - x[:, None]
    S = np.sign(diff)[iu].sum()
    _, c = np.unique(x, return_counts=True)
    varS = (n * (n - 1) * (2 * n + 5) - np.sum(c * (c - 1) * (2 * c + 5))) / 18.0
    slopes = (x[iu[1]] - x[iu[0]]) / (t[iu[1]] - t[iu[0]])
    sen = float(np.median(slopes))
    f = hamed_rao_factor(x, t, sen) if autocorr else 1.0
    z = _z_from_S(S, varS * f)
    lo, hi = _sen_ci(slopes, varS * f)
    return dict(n=n, S=S, tau=S / (n * (n - 1) / 2), z=z, p=2 * stats.norm.sf(abs(z)),
                slope=sen, ci_lo=lo, ci_hi=hi, var_factor=f)


def seasonal_mk(M, t):
    """Seasonal Mann-Kendall (Hirsch & Slack 1984) incl. covariance between seasons."""
    n, g = M.shape
    iu = np.triu_indices(n, 1)
    sgn = np.sign(M[None, :, :] - M[:, None, :])[iu]
    S = sgn.sum()
    K = sgn.T @ sgn
    R = np.column_stack([stats.rankdata(M[:, k]) for k in range(g)])
    varS = ((K + 4 * R.T @ R - n * (n + 1) ** 2) / 3.0).sum()
    dt = (t[iu[1]] - t[iu[0]]).astype(float)[:, None]
    slopes = ((M[iu[1], :] - M[iu[0], :]) / dt).ravel()
    sen = float(np.median(slopes))
    z = _z_from_S(S, varS)
    lo, hi = _sen_ci(slopes, varS)
    return dict(n=n, S=S, tau=S / (g * n * (n - 1) / 2), z=z, p=2 * stats.norm.sf(abs(z)),
                slope=sen, ci_lo=lo, ci_hi=hi, var_factor=np.nan)


def pettitt(x):
    x = np.asarray(x, float)
    n = len(x)
    U = np.array([np.sign(x[: k + 1, None] - x[None, k + 1:]).sum() for k in range(n - 1)])
    K = np.abs(U).max()
    return int(np.abs(U).argmax()) + 1, min(1.0, 2 * np.exp(-6 * K ** 2 / (n ** 3 + n ** 2)))


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def drop_extreme_check(ser):
    yr = (ser - ser.median()).abs().idxmax()
    s2 = ser.drop(yr)
    r2 = mk_test(s2.values, s2.index.values.astype(float))
    return dict(extreme_year=int(yr), p_without_extreme=r2["p"], slope_without_extreme=r2["slope"] * 10)


# ----------------------------------------------------------------- analysis --
def analyse(monthly, info):
    rows, monthly_rows = [], []
    anom = deseasonalise(monthly)
    base = dict(Variable=info["label"], Unit=info["unit"])

    pv = monthly.dropna().to_frame("v")
    pv["yr"], pv["mo"] = pv.index.year, pv.index.month
    pivot = pv.pivot(index="yr", columns="mo", values="v")
    wide = pivot.dropna()
    if len(wide) < MIN_YEARS:
        rows.append({**base, "Test": "ALL", "n": len(wide),
                     "note": f"record too short ({len(wide)} complete years < {MIN_YEARS})"})
        return rows, monthly_rows, anom, None, None

    def add(test, r, mean, extra=None):
        pct = (r["slope"] * 10 / abs(mean) * 100) if info["pct_ok"] and mean else np.nan
        rows.append({**base, "Test": test, "n": r["n"], "tau": r["tau"], "z": r["z"], "p_raw": r["p"],
                     "slope_per_decade": r["slope"] * 10, "ci_lo_decade": r["ci_lo"] * 10,
                     "ci_hi_decade": r["ci_hi"] * 10, "pct_of_mean_per_decade": pct,
                     "var_inflation": r["var_factor"], **(extra or {})})

    smk = seasonal_mk(wide.values, wide.index.values)
    add("Seasonal MK", smk, float(np.nanmean(monthly)))

    hot_series = None
    for name, ser in season_series(monthly).items():
        if len(ser) < MIN_YEARS:
            continue
        r = mk_test(ser.values, ser.index.values.astype(float))
        extra = {**drop_extreme_check(ser), "top3_years": ", ".join(str(y) for y in ser.nlargest(3).index)}
        if name == "Annual":
            k, pp = pettitt(ser.values)
            extra.update(pettitt_p=pp, pettitt_year=int(ser.index[k]),
                         pettitt_shift=ser.values[k:].mean() - ser.values[:k].mean())
        if name == "Bleaching season (JFM)":
            hot_series = ser
        add(name, r, float(ser.mean()), extra)

    if info["key"] == "_sst":                                    # heat-stress proxy
        mmm, hs = heat_stress(monthly)
        if len(hs) >= MIN_YEARS:
            for col, lab in (("stress", "Heat stress above MMM (degC-months)"),
                             ("hot_months", "Months warmer than MMM per summer")):
                ser = hs[col].astype(float)
                r = mk_test(ser.values, ser.index.values.astype(float))
                add(lab, r, float(ser.mean()) or np.nan,
                    {**drop_extreme_check(ser), "MMM_degC": mmm,
                     "top3_years": ", ".join(str(y) for y in ser.nlargest(3).index)})
            info["heat"] = (mmm, hs)

    for mo in range(1, 13):
        col = pivot[mo].dropna()
        if len(col) < MIN_YEARS:
            continue
        r = mk_test(col.values, col.index.values.astype(float), autocorr=False)
        monthly_rows.append({**base, "Month": MONTHS[mo - 1], "z": r["z"], "p_raw": r["p"],
                             "slope_per_decade": r["slope"] * 10})

    a = anom.dropna()
    icpt = np.median(a.values - smk["slope"] * decimal_year(a.index))
    return rows, monthly_rows, anom, (smk["slope"], icpt), hot_series


# ------------------------------------------------------------------- plots ---
def make_plots(panels, res, mres, outdir):
    # 1) deseasonalised anomalies + trend
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.5 * len(panels)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, panels):
        info, anom, line = p["info"], p["anom"].dropna(), p["line"]
        ax.plot(anom.index, anom.values, color="tab:blue", alpha=0.35, lw=0.8, label="Monthly anomaly")
        ax.plot(anom.index, anom.rolling(12, center=True, min_periods=12).mean(), color="navy", lw=1.8,
                label="12-month mean")
        title = f"{info['label']} [{info['unit']}] - record too short for a trend test"
        if line:
            ax.plot(anom.index, line[0] * decimal_year(anom.index) + line[1], color="crimson", lw=2,
                    label="Seasonal Sen trend")
            r = res[(res.Variable == info["label"]) & (res.Test == "Seasonal MK")].iloc[0]
            title = (f"{info['label']} [{info['unit']}]  {r.slope_per_decade:+.3g}/decade "
                     f"(95% CI {r.ci_lo_decade:+.3g} to {r.ci_hi_decade:+.3g}; FDR p={r.p_fdr:.3g}, "
                     f"{'significant' if r.significant_fdr else 'not significant'})")
        ax.set_title(title, fontsize=9, loc="left", fontweight="bold")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel("anomaly", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=7, ncol=3)
    fig.suptitle("Great Barrier Reef - deseasonalised anomalies", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "anomaly_trends.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) bleaching-season mean per year
    hp = [p for p in panels if p["hot"] is not None]
    if hp:
        ncol = 2
        nrow = int(np.ceil(len(hp) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.9 * nrow), squeeze=False)
        for ax in axes.ravel():
            ax.set_visible(False)
        for ax, p in zip(axes.ravel(), hp):
            ax.set_visible(True)
            ser, info = p["hot"], p["info"]
            r = res[(res.Variable == info["label"]) & (res.Test == "Bleaching season (JFM)")].iloc[0]
            cols = ["crimson" if y == r.extreme_year else "tab:cyan" for y in ser.index]
            ax.bar(ser.index, ser.values, color=cols)
            b = r.slope_per_decade / 10
            ax.plot(ser.index, b * ser.index + np.median(ser.values - b * ser.index.values), color="navy", lw=2)
            lo, hi = ser.min(), ser.max()
            ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.1 * (hi - lo))
            ax.set_title(f"{info['label']} [{info['unit']}]\n{r.slope_per_decade:+.3g}/decade, FDR p={r.p_fdr:.3g}; "
                         f"without {int(r.extreme_year)} (red): p={r.p_without_extreme:.3g}", fontsize=8, loc="left")
            ax.grid(alpha=0.3)
        fig.suptitle("Great Barrier Reef - bleaching-season (Jan-Mar) mean by year", fontsize=11, y=1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "bleaching_season_by_year.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3) heat-stress proxy
    sst = next((p for p in panels if p["info"].get("heat")), None)
    if sst:
        mmm, hs = sst["info"]["heat"]
        fig, ax = plt.subplots(figsize=(11, 3.4))
        ax.bar(hs.index, hs["stress"], color="tab:red")
        ax.set_ylabel("degC-months above MMM")
        ax.set_title(f"Summer heat-stress proxy: sum of monthly SST above MMM = {mmm:.2f} degC "
                     "(box-average, monthly; indicator only, not a bleaching alert)", fontsize=9, loc="left")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "sst_heat_stress.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 4) month x variable heatmap
    if len(mres):
        z = mres.pivot(index="Variable", columns="Month", values="z").reindex(columns=MONTHS)
        pf = mres.pivot(index="Variable", columns="Month", values="p_fdr").reindex(columns=MONTHS)
        fig, ax = plt.subplots(figsize=(11, 0.6 * len(z) + 1.8))
        im = ax.imshow(z.values, cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
        ax.set_xticks(range(12), MONTHS)
        ax.set_yticks(range(len(z)), z.index)
        for i in range(z.shape[0]):
            for j in range(12):
                if not np.isnan(z.values[i, j]):
                    ax.text(j, i, f"{z.values[i, j]:+.1f}" + ("*" if pf.values[i, j] < ALPHA else ""),
                            ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="Mann-Kendall z  (red = rising, blue = falling)")
        ax.set_title("Great Barrier Reef - which months drive the trend?  (* = significant after FDR)",
                     fontsize=10, loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "monthly_trend_heatmap.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


# --------------------------------------------------------------------- main --
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        sys.exit(f"No CSV files found in {DATA_DIR!r}")

    all_rows, all_mrows, panels, boxes = [], [], [], set()
    for f in files:
        try:
            monthly, info = load_giovanni(f)
        except Exception as e:                           # noqa
            print(f"!! skipped {os.path.basename(f)}: {e}")
            continue
        span = f"{monthly.dropna().index.min():%Y-%m} to {monthly.dropna().index.max():%Y-%m}"
        print(f"loaded {info['label']:38s} {span}  [{info['cadence']}]  box={info['user_box']}")
        for w in box_warnings(info):
            print(f"   !! {w}")
        boxes.add(info["user_box"])
        rows, mrows, anom, line, hot = analyse(monthly, info)
        all_rows += rows
        all_mrows += mrows
        panels.append(dict(info=info, anom=anom, line=line, hot=hot))
    if len(boxes) > 1:
        print(f"   !! WARNING: variables use different boxes {sorted(boxes, key=str)}")
    if not all_rows:
        sys.exit("No data was loaded.")

    res = pd.DataFrame(all_rows)
    for c in ("p_raw", "note", "p_without_extreme", "pettitt_p"):
        if c not in res.columns:
            res[c] = np.nan
    ok = res["p_raw"].notna()
    res.loc[ok, "p_fdr"] = bh_fdr(res.loc[ok, "p_raw"])
    res["significant_fdr"] = res["p_fdr"] < ALPHA
    res["survives_without_extreme_year"] = (res["p_without_extreme"] < ALPHA).where(res["p_without_extreme"].notna())
    res.to_csv(os.path.join(OUT_DIR, "trend_results.csv"), index=False)
    mres = pd.DataFrame(all_mrows)
    if len(mres):
        mres["p_fdr"] = bh_fdr(mres["p_raw"])
        mres.to_csv(os.path.join(OUT_DIR, "monthly_trends.csv"), index=False)

    order = ["Seasonal MK", "Annual"] + list(SEASONS) + [PEAK]
    tab = res[ok & res.Test.isin(order)].copy()
    tab["cell"] = tab.apply(lambda r: f"{r.slope_per_decade:+.3g}" + ("*" if r.significant_fdr else ""), axis=1)
    pv = tab.pivot(index="Variable", columns="Test", values="cell")
    pv = pv.reindex(columns=[c for c in order if c in pv.columns]).fillna("-")
    print("\nSen slope per decade   (* = significant after FDR correction, alpha=0.05)")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(pv.to_string())

    print("\nBLEACHING SEASON (Jan-Mar) mean: does the trend survive without the most extreme year?")
    for _, r in res[res.Test == "Bleaching season (JFM)"].iterrows():
        print(f"  {r.Variable:38s} {r.slope_per_decade:+.3g} {r.Unit}/decade  FDR p={r.p_fdr:.3g}"
              f" | extreme year {int(r.extreme_year)}: p without it = {r.p_without_extreme:.3g}"
              f" -> {'still significant' if r.survives_without_extreme_year else 'NOT significant'}"
              f" | highest years: {r.top3_years}")
    hs = res[res.Test.str.startswith(("Heat stress", "Months warmer"), na=False)]
    if len(hs):
        print(f"\nSST HEAT-STRESS PROXY (MMM = {hs.MMM_degC.iloc[0]:.2f} degC):")
        for _, r in hs.iterrows():
            print(f"  {r.Test:40s} {r.slope_per_decade:+.3g}/decade  FDR p={r.p_fdr:.3g}  worst summers: {r.top3_years}")
    for _, r in res.loc[~ok].iterrows():
        print(f"  NOT TESTED  {r.Variable}: {r.note}")
    for _, r in res[(res.Test == "Annual") & ok & (res.pettitt_p < ALPHA)].iterrows():
        print(f"  Possible step change: {r.Variable} around {int(r.pettitt_year)} "
              f"(shift {r.pettitt_shift:+.3g} {r.Unit}, Pettitt p={r.pettitt_p:.3f})")

    make_plots(panels, res, mres, OUT_DIR)
    print(f"\nSaved results and figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()
