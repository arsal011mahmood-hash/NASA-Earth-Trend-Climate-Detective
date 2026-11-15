#!/usr/bin/env python3
"""
Boreal fire-region trend detective   (Canada / Russia)
======================================================
Usage (Windows or Linux), one folder of Giovanni CSVs per region:

    python boreal_fire_trends.py canada russia --out boreal_outputs

The folder name is the region name ("canada", "russia"). Each folder must hold
CSVs downloaded with a bounding box for THAT region (see EXPECTED_BOX below).

For every variable it:
  1. loads the Giovanni CSV, masks fill values, converts units, builds a complete
     monthly calendar (daily data -> monthly means);
  2. checks the bounding box (whole-globe-band boxes, partial product coverage);
  3. tests trends in a way that respects the seasons:
       - Seasonal Mann-Kendall (Hirsch-Slack, accounts for correlation between months)
       - annual mean, DJF/MAM/JJA/SON, the FIRE SEASON (May-Sep) mean and the
         FIRE-SEASON PEAK month, each with Hamed-Rao autocorrelation correction
       - month-by-month trend heatmap
  4. Sen's slope per decade with 95% CI;
  5. Pettitt step-change test;
  6. a "drop the most extreme year" check - does the trend survive without one
     spike year (e.g. a record fire year)?  Wildfire series are spiky, so this matters;
  7. Benjamini-Hochberg FDR correction across ALL tests and ALL regions together.
"""
import argparse
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
ALPHA = 0.05
MIN_YEARS = 10                       # fewer complete years than this -> no trend test
FIRE_MONTHS = (5, 6, 7, 8, 9)        # change to (4,...,9) if your region burns from April
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11),
           "Fire season (MJJAS)": FIRE_MONTHS}
PEAK = "Fire-season peak (max month)"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Suggested Giovanni boxes (West, South, East, North). These are rectangles, so they still
# include some non-forest (Hudson Bay, tundra edge, steppe edge). They are my suggestion,
# not an official boreal-forest definition. Adjust if you prefer other limits.
EXPECTED_BOX = {"canada": (-141.0, 50.0, -52.0, 65.0), "russia": (60.0, 52.0, 180.0, 68.0)}

# key found in the Giovanni column name ->
#   (label, unit, scale, offset, percent_makes_sense, months_that_make_physical_sense or None)
# value_shown = raw * scale + offset.   Units are ASSUMED from product names - verify in Giovanni.
VARIABLES = {
    # --- boreal / fire set
    "CH4_VMR":  ("Methane (AIRS)", "ppb", 1.0, 0.0, True, None),
    "_O3":      ("Ozone mixing ratio (MERRA-2)", "ppb", 28.97 / 47.998 * 1e9, 0.0, True, None),
    "BCCMASS":  ("Black carbon column mass (MERRA-2)", "mg/m2", 1e6, 0.0, True, None),
    "AODANA":   ("Aerosol optical depth (MERRA-2)", "AOD", 1.0, 0.0, True, None),
    "XCO2":     ("Column CO2 (OCO-2 GEOS)", "ppm", 1e6, 0.0, True, None),
    "TVeg":     ("Transpiration (NLDAS)", "W/m2", 1.0, 0.0, True, FIRE_MONTHS),   # ~0 when frozen
    "OCCMASS":  ("Organic carbon column mass (MERRA-2)", "mg/m2", 1e6, 0.0, True, None),
    "NDVI":     ("NDVI vegetation greenness (MODIS)", "index", 1.0, 0.0, True, FIRE_MONTHS),
    "EVI":      ("EVI vegetation greenness (MODIS)", "index", 1.0, 0.0, True, FIRE_MONTHS),
    # Any other Giovanni variable (CO, AIRS CO, MODIS AOD, fire radiative power ...) is also
    # accepted: it is analysed in its raw units (see the fallback in load_giovanni).
    # --- Third Pole set (kept so one script serves both projects)
    "TWS":      ("Terrestrial water storage", "mm", 1.0, 0.0, True, None),
    "SWE":      ("Snow water equivalent", "mm", 1.0, 0.0, True, None),
    "Psurf":    ("Surface air pressure", "hPa", 0.01, 0.0, True, None),
    "ASNOW_GL": ("Snow cover on glaciated surface", "fraction", 1.0, 0.0, True, None),
    "QV10M":    ("10 m specific humidity", "g/kg", 1000.0, 0.0, True, None),
    "Tair":     ("Surface air temperature", "degC", 1.0, -273.15, False, None),
    "AvgSurfT": ("Surface skin temperature", "degC", 1.0, -273.15, False, None),
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
    else:                                               # mid-month stamps (CO2) are fine here
        s.index = s.index.to_period("M").to_timestamp()
        monthly = s.groupby(level=0).mean().asfreq("MS")

    key = next((k for k in VARIABLES if k in col), None)
    if key is None:                     # unknown variable: analyse in raw units, still valid
        short = col.replace("mean_", "")
        label, unit, scale, offset, pct_ok, months = (short, "raw units", 1.0, 0.0, True, None)
        print(f"   note: {short} is not in VARIABLES; using raw units (check them in Giovanni)")
    else:
        label, unit, scale, offset, pct_ok, months = VARIABLES[key]
    monthly = monthly * scale + offset
    info = dict(file=os.path.basename(path), column=col, label=label, unit=unit, pct_ok=pct_ok,
                months=months, title=meta.get("Title", ""),
                user_box=parse_box(meta.get("User Bounding Box")),
                data_box=parse_box(meta.get("Data Bounding Box")),
                cadence="daily->monthly" if daily else "monthly")
    return monthly, info


def box_warnings(info, region):
    """Catch the accuracy traps: wrong region, or a product that doesn't cover the box."""
    out, ub, db = [], info["user_box"], info["data_box"]
    if not ub:
        return out
    W, S, E, N = ub
    if E - W >= 300:
        out.append("box spans ALL longitudes: this is a circumpolar band (oceans, Europe, Alaska...), "
                   "not a forest region")
    exp = next((b for k, b in EXPECTED_BOX.items() if k in region.lower()), None)
    if exp and not (W >= exp[0] - 1 and S >= exp[1] - 1 and E <= exp[2] + 1 and N <= exp[3] + 1):
        out.append(f"box {ub} is not inside the suggested {region} box {exp}")
    if db:
        ua, da = (E - W) * (N - S), (db[2] - db[0]) * (db[3] - db[1])
        if ua > 0 and da / ua < 0.6:
            out.append(f"product only covers {100 * da / ua:.0f}% of the requested box "
                       f"(data box {db}); the average is NOT for your whole region")
    return out


# ---------------------------------------------------------------- calendar ---
def deseasonalise(m):
    return m - m.groupby(m.index.month).transform("mean")


def decimal_year(idx):
    return idx.year + (idx.month - 0.5) / 12.0


def season_series(m):
    """Annual, seasonal and fire-season means (+ fire-season peak); complete seasons only.
    December belongs to the following year's DJF."""
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
    x = d[d.mo.isin(FIRE_MONTHS)]
    g = x.groupby("yr").v.agg(["max", "count"])
    out[PEAK] = g.loc[g["count"] == len(FIRE_MONTHS), "max"]
    return out


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
    """Re-test after removing the single most extreme year (largest |value - median|)."""
    yr = (ser - ser.median()).abs().idxmax()
    s2 = ser.drop(yr)
    r2 = mk_test(s2.values, s2.index.values.astype(float))
    return dict(extreme_year=int(yr), p_without_extreme=r2["p"], slope_without_extreme=r2["slope"] * 10)


# ----------------------------------------------------------------- analysis --
def analyse(monthly, info, region):
    rows, monthly_rows = [], []
    anom = deseasonalise(monthly)
    allowed = sorted(info["months"] or range(1, 13))
    base = dict(Region=region, Variable=info["label"], Unit=info["unit"])

    pv = monthly.dropna().to_frame("v")
    pv["yr"], pv["mo"] = pv.index.year, pv.index.month
    pivot = pv.pivot(index="yr", columns="mo", values="v")
    wide = pivot[allowed].dropna()                       # complete years for the months used
    if len(wide) < MIN_YEARS:
        rows.append({**base, "Test": "ALL", "n": len(wide),
                     "note": f"record too short ({len(wide)} complete years < {MIN_YEARS})"})
        return rows, monthly_rows, anom, None, allowed, None

    def add(test, r, mean, extra=None):
        pct = (r["slope"] * 10 / abs(mean) * 100) if info["pct_ok"] and mean else np.nan
        rows.append({**base, "Test": test, "n": r["n"], "tau": r["tau"], "z": r["z"], "p_raw": r["p"],
                     "slope_per_decade": r["slope"] * 10, "ci_lo_decade": r["ci_lo"] * 10,
                     "ci_hi_decade": r["ci_hi"] * 10, "pct_of_mean_per_decade": pct,
                     "var_inflation": r["var_factor"], **(extra or {})})

    smk = seasonal_mk(wide.values, wide.index.values)
    mu = float(np.nanmean(monthly[monthly.index.month.isin(allowed)]))
    months_txt = "all 12" if len(allowed) == 12 else "-".join(MONTHS[m - 1] for m in (allowed[0], allowed[-1]))
    add("Seasonal MK", smk, mu, {"months_used": months_txt})

    fs_series = None
    for name, ser in season_series(monthly).items():
        if name in SEASONS and not set(SEASONS[name]) <= set(allowed):
            continue                                     # e.g. winter transpiration ~ 0 (frozen)
        if name == PEAK and not set(FIRE_MONTHS) <= set(allowed):
            continue
        if len(ser) < MIN_YEARS:
            continue
        r = mk_test(ser.values, ser.index.values.astype(float))
        extra = {**drop_extreme_check(ser), "top3_years": ", ".join(str(y) for y in ser.nlargest(3).index)}
        if name == "Annual":
            k, pp = pettitt(ser.values)
            extra.update(pettitt_p=pp, pettitt_year=int(ser.index[k]),
                         pettitt_shift=ser.values[k:].mean() - ser.values[:k].mean())
        if name == "Fire season (MJJAS)":
            fs_series = ser
        add(name, r, float(ser.mean()), extra)

    for mo in allowed:                                   # month-by-month (heatmap)
        col = pivot[mo].dropna()
        if len(col) < MIN_YEARS:
            continue
        r = mk_test(col.values, col.index.values.astype(float), autocorr=False)
        monthly_rows.append({**base, "Month": MONTHS[mo - 1], "z": r["z"], "p_raw": r["p"],
                             "slope_per_decade": r["slope"] * 10})

    a = anom[anom.index.month.isin(allowed)].dropna()
    icpt = np.median(a.values - smk["slope"] * decimal_year(a.index))
    return rows, monthly_rows, anom, (smk["slope"], icpt), allowed, fs_series


# ------------------------------------------------------------------- plots ---
def plot_region(region, panels, res, mres, outdir):
    os.makedirs(outdir, exist_ok=True)
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
    fig.suptitle(f"{region.title()} - deseasonalised anomalies", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "anomaly_trends.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) fire-season mean per year  (is it "increasing every year" or one or two spike years?)
    fsp = [p for p in panels if p["fs"] is not None]
    if fsp:
        ncol = 2
        nrow = int(np.ceil(len(fsp) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.9 * nrow), squeeze=False)
        for ax in axes.ravel():
            ax.set_visible(False)
        for ax, p in zip(axes.ravel(), fsp):
            ax.set_visible(True)
            ser, info = p["fs"], p["info"]
            r = res[(res.Variable == info["label"]) & (res.Test == "Fire season (MJJAS)")].iloc[0]
            cols = ["crimson" if y == r.extreme_year else "tab:orange" for y in ser.index]
            ax.bar(ser.index, ser.values, color=cols)
            ax.plot(ser.index, r.slope_per_decade / 10 * ser.index +
                    np.median(ser.values - r.slope_per_decade / 10 * ser.index.values), color="navy", lw=2)
            lo, hi = ser.min(), ser.max()
            ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.1 * (hi - lo))
            ax.set_title(f"{info['label']} [{info['unit']}]\n{r.slope_per_decade:+.3g}/decade, FDR p={r.p_fdr:.3g}; "
                         f"without {int(r.extreme_year)} (red): p={r.p_without_extreme:.3g}", fontsize=8, loc="left")
            ax.grid(alpha=0.3)
        fig.suptitle(f"{region.title()} - fire-season (May-Sep) mean by year", fontsize=11, y=1.0)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fire_season_by_year.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3) month x variable heatmap
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
        ax.set_title(f"{region.title()} - which months drive the trend?  (* = significant after FDR)",
                     fontsize=10, loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "monthly_trend_heatmap.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


# --------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", help="one folder of Giovanni CSVs per region (folder name = region)")
    ap.add_argument("--out", default="boreal_outputs", help="output folder")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    all_rows, all_mrows, region_panels = [], [], {}
    for folder in args.folders:
        region = os.path.basename(os.path.abspath(folder))
        files = sorted(glob.glob(os.path.join(folder, "*.csv")))
        if not files:
            print(f"!! no CSV files in {folder!r}")
            continue
        print(f"\n===== REGION: {region}  ({len(files)} files) =====")
        boxes, panels = set(), []
        for f in files:
            try:
                monthly, info = load_giovanni(f)
            except Exception as e:                       # noqa
                print(f"!! skipped {os.path.basename(f)}: {e}")
                continue
            span = f"{monthly.dropna().index.min():%Y-%m} to {monthly.dropna().index.max():%Y-%m}"
            print(f"loaded {info['label']:38s} {span}  [{info['cadence']}]  box={info['user_box']}")
            for w in box_warnings(info, region):
                print(f"   !! {w}")
            boxes.add(info["user_box"])
            rows, mrows, anom, line, allowed, fs = analyse(monthly, info, region)
            all_rows += rows
            all_mrows += mrows
            panels.append(dict(info=info, anom=anom, line=line, allowed=allowed, fs=fs))
        if len(boxes) > 1:
            print(f"   !! WARNING: variables in {region} use different boxes {sorted(boxes, key=str)}; "
                  "re-download with one common box.")
        region_panels[region] = panels

    if not all_rows:
        sys.exit("\nNo data was loaded, so there is nothing to analyse.\n"
                 "Check that each folder you passed exists and contains the .csv files "
                 "(run:  dir foldername  to see what is inside).")
    res = pd.DataFrame(all_rows)
    for c in ("p_raw", "note", "p_without_extreme", "pettitt_p"):   # exist even if nothing was skipped
        if c not in res.columns:
            res[c] = np.nan
    ok = res["p_raw"].notna()
    res.loc[ok, "p_fdr"] = bh_fdr(res.loc[ok, "p_raw"])      # one FDR family across all regions
    res["significant_fdr"] = res["p_fdr"] < ALPHA
    res["survives_without_extreme_year"] = (res["p_without_extreme"] < ALPHA).where(res["p_without_extreme"].notna())
    res.to_csv(os.path.join(args.out, "trend_results.csv"), index=False)
    mres = pd.DataFrame(all_mrows)
    if len(mres):
        mres["p_fdr"] = bh_fdr(mres["p_raw"])
        mres.to_csv(os.path.join(args.out, "monthly_trends.csv"), index=False)

    # ---- console summary
    order = ["Seasonal MK", "Annual", "DJF", "MAM", "JJA", "SON", "Fire season (MJJAS)", PEAK]
    tab = res[ok].copy()
    tab["cell"] = tab.apply(lambda r: f"{r.slope_per_decade:+.3g}" + ("*" if r.significant_fdr else ""), axis=1)
    pv = tab.pivot(index=["Region", "Variable"], columns="Test", values="cell")
    pv = pv.reindex(columns=[c for c in order if c in pv.columns]).fillna("-")
    print("\nSen slope per decade   (* = significant after FDR correction, alpha=0.05; '-' = not tested)")
    print(pv.to_string())

    print("\nFIRE-SEASON (May-Sep) mean: does the trend survive without the single most extreme year?")
    fs = res[res.Test == "Fire season (MJJAS)"]
    for _, r in fs.iterrows():
        print(f"  {r.Region:8s} {r.Variable:38s} {r.slope_per_decade:+.3g} {r.Unit}/decade  FDR p={r.p_fdr:.3g}"
              f" | extreme year {int(r.extreme_year)}: p without it = {r.p_without_extreme:.3g}"
              f" -> {'still significant' if r.survives_without_extreme_year else 'NOT significant'}"
              f" | highest years: {r.top3_years}")
    notes = res.loc[~ok, ["Region", "Variable", "note"]]
    for _, r in notes.iterrows():
        print(f"  NOT TESTED  {r.Region} / {r.Variable}: {r.note}")
    ann = res[(res.Test == "Annual") & ok & (res.pettitt_p < ALPHA)]
    for _, r in ann.iterrows():
        print(f"  Possible step change: {r.Region} / {r.Variable} around {int(r.pettitt_year)} "
              f"(shift {r.pettitt_shift:+.3g} {r.Unit}, Pettitt p={r.pettitt_p:.3f})")

    for region, panels in region_panels.items():
        plot_region(region, panels, res[res.Region == region],
                    mres[mres.Region == region] if len(mres) else mres, os.path.join(args.out, region))
    print(f"\nSaved results and figures to {args.out}/")


if __name__ == "__main__":
    main()