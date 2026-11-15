#!/usr/bin/env python3
"""
Third Pole trend detective  (improved version)
==============================================
Usage:   python third_pole_trends.py  [folder_with_giovanni_csvs]  [output_folder]

What it does, per variable
  1. Loads the Giovanni CSV, masks fill values, converts to friendly units.
  2. Puts everything on a complete monthly calendar (daily data -> monthly means).
  3. Trend tests that respect the seasons:
       - Seasonal Mann-Kendall (Hirsch-Slack) over all 12 months, INCLUDING the
         correlation between months (the old code assumed they were independent).
       - Annual mean and each season (DJF/MAM/JJA/SON) tested separately with the
         Hamed-Rao correction for year-to-year autocorrelation.
       - A month-by-month trend heatmap (which months drive the trend?).
  4. Sen's slope with 95% confidence interval, shown per decade.
  5. Pettitt test for a step change (regime shift) in the annual series.
  6. Benjamini-Hochberg FDR across all tests (we run ~40, so some "hits" are luck).
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
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "trend_outputs"
ALPHA = 0.05
MIN_YEARS = 10          # fewer complete years than this -> no trend test
# Edit these to change the season definitions (e.g. monsoon = JJAS).
SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# key found in the Giovanni column name -> (label, unit, scale, offset, percent_makes_sense)
# value_shown = raw * scale + offset.  (Units are assumed from the product names.)
VARIABLES = {
    "TWS":      ("Terrestrial water storage", "mm", 1.0, 0.0, True),
    "SWE":      ("Snow water equivalent", "mm", 1.0, 0.0, True),
    "Psurf":    ("Surface air pressure", "hPa", 0.01, 0.0, True),
    "ASNOW_GL": ("Snow cover on glaciated surface", "fraction", 1.0, 0.0, True),
    "QV10M":    ("10 m specific humidity", "g/kg", 1000.0, 0.0, True),
    "Tair":     ("Surface air temperature", "degC", 1.0, -273.15, False),
    "AvgSurfT": ("Surface skin temperature", "degC", 1.0, -273.15, False),
}


# ------------------------------------------------------------------ loading --
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
        v = v.mask(np.isclose(v, fill))          # Giovanni fill values -> NaN
    v = v.mask(v.abs() > 1e14)                   # safety net for 1e15-style fills

    s = pd.Series(v.values, index=pd.to_datetime(df["time"]))
    step = s.index.to_series().diff().median()
    if step <= pd.Timedelta(days=2):             # daily -> monthly mean (>=80% coverage)
        g = s.resample("MS").agg(["mean", "count"])
        monthly = g["mean"].where(g["count"] >= 0.8 * g.index.days_in_month)
    else:                                        # already monthly
        s.index = s.index.to_period("M").to_timestamp()
        monthly = s.groupby(level=0).mean().asfreq("MS")

    key = next(k for k in VARIABLES if k in col)
    label, unit, scale, offset, pct_ok = VARIABLES[key]
    monthly = monthly * scale + offset
    info = dict(file=os.path.basename(path), column=col, label=label, unit=unit,
                pct_ok=pct_ok, bbox=meta.get("User Bounding Box", "?"),
                cadence="daily->monthly" if step <= pd.Timedelta(days=2) else "monthly")
    return monthly, info


def deseasonalise(m):
    """Monthly anomalies: subtract each calendar month's long-term mean."""
    return m - m.groupby(m.index.month).transform("mean")


def decimal_year(idx):
    return idx.year + (idx.month - 0.5) / 12.0


def season_series(m):
    """Annual and seasonal means; only COMPLETE seasons/years are kept.
    Dec belongs to the following year's DJF."""
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
    return (S - np.sign(S)) / np.sqrt(varS)      # continuity correction


def hamed_rao_factor(x, t, sen):
    """Variance inflation for autocorrelation (Hamed & Rao 1998).
    Clipped at 1 so negative autocorrelation never makes a trend look MORE significant."""
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
    """Seasonal Mann-Kendall (Hirsch & Slack 1984) with the covariance between seasons.
    M: (n_years, n_seasons) complete matrix, t: year of each row.
    The old code summed the 12 monthly variances as if months were independent;
    months move together, so that overstates the evidence (z-scores of 10-16)."""
    n, g = M.shape
    iu = np.triu_indices(n, 1)
    sgn = np.sign(M[None, :, :] - M[:, None, :])[iu]                 # (pairs, seasons)
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
    """Pettitt change-point test -> (index of first point after the shift, p-value)."""
    x = np.asarray(x, float)
    n = len(x)
    U = np.array([np.sign(x[: k + 1, None] - x[None, k + 1:]).sum() for k in range(n - 1)])
    K = np.abs(U).max()
    return int(np.abs(U).argmax()) + 1, min(1.0, 2 * np.exp(-6 * K ** 2 / (n ** 3 + n ** 2)))


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


# --------------------------------------------------------------------- main --
def analyse(monthly, info):
    rows, monthly_rows = [], []
    m = monthly.copy()
    anom = deseasonalise(m)
    pivot = m.dropna().to_frame("v")
    pivot["yr"], pivot["mo"] = pivot.index.year, pivot.index.month
    wide = pivot.pivot(index="yr", columns="mo", values="v")
    wide = wide.dropna()                                   # keep complete calendar years
    n_years = len(wide)
    base = dict(Variable=info["label"], Unit=info["unit"])
    if n_years < MIN_YEARS:
        rows.append({**base, "Test": "ALL", "n": n_years, "note": f"record too short ({n_years} complete years < {MIN_YEARS})"})
        return rows, monthly_rows, anom, None

    def add(test, r, series_mean, extra=None):
        pct = (r["slope"] * 10 / abs(series_mean) * 100) if info["pct_ok"] and series_mean else np.nan
        rows.append({**base, "Test": test, "n": r["n"], "tau": r["tau"], "z": r["z"], "p_raw": r["p"],
                     "slope_per_decade": r["slope"] * 10, "ci_lo_decade": r["ci_lo"] * 10,
                     "ci_hi_decade": r["ci_hi"] * 10, "pct_of_mean_per_decade": pct,
                     "var_inflation": r["var_factor"], **(extra or {})})

    # (a) seasonal MK over all 12 months
    smk = seasonal_mk(wide.values, wide.index.values)
    add("All months (Seasonal MK)", smk, float(np.nanmean(m)))

    # (b) annual + each season, with autocorrelation correction
    ss = season_series(m)
    extra_annual = {}
    for name, ser in ss.items():
        if len(ser) < MIN_YEARS:
            continue
        r = mk_test(ser.values, ser.index.values)
        extra = {}
        if name == "Annual":
            k, pp = pettitt(ser.values)
            yr = int(ser.index[k])
            diff = ser.values[k:].mean() - ser.values[:k].mean()
            extra = {"pettitt_p": pp, "pettitt_year": yr, "pettitt_shift": diff}
        add(name, r, float(ser.mean()), extra)

    # (c) month-by-month trends (for the heatmap)
    for mo in range(1, 13):
        col = wide[mo]
        r = mk_test(col.values, col.index.values, autocorr=False)
        monthly_rows.append({**base, "Month": MONTHS[mo - 1], "z": r["z"], "p_raw": r["p"],
                             "slope_per_decade": r["slope"] * 10})

    # trend line for the plot (Seasonal Sen slope on the monthly anomalies)
    t = decimal_year(anom.dropna().index)
    icpt = np.median(anom.dropna().values - smk["slope"] * t)
    return rows, monthly_rows, anom, (smk["slope"], icpt)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        sys.exit(f"No CSV files found in {DATA_DIR!r}")

    all_rows, all_monthly, panels, infos = [], [], [], []
    for f in files:
        try:
            monthly, info = load_giovanni(f)
        except Exception as e:                      # noqa
            print(f"!! skipped {os.path.basename(f)}: {e}")
            continue
        rows, mrows, anom, line = analyse(monthly, info)
        all_rows += rows
        all_monthly += mrows
        panels.append((info, monthly, anom, line))
        infos.append(info)
        span = f"{monthly.dropna().index.min():%Y-%m} to {monthly.dropna().index.max():%Y-%m}"
        print(f"loaded {info['label']:34s} {span}  [{info['cadence']}]  bbox={info['bbox']}")

    boxes = {i["bbox"] for i in infos}
    if len(boxes) > 1:
        print(f"\n!! WARNING: datasets use different bounding boxes {sorted(boxes)}; "
              "trends are not strictly comparable. Re-download with one common box.")

    res = pd.DataFrame(all_rows)
    ok = res["p_raw"].notna()
    res.loc[ok, "p_fdr"] = bh_fdr(res.loc[ok, "p_raw"])
    res["significant_fdr"] = res["p_fdr"] < ALPHA
    res["direction"] = np.where(res["slope_per_decade"] > 0, "up", "down")
    res.to_csv(os.path.join(OUT_DIR, "trend_results.csv"), index=False)

    mres = pd.DataFrame(all_monthly)
    mres["p_fdr"] = bh_fdr(mres["p_raw"])
    mres.to_csv(os.path.join(OUT_DIR, "monthly_trends.csv"), index=False)

    # ---- console summary
    order = ["All months (Seasonal MK)", "Annual"] + list(SEASONS)
    tab = res[ok].copy()
    tab["cell"] = tab.apply(lambda r: f"{r.slope_per_decade:+.3g}" + ("*" if r.significant_fdr else ""), axis=1)
    pv = tab.pivot(index="Variable", columns="Test", values="cell")[order]
    print("\nSen slope per decade   (* = significant after FDR correction, alpha=0.05)")
    print(pv.to_string())
    print("\nNotes:", *res.loc[~ok, ["Variable", "note"]].apply(lambda r: f"{r.Variable}: {r.note}", axis=1))
    ann = res[(res.Test == "Annual") & ok & (res.pettitt_p < ALPHA)]
    for _, r in ann.iterrows():
        print(f"Possible step change: {r.Variable} around {int(r.pettitt_year)} "
              f"(shift {r.pettitt_shift:+.3g} {r.Unit}, Pettitt p={r.pettitt_p:.3f})")

    # ---- figure 1: deseasonalised anomalies + trend line
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.5 * len(panels)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (info, monthly, anom, line) in zip(axes, panels):
        a = anom.dropna()
        ax.plot(a.index, a.values, color="tab:blue", alpha=0.35, lw=0.8, label="Monthly anomaly")
        ax.plot(a.index, a.rolling(12, center=True, min_periods=12).mean(), color="navy", lw=1.8,
                label="12-month mean")
        title = f"{info['label']} [{info['unit']}] - record too short for a trend test"
        if line:
            ax.plot(a.index, line[0] * decimal_year(a.index) + line[1], color="crimson", lw=2,
                    label="Seasonal Sen trend")
            r = res[(res.Variable == info["label"]) & (res.Test == "All months (Seasonal MK)")].iloc[0]
            star = "significant" if r.significant_fdr else "not significant"
            title = (f"{info['label']} [{info['unit']}]   {r.slope_per_decade:+.3g} {info['unit']}/decade "
                     f"(95% CI {r.ci_lo_decade:+.3g} to {r.ci_hi_decade:+.3g}; FDR p={r.p_fdr:.3g}, {star})")
        ax.set_title(title, fontsize=9, loc="left", fontweight="bold")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel("anomaly", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "anomaly_trends.png"), dpi=170)
    plt.close(fig)

    # ---- figure 2: month x variable heatmap of trend z-scores
    if len(mres):
        z = mres.pivot(index="Variable", columns="Month", values="z")[MONTHS]
        pf = mres.pivot(index="Variable", columns="Month", values="p_fdr")[MONTHS]
        fig, ax = plt.subplots(figsize=(11, 0.6 * len(z) + 1.8))
        im = ax.imshow(z.values, cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
        ax.set_xticks(range(12), MONTHS)
        ax.set_yticks(range(len(z)), z.index)
        for i in range(z.shape[0]):
            for j in range(12):
                ax.text(j, i, f"{z.values[i, j]:+.1f}" + ("*" if pf.values[i, j] < ALPHA else ""),
                        ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="Mann-Kendall z  (red = rising, blue = falling)")
        ax.set_title("Which months drive the trend?  (* = significant after FDR)", fontsize=10, loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "monthly_trend_heatmap.png"), dpi=170)
        plt.close(fig)

    print(f"\nSaved: trend_results.csv, monthly_trends.csv, anomaly_trends.png, monthly_trend_heatmap.png -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
