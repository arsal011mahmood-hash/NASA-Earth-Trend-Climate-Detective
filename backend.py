"""
backend.py - loads Giovanni / AppEEARS CSVs and computes the trend statistics.
No Streamlit in here, so you can test it alone:  python backend.py

Put one CSV per variable in  data/karakoram/<key>.csv  (keys below). Accepted:
  * the simple format:  date,value
  * a raw Giovanni time-series CSV (metadata header lines are skipped)
  * an AppEEARS *-Statistics.csv (the "Mean" column is used)
Daily / 8-day / 16-day data is averaged to monthly automatically.
If a file is missing, clearly-labelled SYNTHETIC DEMO data is used for that
variable so the app can be tried before the real files arrive.
"""
import io
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
YEARS = (2000, 2026)          # slider range
MIN_YEARS = 5                 # shortest window the slider allows
REGION = dict(id="karakoram", name="Karakoram", bbox=[34.0, 37.5, 73.0, 79.0])  # S,N,W,E


def _k2c(s):  # Kelvin -> Celsius only if the values look like Kelvin
    return s - 273.15 if s.mean() > 150 else s


# demo=(base, seasonal amplitude, trend per year, noise sd)  -> synthetic only
VARIABLES = [
    dict(key="t2m", label="Air temperature (2 m)", unit="°C", fix=_k2c, demo=(-2, 12, 0.04, 1.2),
         source="MERRA-2 monthly, Giovanni: M2TMNXSLV 'T2M'",
         definition="How warm the air is about 2 metres above the ground, averaged over the Karakoram box.",
         why="Warmer air melts snow and glacier ice faster."),
    dict(key="lst", label="Land surface temperature", unit="°C",
         fix=lambda s: _k2c(s * 0.02 if s.mean() > 1000 else s), demo=(5, 18, 0.05, 1.5),
         source="MODIS LST, AppEEARS: MOD11A2 or MOD11C3 'LST_Day'",
         definition="The temperature of the ground or snow surface itself, measured by satellite infrared sensors.",
         why="A warmer surface speeds up melting of snow and ice."),
    dict(key="snow", label="Snow cover", unit="%", fix=lambda s: s, demo=(45, 30, -0.25, 6),
         source="MODIS NDSI snow cover, AppEEARS: MOD10A1 / MOD10CM (check availability)",
         definition="The share of the land covered by snow, from daily MODIS satellite images.",
         why="Snow reflects sunlight, feeds glaciers and supplies rivers like the Indus."),
    dict(key="qv2m", label="Specific humidity", unit="g/kg", fix=lambda s: s * 1000 if s.mean() < 0.1 else s,
         demo=(3, 2, 0.01, 0.4), source="MERRA-2 monthly, Giovanni: M2TMNXSLV 'QV2M'",
         definition="How much water vapour is in the air: grams of vapour per kilogram of air.",
         why="Warmer air holds more vapour, which affects heat trapping and snowfall."),
    dict(key="ndvi", label="Vegetation greenness (NDVI)", unit="index",
         fix=lambda s: s * 0.0001 if s.mean() > 1.5 else s, demo=(0.12, 0.05, 0.0004, 0.02),
         source="MODIS vegetation indices, AppEEARS: MOD13A3 or MOD13C2 'NDVI'",
         definition="A 'greenness' score from satellite light measurements. Higher means more plant growth.",
         why="Greener high-altitude land can signal warming and a retreating snow line."),
    dict(key="precip", label="Precipitation", unit="mm/day",
         fix=lambda s: s * 24,  # Giovanni IMERG monthly is mm/hr. Use `s` if you downloaded mm/day.
         demo=(1.5, 1.0, 0.0, 0.4), source="GPM IMERG monthly, Giovanni: GPM_3IMERGM 'precipitation'",
         definition="Rain and snowfall water reaching the ground, in millimetres per day.",
         why="Glaciers need snowfall to stay in balance, but precipitation varies a lot year to year."),
    dict(key="cloud", label="Cloud fraction", unit="%", fix=lambda s: s * 100 if s.max() <= 1.5 else s,
         demo=(45, 15, 0.0, 6), source="MODIS Terra cloud fraction, Giovanni: MOD08_M3",
         definition="The fraction of the sky covered by cloud, as seen by MODIS.",
         why="Clouds shade the surface and trap heat, but this signal is noisy."),
]


# ------------------------------------------------------------------ loading
def read_csv(path):
    lines = Path(path).read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    head = next((i for i, l in enumerate(lines)
                 if l.split(",")[0].strip().strip('"').lower() in ("time", "date", "timestamp")), 0)
    df = pd.read_csv(io.StringIO("\n".join(lines[head:])))
    df.columns = [str(c).strip() for c in df.columns]
    dcol = next((c for c in df.columns if c.lower() in ("time", "date", "timestamp")), df.columns[0])
    skip = ("latitude", "longitude", "count", "std", "minimum", "maximum", "median",
            "variance", "range", "pixel", "qa", "quality")
    cands = [c for c in df.columns if c != dcol and c.lower() not in ("id", "aid")
             and not any(k in c.lower() for k in skip)]
    for c in [c for c in cands if "mean" in c.lower()] + cands:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().sum() > 12:
            break
    else:
        raise ValueError("no numeric value column found")
    d = pd.to_datetime(df[dcol], errors="coerce", utc=True).dt.tz_localize(None)
    s = pd.Series(v.to_numpy(), index=d)
    s = s[~s.index.isna()].dropna()
    return s[(s > -9000) & (s < 1e20)].sort_index()      # drop fill values


def monthly(s):
    return s.resample("MS").mean().dropna()


def demo_series(var):
    base, amp, trend, noise = var["demo"]
    rng = np.random.default_rng(zlib.crc32(var["key"].encode()))
    idx = pd.date_range("2000-01-01", "2026-10-01", freq="MS")
    t = idx.year.to_numpy() + (idx.month.to_numpy() - 0.5) / 12
    e = np.zeros(len(t))
    for i in range(1, len(t)):
        e[i] = 0.5 * e[i - 1] + rng.normal(0, noise)
    return pd.Series(base + amp * np.sin(2 * np.pi * (t - 0.3)) + trend * (t - 2000) + e, index=idx)


# ------------------------------------------------------------------ statistics
def to_arrays(s):
    """Monthly series -> (decimal year, anomaly, year). Anomaly = value minus that calendar month's mean."""
    clim = s.groupby(s.index.month).mean()
    anom = s.to_numpy() - clim.reindex(s.index.month).to_numpy()
    yr = s.index.year.to_numpy()
    return yr + (s.index.month.to_numpy() - 0.5) / 12, anom, yr


def mann_kendall(x):
    n = len(x)
    s = sum(np.sign(x[i + 1:] - x[i]).sum() for i in range(n - 1))
    _, c = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(c * (c - 1) * (2 * c + 5))) / 18
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0
    return s / (n * (n - 1) / 2), 2 * (1 - stats.norm.cdf(abs(z)))      # tau, p


def window_stats(t, a, yr, y0, y1):
    """[slope/yr, ci_lo, ci_hi, kendall_tau, mk_p, n_years, intercept, n_eff] for years y0..y1, or None."""
    m = (yr >= y0) & (yr <= y1)
    t_, a_, y_ = t[m], a[m], yr[m]
    ann = [a_[y_ == y].mean() for y in np.unique(y_) if (y_ == y).sum() >= 6]
    if m.sum() < 60 or len(ann) < MIN_YEARS:
        return None
    fit = stats.linregress(t_, a_)
    n = len(a_)
    resid = a_ - (fit.intercept + fit.slope * t_)
    r1 = float(np.clip(np.corrcoef(resid[:-1], resid[1:])[0, 1], 0, 0.95))   # month-to-month autocorrelation
    neff = max(4.0, n * (1 - r1) / (1 + r1))                                   # fewer "independent" points
    se = fit.stderr * np.sqrt((n - 2) / (neff - 2))
    tc = stats.t.ppf(0.975, neff - 2)
    tau, p = mann_kendall(np.array(ann))                                       # on yearly means
    g = lambda x: float(f"{x:.6g}")
    return [g(fit.slope), g(fit.slope - tc * se), g(fit.slope + tc * se), g(tau), g(p),
            len(ann), g(fit.intercept), round(float(neff), 1)]


# ------------------------------------------------------------------ payload for the front end
def build_payload():
    out, warnings = [], []
    for v in VARIABLES:
        f = DATA_DIR / REGION["id"] / f"{v['key']}.csv"
        try:
            demo = not f.exists()
            s = monthly(v["fix"](demo_series(v) if demo else read_csv(f)))
            if len(s) < 60:
                raise ValueError(f"only {len(s)} monthly values (need 60+)")
        except Exception as e:
            warnings.append(f"{v['label']}: could not use {f.name} ({e}). Skipped.")
            continue
        t, a, yr = to_arrays(s)
        table = {}
        for y0 in range(YEARS[0], YEARS[1] - MIN_YEARS + 2):
            for y1 in range(y0 + MIN_YEARS - 1, YEARS[1] + 1):
                r = window_stats(t, a, yr, y0, y1)
                if r:
                    table[f"{y0}-{y1}"] = r
        out.append(dict(
            key=v["key"], label=v["label"], unit=v["unit"], definition=v["definition"], why=v["why"],
            source=v["source"], demo=demo, file=f"data/{REGION['id']}/{v['key']}.csv",
            dates=[d.strftime("%Y-%m") for d in s.index],
            raw=[round(float(x), 4) for x in s], anom=[round(float(x), 4) for x in a], stats=table))
    demos = [v["label"] for v in out if v["demo"]]
    if demos:
        warnings.insert(0, "DEMO DATA (synthetic, not real observations) is being used for: " + ", ".join(demos)
                        + f". Put the real CSVs in data/{REGION['id']}/ named <key>.csv.")
    return dict(region=REGION, years=list(YEARS), variables=out), warnings


def data_signature():
    d = DATA_DIR / REGION["id"]
    return tuple((p.name, p.stat().st_mtime) for p in sorted(d.glob("*.csv"))) if d.exists() else ()


if __name__ == "__main__":
    payload, warns = build_payload()
    print(*warns, sep="\n")
    for v in payload["variables"]:
        r = v["stats"].get(f"{YEARS[0]}-{YEARS[1]}")
        print(f"{v['key']:7s} demo={v['demo']} slope={r[0]} CI=({r[1]},{r[2]}) tau={r[3]} p={r[4]}")
