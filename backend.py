"""
backend.py - loads the real NASA Giovanni CSVs for three regions and computes every
trend statistic the app shows. No Streamlit in here, so you can test it alone:

    python backend.py

Folder layout (next to this file):
    data/third_pole_depletion/*.csv
    data/boreal_fire_trends/*.csv
    data/great_barrier_reef/*.csv

File NAMES do not matter: each CSV is recognised from the Giovanni variable name in its
header (e.g. "..._TWS_tavg", "..._sst"). Any other Giovanni area-averaged time series
dropped into a folder is also analysed, in its raw units.

Statistics (same family of methods as the three stand-alone scripts):
  * anomaly = monthly value minus that calendar month's long-term mean (removes seasons)
  * rate of change = least-squares slope of the anomalies, with a 95% confidence interval
    and a t-test widened for month-to-month autocorrelation (effective sample size)
  * Mann-Kendall test + Sen's slope on the yearly mean anomalies
  * full-record extra checks: Seasonal Mann-Kendall (Hirsch-Slack, with the covariance
    between months), a key-season test with Hamed-Rao correction, "drop the most extreme
    year" robustness check, Pettitt change-point test, month-by-month trends, and
    Benjamini-Hochberg false-discovery-rate correction across each region.
"""
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
MIN_YEARS = 5          # shortest window the slider allows
RELIABLE_YEARS = 10    # fewer years than this -> shown, but flagged "short record"
ALPHA = 0.05
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ---------------------------------------------------------------- regions
# bbox is [South, North, West, East] = the box the CSVs were actually downloaded with.
REGIONS = [
    dict(
        id="third_pole_depletion", name="Third Pole", title="Third Pole Depletion",
        bbox=[25.0, 50.0, 65.0, 105.0], focus=[85.0, 36.0],
        key_season=("Melt season (Jun–Sep)", (6, 7, 8, 9)),
        intro=("The Third Pole is the Hindu Kush–Karakoram–Himalaya–Tibetan Plateau region. Outside the Arctic "
               "and Antarctica it holds more ice than anywhere else on Earth, and its snow and glaciers feed "
               "rivers such as the Indus, Ganges, Brahmaputra and Yangtze that well over a billion people rely on. "
               "“Depletion” means that this stored ice, snow and water is being lost. "
               "The box used here (25–50°N, 65–105°E) covers the plateau and mountains but also nearby deserts "
               "and plains, so every number is an average over that whole rectangle."),
    ),
    dict(
        id="boreal_fire_trends", name="Boreal belt", title="Boreal Fire Trends",
        bbox=[48.0, 70.0, -180.0, 180.0], focus=[100.0, 70.0],
        key_season=("Fire season (May–Sep)", (5, 6, 7, 8, 9)),
        intro=("The boreal (northern) forest is the ring of conifer forest across Canada, Alaska, Scandinavia "
               "and Russia. It stores huge amounts of carbon in trees and frozen soils, and its fire seasons have "
               "become more extreme. These datasets track smoke, soot and greenhouse gases. "
               "Important: the files were downloaded for the whole 48–70°N ring around the planet (every "
               "longitude), so the averages include oceans, Europe and tundra, not only forest."),
    ),
    dict(
        id="great_barrier_reef", name="Great Barrier Reef", title="Great Barrier Reef Under Heat",
        bbox=[-24.0, -10.0, 142.0, 154.0], focus=[148.0, -17.0],
        key_season=("Bleaching season (Jan–Mar)", (1, 2, 3)),
        intro=("The Great Barrier Reef off Queensland, Australia, is the largest coral reef system on Earth. "
               "Corals bleach (expel the algae that feed them) when the sea stays unusually warm for weeks, which "
               "happens mostly in the southern summer, January to March. Mass bleaching hit the reef in 2016, 2017, "
               "2020, 2022, 2024 and 2025. Here sea temperature, air temperature, wind, cloud, rain and ocean "
               "greenness are tested for trends over the box 10–24°S, 142–154°E."),
    ),
]

# ---------------------------------------------------------------- variables
# Matched (in this order) against the Giovanni column name.  value_shown = raw*scale + offset.
# Units were checked against the actual files (e.g. IMERG was downloaded as mm/day,
# NLDAS transpiration is W/m2, MERRA-2 ozone is a mass mixing ratio in kg/kg).
VARS = [
    # ---------------- Third Pole
    dict(match="TWS", label="Terrestrial water storage", unit="mm",
         source="GLDAS Catchment model v2.2 with GRACE data assimilation, daily 0.25° (GLDAS_CLSM025_DA1_D 'TWS')",
         definition="All the water held on and under the land, added together: snow, soil moisture, surface water "
                    "and groundwater, written as the depth of water it would make (in mm) if spread evenly.",
         why="The most direct measure of “depletion”. A falling value means the region is losing water "
             "overall, through glacier and snow loss, drying soils or groundwater pumping. This model is nudged "
             "toward GRACE satellite gravity measurements, which weigh the water from space."),
    dict(match="SWE", label="Snow water equivalent", unit="mm",
         source="GLDAS Noah model v2.1, monthly 1° (GLDAS_NOAH10_M 'SWE_inst')",
         definition="How much water you would get if you melted all the snow on the ground, as a depth in mm. "
                    "It measures the snowpack's water content, not its height.",
         why="The snowpack is a natural reservoir: it stores winter precipitation and releases it as meltwater in "
             "spring and summer, when farms and rivers downstream need it most."),
    dict(match="ASNOW_GL", label="Snow cover on glaciers", unit="%", scale=100.0,
         source="MERRA-2 reanalysis, monthly 0.5°×0.625° (M2TMNXGLC 'ASNOW_GL')",
         definition="The share of glacier-covered ground (in the MERRA-2 model) that also has snow lying on top of it.",
         why="Fresh snow on a glacier is bright and reflects sunlight, shielding the ice below. Bare, darker ice "
             "absorbs more sunlight and melts faster."),
    dict(match="Tair", label="Surface air temperature", unit="°C", offset=-273.15,
         source="FLDAS Noah model, monthly 0.1° (FLDAS_NOAH01_C_GL_M 'Tair_f_tavg')",
         definition="The temperature of the air close to the ground (about 2 m up), averaged over the whole box.",
         why="Warmer air delivers more heat to snow and ice, so it melts earlier and faster, and more "
             "precipitation falls as rain instead of snow."),
    dict(match="AvgSurfT", label="Surface skin temperature", unit="°C", offset=-273.15,
         source="GLDAS Noah model v2.1, monthly 0.25° (GLDAS_NOAH025_M 'AvgSurfT_inst')",
         definition="The temperature of the very top of the land, snow or ice surface (its “skin”), "
                    "rather than of the air above it.",
         why="The surface temperature decides whether snow and ice can melt at all (it must reach 0 °C) and how "
             "much heat the ground gives back to the air.",
         caveat="GLDAS v2.1 is a model driven by input data that changed over time. A sudden jump flagged by the "
                "change-point test can be partly an artefact of those inputs, not only real warming. Compare it "
                "with the air-temperature series from a different model (FLDAS)."),
    dict(match="Psurf", label="Surface air pressure", unit="hPa", scale=0.01,
         source="GLDAS Noah model v2.1, monthly 0.25° (GLDAS_NOAH025_M 'Psurf_f_inst')",
         definition="The weight of the whole column of air pressing down on the ground, in hectopascals. It is low "
                    "here (about 755 hPa, against 1013 at sea level) because much of the box is high mountain land.",
         why="Pressure is mainly a check on the weather patterns (circulation) over the region. It is not a direct "
             "measure of ice loss."),
    dict(match="QV10M", label="10 m specific humidity", unit="g/kg", scale=1000.0,
         source="MERRA-2 reanalysis",
         definition="Grams of water vapour in each kilogram of air, 10 m above the ground.",
         why="Warmer air can hold more water vapour, which traps heat and changes snowfall."),
    # ---------------- Boreal
    dict(match="AODANA", label="Aerosol optical depth", unit="AOD",
         source="MERRA-2 reanalysis (assimilated), monthly 0.5°×0.625° (M2IMNXGAS 'AODANA')",
         definition="How much sunlight is blocked by tiny particles (smoke, dust, sea salt, pollution) in the whole "
                    "air column. No units: about 0.05 is very clean air, above 0.5 is a thick haze.",
         why="Wildfire smoke shows up as AOD spikes. More smoke means worse air quality and less sunlight reaching "
             "the ground."),
    dict(match="BCCMASS", label="Black carbon (soot)", unit="mg/m²", scale=1e6,
         source="MERRA-2 reanalysis, monthly 0.5°×0.625° (M2TMNXAER 'BCCMASS')",
         definition="The mass of black carbon (soot) in the air column above each square metre of ground, in milligrams.",
         why="Soot comes from burning vegetation and fuels. It absorbs sunlight and warms the air, and when it "
             "settles on snow and ice it darkens them and speeds up melting."),
    dict(match="XCO2", label="Carbon dioxide (column)", unit="ppm", scale=1e6,
         source="OCO-2 GEOS Level 3, monthly 0.5°×0.625° (OCO2_GEOS_L3CO2_MONTH 'XCO2')",
         definition="The average carbon dioxide concentration in the air column from the ground to the top of the "
                    "atmosphere, in parts per million (molecules of CO₂ per million molecules of dry air).",
         why="CO₂ is the main greenhouse gas driving warming. Fires and thawing soils add to it. Its rise is "
             "global, so this mostly shows the worldwide increase."),
    dict(match="CH4_VMR", label="Methane", unit="ppb",
         source="AIRS v7 (AIRS-only, daytime), monthly 1°, 1000 hPa level (AIRS3STM 'CH4_VMR_A')",
         definition="Methane molecules per billion molecules of air, as retrieved by the AIRS infrared sounder.",
         why="Methane is a powerful greenhouse gas. Northern wetlands, thawing permafrost and fires release it.",
         caveat="AIRS measures methane mostly in the middle troposphere. The 1000 hPa (“near-surface”) "
                "level is largely filled in from higher up, so this is not a true ground-level measurement."),
    dict(match="_O3", label="Near-surface ozone", unit="ppb", scale=28.97 / 47.998 * 1e9,
         source="MERRA-2 reanalysis, monthly, 1000 hPa level (M2IMNPANA 'O3', converted from kg/kg to ppb)",
         definition="Ozone in the air near the ground (1000 hPa level), in parts per billion. This is the “bad” "
                    "ozone at breathing height, not the protective ozone layer high above.",
         why="Fire smoke contains gases that make ozone in sunlight. Ozone near the ground harms lungs, crops and trees."),
    dict(match="TVeg", label="Plant transpiration", unit="W/m²", months=(5, 6, 7, 8, 9),
         source="NLDAS-2 Noah model, monthly 0.125° (NLDAS_NOAH0125_M 'TVeg')",
         definition="Water that plants pull from the soil and release through their leaves, given as the energy it "
                    "uses (1 W/m² is about 0.035 mm of water per day). Tested on May–September only, because "
                    "frozen winter months are close to zero.",
         why="Higher transpiration means more active plants or a longer growing season, but it also dries soils, "
             "which can raise fire risk.",
         caveat="NLDAS only covers North America up to about 53°N. Its data box is a thin strip along the "
                "US–Canada border (48–53°N, 125–67°W), about 4% of the boreal box, so this is NOT a boreal average."),
    # ---------------- Great Barrier Reef
    dict(match="_sst", label="Sea surface temperature", unit="°C", heat_stress=True,
         source="MODIS-Aqua, 11 µm daytime, monthly 4 km (MODISA_L3m_SST 'sst')",
         definition="The temperature of the top millimetre of the ocean, measured from space by the MODIS "
                    "infrared sensor during the day, averaged over all ocean pixels in the box.",
         why="Corals bleach when the water stays about 1 °C or more above their usual summer maximum for weeks. "
             "Long-term warming makes those heatwaves more frequent.",
         caveat="MODIS-Aqua's orbit has been drifting since 2022 (later overpass time), which may add small "
                "artefacts to the most recent years."),
    dict(match="chlor_a", label="Chlorophyll-a (ocean greenness)", unit="mg/m³",
         source="MODIS-Aqua ocean colour, monthly 4 km (MODISA_L3m_CHL 'chlor_a')",
         definition="The concentration of chlorophyll, the green pigment in tiny ocean plants (phytoplankton), in "
                    "milligrams per cubic metre of seawater.",
         why="Higher values can mean more nutrients from river floods and farm runoff, which feeds algae and the "
             "coral-eating crown-of-thorns starfish. Very clear water usually has low chlorophyll.",
         caveat="Near the coast, river mud can be mistaken for chlorophyll by the satellite. Values naturally swing "
                "a lot, so trends are hard to detect."),
    dict(match="_T2M", label="Air temperature (2 m)", unit="°C", offset=-273.15,
         source="MERRA-2 reanalysis, monthly 0.5°×0.625° (M2IMNXASM 'T2M')",
         definition="The temperature of the air 2 metres above the land or sea surface.",
         why="Warmer air heats the ocean surface and reduces the cooling the sea gets from the atmosphere."),
    dict(match="T2M", label="Air temperature (2 m)", unit="°C", offset=-273.15,
         source="MERRA-2 reanalysis",
         definition="The temperature of the air 2 metres above the ground.",
         why="Warmer air melts snow and heats the surface."),
    dict(match="SPEED", label="Surface wind speed", unit="m/s",
         source="MERRA-2 reanalysis, monthly 0.5°×0.625° (M2TMNXFLX 'SPEED')",
         definition="How fast the wind blows near the surface (10 m), in metres per second.",
         why="Wind mixes the warm surface water with cooler water below and cools it by evaporation. Calm, still "
             "weeks are when marine heatwaves build up."),
    dict(match="Cloud_Fraction", label="Cloud fraction", unit="%", scale=100.0,
         source="MODIS-Aqua atmosphere, monthly 1° (MYD08_M3 'Cloud_Fraction_Day_Mean')",
         definition="The share of the sky covered by cloud during the daytime MODIS overpass, in percent.",
         why="Clouds shade the sea. Clear, sunny summer weeks let the water heat up and give corals extra light "
             "stress, which makes bleaching worse."),
    dict(match="precipitation", label="Precipitation", unit="mm/day",
         source="GPM IMERG Final Run v07, monthly 0.1° (GPM_3IMERGM 'precipitation', downloaded in mm/day)",
         definition="Rain falling on the box, in millimetres per day (averaged over the month).",
         why="Heavy wet-season rain floods rivers that carry mud, fertiliser and fresh water onto inshore reefs; "
             "low rainfall often comes with clear, hot bleaching conditions."),
]


# ---------------------------------------------------------------- loading
def read_giovanni(path):
    """Giovanni area-averaged CSV -> (raw Series on its own time stamps, meta dict, column name)."""
    lines = Path(path).read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    hdr = next(i for i, ln in enumerate(lines) if ln.split(",")[0].strip().strip('"').lower() in ("time", "date"))
    meta = {}
    for ln in lines[:hdr]:
        k, _, v = ln.partition(",")
        meta[k.strip().rstrip(":")] = v.strip().strip('"').strip()
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])))
    df.columns = [str(c).strip() for c in df.columns]
    tcol = df.columns[0]
    col = next(c for c in df.columns if c != tcol)
    v = pd.to_numeric(df[col], errors="coerce")
    fill = next((float(x) for k, x in meta.items() if k.startswith("Fill Value")), None)
    if fill is not None:
        v = v.mask(np.isclose(v, fill))
    v = v.mask(v.abs() > 1e14)                                # 1e15-style fills
    s = pd.Series(v.to_numpy(), index=pd.to_datetime(df[tcol], errors="coerce"))
    s = s[~s.index.isna()].dropna().sort_index()
    return s, meta, col


def to_monthly(s):
    step = s.index.to_series().diff().median()
    if step <= pd.Timedelta(days=2):                          # daily -> monthly, need 80% of days
        g = s.resample("MS").agg(["mean", "count"])
        m = g["mean"].where(g["count"] >= 0.8 * g.index.days_in_month)
        return m.dropna(), "daily → monthly"
    s = s.copy()                                              # mid-month stamps -> that month
    s.index = s.index.to_period("M").to_timestamp()
    return s.groupby(level=0).mean().dropna(), "monthly"


def parse_box(txt):
    try:
        w, s, e, n = [float(x) for x in str(txt).split(",")]
        return [w, s, e, n]
    except Exception:                                         # noqa
        return None


def find_var(col):
    for v in VARS:
        if v["match"] in col:
            return v
    short = col.replace("mean_", "")
    return dict(match=short, label=short, unit="raw units", source=f"Giovanni '{short}'",
                definition="A Giovanni variable that is not in this app's list, analysed in its original units.",
                why="Check the units and meaning on the Giovanni page for this variable.")


# ---------------------------------------------------------------- statistics
def to_arrays(s):
    """Monthly series -> (decimal year, anomaly, year). Anomaly = value minus that calendar month's mean."""
    clim = s.groupby(s.index.month).mean()
    anom = s.to_numpy() - clim.reindex(s.index.month).to_numpy()
    yr = s.index.year.to_numpy()
    return yr + (s.index.month.to_numpy() - 0.5) / 12, anom, yr


def _z(S, var):
    return 0.0 if (var <= 0 or S == 0) else (S - np.sign(S)) / math.sqrt(var)


def _sen_ci(slopes, varS):
    N = len(slopes)
    C = stats.norm.ppf(1 - ALPHA / 2) * math.sqrt(max(varS, 0))
    srt = np.sort(slopes)
    return srt[max(int(np.floor((N - C) / 2)) - 1, 0)], srt[min(int(np.ceil((N + C) / 2)), N - 1)]


def hamed_rao_factor(x, t, sen):
    n = len(x)
    r = stats.rankdata(x - sen * t)
    r = r - r.mean()
    d = (r ** 2).sum()
    if d == 0 or n < 4:
        return 1.0
    k = np.arange(1, n)
    acf = np.array([(r[:-kk] * r[kk:]).sum() / d for kk in k])
    sig = np.abs(acf) > stats.norm.ppf(1 - ALPHA / 2) / math.sqrt(n)
    w = (n - k) * (n - k - 1) * (n - k - 2)
    return max(1 + 2.0 / (n * (n - 1) * (n - 2)) * np.sum(w[sig] * acf[sig]), 1.0)


def mk_test(x, t, autocorr=False):
    """Mann-Kendall (tie-corrected) + Sen's slope per unit t; optional Hamed-Rao correction."""
    x, t = np.asarray(x, float), np.asarray(t, float)
    n = len(x)
    iu = np.triu_indices(n, 1)
    S = np.sign(x[None, :] - x[:, None])[iu].sum()
    _, c = np.unique(x, return_counts=True)
    varS = (n * (n - 1) * (2 * n + 5) - np.sum(c * (c - 1) * (2 * c + 5))) / 18.0
    slopes = (x[iu[1]] - x[iu[0]]) / (t[iu[1]] - t[iu[0]])
    sen = float(np.median(slopes))
    f = hamed_rao_factor(x, t, sen) if autocorr else 1.0
    z = _z(S, varS * f)
    lo, hi = _sen_ci(slopes, varS * f)
    return dict(n=n, tau=S / (n * (n - 1) / 2), z=z, p=2 * stats.norm.sf(abs(z)), slope=sen, lo=lo, hi=hi)


def seasonal_mk(M, t):
    """Seasonal Mann-Kendall (Hirsch & Slack 1984) incl. covariance between months."""
    n, g = M.shape
    iu = np.triu_indices(n, 1)
    sgn = np.sign(M[None, :, :] - M[:, None, :])[iu]
    S = sgn.sum()
    K = sgn.T @ sgn
    R = np.column_stack([stats.rankdata(M[:, k]) for k in range(g)])
    varS = ((K + 4 * R.T @ R - n * (n + 1) ** 2) / 3.0).sum()
    slopes = ((M[iu[1], :] - M[iu[0], :]) / (t[iu[1]] - t[iu[0]]).astype(float)[:, None]).ravel()
    z = _z(S, varS)
    lo, hi = _sen_ci(slopes, varS)
    return dict(n=n, tau=S / (g * n * (n - 1) / 2), z=z, p=2 * stats.norm.sf(abs(z)),
                slope=float(np.median(slopes)), lo=lo, hi=hi)


def pettitt(x):
    x = np.asarray(x, float)
    n = len(x)
    U = np.array([np.sign(x[: k + 1, None] - x[None, k + 1:]).sum() for k in range(n - 1)])
    K = np.abs(U).max()
    return int(np.abs(U).argmax()) + 1, min(1.0, 2 * math.exp(-6 * K ** 2 / (n ** 3 + n ** 2)))


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    adj = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def window_stats(t, a, yr, y0, y1):
    """Stats for years y0..y1, or None.
    -> [slope/yr, ci_lo, ci_hi, kendall_tau, mk_p, n_years, intercept, n_eff, t_test_p, sen_slope/yr]"""
    m = (yr >= y0) & (yr <= y1)
    t_, a_, y_ = t[m], a[m], yr[m]
    yrs = [y for y in np.unique(y_) if (y_ == y).sum() >= 6]
    if len(yrs) < MIN_YEARS or m.sum() < 36:
        return None
    ann = np.array([a_[y_ == y].mean() for y in yrs])
    fit = stats.linregress(t_, a_)
    n = len(a_)
    resid = a_ - (fit.intercept + fit.slope * t_)
    r1 = float(np.clip(np.corrcoef(resid[:-1], resid[1:])[0, 1], 0, 0.95))   # month-to-month autocorrelation
    neff = max(4.0, n * (1 - r1) / (1 + r1))                                   # "independent" points
    se = fit.stderr * math.sqrt((n - 2) / (neff - 2))
    tc = stats.t.ppf(0.975, neff - 2)
    tp = 2 * stats.t.sf(abs(fit.slope / se), neff - 2) if se > 0 else 1.0
    mk = mk_test(ann, np.array(yrs, float))
    g = lambda x: float(f"{x:.6g}")
    return [g(fit.slope), g(fit.slope - tc * se), g(fit.slope + tc * se), g(mk["tau"]), g(mk["p"]),
            len(yrs), g(fit.intercept), round(float(neff), 1), g(tp), g(mk["slope"])]


def deep_checks(s, months, key_season):
    """Full-record checks. s = monthly Series. Returns dict or None if < RELIABLE_YEARS complete years."""
    allowed = list(months or range(1, 13))
    d = s.to_frame("v")
    d["yr"], d["mo"] = d.index.year, d.index.month
    piv = d.pivot(index="yr", columns="mo", values="v").reindex(columns=range(1, 13))
    wide = piv[allowed].dropna()
    if len(wide) < RELIABLE_YEARS:
        return None
    out = {}
    smk = seasonal_mk(wide.to_numpy(), wide.index.to_numpy())
    out["smk"] = dict(slope=smk["slope"], lo=smk["lo"], hi=smk["hi"], p=smk["p"], tau=smk["tau"], n=smk["n"],
                      months="all 12 months" if len(allowed) == 12 else f"{MONTHS[allowed[0]-1]}–{MONTHS[allowed[-1]-1]}")
    # key season (Dec counts toward the next year when the season wraps through January)
    name, sm = key_season
    if set(sm) <= set(allowed):
        x = d[d.mo.isin(sm)].copy()
        x["sy"] = x.yr + ((x.mo == 12) & (1 in sm)).astype(int)
        gs = x.groupby("sy").v.agg(["mean", "count"])
        ser = gs.loc[gs["count"] == len(sm), "mean"]
        if len(ser) >= RELIABLE_YEARS:
            r = mk_test(ser.to_numpy(), ser.index.to_numpy(float), autocorr=True)
            ey = int((ser - ser.median()).abs().idxmax())
            s2 = ser.drop(ey)
            r2 = mk_test(s2.to_numpy(), s2.index.to_numpy(float), autocorr=True)
            out["season"] = dict(name=name, slope=r["slope"], lo=r["lo"], hi=r["hi"], p=r["p"], n=r["n"],
                                 extreme_year=ey, p_without=r2["p"], slope_without=r2["slope"],
                                 years=[int(y) for y in ser.index], values=[float(v) for v in ser])
    # Pettitt on yearly means of complete years
    ann = wide.mean(axis=1)
    k, pp = pettitt(ann.to_numpy())
    out["pettitt"] = dict(year=int(ann.index[k]), p=pp, shift=float(ann.iloc[k:].mean() - ann.iloc[:k].mean()))
    # month-by-month
    mz, mp = [], []
    for mo in range(1, 13):
        col = piv[mo].dropna()
        if mo not in allowed or len(col) < RELIABLE_YEARS:
            mz.append(None); mp.append(None); continue
        r = mk_test(col.to_numpy(), col.index.to_numpy(float))
        mz.append(r["z"]); mp.append(r["p"])
    out["months_z"], out["months_p"] = mz, mp
    return out


def heat_stress(s):
    """SST heat-stress proxy. MMM = warmest climatological month. Per summer (Dec-Apr, Dec->next year):
    sum of (SST - MMM) over months above MMM, in degC-months."""
    clim = s.groupby(s.index.month).mean()
    mmm = float(clim.max())
    d = s.to_frame("v")
    d = d[d.index.month.isin((12, 1, 2, 3, 4))].copy()
    d["sy"] = d.index.year + (d.index.month == 12).astype(int)
    d["hot"] = (d.v - mmm).clip(lower=0)
    g = d.groupby("sy").agg(stress=("hot", "sum"), n=("v", "count"))
    g = g[g.n == 5]
    if len(g) < RELIABLE_YEARS:
        return None
    r = mk_test(g.stress.to_numpy(), g.index.to_numpy(float), autocorr=True)
    return dict(mmm=mmm, years=[int(y) for y in g.index], values=[round(float(v), 3) for v in g.stress],
                slope=r["slope"], p=r["p"], worst=[int(y) for y in g.stress.nlargest(3).index])


def _clean(o):
    """numpy -> plain python, NaN -> None, so the payload is JSON-safe."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if not math.isfinite(f) else float(f"{f:.6g}")
    return o


# ---------------------------------------------------------------- payload for the front end
def build_region(rg):
    folder = DATA_DIR / rg["id"]
    out, warnings = [], []
    files = sorted(folder.glob("*.csv")) if folder.exists() else []
    if not files:
        warnings.append(f"{rg['title']}: no CSV files found in data/{rg['id']}/")
    S, N, W, E = rg["bbox"]
    for f in files:
        try:
            raw, meta, col = read_giovanni(f)
            spec = find_var(col)
            s, cadence = to_monthly(raw)
            s = s * spec.get("scale", 1.0) + spec.get("offset", 0.0)
            if len(s) < 36:
                raise ValueError(f"only {len(s)} monthly values")
        except Exception as e:                                 # noqa
            warnings.append(f"{rg['title']}: could not use {f.name} ({e}). Skipped.")
            continue

        t, a, yr = to_arrays(s)
        y_first, y_last = int(yr.min()), int(yr.max())
        table = {}
        for y0 in range(y_first, y_last - MIN_YEARS + 2):
            for y1 in range(y0 + MIN_YEARS - 1, y_last + 1):
                r = window_stats(t, a, yr, y0, y1)
                if r:
                    table[f"{y0}-{y1}"] = r
        full = f"{y_first}-{y_last}"
        n_good = sum(1 for y in np.unique(yr) if (yr == y).sum() >= 6)

        # ---- data-quality notes
        notes = []
        ub, db = parse_box(meta.get("User Bounding Box")), parse_box(meta.get("Data Bounding Box"))
        if ub and ub[2] - ub[0] >= 300:
            notes.append("The download box spans every longitude (a ring around the planet), so this average "
                         "includes oceans and non-forest land.")
        if ub and db:
            ua, da = (ub[2] - ub[0]) * (ub[3] - ub[1]), (db[2] - db[0]) * (db[3] - db[1])
            if ua > 0 and da / ua < 0.6:
                notes.append(f"The product only covers {100 * da / ua:.0f}% of the requested box "
                             f"(data box {db[0]:g}…{db[2]:g}°E, {db[1]:g}…{db[3]:g}°N).")
        if ub and (abs(ub[1] - S) > 1 or abs(ub[3] - N) > 1 or abs(ub[0] - W) > 1 or abs(ub[2] - E) > 1):
            notes.append(f"Downloaded with a different box ({ub[1]:g}–{ub[3]:g}°N, {ub[0]:g}–{ub[2]:g}°E) "
                         "from the rest of this region, so it is not strictly comparable.")
        if n_good < RELIABLE_YEARS:
            notes.append(f"Short record: only {n_good} years ({s.index.min():%b %Y} – {s.index.max():%b %Y}). "
                         f"Fewer than {RELIABLE_YEARS} years is too short for a reliable climate trend, so treat "
                         "any result as indicative only.")
        if spec.get("caveat"):
            notes.append(spec["caveat"])

        deep = deep_checks(s, spec.get("months"), rg["key_season"])
        heat = heat_stress(s) if spec.get("heat_stress") else None
        out.append(dict(
            key=spec["match"].strip("_").lower(), label=spec["label"], unit=spec["unit"],
            definition=spec["definition"], why=spec["why"], source=spec["source"],
            file=f"data/{rg['id']}/{f.name}", cadence=cadence, years=[y_first, y_last], full=full,
            span=f"{s.index.min():%b %Y} – {s.index.max():%b %Y}", n_years=n_good,
            short=n_good < RELIABLE_YEARS, notes=notes,
            dates=[d.strftime("%Y-%m") for d in s.index],
            raw=[round(float(x), 5) for x in s], anom=[round(float(x), 5) for x in a],
            stats=table, deep=deep, heat=heat))

    # FDR across the region's full-record tests (seasonal MK + key season + heat stress)
    ps, where = [], []
    for i, v in enumerate(out):
        if v["deep"]:
            ps.append(v["deep"]["smk"]["p"]); where.append((i, "smk"))
            if "season" in v["deep"]:
                ps.append(v["deep"]["season"]["p"]); where.append((i, "season"))
        if v["heat"]:
            ps.append(v["heat"]["p"]); where.append((i, "heat"))
    for (i, k), q in zip(where, bh_fdr(ps)):
        tgt = out[i]["heat"] if k == "heat" else out[i]["deep"][k]
        tgt["p_fdr"] = float(q)
    out.sort(key=lambda v: v["short"])
    y0 = min((v["years"][0] for v in out), default=2000)
    y1 = max((v["years"][1] for v in out), default=2026)
    reg = dict(id=rg["id"], name=rg["name"], title=rg["title"], bbox=rg["bbox"], focus=rg["focus"],
               intro=rg["intro"], key_season=rg["key_season"][0], years=[y0, y1], variables=out)
    return _clean(reg), warnings


def build_payload():
    regions, warnings = [], []
    for rg in REGIONS:
        r, w = build_region(rg)
        warnings += w
        if r["variables"]:
            regions.append(r)
    return dict(regions=regions, min_years=MIN_YEARS, reliable_years=RELIABLE_YEARS), warnings


def data_signature():
    if not DATA_DIR.exists():
        return ()
    return tuple((str(p.relative_to(DATA_DIR)), p.stat().st_mtime) for p in sorted(DATA_DIR.glob("*/*.csv")))


if __name__ == "__main__":
    import json, time
    t0 = time.time()
    payload, warns = build_payload()
    print(*warns, sep="\n")
    for rg in payload["regions"]:
        print(f"\n=== {rg['title']}  ({rg['years'][0]}-{rg['years'][1]})")
        for v in rg["variables"]:
            r = v["stats"].get(v["full"])
            d = v["deep"]
            extra = (f" | SMK {d['smk']['slope']*10:+.3g}/dec p_fdr={d['smk'].get('p_fdr', float('nan')):.3g}"
                     f" | Pettitt {d['pettitt']['year']} p={d['pettitt']['p']:.3g}") if d else " | (short: no deep checks)"
            print(f"{v['label']:32s} {v['span']:20s} slope={r[0]:+.4g} {v['unit']}/yr "
                  f"CI=({r[1]:+.3g},{r[2]:+.3g}) t_p={r[8]:.3g} MK_p={r[4]:.3g} tau={r[3]:+.2f}{extra}")
            for n in v["notes"]:
                print(f"     ! {n[:120]}")
    print(f"\npayload {len(json.dumps(payload)) / 1e6:.2f} MB, built in {time.time() - t0:.1f}s")
