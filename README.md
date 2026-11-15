# Earth System Trend Detective

Three regions, 19 real NASA Giovanni time series, every trend tested.

## Run the app
    pip install -r requirements.txt
    streamlit run app.py

Click a glowing region on the globe (or a button), then click a bar to open a variable.
Needs internet for the globe libraries (d3, Plotly) and satellite imagery.

## Run the stand-alone reports (PNG figures + CSV tables)
    python third_pole_depletion.py      data/third_pole_depletion  outputs/third_pole
    python boreal_fire_trends.py        data/boreal_fire_trends    --out outputs/boreal
    python great_barrier_reef_trends.py data/great_barrier_reef    outputs/great_barrier_reef

Results from the real data are already in `outputs/`.

## Files
- `app.py` - globe, region pages, per-variable charts, slider, extra checks, glossary, results table + CSV download
- `backend.py` - loads the CSVs (recognised by Giovanni variable name, so file names don't matter), converts units, computes all statistics
- `data/<region>/*.csv` - drop any new Giovanni area-averaged CSV in a region folder and it is picked up automatically

## Methods (covers the challenge objectives)
1. Variables: 6 per region, chosen for that region's story
2. Data: Giovanni area-averaged monthly series per region box
3. Time series + visualisation: raw values, deseasonalised anomalies, yearly means, trend line
4. Statistics: rate of change (least squares, autocorrelation-adjusted 95% CI and t-test),
   Mann-Kendall + Sen's slope, Seasonal Mann-Kendall, key-season test with Hamed-Rao correction,
   drop-the-extreme-year check, Pettitt change point, Benjamini-Hochberg FDR

## Known data limits (shown inside the app too)
- Boreal files cover the whole 48-70N ring (all longitudes), not only forest.
- Boreal transpiration (NLDAS) only covers a US-Canada border strip (~4% of the box).
- Boreal CO2 (2015-2022) and Third Pole surface pressure (2020-2026) are too short for reliable trends.
- Third Pole surface pressure used a different box (25-40N).
