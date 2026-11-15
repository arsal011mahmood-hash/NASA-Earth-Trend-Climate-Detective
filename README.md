EARTH SCIENCE TREND DETECTIVE
=============================

An interactive toolkit for trend analysis of NASA GIOVANNI area-averaged
monthly time series. Ships a Streamlit dashboard, a 3-D Globe Explorer,
and two command-line batch tools that all share the same statistical
engine.

Real data only. No synthetic series. No demo mode.


WHAT IT DOES
------------

- Loads real NASA GIOVANNI CSVs (MERRA-2, AIRS, OCO-2 GEOS, NLDAS)
- Detects the region from each file's bounding-box metadata
- Runs Seasonal Mann-Kendall, Sen's slope with 95% CI, Hamed-Rao
  autocorrelation correction, Pettitt step-change, drop-extreme-year
  robustness, and Benjamini-Hochberg FDR
- Renders anomaly-trend plots, monthly MK heatmaps, and seasonal panels
- Summarises each result in one plain-English sentence


THREE WAYS TO USE IT
--------------------

1. Dashboard      streamlit run app.py
                  Point-and-click trend explorer, one tab per CSV.

2. Globe Explorer Inside the dashboard. A rotating 3-D Earth with four
                  clickable sectors (Northern Band, Third Pole, Boreal
                  Canada, Boreal Russia). Click a sector to open a
                  regional map plus a variable-impact panel showing
                  Kendall tau, Spearman rho, Pearson r, lagged
                  correlations, and standardised beta against the
                  sector's master trend.

3. CLI            python boreal_fire_trends.py data/canada --out out/
                  python third_pole_trends.py data/third_pole --out out/
                  Batch runs. Writes trend_results.csv,
                  monthly_trends.csv, and three PNGs per region.


DATA
----

data/aerosol.csv              AODANA    Northern band 48-70 N
data/black carbon.csv         BCCMASS   Northern band
data/co2.csv                  XCO2      Northern band (2015-2022)
data/methane.csv              CH4_VMR   Northern band (2002-2014)
data/glaciated_snowcover.csv  ASNOW_GL  Third Pole 65-105 E / 25-50 N

Every CSV carries a "URL to Reproduce Results" line in its header.
Paste that URL into a browser to regenerate the exact series from
NASA GIOVANNI.


INSTALL
-------

    pip install -r requirements.txt

requirements.txt:
    streamlit>=1.30
    pandas>=2.0
    numpy>=1.24
    scipy>=1.10
    matplotlib>=3.7
    plotly>=5.20
    requests>=2.31


QUICKSTART
----------

    git clone https://github.com/<your-org>/earth-science-trend-detective
    cd earth-science-trend-detective
    pip install -r requirements.txt
    streamlit run app.py

Open http://localhost:8501


STATISTICS
----------

    Mann-Kendall              non-parametric monotonic trend test
    Seasonal Mann-Kendall     Hirsch-Slack variant across months
    Sen's slope               median of all pairwise slopes
    Hamed-Rao correction      variance inflation for autocorrelation
    Pettitt test              single change-point detection
    Benjamini-Hochberg FDR    multiple-comparison correction
    Drop-extreme-year         re-tests without the largest year

Default alpha = 0.05. All formulas follow the original papers
(Mann 1945, Kendall 1975, Hirsch & Slack 1984, Sen 1968,
Hamed & Rao 1998, Pettitt 1979, Benjamini & Hochberg 1995).


REPOSITORY LAYOUT
-----------------

    app.py                     Streamlit dashboard (main entry)
    globe_explorer.py          Globe -> map -> variable-impact panel
    boreal_fire_trends.py      CLI for boreal fire regions
    third_pole_trends.py       CLI for High Mountain Asia
    DEFINITIONS.md             Glossary of variables and methods
    MASTER_PROMPT.md           One-shot spec for regeneration
    README.txt                 This file
    data/                      Real Giovanni CSVs
    assets/world.geojson       Cached coastline


CONFIGURATION
-------------

Edit the constants at the top of each script:

    ALPHA         = 0.05              significance threshold
    MIN_YEARS     = 6 (app), 8 (CLI)  minimum complete years
    FIRE_MONTHS   = (5, 6, 7, 8, 9)   boreal fire season
    MELT_MONTHS   = (5, 6, 7, 8, 9)   Third Pole melt season
    EXPECTED_BOX  = { ... }           per-region bounding boxes
    VARIABLES     = { ... }           label, unit, scale, offset

Add new sectors in globe_explorer.py under SECTORS.


ACCEPTANCE TESTS
----------------

    [x] app.py loads all five CSVs, including the truncated CO2 and
        CH4 files, without crashing.
    [x] Overview table shows five rows: four "Northern band", one
        "Third Pole".
    [x] Every file with >= 6 complete years shows anomaly + trend,
        monthly heatmap, seasonal panels, and a plain-English
        interpretation.
    [x] FDR p-values computed once across all files.
    [x] Pettitt step changes reported where p < 0.05.
    [x] Drop-extreme-year note appears on every MJJAS panel.
    [x] Globe auto-rotates at idle and drills into a flat regional
        map on click.
    [x] Variable-impact panel ranks drivers by |tau| and expands
        into scatter + correlation + lag + trend + beta.
    [x] third_pole_trends.py reproduces the exact Sen slope and FDR
        p-value that the app shows for ASNOW_GL.
    [x] Both CLIs produce numerically identical statistics on the
        same input.


OUTPUTS (CLI)
-------------

    outputs/
        trend_results.csv       every test x every variable x region
        monthly_trends.csv      month-by-month MK z, p, FDR p
        <region>/
            anomaly_trends.png
            fire_season_by_year.png    (or melt_season_by_year.png)
            monthly_trend_heatmap.png


CONTRIBUTING
------------

    Fork, branch, commit, push, open a Pull Request.
    Please ensure black and ruff are clean and pytest passes.


CITATION
--------

    @software{earth_science_trend_detective,
      title  = {Earth Science Trend Detective},
      author = {<Your Name>},
      year   = {2026},
      url    = {https://github.com/<your-org>/earth-science-trend-detective}
    }

Also cite the relevant NASA GIOVANNI product and the statistical
papers listed in DEFINITIONS.md.


ACKNOWLEDGEMENTS
----------------

    Data          NASA GIOVANNI / GES DISC
                  MERRA-2, AIRS, OCO-2 GEOS, NLDAS

    Statistics    Mann 1945, Kendall 1975, Hirsch & Slack 1984,
                  Sen 1968, Hamed & Rao 1998, Pettitt 1979,
                  Benjamini & Hochberg 1995

    Visualisation Streamlit, Plotly, globe.gl, Matplotlib


LICENSE
-------

MIT. See LICENSE file.
