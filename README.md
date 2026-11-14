# Third Pole Trend Detective — NASA Space Apps Challenge 2026

An interactive Earth-science trend analysis tool built for students, educators, and the public to investigate **what is changing, where, how fast, and whether the signal is statistically significant** across the Hindu Kush–Himalaya–Tibet ("Third Pole").

## Why This Matters
Global averages hide regional contrasts. Across the Third Pole, high-altitude warming and seasonal snow decline are strongly positive/negative in the Central and Eastern Himalaya, whereas the **Karakoram** exhibits near-stable snow/glacier behavior and **total precipitation** is dominated by large interannual monsoon variability. Our tool prevents false alarms and cherry-picking by separating seasonal cycles and natural noise from statistically significant trends.

## Methodology
1. **Deseasonalization:** Subtracts each calendar month's multi-year climatology (`anomaly = value - monthly_mean`) so seasonal winter/summer swings do not mask long-term trends.
2. **Autocorrelation-Safe Linear Regression:** Estimates the rate of change per decade and 95% confidence intervals using Ordinary Least Squares with **Newey-West HAC standard errors** (`maxlags=12`), preventing monthly persistence from inflating significance.
3. **Non-Parametric Cross-Check:** Runs the **Mann-Kendall trend test** (`p < 0.05`) to verify robustness against outliers.
4. **Honest Uncertainty:** Explicitly distinguishes *"no statistically significant linear trend detected"* from *"nothing is changing"* and allows sub-period slicing (e.g., 2010–present) to inspect recent acceleration.

## NASA Data Sources Credited
* **NASA GES DISC / Giovanni:** MERRA-2 Monthly Single-Level Diagnostics (`M2TMNXSLV` — 2m Air Temperature `T2M` & Specific Humidity `QV2M`) and GPM IMERG Final Monthly Precipitation (`GPM_3IMERGM`).
* **NASA LP DAAC & NSIDC / AppEEARS:** MODIS/Terra Monthly Snow Cover (`MOD10CM` v6.1) and Monthly Vegetation Indices (`MOD13C2` v6.1).

## Quickstart
```bash
pip install streamlit pandas numpy scipy statsmodels plotly pymannkendall
streamlit run app.py