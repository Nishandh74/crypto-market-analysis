"""
STEP 10 - SIDE-BY-SIDE TABLE: WEEKLY (main analysis) vs DAILY (robustness check)
Reads the CSV files already made by steps 2, 5 (tables/) and 8, 9 (tables_daily/). Nothing is recalculated.
Output: tables_daily/10_daily_vs_weekly.csv   -> paste into the report as the robustness table.
"""
import pandas as pd
W, D = 'tables/', 'tables_daily/'
rd = lambda f: pd.read_csv(f, index_col=0)
rows = []
P = lambda p: '<0.001' if p < 0.001 else round(p, 4)
def add(name, w, d): rows.append({'Measure': name, 'Weekly (main)': w, 'Daily (check)': d})

# sample size
import numpy as np
nw = len(pd.read_excel('Final_Weekly_Analysis_Table.xlsx', sheet_name='Analysis table'))
nd = len(pd.read_excel('Final_Daily_Analysis_Table.xlsx', sheet_name='Analysis table'))
add('Observations', nw, nd)

# correlation of the crypto index with each market
pw, pdy = rd(W + '3_pearson.csv')['Crypto_Index'], rd(D + '3_corr_with_index.csv')['pearson_r']
for k, lab in [('SP500', 'S&P 500'), ('FTSE100', 'FTSE 100'), ('Gold', 'Gold'), ('US10Y_change', 'US 10Y yield change')]:
    add(f'Correlation with {lab}', round(pw[k], 3), round(pdy[k], 3))

# regression
rw, rdy = rd(W + '4_regression_full.csv'), rd(D + '4_regression_full.csv')
fit_d = rd(D + '4_regression_fit.csv').iloc[:, 0]
import statsmodels.api as sm
wk = pd.read_excel('Final_Weekly_Analysis_Table.xlsx', sheet_name='Analysis table', index_col='Date')
X = ['SP500', 'FTSE100', 'Gold', 'US10Y_change']
r2w = sm.OLS(wk['Crypto_Index'], sm.add_constant(wk[X])).fit().rsquared
add('Regression R-squared (all four markets)', round(r2w, 3), round(fit_d['R2'], 3))
for k, lab in [('SP500', 'S&P 500'), ('FTSE100', 'FTSE 100'), ('Gold', 'Gold')]:
    add(f'  {lab}: coefficient', round(rw.loc[k, 'coef'], 3), round(rdy.loc[k, 'coef'], 3))
    add(f'  {lab}: p-value (robust HAC)', P(rw.loc[k, 'p_HAC']), P(rdy.loc[k, 'p_HAC']))

sw, sd = pd.read_csv(W + '4_regression_subperiods.csv', header=[0, 1], index_col=0), pd.read_csv(D + '4_regression_subperiods.csv', header=[0, 1], index_col=0)
for per in ['2018-2019', '2020-2026']:
    add(f'R-squared {per}', round(float(sw[(per, 'R2')].iloc[0]), 3), round(float(sd[(per, 'R2')].iloc[0]), 3))
add('S&P 500 p-value (HAC) 2020-2026', P(float(sw[('2020-2026', 'p_HAC')]['SP500'])), P(float(sd[('2020-2026', 'p_HAC')]['SP500'])))

# forecasting
fw, fd = rd(W + '5_forecast_accuracy.csv'), rd(D + '5_forecast_accuracy.csv')
for a, b, lab in [('A: ARIMA', 'A: ARIMA', 'ARIMA (own past)'), ('B: ARIMAX (lagged)', 'B: ARIMAX (lagged)', 'ARIMAX, lagged markets'),
                  ('C: ARIMAX (same week, conditional)', 'C: ARIMAX (same day, conditional)', 'ARIMAX, same-period markets (not a true forecast)')]:
    add(f'Forecast RMSE vs mean benchmark: {lab}', f"{fw.loc[a, 'RMSE_vs_mean_%']:+.2f}%", f"{fd.loc[b, 'RMSE_vs_mean_%']:+.2f}%")

# Granger
gw, gd = rd(W + '8_granger.csv'), rd(D + '9_granger.csv')
for dirn in ['SP500 -> Crypto', 'Crypto -> SP500', 'FTSE100 -> Crypto', 'Crypto -> FTSE100', 'Gold -> Crypto', 'Crypto -> Gold']:
    pc = lambda g: g.loc[dirn, [c for c in g.columns if c.startswith('p (lag')]].astype(float).min()
    add(f'Granger {dirn}: smallest p over lags', P(pc(gw)), P(pc(gd)))

# volatility
pw_, pd_ = rd(W + '8_garch_params.csv'), rd(D + '9_garch_params.csv')
add('GARCH persistence (alpha + beta)', round(pw_.loc['alpha[1]', 'coef'] + pw_.loc['beta[1]', 'coef'], 3), round(pd_.loc['alpha[1]', 'coef'] + pd_.loc['beta[1]', 'coef'], 3))

out = pd.DataFrame(rows).set_index('Measure')
out.to_csv(D + '10_daily_vs_weekly.csv'); print(out.to_string()); print('\nSaved', D + '10_daily_vs_weekly.csv')
