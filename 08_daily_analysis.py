"""
STEP 8 - DAILY ANALYSIS (robustness check of the weekly results)
Input : Final_Daily_Analysis_Table.xlsx (made by 07_build_daily_table.py)
Output: tables_daily/*.csv and figures_daily/*.png   (weekly outputs in tables/ and figures/ are NOT touched)
Same steps as the weekly 02_analysis.py: descriptives, normality, stationarity, correlation, OLS, sub-periods,
rolling correlation, and out-of-sample forecasting (train 2018-2023, test 2024 -> Sep 2026).
Run:  python 08_daily_analysis.py
"""
import os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
warnings.filterwarnings('ignore')
T_DIR, F_DIR = 'tables_daily/', 'figures_daily/'
os.makedirs(T_DIR, exist_ok=True); os.makedirs(F_DIR, exist_ok=True)

full = pd.read_excel('Final_Daily_Analysis_Table.xlsx', sheet_name='Analysis table', index_col='Date')
X_COLS = ['SP500', 'FTSE100', 'Gold', 'US10Y_change']; Y = 'Crypto_Index'
df = full[[Y] + X_COLS]
print(f'{len(df)} common trading days, {df.index.min().date()} to {df.index.max().date()}\n')

# ---------------------------------------------------------------- 1. DESCRIPTIVES + NORMALITY
desc = df.describe().T
desc['skew'] = df.skew(); desc['excess_kurtosis'] = df.kurt()
desc['shapiro_p'] = [stats.shapiro(df[c])[1] for c in df]
desc['jarque_bera_p'] = [stats.jarque_bera(df[c])[1] for c in df]
desc.round(4).to_csv(T_DIR + '1_descriptives.csv'); print(desc.round(4), '\n')

# ---------------------------------------------------------------- 2. STATIONARITY
adf = pd.DataFrame({c: dict(zip(['ADF_stat', 'p_value'], adfuller(df[c], autolag='AIC')[:2])) for c in df}).T
adf.round(4).to_csv(T_DIR + '2_adf.csv'); print('ADF (p<0.05 = stationary)\n', adf.round(4), '\n')

# ---------------------------------------------------------------- 3. CORRELATION
pear, spear = df.corr('pearson'), df.corr('spearman')
pear.round(3).to_csv(T_DIR + '3_pearson.csv'); spear.round(3).to_csv(T_DIR + '3_spearman.csv')
cp = pd.DataFrame({c: dict(zip(['pearson_r', 'p_value'], stats.pearsonr(df[Y], df[c]))) for c in X_COLS}).T
cp['spearman_rho'] = [stats.spearmanr(df[Y], df[c])[0] for c in X_COLS]
cp.round(4).to_csv(T_DIR + '3_corr_with_index.csv')
print('Pearson\n', pear.round(3), '\nCorrelation of the crypto index with each market\n', cp.round(4), '\n')
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(pear, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(pear))); ax.set_xticklabels(pear.columns, rotation=45, ha='right')
ax.set_yticks(range(len(pear))); ax.set_yticklabels(pear.columns)
for i in range(len(pear)):
    for j in range(len(pear)): ax.text(j, i, f'{pear.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8)
fig.colorbar(im); ax.set_title('Pearson correlation of daily returns'); fig.tight_layout()
fig.savefig(F_DIR + '3_corr_heatmap.png', dpi=200); plt.close()

# ---------------------------------------------------------------- 4. REGRESSION
HAC = 5
def ols(data, label, quiet=False):
    Xm = sm.add_constant(data[X_COLS]); m = sm.OLS(data[Y], Xm).fit()
    mh = sm.OLS(data[Y], Xm).fit(cov_type='HAC', cov_kwds={'maxlags': HAC})
    if not quiet:
        print(f'--- OLS: {label} (n={len(data)}) ---\n', m.summary().tables[1], f'\nR2={m.rsquared:.4f} adjR2={m.rsquared_adj:.4f} F-p={m.f_pvalue:.4g}')
        print('HAC robust p-values:\n', mh.pvalues.round(4).to_string(), '\n')
    return m, mh
m_all, mh_all = ols(df, 'full sample')
vif = pd.Series([variance_inflation_factor(sm.add_constant(df[X_COLS]).values, i + 1) for i in range(len(X_COLS))], index=X_COLS)
bp = het_breuschpagan(m_all.resid, m_all.model.exog)
print('VIF (>5 = concern):\n', vif.round(2).to_string(), f'\nBreusch-Pagan p={bp[1]:.4f}  Durbin-Watson={durbin_watson(m_all.resid):.3f}\n')
pd.DataFrame({'coef': m_all.params, 'se': m_all.bse, 'p_OLS': m_all.pvalues, 'p_HAC': mh_all.pvalues}).round(4).to_csv(T_DIR + '4_regression_full.csv')
pd.Series({'R2': m_all.rsquared, 'adjR2': m_all.rsquared_adj, 'n': len(df)}).round(4).to_csv(T_DIR + '4_regression_fit.csv')
vif.round(2).to_csv(T_DIR + '4_vif.csv')
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sm.qqplot(m_all.resid, line='s', ax=ax[0]); ax[0].set_title('Residual Q-Q plot')
ax[1].scatter(m_all.fittedvalues, m_all.resid, s=5); ax[1].axhline(0, c='k'); ax[1].set_title('Residuals vs fitted')
fig.tight_layout(); fig.savefig(F_DIR + '4_residuals.png', dpi=200); plt.close()

SPLIT = '2020-01-01'
res = {}
for lab, d in [('2018-2019', df[df.index < SPLIT]), ('2020-2026', df[df.index >= SPLIT])]:
    m, mh = ols(d, lab); res[lab] = pd.DataFrame({'coef': m.params, 'p_HAC': mh.pvalues, 'R2': m.rsquared})
pd.concat(res, axis=1).round(4).to_csv(T_DIR + '4_regression_subperiods.csv')

# sensitivity: only ordinary one-day intervals (drops Mondays after weekends and holiday-spanning returns)
d1 = full[full['Gap_days'] == 1][[Y] + X_COLS]
m1, mh1 = ols(d1, 'one-day intervals only')
pd.DataFrame({'coef': m1.params, 'p_HAC': mh1.pvalues, 'R2': m1.rsquared, 'n': len(d1)}).round(4).to_csv(T_DIR + '4_regression_1day_only.csv')

roll = df[Y].rolling(250).corr(df['SP500'])
fig, ax = plt.subplots(figsize=(9, 3.5)); ax.plot(roll); ax.axhline(0, c='k', lw=.5)
ax.set_title('Rolling 250-trading-day correlation: crypto index vs S&P 500'); fig.tight_layout()
fig.savefig(F_DIR + '4_rolling_corr.png', dpi=200); plt.close()

# ---------------------------------------------------------------- 5. FORECASTING
TRAIN_END = '2023-12-31'
train, test = df[df.index <= TRAIN_END], df[df.index > TRAIN_END]
print(f'Train {len(train)} days, test {len(test)} days\n')
# trading days are not evenly spaced (holidays), so the time series models use a plain 0,1,2,... index
def plain(d, start=0):
    d = d.copy(); d.index = range(start, start + len(d)); return d
ytr, yte = plain(train[Y]), plain(test[Y], len(train))
xtr, xte = plain(train[X_COLS]), plain(test[X_COLS], len(train))
best = min(((ARIMA(ytr, order=(p, 0, q)).fit().aic, (p, 0, q)) for p in range(4) for q in range(4)))
order = best[1]; print('Best ARIMA order by AIC (train only):', order, '\n')

def rolling_forecast(fit, ytest, EX=None):
    """1-step-ahead: parameters fixed from TRAIN, each new day appended after it is forecast."""
    preds = []
    for t in range(len(ytest)):
        ex_t = EX.iloc[[t]] if EX is not None else None
        preds.append(np.asarray(fit.forecast(1, exog=ex_t))[0])
        fit = fit.append(ytest.iloc[[t]], exog=ex_t, refit=False)
    return np.array(preds)

y = test[Y].values; P = {}
P['Benchmark: train mean'] = np.full(len(y), train[Y].mean()); P['Benchmark: zero'] = np.zeros(len(y))
P['A: ARIMA'] = rolling_forecast(ARIMA(ytr, order=order).fit(), yte)
# Model B: lagged traditional returns (a genuine forecast: the previous trading day's values are known)
lag = plain(df[X_COLS].shift(1).add_suffix('_lag1'))
ltr, lte = lag.iloc[1:len(train)], lag.iloc[len(train):]                    # first row has no lag -> dropped from training
P['B: ARIMAX (lagged)'] = rolling_forecast(ARIMA(ytr.iloc[1:], exog=ltr, order=order).fit(), yte, lte)
# Model C: SAME-DAY returns - NOT a true forecast (conditional / explanatory only)
P['C: ARIMAX (same day, conditional)'] = rolling_forecast(ARIMA(ytr, exog=xtr, order=order).fit(), yte, xte)

rows = []
for k, p in P.items():
    e = y - p; rows.append({'model': k, 'RMSE': np.sqrt((e ** 2).mean()), 'MAE': np.abs(e).mean()})
acc = pd.DataFrame(rows).set_index('model'); acc['RMSE_vs_mean_%'] = (acc.RMSE / acc.loc['Benchmark: train mean', 'RMSE'] - 1) * 100
acc.round(5).to_csv(T_DIR + '5_forecast_accuracy.csv'); print(acc.round(5), '\n')

def dm(e1, e2):
    return stats.ttest_1samp(e1 ** 2 - e2 ** 2, 0)
dmrows = []
for k in ['A: ARIMA', 'B: ARIMAX (lagged)']:
    s = dm(y - P[k], y - P['Benchmark: train mean']); dmrows.append({'model': k, 't': s.statistic, 'p': s.pvalue})
    print(f'DM-type test {k} vs mean benchmark: t={s.statistic:.3f}, p={s.pvalue:.3f}')
pd.DataFrame(dmrows).round(4).to_csv(T_DIR + '5_dm_tests.csv', index=False)

fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(test.index, y, c='lightgrey', lw=.8, label='Actual')
for k in ['A: ARIMA', 'B: ARIMAX (lagged)', 'C: ARIMAX (same day, conditional)']: ax.plot(test.index, P[k], lw=1, label=k)
ax.legend(fontsize=7); ax.set_title('Test period: actual vs forecast daily index return'); fig.tight_layout()
fig.savefig(F_DIR + '5_forecast.png', dpi=200); plt.close()

da = {k: (np.sign(P[k]) == np.sign(y)).mean() for k in ['A: ARIMA', 'B: ARIMAX (lagged)', 'C: ARIMAX (same day, conditional)']}
for k, v in da.items(): print(f'Directional accuracy {k}: {v:.1%}')
pd.Series(da, name='directional_accuracy').round(4).to_csv(T_DIR + '5_directional_accuracy.csv')
print('\nDone. See figures_daily/ and tables_daily/')
