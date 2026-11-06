"""
STEP 2 - STATISTICAL ANALYSIS (Objective 2) AND FORECASTING (Objective 3)
Input : Final_Weekly_Analysis_Table.xlsx (sheet "Analysis table")
Output: charts (PNG) in figures/ and tables (CSV) in tables/  -> screenshot / paste these in your report.
Normality tests (Shapiro-Wilk, Q-Q) are done in SPSS, but a quick Python check is included.
Run:  python 02_analysis.py
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
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True); os.makedirs('tables', exist_ok=True)

df = pd.read_excel('Final_Weekly_Analysis_Table.xlsx', sheet_name='Analysis table', index_col='Date')
X_COLS = ['SP500', 'FTSE100', 'Gold', 'US10Y_change']
Y = 'Crypto_Index'

# ---------------------------------------------------------------- 1. DESCRIPTIVES
desc = df.describe().T
desc['skew'] = df.skew(); desc['excess_kurtosis'] = df.kurt()
desc['shapiro_p'] = [stats.shapiro(df[c])[1] for c in df]       # quick check; confirm in SPSS
desc.round(4).to_csv('tables/1_descriptives.csv'); print(desc.round(4), '\n')

# ---------------------------------------------------------------- 2. STATIONARITY (ADF)
adf = pd.DataFrame({c: dict(zip(['ADF_stat', 'p_value'], adfuller(df[c], autolag='AIC')[:2])) for c in df}).T
adf.round(4).to_csv('tables/2_adf.csv'); print('ADF (p<0.05 = stationary)\n', adf.round(4), '\n')

# ---------------------------------------------------------------- 3. CORRELATION
pear, spear = df.corr('pearson'), df.corr('spearman')
pear.round(3).to_csv('tables/3_pearson.csv'); spear.round(3).to_csv('tables/3_spearman.csv')
pvals = {c: stats.pearsonr(df[Y], df[c])[1] for c in X_COLS}
print('Pearson\n', pear.round(3), '\nSpearman\n', spear.round(3), '\nPearson p-values vs index', pvals, '\n')
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(pear, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(pear))); ax.set_xticklabels(pear.columns, rotation=45, ha='right')
ax.set_yticks(range(len(pear))); ax.set_yticklabels(pear.columns)
for i in range(len(pear)):
    for j in range(len(pear)): ax.text(j, i, f'{pear.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8)
fig.colorbar(im); ax.set_title('Pearson correlation of weekly returns'); fig.tight_layout()
fig.savefig('figures/3_corr_heatmap.png', dpi=200); plt.close()

# ---------------------------------------------------------------- 4. REGRESSION (OLS)
def ols(data, label):
    Xm = sm.add_constant(data[X_COLS]); m = sm.OLS(data[Y], Xm).fit()
    mh = sm.OLS(data[Y], Xm).fit(cov_type='HAC', cov_kwds={'maxlags': 4})      # robust (Newey-West)
    print(f'--- OLS: {label} ---\n', m.summary().tables[1], f'\nR2={m.rsquared:.4f} adjR2={m.rsquared_adj:.4f} F-p={m.f_pvalue:.4g}')
    print('HAC robust p-values:\n', mh.pvalues.round(4).to_string(), '\n')
    return m, mh
m_all, mh_all = ols(df, 'full sample')
vif = pd.Series([variance_inflation_factor(sm.add_constant(df[X_COLS]).values, i + 1) for i in range(len(X_COLS))], index=X_COLS)
bp = het_breuschpagan(m_all.resid, m_all.model.exog)
print('VIF (>5 = concern):\n', vif.round(2).to_string(), f'\nBreusch-Pagan p={bp[1]:.4f}  Durbin-Watson={durbin_watson(m_all.resid):.3f}\n')
pd.DataFrame({'coef': m_all.params, 'se': m_all.bse, 'p_OLS': m_all.pvalues, 'p_HAC': mh_all.pvalues}).round(4).to_csv('tables/4_regression_full.csv')
vif.round(2).to_csv('tables/4_vif.csv')
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sm.qqplot(m_all.resid, line='s', ax=ax[0]); ax[0].set_title('Residual Q-Q plot')
ax[1].scatter(m_all.fittedvalues, m_all.resid, s=8); ax[1].axhline(0, c='k'); ax[1].set_title('Residuals vs fitted')
fig.tight_layout(); fig.savefig('figures/4_residuals.png', dpi=200); plt.close()

# before / after 2020 (split date = your choice; justify in report)
SPLIT = '2020-01-01'
res = {}
for lab, d in [('2018-2019', df[df.index < SPLIT]), ('2020-2026', df[df.index >= SPLIT])]:
    m, mh = ols(d, lab); res[lab] = pd.DataFrame({'coef': m.params, 'p_HAC': mh.pvalues, 'R2': m.rsquared})
pd.concat(res, axis=1).round(4).to_csv('tables/4_regression_subperiods.csv')

# rolling 52-week correlation with S&P 500 (shows relationship changing over time)
roll = df[Y].rolling(52).corr(df['SP500'])
fig, ax = plt.subplots(figsize=(9, 3.5)); ax.plot(roll); ax.axhline(0, c='k', lw=.5)
ax.set_title('Rolling 52-week correlation: crypto index vs S&P 500'); fig.tight_layout()
fig.savefig('figures/4_rolling_corr.png', dpi=200); plt.close()

# ---------------------------------------------------------------- 5. FORECASTING
TRAIN_END = '2023-12-31'
train, test = df[df.index <= TRAIN_END], df[df.index > TRAIN_END]
print(f'Train {len(train)} weeks, test {len(test)} weeks\n')
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
plot_acf(train[Y], lags=20, ax=ax[0]); plot_pacf(train[Y], lags=20, ax=ax[1]); fig.tight_layout()
fig.savefig('figures/5_acf_pacf.png', dpi=200); plt.close()

# choose ARIMA order on TRAIN data only (AIC), small grid
best = min(((ARIMA(train[Y], order=(p, 0, q)).fit().aic, (p, 0, q)) for p in range(4) for q in range(4)))
order = best[1]; print('Best ARIMA order by AIC (train only):', order, '\n')

def rolling_forecast(model_fn):
    """1-step-ahead forecast, refit-free: parameters fitted on train, then new data appended each week."""
    preds, fit = [], model_fn(train)
    for t in range(len(test)):
        preds.append(fit.forecast(1, exog=EX.iloc[[t]] if EX is not None else None).iloc[0])
        fit = fit.append(test[[Y]].iloc[[t]], exog=(EX.iloc[[t]] if EX is not None else None), refit=False)
    return np.array(preds)

errs, P = {}, {}
y = test[Y].values
# Benchmark 1: train mean;  Benchmark 2: zero (random walk on price)
P['Benchmark: train mean'] = np.full(len(y), train[Y].mean()); P['Benchmark: zero'] = np.zeros(len(y))
# Model A: ARIMA, own past only
EX = None; P['A: ARIMA'] = rolling_forecast(lambda d: ARIMA(d[Y], order=order).fit())
# Model B: ARIMAX with LAGGED traditional returns (a genuine forecast: last week's values are known)
lagged = df[X_COLS].shift(1).add_suffix('_lag1').dropna(); lagged = lagged.loc[df.index.intersection(lagged.index)]
tr_ex, EX = lagged.loc[train.index.intersection(lagged.index)], lagged.loc[test.index]
fitB0 = lambda d: ARIMA(d[Y].loc[tr_ex.index], exog=tr_ex, order=order).fit()
P['B: ARIMAX (lagged)'] = rolling_forecast(fitB0)
# Model C: ARIMAX with SAME-WEEK returns - NOT a true forecast (conditional / explanatory only)
tr_ex2, EX = train[X_COLS], test[X_COLS]
P['C: ARIMAX (same week, conditional)'] = rolling_forecast(lambda d: ARIMA(d[Y], exog=tr_ex2, order=order).fit())

rows = []
for k, p in P.items():
    e = y - p; rows.append({'model': k, 'RMSE': np.sqrt((e ** 2).mean()), 'MAE': np.abs(e).mean()})
acc = pd.DataFrame(rows).set_index('model'); acc['RMSE_vs_mean_%'] = (acc.RMSE / acc.loc['Benchmark: train mean', 'RMSE'] - 1) * 100
acc.round(5).to_csv('tables/5_forecast_accuracy.csv'); print(acc.round(5), '\n')

# Diebold-Mariano style test (A or B vs mean benchmark), simple version
def dm(e1, e2):
    d = e1 ** 2 - e2 ** 2; return stats.ttest_1samp(d, 0)
for k in ['A: ARIMA', 'B: ARIMAX (lagged)']:
    s = dm(y - P[k], y - P['Benchmark: train mean']); print(f'DM-type test {k} vs mean benchmark: t={s.statistic:.3f}, p={s.pvalue:.3f}')

fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(test.index, y, c='k', lw=1, label='Actual')
for k in ['A: ARIMA', 'B: ARIMAX (lagged)', 'C: ARIMAX (same week, conditional)']: ax.plot(test.index, P[k], lw=1, label=k)
ax.legend(fontsize=7); ax.set_title('Test period: actual vs forecast weekly index return'); fig.tight_layout()
fig.savefig('figures/5_forecast.png', dpi=200); plt.close()

# directional accuracy (did we get the sign right?)
for k in ['A: ARIMA', 'B: ARIMAX (lagged)', 'C: ARIMAX (same week, conditional)']:
    print(f'Directional accuracy {k}: {(np.sign(P[k]) == np.sign(y)).mean():.1%}')
print('\nDone. See figures/ and tables/')
