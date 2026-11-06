"""
STEP 5 - (A) GRANGER CAUSALITY and (B) VOLATILITY ANALYSIS + FORECASTING (GARCH)
Needs: pip install arch
Input : Final_Weekly_Analysis_Table.xlsx (sheet "Analysis table")
Output: tables/8_*.csv and figures/8_*.png

(A) Granger: does last weeks' return of a traditional market help predict crypto (and the reverse)?
(B) Volatility: weekly returns are almost unpredictable, but the SIZE of the swings (volatility) usually is.
    We fit GARCH(1,1) on 2018-2023 and forecast volatility week by week for 2024 -> Sep 2026,
    and compare with simple benchmarks.
"""
import os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model
warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True); os.makedirs('tables', exist_ok=True)

df = pd.read_excel('Final_Weekly_Analysis_Table.xlsx', sheet_name='Analysis table', index_col='Date')
Y = 'Crypto_Index'; TRAD = ['SP500', 'FTSE100', 'Gold', 'US10Y_change']

# =============================================================== A. GRANGER CAUSALITY
MAXLAG = 4
rows = []
for x in TRAD:
    for direction, (cause, effect) in {f'{x} -> Crypto': (x, Y), f'Crypto -> {x}': (Y, x)}.items():
        res = grangercausalitytests(df[[effect, cause]], maxlag=MAXLAG)   # tests: does 2nd column cause 1st?
        r = {'Direction': direction}
        for L in range(1, MAXLAG + 1): r[f'p (lag {L})'] = res[L][0]['ssr_ftest'][1]
        r['Significant at 5% (any lag)?'] = 'Yes' if min(r[f'p (lag {L})'] for L in range(1, MAXLAG + 1)) < 0.05 else 'No'
        rows.append(r)
gr = pd.DataFrame(rows).set_index('Direction'); gr.round(4).to_csv('tables/8_granger.csv')
print('GRANGER CAUSALITY (p<0.05 = past values help predict)\n', gr.round(4).to_string(), '\n')

# =============================================================== B. VOLATILITY
r = df[Y] * 100                               # percent returns (helps the optimiser)
# B1. Is there volatility clustering? ARCH-LM test on index returns
lm = het_arch(r - r.mean(), nlags=4)
print(f'ARCH-LM test (clustering of volatility): stat={lm[0]:.2f}, p={lm[1]:.4f}\n')

# B2. Does traditional-market turbulence spill over to crypto turbulence? (|return| regressed on lagged |returns|)
A = df.abs(); Xv = sm.add_constant(A[TRAD].shift(1)).dropna(); yv = A[Y].loc[Xv.index]
sp = sm.OLS(yv, Xv).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
print('Volatility spillover: |crypto| on LAST WEEK\'s |traditional| (HAC)\n', sp.summary().tables[1])
pd.DataFrame({'coef': sp.params, 'p_HAC': sp.pvalues}).round(4).to_csv('tables/8_vol_spillover.csv')
print(f'R2 = {sp.rsquared:.3f}\n')

# B3. GARCH(1,1), Student-t errors, fitted on TRAIN only
TRAIN_END = '2023-12-31'
n_train = int((r.index <= TRAIN_END).sum())
am = arch_model(r, mean='Constant', vol='GARCH', p=1, q=1, dist='t')
fit = am.fit(last_obs=r.index[n_train - 1], disp='off')
print(fit.summary(), '\n')
pd.DataFrame({'coef': fit.params, 'p': fit.pvalues}).round(4).to_csv('tables/8_garch_params.csv')
persist = fit.params['alpha[1]'] + fit.params['beta[1]']; print(f'Persistence (alpha+beta) = {persist:.3f}\n')

# one-step-ahead variance forecasts for every test week (parameters fixed, data updated each week)
fc = fit.forecast(start=r.index[n_train - 1], horizon=1, reindex=False)
garch_var = fc.variance['h.1'].iloc[:-1].copy(); garch_var.index = r.index[n_train:]   # row = forecast ORIGIN date; shift so index = week being forecast (made the week before)

# benchmarks (all use only past information)
test = r.iloc[n_train:]; real = test ** 2        # squared return = noisy proxy of true variance
bench = {}
bench['Constant (train variance)'] = pd.Series(r.iloc[:n_train].var(), index=test.index)
bench['Rolling 26-week variance'] = (r ** 2).rolling(26).mean().shift(1).loc[test.index]
ew = (r ** 2).ewm(alpha=0.06, adjust=False).mean().shift(1)            # RiskMetrics-style EWMA (lambda 0.94)
bench['EWMA (lambda 0.94)'] = ew.loc[test.index]
allf = {'GARCH(1,1)': garch_var, **bench}

def qlike(f, y):  return np.mean(np.log(f) + y / f)
rows = []
for k, f in allf.items():
    rows.append({'Model': k, 'MSE (on squared return)': np.mean((real - f) ** 2),
                 'MAE (on |return|, using sqrt forecast)': np.mean(np.abs(test.abs() - np.sqrt(f))),
                 'QLIKE (lower=better)': qlike(f, real),
                 'Correlation with realised |return|': np.corrcoef(np.sqrt(f), test.abs())[0, 1]})
vt = pd.DataFrame(rows).set_index('Model'); vt.round(4).to_csv('tables/8_vol_forecast_accuracy.csv')
print('VOLATILITY FORECAST ACCURACY, test period', test.index[0].date(), 'to', test.index[-1].date(), '\n', vt.round(4).to_string(), '\n')

# Mincer-Zarnowitz style check: does GARCH forecast explain realised squared returns?
mz = sm.OLS(real, sm.add_constant(garch_var)).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
print(f'Realised variance on GARCH forecast: slope={mz.params.iloc[1]:.3f} (p={mz.pvalues.iloc[1]:.4f}), R2={mz.rsquared:.3f}')

# charts
cv = fit.conditional_volatility.reindex(r.index)
fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(r.index, r.abs(), lw=.5, c='lightgrey', label='|weekly return| %')
ax.plot(cv.index, cv, c='C0', lw=1.2, label='GARCH volatility (in-sample to 2023)')
ax.plot(garch_var.index, np.sqrt(garch_var), c='C3', lw=1.2, label='GARCH forecast (out of sample)')
ax.axvline(pd.Timestamp(TRAIN_END), c='k', ls=':'); ax.legend(fontsize=8); ax.set_ylabel('% per week')
ax.set_title('Crypto index volatility: clustering and out-of-sample forecast'); fig.tight_layout()
fig.savefig('figures/8_volatility_garch.png', dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(test.index, test.abs(), c='lightgrey', label='Realised |return| %')
for k, f in allf.items(): ax.plot(f.index, np.sqrt(f), lw=1.1, label=k)
ax.legend(fontsize=7); ax.set_title('Test period: realised vs forecast volatility'); fig.tight_layout()
fig.savefig('figures/8_vol_forecast_compare.png', dpi=200); plt.close()
print('\nDone: tables/8_*, figures/8_*')
