"""
STEP 9 - DAILY GRANGER CAUSALITY, VOLATILITY (GARCH) AND EVENT WINDOWS
Needs: pip install arch      Input: Final_Daily_Analysis_Table.xlsx      Output: tables_daily/9_*.csv, figures_daily/9_*.png

A. Granger causality on the common-trading-day table (lags 1-5), both directions.
   Why it matters: the weekly test said crypto 'Granger-causes' the S&P 500, which may only be a timing effect
   (crypto's week ends on Sunday, stock markets' week ends on Friday). Daily data lets us check that.
B. Volatility: ARCH-LM test, spillover regression, GARCH(1,1) forecast vs simple benchmarks.
   GARCH uses the crypto index on EVERY calendar day (equal spacing); the spillover regression uses the common-day table.
C. Event windows (COVID, Russia-Ukraine, Hamas attack) with daily detail: cumulative returns from the day before the event.
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
T_DIR, F_DIR = 'tables_daily/', 'figures_daily/'
os.makedirs(T_DIR, exist_ok=True); os.makedirs(F_DIR, exist_ok=True)

F = 'Final_Daily_Analysis_Table.xlsx'
df = pd.read_excel(F, sheet_name='Analysis table', index_col='Date')
cal = pd.read_excel(F, sheet_name='Crypto calendar-day', index_col='Date')['Crypto_Index']
LV = pd.read_excel(F, sheet_name='Traditional levels', index_col='Date')
Y = 'Crypto_Index'; TRAD = ['SP500', 'FTSE100', 'Gold', 'US10Y_change']

# =============================================================== A. GRANGER CAUSALITY
MAXLAG = 5
rows = []
for x in TRAD:
    for direction, (cause, effect) in {f'{x} -> Crypto': (x, Y), f'Crypto -> {x}': (Y, x)}.items():
        res = grangercausalitytests(df[[effect, cause]].reset_index(drop=True), maxlag=MAXLAG)
        r = {'Direction': direction}
        for L in range(1, MAXLAG + 1): r[f'p (lag {L})'] = res[L][0]['ssr_ftest'][1]
        r['Significant at 5% (any lag)?'] = 'Yes' if min(r[f'p (lag {L})'] for L in range(1, MAXLAG + 1)) < 0.05 else 'No'
        rows.append(r)
gr = pd.DataFrame(rows).set_index('Direction'); gr.round(4).to_csv(T_DIR + '9_granger.csv')
print('DAILY GRANGER CAUSALITY (p<0.05 = past values help predict)\n', gr.round(4).to_string(), '\n')

# =============================================================== B. VOLATILITY
r = cal * 100                                     # % per calendar day
lm = het_arch(r - r.mean(), nlags=5)
print(f'ARCH-LM test (volatility clustering, calendar-day crypto index): stat={lm[0]:.2f}, p={lm[1]:.4g}\n')
pd.Series({'ARCH_LM_stat': lm[0], 'ARCH_LM_p': lm[1]}).to_csv(T_DIR + '9_arch_lm.csv')

A = df[[Y] + TRAD].abs(); Xv = sm.add_constant(A[TRAD].shift(1)).dropna(); yv = A[Y].loc[Xv.index]
sp = sm.OLS(yv, Xv).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
print("Volatility spillover: |crypto| on the PREVIOUS common day's |traditional| (HAC)\n", sp.summary().tables[1], f'\nR2 = {sp.rsquared:.3f}\n')
pd.DataFrame({'coef': sp.params, 'p_HAC': sp.pvalues}).round(4).to_csv(T_DIR + '9_vol_spillover.csv')
# same-day version (volatility moves together?)
Xs = sm.add_constant(A[TRAD]); ss = sm.OLS(A[Y], Xs).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
print('Same-day volatility link: |crypto| on same-day |traditional| (HAC)\n', ss.summary().tables[1], f'\nR2 = {ss.rsquared:.3f}\n')
pd.DataFrame({'coef': ss.params, 'p_HAC': ss.pvalues}).round(4).to_csv(T_DIR + '9_vol_same_day.csv')

TRAIN_END = '2023-12-31'
n_train = int((r.index <= TRAIN_END).sum())
am = arch_model(r, mean='Constant', vol='GARCH', p=1, q=1, dist='t')
fit = am.fit(last_obs=r.index[n_train - 1], disp='off')
print(fit.summary(), '\n')
pd.DataFrame({'coef': fit.params, 'p': fit.pvalues}).round(4).to_csv(T_DIR + '9_garch_params.csv')
persist = fit.params['alpha[1]'] + fit.params['beta[1]']; print(f'Persistence (alpha+beta) = {persist:.3f}\n')

fc = fit.forecast(start=r.index[n_train - 1], horizon=1, reindex=False)
garch_var = fc.variance['h.1'].iloc[:-1].copy(); garch_var.index = r.index[n_train:]
test = r.iloc[n_train:]; real = test ** 2
bench = {'Constant (train variance)': pd.Series(r.iloc[:n_train].var(), index=test.index),
         'Rolling 60-day variance': (r ** 2).rolling(60).mean().shift(1).loc[test.index],
         'EWMA (lambda 0.94)': (r ** 2).ewm(alpha=0.06, adjust=False).mean().shift(1).loc[test.index]}
allf = {'GARCH(1,1)': garch_var, **bench}
qlike = lambda f, y_: np.mean(np.log(f) + y_ / f)
vrows = []
for k, f in allf.items():
    vrows.append({'Model': k, 'MSE (on squared return)': np.mean((real - f) ** 2),
                  'MAE (on |return|, using sqrt forecast)': np.mean(np.abs(test.abs() - np.sqrt(f))),
                  'QLIKE (lower=better)': qlike(f, real),
                  'Correlation with realised |return|': np.corrcoef(np.sqrt(f), test.abs())[0, 1]})
vt = pd.DataFrame(vrows).set_index('Model'); vt.round(4).to_csv(T_DIR + '9_vol_forecast_accuracy.csv')
print('VOLATILITY FORECAST ACCURACY, test', test.index[0].date(), 'to', test.index[-1].date(), '\n', vt.round(4).to_string(), '\n')
mz = sm.OLS(real, sm.add_constant(garch_var)).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
print(f'Realised variance on GARCH forecast: slope={mz.params.iloc[1]:.3f} (p={mz.pvalues.iloc[1]:.4g}), R2={mz.rsquared:.3f}')
pd.Series({'slope': mz.params.iloc[1], 'p': mz.pvalues.iloc[1], 'R2': mz.rsquared}).round(4).to_csv(T_DIR + '9_garch_mincer.csv')

cv = fit.conditional_volatility.reindex(r.index)
fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(r.index, r.abs(), lw=.4, c='lightgrey', label='|daily return| %')
ax.plot(cv.index, cv, c='C0', lw=1.1, label='GARCH volatility (in-sample to 2023)')
ax.plot(garch_var.index, np.sqrt(garch_var), c='C3', lw=1.1, label='GARCH forecast (out of sample)')
ax.axvline(pd.Timestamp(TRAIN_END), c='k', ls=':'); ax.legend(fontsize=8); ax.set_ylabel('% per day')
ax.set_title('Crypto index volatility (daily): clustering and out-of-sample forecast'); fig.tight_layout()
fig.savefig(F_DIR + '9_volatility_garch.png', dpi=200); plt.close()
fig, ax = plt.subplots(figsize=(10, 4)); ax.plot(test.index, test.abs(), c='lightgrey', lw=.6, label='Realised |return| %')
for k, f in allf.items(): ax.plot(f.index, np.sqrt(f), lw=1, label=k)
ax.legend(fontsize=7); ax.set_title('Test period: realised vs forecast daily volatility'); fig.tight_layout()
fig.savefig(F_DIR + '9_vol_forecast_compare.png', dpi=200); plt.close()

# =============================================================== C. EVENT WINDOWS (daily)
EVENTS = {'COVID-19 crash': '2020-03-12', 'Russia invades Ukraine': '2022-02-24', 'Hamas attack / Israel-Gaza war': '2023-10-07'}
PRE, POST = 5, 15
crypto_level = cal.fillna(0).cumsum()
lv = LV.reindex(pd.date_range(LV.index.min(), LV.index.max(), freq='D')).ffill()      # weekends/holidays carry the last close
lg = pd.DataFrame({'Crypto_Index': crypto_level, 'SP500': np.log(lv['SP500']), 'FTSE100': np.log(lv['FTSE100']),
                   'Gold': np.log(lv['Gold'])}).dropna()
fig, axes = plt.subplots(1, len(EVENTS), figsize=(5 * len(EVENTS), 4), squeeze=False)
erows = []
for ax, (name, date) in zip(axes[0], EVENTS.items()):
    d0 = pd.Timestamp(date); base = lg.loc[d0 - pd.Timedelta(days=1)]
    win = (lg.loc[d0 - pd.Timedelta(days=PRE): d0 + pd.Timedelta(days=POST)] - base) * 100
    win.index = (win.index - d0).days
    for c in win: ax.plot(win.index, win[c], label=c)
    ax.axvline(0, c='k', ls=':'); ax.set_title(f'{name}\n(event day {date})', fontsize=9)
    ax.set_xlabel('Days from event (0 = event day)'); ax.set_ylabel('Cumulative log return since day before, %')
    row = {'Event': name, 'Event date': date}
    for c in win:
        row[f'{c} | day 0'] = win.loc[0, c]; row[f'{c} | +5 days'] = win.loc[5, c]; row[f'{c} | +15 days'] = win.loc[15, c]
    erows.append(row)
axes[0][0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(F_DIR + '9_event_windows.png', dpi=200); plt.close()
ev = pd.DataFrame(erows).set_index('Event').T
ev.to_csv(T_DIR + '9_event_table.csv'); print('\nEVENTS (cumulative log return in %, base = close of day before the event)\n', ev.round(2).to_string())
print('\nDone: tables_daily/9_*, figures_daily/9_*')
