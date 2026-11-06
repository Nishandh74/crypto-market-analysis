"""
STEP 3 - EVENT ANALYSIS (geopolitical / crisis) + EXTRA CHARTS + ROBUSTNESS
Input : Final_Weekly_Analysis_Table.xlsx (all sheets)
Output: figures/ and tables/ (same folders as step 2)
EDIT the EVENTS dictionary: use dates YOU have verified from news sources and cite them in the report.
"""
import os, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
warnings.filterwarnings('ignore')
os.makedirs('figures', exist_ok=True); os.makedirs('tables', exist_ok=True)

F = 'Final_Weekly_Analysis_Table.xlsx'
df = pd.read_excel(F, sheet_name='Analysis table', index_col='Date')
W  = pd.read_excel(F, sheet_name='Index weights', index_col='Date')
CR = pd.read_excel(F, sheet_name='Coin returns', index_col='Date')
MK = ['Crypto_Index', 'SP500', 'FTSE100', 'Gold', 'US10Y_change']

# ---- 1. EVENTS: name -> date the event started (CHECK these and add/remove as you decide)
EVENTS = {'COVID-19 crash': '2020-03-12',
          'Russia invades Ukraine': '2022-02-24',
          'Hamas attack / Israel-Gaza war': '2023-10-07'}
#  e.g. add a Hormuz / Middle East event only after you have a reliable source for its date:
#  EVENTS['Strait of Hormuz tension'] = 'YYYY-MM-DD'
PRE, POST = 4, 4          # weeks shown before / after the event week

def event_week(d):        # the weekly (Sunday-ending) observation that contains the date
    d = pd.Timestamp(d); pos = df.index.searchsorted(d); return df.index[pos]

rows, fig, axes = [], *plt.subplots(1, len(EVENTS), figsize=(5 * len(EVENTS), 4), squeeze=False)
for ax, (name, date) in zip(axes[0], EVENTS.items()):
    ew = event_week(date); i = df.index.get_loc(ew)
    lo = max(i - PRE, 0)
    win = df[MK[:4]].iloc[lo: i + POST + 1]
    cum = win.cumsum()
    cum = cum - cum.iloc[i - 1 - lo]          # 0 = end of the week BEFORE the event week
    cum.index = range(lo - i, lo - i + len(win))
    for c in cum: ax.plot(cum.index, cum[c] * 100, label=c)
    ax.axvline(0, c='k', ls=':'); ax.set_title(f'{name}\n(event week {ew.date()})', fontsize=9)
    ax.set_xlabel('Weeks from event week (0 = week before)'); ax.set_ylabel('Cumulative log return, %')
    r = {'Event': name, 'Event week': ew.date()}
    for c in MK:
        r[f'{c} | event wk'] = df[c].iloc[i]; r[f'{c} | next {POST} wks'] = df[c].iloc[i + 1:i + 1 + POST].sum()
    rows.append(r)
axes[0][0].legend(fontsize=7); fig.tight_layout(); fig.savefig('figures/6_event_windows.png', dpi=200); plt.close()
ev = pd.DataFrame(rows).set_index('Event'); ev.round(4).T.to_csv('tables/6_event_table.csv'); print(ev.round(4).T, '\n')

# event dummies in the regression: 1 for the event week and the following 3 weeks
dm = pd.Series(0, index=df.index)
for d in EVENTS.values():
    i = df.index.get_loc(event_week(d)); dm.iloc[i:i + 4] = 1
X = sm.add_constant(df[['SP500', 'FTSE100', 'Gold', 'US10Y_change']].assign(Event=dm))
m = sm.OLS(df['Crypto_Index'], X).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
print('Regression with event dummy (HAC):\n', m.summary().tables[1]); 
pd.DataFrame({'coef': m.params, 'p_HAC': m.pvalues}).round(4).to_csv('tables/6_event_regression.csv')

# ---- 2. EXTRA CHARTS
lvl = np.exp(df['Crypto_Index'].cumsum()) * 100
fig, ax = plt.subplots(figsize=(9, 3.5)); ax.plot(lvl); ax.set_yscale('log')
ax.set_title('Crypto index level (start = 100, log scale)'); fig.tight_layout(); fig.savefig('figures/2_index_level.png', dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(9, 3.8)); ax.stackplot(W.index, W.T.values, labels=W.columns)
ax.legend(loc='upper left', fontsize=7, ncol=4); ax.set_ylim(0, 1); ax.set_title('Index weights (previous-week market-cap share)')
fig.tight_layout(); fig.savefig('figures/2_index_weights.png', dpi=200); plt.close()

fig, axs = plt.subplots(5, 1, figsize=(9, 9), sharex=True)
for a, c in zip(axs, MK): a.plot(df[c], lw=.7); a.set_ylabel(c, fontsize=8)
axs[0].set_title('Weekly returns (US10Y: change in percentage points)'); fig.tight_layout()
fig.savefig('figures/2_weekly_returns.png', dpi=200); plt.close()

# ---- 3. ROBUSTNESS: does the result depend on the index? (BTC only, BTC+ETH, ex-Bitcoin)
alt = {'Main index': df['Crypto_Index'], 'BTC only': CR['BTC'], 'BTC+ETH (cap-weighted)': None, 'Excl. BTC': None}
w2 = W[['BTC', 'ETH']].div(W[['BTC', 'ETH']].sum(axis=1), axis=0); alt['BTC+ETH (cap-weighted)'] = (w2 * CR[['BTC', 'ETH']]).sum(axis=1)
oth = [c for c in W.columns if c != 'BTC']; w3 = W[oth].div(W[oth].sum(axis=1), axis=0); alt['Excl. BTC'] = (w3 * CR[oth]).sum(axis=1)
out = {}
for k, y in alt.items():
    d = pd.concat([y.rename('y'), df[['SP500', 'FTSE100', 'Gold', 'US10Y_change']]], axis=1).dropna()
    r = sm.OLS(d['y'], sm.add_constant(d.iloc[:, 1:])).fit(cov_type='HAC', cov_kwds={'maxlags': 4})
    out[k] = pd.concat([r.params.add_suffix(' coef'), r.pvalues.add_suffix(' p'), pd.Series({'R2': r.rsquared})])
rb = pd.DataFrame(out).round(4); rb.to_csv('tables/7_robustness.csv'); print('\nRobustness\n', rb)
print('\nDone: figures/6_*, 2_*, tables/6_*, 7_*')
