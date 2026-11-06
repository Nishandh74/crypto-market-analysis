"""
STEP 11 - RISK TABLE: how risky was each asset? (weekly data, 2018 - Sep 2026)
Input : Final_Weekly_Analysis_Table.xlsx   (nothing new to download)
Output: tables/11_risk_table.csv, tables/11_drawdown_dates.csv, figures/11_drawdown.png

Measures (all from weekly log returns):
  Average yearly return   = growth rate per year (geometric), exp(mean weekly log return x 52) - 1
  Yearly volatility       = std of weekly log returns x sqrt(52)
  Sharpe ratio            = average yearly log return / yearly volatility   (risk-free rate assumed 0, so it is a simple 'return per unit of risk')
  Maximum drawdown        = biggest fall from a previous peak, in %
  Worst week              = the single worst weekly return, in %
  Weeks with a loss       = share of weeks with a negative return
US10Y is left out: it is a change in yield, not an asset price return.
S&P 500 and FTSE 100 are price indices (no dividends).
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('tables', exist_ok=True); os.makedirs('figures', exist_ok=True)
df = pd.read_excel('Final_Weekly_Analysis_Table.xlsx', sheet_name='Analysis table', index_col='Date')
ASSETS = {'Crypto_Index': 'Crypto index', 'SP500': 'S&P 500', 'FTSE100': 'FTSE 100', 'Gold': 'Gold'}
W = 52

rows, dd_rows, dd_series = {}, [], {}
for col, lab in ASSETS.items():
    r = df[col]
    mean_w, sd_w = r.mean(), r.std()
    level = r.cumsum()                                   # log level, starts at 0
    peak = level.cummax().clip(lower=0)                  # the starting level (0) counts as a peak
    dd = np.exp(level - peak) - 1                        # drawdown from the previous peak
    trough = dd.idxmin()
    peak_date = level.loc[:trough].idxmax() if level.loc[:trough].max() > 0 else df.index[0]
    rec = level.loc[trough:][level.loc[trough:] >= level.loc[peak_date]]
    dd_series[lab] = dd
    rows[lab] = {
        'Average yearly return (%)': (np.exp(mean_w * W) - 1) * 100,
        'Yearly volatility (%)': sd_w * np.sqrt(W) * 100,
        'Sharpe ratio (risk-free = 0)': (mean_w * W) / (sd_w * np.sqrt(W)),
        'Maximum drawdown (%)': dd.min() * 100,
        'Worst week (%)': (np.exp(r.min()) - 1) * 100,
        'Weeks with a loss (%)': (r < 0).mean() * 100,
        'Weeks': len(r)}
    dd_rows.append({'Asset': lab, 'Peak before the fall': peak_date.date(), 'Trough (lowest point)': trough.date(),
                    'Maximum drawdown (%)': round(dd.min() * 100, 1),
                    'Back to previous peak': rec.index[0].date() if len(rec) else 'not yet recovered'})

tab = pd.DataFrame(rows).round(2)
tab.to_csv('tables/11_risk_table.csv')
dd_tab = pd.DataFrame(dd_rows).set_index('Asset'); dd_tab.to_csv('tables/11_drawdown_dates.csv')
print(f'RISK TABLE, {df.index.min().date()} to {df.index.max().date()} (weekly returns)\n')
print(tab.to_string(), '\n')
print('WHEN WAS THE BIGGEST FALL?\n', dd_tab.to_string(), '\n')

fig, ax = plt.subplots(figsize=(10, 4))
for lab, s in dd_series.items(): ax.plot(s.index, s * 100, lw=1, label=lab)
ax.set_ylabel('Fall from previous peak, %'); ax.set_title('Drawdowns: how far each asset was below its previous high')
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig('figures/11_drawdown.png', dpi=200); plt.close()
print('Saved tables/11_risk_table.csv, tables/11_drawdown_dates.csv, figures/11_drawdown.png')
