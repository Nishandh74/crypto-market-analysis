"""
STEP 7 - BUILD THE DAILY ANALYSIS TABLE (robustness check for the weekly analysis)
Input : the 11 daily files in data_daily/  (and daily_loader.py)
Output: Final_Daily_Analysis_Table.xlsx

Method (same logic as the weekly table, so results are comparable)
 1. Crypto trades every day, stock markets and bonds do not. So we only use COMMON TRADING DAYS:
    days when the S&P 500, FTSE 100, gold and the US 10-year yield all have a value.
 2. Crypto index = market-cap weighted, weights = PREVIOUS day's market-cap share (no look-ahead).
    Index daily return = ln(1 + weighted simple return), exactly as in the weekly table.
 3. For each pair of consecutive common days (e.g. Friday -> Monday) we measure BOTH sides over the SAME interval:
    crypto = sum of the daily index log returns in the interval, traditional = log change of the level
    (US10Y = change in yield). So a Monday return covers Friday close -> Monday close for everything.
 4. 'Gap_days' = length of the interval (1 normally, 3 after a weekend, more around holidays).
"""
import numpy as np, pandas as pd
from daily_loader import COIN_PREFIXES, TRAD_PREFIXES, load_coin, load_trad

OUT = 'Final_Daily_Analysis_Table.xlsx'
FIRST_RET, LAST = '2018-01-08', '2026-09-27'          # same study window as the weekly analysis

# ---------------- coins on EVERY calendar day (levels from 1 Jan 2018 so the first return exists)
cal = pd.date_range('2018-01-01', '2026-09-30', freq='D')
price, mcap = {}, {}
for c, pre in COIN_PREFIXES.items():
    d, _ = load_coin(pre)
    d = d.reindex(cal)
    assert d.notna().all().all(), f'Missing days in {c} - run 06_check_daily_data.py first'
    price[c], mcap[c] = d['Closing Price'], d['Market Cap']
price, mcap = pd.DataFrame(price), pd.DataFrame(mcap)

ret = price.pct_change()                                           # simple daily returns
w = mcap.shift(1).div(mcap.shift(1).sum(axis=1), axis=0)           # lagged weights
idx_ret = np.log1p((w * ret).sum(axis=1, min_count=1))             # daily log return of the index (calendar days)
idx_level = idx_ret.fillna(0).cumsum()                             # log level of the index

# ---------------- traditional markets on trading days
trad = {k: load_trad(p)[0] for k, p in TRAD_PREFIXES.items()}
L = pd.DataFrame(trad).reindex(cal)
common = L.dropna().index                                          # all four markets open
print(f'Common trading days (2018-01-01 to 2026-09-30): {len(common)}')

lv = L.loc[common]
T = pd.DataFrame({'Crypto_Index': idx_level.reindex(common).diff(),
                  'SP500': np.log(lv['SP500']).diff(),
                  'FTSE100': np.log(lv['FTSE100']).diff(),
                  'Gold': np.log(lv['Gold']).diff(),
                  'US10Y_change': lv['US10Y'].diff(),
                  'Gap_days': pd.Series(common, index=common).diff().dt.days})
T = T.loc[FIRST_RET:LAST].dropna()
T.index.name = 'Date'

print(T.shape, T.index.min().date(), 'to', T.index.max().date())
print('Interval length (days):', T['Gap_days'].value_counts().sort_index().to_dict())
print(T.describe().round(4))
print('Average index weights:\n', w.mean().round(3))

# ---------------- consistency check against the weekly table (if you have it in this folder)
try:
    wk = pd.read_excel('Final_Weekly_Analysis_Table.xlsx', sheet_name='Analysis table', index_col='Date')
    d = idx_ret.loc[FIRST_RET:LAST]
    wsum = d.resample('W-SUN', label='right', closed='right').sum()
    both = pd.concat([wsum, wk['Crypto_Index']], axis=1, join='inner').dropna()
    print(f'\nCheck: weekly sums of the DAILY index vs the WEEKLY index  ->  correlation {both.iloc[:, 0].corr(both.iloc[:, 1]):.4f} '
          f'({len(both)} weeks)  [should be about 0.99+; small differences = daily vs weekly rebalancing]')
except FileNotFoundError:
    print('\n(Final_Weekly_Analysis_Table.xlsx not found here - skipped the weekly consistency check)')

# ---------------- save
cal_out = pd.DataFrame({'Crypto_Index': idx_ret}).loc[FIRST_RET:LAST]; cal_out.index.name = 'Date'
W_out = w.loc[FIRST_RET:LAST].round(6); W_out.index.name = 'Date'
L_out = L.loc[:LAST]; L_out.index.name = 'Date'
with pd.ExcelWriter(OUT) as xw:
    T.to_excel(xw, sheet_name='Analysis table')
    cal_out.to_excel(xw, sheet_name='Crypto calendar-day')
    W_out.to_excel(xw, sheet_name='Index weights')
    L_out.to_excel(xw, sheet_name='Traditional levels')
print('Saved', OUT)
