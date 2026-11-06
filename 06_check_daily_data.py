"""
CHECK THE DAILY DATA FOR MISSING DAYS, DUPLICATES, MIX-UPS AND ODD VALUES
Put all 11 daily files in the folder  data_daily/  (next to this script), together with daily_loader.py.
File names must START with: BTC (or Bitcoin), ETH, XRP, BCH, ADA, NEM, LTC, SP500, FTSE, Gold (or XAU), US10Y (or US_treasury).
Run:  python 06_check_daily_data.py     -> it prints ALL CLEAN or the list of files to look at.
"""
import numpy as np, pandas as pd
from daily_loader import COIN_PREFIXES, TRAD_PREFIXES, load_coin, load_trad

START, END = '2018-01-07', '2026-09-27'
# rough price on 7 Jan 2018 for each coin - catches a file saved under the wrong coin name
FIRST_DAY_RANGE = {'BTC': (10000, 20000), 'ETH': (800, 1500), 'XRP': (1.0, 4.0), 'BCH': (1500, 3500),
                   'ADA': (0.5, 1.6), 'XEM': (0.5, 2.5), 'LTC': (150, 300)}
problems = []

print('=== COINS (must have EVERY calendar day) ===')
all_days = pd.date_range(START, END, freq='D')
for coin, pre in COIN_PREFIXES.items():
    d, files = load_coin(pre)
    if d is None:
        print(f'{coin}: NO FILE FOUND (name must start with {pre})'); problems.append(coin); continue
    inside = d.loc[START:END]
    missing = all_days.difference(inside.index)
    nulls = int(inside.isna().sum().sum())
    zeros = int((inside <= 0).sum().sum())
    jumps = inside['Market Cap'].pct_change().abs().loc[lambda s: s > 0.5].index.strftime('%Y-%m-%d').tolist()
    first = inside['Closing Price'].iloc[0]; lo, hi = FIRST_DAY_RANGE[coin]
    ok_first = lo <= first <= hi
    print(f'{coin}: {len(files)} file(s) | rows {len(inside)}/{len(all_days)} | missing days {len(missing)} | duplicates {d.attrs["duplicates"]} '
          f'| empty {nulls} | zero/neg {zeros} | >50% mkt-cap jumps {len(jumps)} | price on {inside.index[0].date()}: {first:,.4f} '
          f'{"OK" if ok_first else "<-- WRONG COIN?"}')
    if len(missing): print('   missing (first 10):', [x.strftime('%Y-%m-%d') for x in missing[:10]]); problems.append(coin)
    if jumps: print('   jumps are usually real moves in small coins - glance at:', jumps[:6])
    if d.attrs['duplicates'] or zeros or nulls or not ok_first: problems.append(coin)

print('\n=== TRADITIONAL MARKETS (holidays are normal; look for gaps and thin weekdays) ===')
for name, pre in TRAD_PREFIXES.items():
    s, files = load_trad(pre)
    if s is None:
        print(f'{name}: NO FILE FOUND (name must start with {pre})'); problems.append(name); continue
    a = s.loc[START:END]
    gaps = a.index.to_series().diff().dt.days
    wd = a.index.dayofweek.value_counts().reindex(range(5), fill_value=0)
    thin = [n for n, c in zip(['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], wd.values) if c < 0.85 * wd.max()]
    print(f'{name}: {len(files)} file(s) | trading days {len(a)} | first {a.index.min().date()} last {a.index.max().date()} '
          f'| blank holiday rows {s.attrs["blank_rows"]} | weekend rows {s.attrs["weekend_rows"]} | duplicates {s.attrs["duplicates"]} '
          f'| gaps over 5 days {int((gaps > 5).sum())} | zero/neg {int((a <= 0).sum())}')
    print(f'   rows per weekday Mon..Fri: {wd.tolist()}' + (f'  <-- THIN: {thin} (missing days?)' if thin else ''))
    if (a.index.min() > pd.Timestamp('2018-01-10')) or (a.index.max() < pd.Timestamp('2026-09-24')): problems.append(name)
    if (gaps > 5).sum() or (a <= 0).sum() or thin or s.attrs['duplicates'] or s.attrs['weekend_rows']: problems.append(name)

print('\n=== RESULT ===')
print('ALL CLEAN' if not problems else 'CHECK THESE: ' + ', '.join(sorted(set(problems))))
