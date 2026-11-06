"""
daily_loader.py - shared helper for the daily scripts (06, 07, 08, 09).  Keep it in the same folder as them.
You do not run this file. It reads every daily file format you downloaded:
  * raw CoinMarketCap csv        (timeClose, close, marketCap)
  * FRED csv (S&P 500, US10Y)    (observation_date, SP500 / DGS10; blank rows = holidays)
  * Investing.com csv (FTSE, gold) (Date as month/day/year, Price with commas "10,606.00")
  * your own Excel files         (Date, Closing Price, Market Cap)
File names only need to START with one of the prefixes below (case does not matter on Windows).
"""
import glob
import numpy as np, pandas as pd

DIR = 'daily_data/'pythipyht
COIN_PREFIXES = {'BTC': ['BTC', 'Bitcoin'], 'ETH': ['ETH', 'Ethereum'], 'XRP': ['XRP'], 'BCH': ['BCH'],
                 'ADA': ['ADA', 'Cardano'], 'XEM': ['NEM', 'XEM'], 'LTC': ['LTC', 'Litecoin']}
TRAD_PREFIXES = {'SP500': ['SP500'], 'FTSE100': ['FTSE'], 'Gold': ['Gold', 'XAU'],
                 'US10Y': ['US10Y', 'US_treasury', 'DGS10']}


def _files(prefixes):
    out = []
    for p in prefixes:
        out += glob.glob(DIR + p + '*.csv') + glob.glob(DIR + p + '*.xlsx')
    return sorted({f for f in out if '~$' not in f})


def _read_file(f):
    if f.lower().endswith('xlsx'):
        return pd.read_excel(f, dtype=object)
    with open(f, encoding='utf-8-sig') as fh:
        head = fh.readline()
    sep = ';' if head.count(';') > head.count(',') else ','
    return pd.read_csv(f, sep=sep, encoding='utf-8-sig', dtype=str)


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(',', '', regex=False).str.strip(), errors='coerce')


def _dates(s):
    s = s.astype(str).str.strip()
    if s.str.match(r'^\d{1,2}/\d{1,2}/\d{4}$').all():                  # Investing.com: month/day/year
        return pd.to_datetime(s, format='%m/%d/%Y')
    d = pd.to_datetime(s, utc=True, errors='coerce')                    # ISO dates or CoinMarketCap timestamps (UTC)
    return d.dt.tz_localize(None).dt.normalize()


def _col(d, names):
    low = {c.lower().strip(): c for c in d.columns}
    for n in names:
        if n in low: return low[n]
    return None


def load_coin(prefixes):
    """Returns a DataFrame indexed by date with columns 'Closing Price' and 'Market Cap' (None if no file)."""
    files = _files(prefixes)
    if not files: return None, []
    parts = []
    for f in files:
        d = _read_file(f)
        dc, pc, mc = (_col(d, ['timeclose', 'date']), _col(d, ['close', 'closing price', 'price']),
                      _col(d, ['marketcap', 'market cap']))
        if None in (dc, pc, mc): raise ValueError(f'{f}: could not find date/close/marketCap columns in {list(d.columns)}')
        parts.append(pd.DataFrame({'Date': _dates(d[dc]), 'Closing Price': _num(d[pc]), 'Market Cap': _num(d[mc])}))
    d = pd.concat(parts, ignore_index=True)
    n_dup = int(d['Date'].duplicated().sum())
    d = d.drop_duplicates('Date', keep='last').set_index('Date').sort_index()
    d.attrs['duplicates'] = n_dup
    return d, files


def load_trad(prefixes):
    """Returns a Series (trading days only, weekends dropped, blanks dropped), or None if no file."""
    files = _files(prefixes)
    if not files: return None, []
    parts = []
    for f in files:
        d = _read_file(f)
        dc = _col(d, ['date', 'observation_date'])
        vc = _col(d, ['price', 'close', 'closing level']) or [c for c in d.columns if c != dc][0]
        parts.append(pd.Series(_num(d[vc]).values, index=_dates(d[dc]), name='value'))
    s = pd.concat(parts)
    s.attrs['duplicates'] = int(s.index.duplicated().sum())
    s.attrs['blank_rows'] = int(s.isna().sum())
    s.attrs['weekend_rows'] = int((s.dropna().index.dayofweek >= 5).sum())
    s = s[~s.index.duplicated(keep='last')].dropna().sort_index()
    s = s[s.index.dayofweek < 5]
    return s, files
