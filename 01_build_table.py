"""
STEP 1 - BUILD THE WEEKLY ANALYSIS TABLE
Input : the 11 weekly Excel files you downloaded (put them all in the folder DATA_DIR)
Output: Final_Weekly_Analysis_Table.xlsx  (sheet "Analysis table" is what step 2 and SPSS use)

What it does
 1. Reads each coin file (Date, Closing Price, Market Cap) and aligns it to Sundays 7 Jan 2018 - 27 Sep 2026.
 2. Weekly simple returns per coin.
 3. Market-cap weighted index: weights = PREVIOUS week's market cap share (no look-ahead bias).
 4. Index return = ln(1 + weighted return). Traditional markets = weekly log returns; US10Y = change in yield.
"""
import pandas as pd, numpy as np

DATA_DIR = 'data/'          # <- folder with your Excel files
OUT = 'Final_Weekly_Analysis_Table.xlsx'

# file names as saved on your computer (edit if yours differ)
COIN_FILES = {'BTC': 'Bitcoin_Weekly_Data.xlsx', 'ETH': 'Ethereum_Weekly_Data.xlsx',
              'XRP': 'XRP_Weekly_Data.xlsx', 'BCH': 'BCH_Weekly_Data.xlsx',
              'ADA': 'Cardano_Weekly_Data.xlsx', 'XEM': 'NEM_Weekly_Data.xlsx',
              'LTC': 'Litecoin_Weekly_Data.xlsx'}
TRAD_FILES = {'SP500': 'SP500_Weekly_Data.xlsx', 'FTSE100': 'FTSE100_Weekly_Data.xlsx',
              'Gold': 'Gold_Weekly_Data.xlsx', 'US10Y': 'US10Y_Weekly_Data.xlsx'}

weeks = pd.date_range('2018-01-07', '2026-09-27', freq='W-SUN')   # 456 Sundays

def read(path):
    d = pd.read_excel(DATA_DIR + path)
    d['Date'] = pd.to_datetime(d['Date'])
    return d.set_index('Date').reindex(weeks)

# --- coins ---
price, mcap = {}, {}
for c, f in COIN_FILES.items():
    d = read(f)
    assert d.notna().all().all(), f'Missing weeks in {c}'     # stops if any gap
    price[c], mcap[c] = d['Closing Price'], d['Market Cap']
price, mcap = pd.DataFrame(price), pd.DataFrame(mcap)
price['XEM'] = mcap['XEM'] / 9e9     # NEM's price is rounded; fixed 9bn supply so derive it

ret = price.pct_change()                                       # simple weekly returns
w = mcap.shift(1).div(mcap.shift(1).sum(axis=1), axis=0)       # lagged weights
index_ret = np.log1p((w * ret).sum(axis=1, min_count=1))       # log return of the index

# --- traditional markets ---
trad = {k: read(f).iloc[:, 0] for k, f in TRAD_FILES.items()}
for k, s in trad.items():
    assert s.notna().all(), f'Missing weeks in {k}'

T = pd.DataFrame({'Crypto_Index': index_ret,
                  'SP500': np.log(trad['SP500']).diff(),
                  'FTSE100': np.log(trad['FTSE100']).diff(),
                  'Gold': np.log(trad['Gold']).diff(),
                  'US10Y_change': trad['US10Y'].diff()}).dropna()
T.index.name = 'Date'

print(T.shape, T.index.min().date(), 'to', T.index.max().date())
print(T.describe().round(4))
print('Average weights:\n', w.mean().round(3))     # Bitcoin dominance (limitation for report)

W_out, R_out, L_out = w.dropna().round(6), np.log1p(ret).dropna().round(6), pd.DataFrame(trad)
for x in (W_out, R_out, L_out): x.index.name = 'Date'
with pd.ExcelWriter(OUT) as xw:
    T.to_excel(xw, sheet_name='Analysis table')
    W_out.to_excel(xw, sheet_name='Index weights')
    R_out.to_excel(xw, sheet_name='Coin returns')
    L_out.to_excel(xw, sheet_name='Traditional levels')
print('Saved', OUT)
