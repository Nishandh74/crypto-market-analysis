"""
STEP 13 - EXPLORATORY CHART: growth of 100 invested in each market (2018 - Sep 2026)
Input : Final_Weekly_Analysis_Table.xlsx
Output: figures/0_exploratory_growth.png
Each asset is rebased to 100 at the start, using cumulative weekly log returns.
A log scale is used so that all four assets can be compared on one chart despite very different sizes of movement.
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)
d = pd.read_excel('Final_Weekly_Analysis_Table.xlsx', sheet_name='Analysis table', index_col='Date')
assets = {'Crypto index': 'Crypto_Index', 'S&P 500': 'SP500', 'FTSE 100': 'FTSE100', 'Gold': 'Gold'}
col = {'Crypto index': '#2563eb', 'S&P 500': '#f59e0b', 'FTSE 100': '#16a34a', 'Gold': '#dc2626'}

fig, ax = plt.subplots(figsize=(10, 4.5))
for lab, c in assets.items():
    lvl = 100 * np.exp(d[c].cumsum())
    ax.plot(lvl.index, lvl, label=lab, color=col[lab], lw=1.3)
ax.set_yscale('log')
ax.set_ylabel('Value of 100 invested in Jan 2018 (log scale)')
ax.set_title('Growth of 100 invested in each market, 2018-2026')
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig('figures/0_exploratory_growth.png', dpi=200); plt.close()
print('Saved figures/0_exploratory_growth.png')
