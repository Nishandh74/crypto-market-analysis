"""
STEP 4 - NORMALITY TESTS (replaces the SPSS step: Shapiro-Wilk + Q-Q plots + histograms)
Input : Final_Weekly_Analysis_Table.xlsx (sheet "Analysis table")
Output: tables/0_normality.csv, figures/1_qq_plots.png, figures/1_histograms.png
Reading the table: p-value (Shapiro-Wilk) < 0.05 -> reject normality.
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
os.makedirs('figures', exist_ok=True); os.makedirs('tables', exist_ok=True)

df = pd.read_excel('Final_Weekly_Analysis_Table.xlsx', sheet_name='Analysis table', index_col='Date')

rows = []
for c in df:
    w, p = stats.shapiro(df[c]); jb, pjb = stats.jarque_bera(df[c])
    rows.append({'Variable': c, 'N': len(df), 'Shapiro-Wilk W': w, 'Shapiro-Wilk p': p,
                 'Jarque-Bera p': pjb, 'Skewness': stats.skew(df[c]),
                 'Excess kurtosis': stats.kurtosis(df[c]), 'Normal at 5%?': 'Yes' if p >= 0.05 else 'No'})
out = pd.DataFrame(rows).set_index('Variable')
out.round(4).to_csv('tables/0_normality.csv'); print(out.round(4).to_string())

fig, axs = plt.subplots(2, 3, figsize=(11, 7)); axs = axs.ravel()
for ax, c in zip(axs, df): sm.qqplot(df[c], line='s', ax=ax, markersize=3); ax.set_title(f'Normal Q-Q: {c}', fontsize=10)
axs[-1].axis('off'); fig.tight_layout(); fig.savefig('figures/1_qq_plots.png', dpi=200); plt.close()

fig, axs = plt.subplots(2, 3, figsize=(11, 6)); axs = axs.ravel()
for ax, c in zip(axs, df):
    ax.hist(df[c], bins=30, density=True, alpha=.7)
    x = np.linspace(df[c].min(), df[c].max(), 200); ax.plot(x, stats.norm.pdf(x, df[c].mean(), df[c].std()), 'k')
    ax.set_title(c, fontsize=10)
axs[-1].axis('off'); fig.tight_layout(); fig.savefig('figures/1_histograms.png', dpi=200); plt.close()
print('Saved tables/0_normality.csv, figures/1_qq_plots.png, figures/1_histograms.png')
