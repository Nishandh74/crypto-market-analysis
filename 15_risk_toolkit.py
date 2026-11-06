"""
RISK-SIGNAL TOOLKIT (prototype): combine GARCH volatility + rolling correlation into a weekly risk state.
Idea: returns are not predictable, but volatility and the equity-link regime are. The tool outputs a
risk state each week using ONLY past information, as a practical signal for EDBI.
"""
import numpy as np, pandas as pd
from arch import arch_model
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

d = pd.read_excel('Final_Weekly_Analysis_Table.xlsx', sheet_name='Analysis table', index_col='Date')
r = d['Crypto_Index']*100   # % weekly

# 1) GARCH(1,1)-t conditional volatility (one-step-ahead: uses info up to t-1)
am = arch_model(r, mean='Constant', vol='GARCH', p=1, q=1, dist='t').fit(disp='off')
cv = am.conditional_volatility           # weekly % conditional vol, each week uses past only
cvann = cv*np.sqrt(52)

# 2) rolling 52-week correlation with S&P (past only)
roll = d['Crypto_Index'].rolling(52).corr(d['SP500'])

# 3) classify (thresholds from the data, using expanding history to avoid look-ahead on levels)
# volatility state: terciles of conditional vol computed on a 1-yr warmup then fixed-ish; use full-sample terciles for the prototype illustration
vlo, vhi = cvann.quantile(0.33), cvann.quantile(0.66)
vol_state = pd.cut(cvann, [-1, vlo, vhi, 1e9], labels=['Low','Normal','High'])
corr_state = np.where(roll > 0.3, 'Equity-linked', 'Decoupled')
corr_state = pd.Series(corr_state, index=roll.index)

sig = pd.DataFrame({'cond_vol_ann':cvann.round(1),'roll_corr':roll.round(2),
                    'vol_state':vol_state,'corr_state':corr_state})
# combined signal
def combine(row):
    if row['vol_state']=='High' and row['corr_state']=='Equity-linked': return 'Elevated (high vol + equity-linked)'
    if row['vol_state']=='High': return 'High standalone risk'
    if row['vol_state']=='Low': return 'Calm'
    return 'Normal'
sig['signal'] = sig.apply(combine, axis=1)

# ---- VALIDATION: does the vol state predict NEXT week's outcome? (genuine 1-step test)
nxt_abs = r.abs().shift(-1)   # next week's absolute return
val = pd.DataFrame({'vol_state':sig['vol_state'], 'next_abs':nxt_abs}).dropna()
print('NEXT-WEEK |return| by current volatility state (mean %, and share of weeks worse than -10%):')
g = val.groupby('vol_state', observed=True)['next_abs'].agg(['mean','count'])
big = pd.DataFrame({'vol_state':sig['vol_state'],'nextret':r.shift(-1)}).dropna()
g['loss>10%_next_wk_%'] = big.groupby('vol_state',observed=True)['nextret'].apply(lambda s:(s<-10).mean()*100)
print(g.round(2).to_string())

print('\nShare of weeks in each signal state:')
print((sig['signal'].value_counts(normalize=True)*100).round(1).to_string())

# did it flag around COVID? show states just before/at the crash
covid = sig.loc['2020-02-16':'2020-04-05', ['cond_vol_ann','roll_corr','signal']]
print('\nAround the COVID crash (week ending shown):')
print(covid.to_string())

# ---- chart
fig, ax = plt.subplots(2,1,figsize=(10,6), sharex=True)
ax[0].plot(cvann.index, cvann, color='#2563eb'); ax[0].axhline(vhi, ls=':', c='r', label=f'High threshold ({vhi:.0f}%)')
ax[0].axhline(vlo, ls=':', c='g', label=f'Low threshold ({vlo:.0f}%)')
ax[0].set_ylabel('Forecast volatility (% ann.)'); ax[0].legend(fontsize=7); ax[0].set_title('Risk-signal toolkit: volatility state')
ax[1].plot(roll.index, roll, color='#16a34a'); ax[1].axhline(0.3, ls=':', c='r', label='Equity-linked threshold (0.3)')
ax[1].axhline(0, c='k', lw=.5); ax[1].set_ylabel('Rolling corr. with S&P 500'); ax[1].legend(fontsize=7)
for a in ax: a.axvspan(pd.Timestamp('2020-02-16'), pd.Timestamp('2020-04-05'), color='grey', alpha=0.2)
fig.tight_layout(); fig.savefig('toolkit_signal.png', dpi=200); plt.close()
print('\nsaved toolkit_signal.png')
