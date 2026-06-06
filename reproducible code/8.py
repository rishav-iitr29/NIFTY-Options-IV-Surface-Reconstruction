import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

df = pd.read_csv('dataset.csv')
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
cols = [c for c in df if c.startswith('NIFTY')]
calls = sorted([c for c in cols if c.endswith('CE')], key=lambda x: int(x[12:-2]))
puts = sorted([c for c in cols if c.endswith('PE')], key=lambda x: int(x[12:-2]))
call_strikes = np.array([int(c[12:-2]) for c in calls])
put_strikes = np.array([int(c[12:-2]) for c in puts])
expiry_mask = df['datetime'].dt.date == pd.Timestamp('2026-01-27').date()
original_missing = df[cols].isna()
groups = [(calls, call_strikes), (puts, put_strikes)]

filled = df.copy()
for idx in df.index[~expiry_mask]:
    for group, strikes in groups:
        vals = df.loc[idx, group].to_numpy(dtype=float)
        mask = np.isnan(vals)
        if not mask.any():
            continue
        obs = ~mask
        if obs.sum() >= 2:
            filled.loc[idx, np.array(group)[mask]] = np.clip(
                PchipInterpolator(strikes[obs], vals[obs], extrapolate=True)(strikes[mask]),
                0.005,
                10.0,
            )
for idx in df.index[expiry_mask]:
    for group in [calls, puts]:
        vals = df.loc[idx, group].to_numpy(dtype=float)
        mask = np.isnan(vals)
        if mask.any():
            filled.loc[idx, np.array(group)[mask]] = np.nanmedian(vals)
filled[cols] = filled[cols].ffill().bfill()

records = [
    {'id': f"{df.at[row_idx, 'datetime'].strftime('%d-%m-%Y %H:%M')}||{col}", 'value': filled.at[row_idx, col]}
    for col in cols
    for row_idx in df.index[original_missing[col]]
]
pd.DataFrame(records).to_csv('submission_8.csv', index=False)
print('Pipeline complete. Saved as submission_8.csv')