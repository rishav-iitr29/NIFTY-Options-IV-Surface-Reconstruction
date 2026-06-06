import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

df = pd.read_csv('dataset.csv')
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
cols = [c for c in df.columns if c.startswith('NIFTY')]

calls = sorted([c for c in cols if c.endswith('CE')], key=lambda x: int(x[12:-2]))
puts = sorted([c for c in cols if c.endswith('PE')], key=lambda x: int(x[12:-2]))
call_strikes = np.array([int(c[12:-2]) for c in calls])
put_strikes = np.array([int(c[12:-2]) for c in puts])

filled = df.copy()
for idx in range(len(df)):
    for group, strikes in [(calls, call_strikes), (puts, put_strikes)]:
        values = df.loc[idx, group].to_numpy(dtype=float)
        missing = np.isnan(values)
        if not missing.any():
            continue
        observed = ~missing
        if observed.sum() >= 2:
            interp = PchipInterpolator(strikes[observed], values[observed], extrapolate=True)
            filled.loc[idx, np.array(group)[missing]] = np.clip(interp(strikes[missing]), 0.01, 2.0)

still_missing_before = filled[cols].isna().sum().sum()
filled[cols] = filled[cols].ffill().bfill()
still_missing_after = filled[cols].isna().sum().sum()

original_missing = df[cols].isna()
rows = []
for col in cols:
    for idx in df.index[original_missing[col]]:
        dt = df.at[idx, 'datetime'].strftime('%d-%m-%Y %H:%M')
        rows.append({'id': f"{dt}||{col}", 'value': filled.at[idx, col]})

submission = pd.DataFrame(rows, columns=['id', 'value'])
submission.to_csv('submission_spline.csv', index=False)
print('Pipeline complete. Saved as submission_spline.csv')