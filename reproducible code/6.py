import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from sklearn.metrics import mean_squared_error


df = pd.read_csv('dataset.csv')
id_vars = ['datetime', 'underlying_price']
value_vars = [c for c in df if c not in id_vars]

df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='ticker', value_name='IV')
df_long['is_missing_original'] = df_long['IV'].isnull()
df_long['strike'] = df_long['ticker'].str.extract(r'(\d{4,5})(?:CE|PE)').astype(float)
df_long['option_type'] = df_long['ticker'].str.extract(r'(CE|PE)')
df_long['datetime_obj'] = pd.to_datetime(df_long['datetime'], format='%d-%m-%Y %H:%M')
df_long['date'] = df_long['datetime_obj'].dt.date
df_long = df_long.sort_values(['datetime_obj', 'strike', 'option_type']).reset_index(drop=True)
expiry_date = pd.to_datetime('2026-01-27').date()

def apply_normal_spline(group, col):
    observed = group.dropna(subset=[col]).sort_values('strike')
    missing = group[group[col].isnull()]
    if len(observed) < 2 or missing.empty:
        return group
    x_obs = observed['strike'].values
    y_obs = observed[col].values
    if len(observed) >= 3:
        preds = CubicSpline(x_obs, y_obs, bc_type='natural', extrapolate=True)(missing['strike'].values)
    else:
        preds = interp1d(x_obs, y_obs, kind='linear', fill_value='extrapolate')(missing['strike'].values)
    group.loc[missing.index, 'predicted_IV'] = np.clip(preds, 0.0001, 5.0)
    return group

def apply_expiry_linear(group, col):
    observed = group.dropna(subset=[col]).sort_values('strike')
    missing = group[group[col].isnull()]
    if len(observed) < 2 or missing.empty:
        return group
    preds = interp1d(observed['strike'].values, observed[col].values, kind='linear', fill_value='extrapolate')(missing['strike'].values)
    group.loc[missing.index, 'predicted_IV'] = np.clip(preds, 0.0001, 15.0)
    return group

unique_dates = sorted(df_long['date'].unique())
val_dates = unique_dates[-5:]
cv_scores = []
np.random.seed(42)
for val_date in val_dates:
    fold = df_long[df_long['date'] == val_date].copy()
    known = fold[~fold['is_missing_original']].index
    mask = np.random.choice(known, size=int(len(known) * 0.2), replace=False)
    fold['IV_simulated'] = fold['IV']
    fold.loc[mask, 'IV_simulated'] = np.nan
    fold['predicted_IV'] = fold['IV_simulated']
    fn = apply_normal_spline if val_date < expiry_date else apply_expiry_linear
    fold = fold.groupby(['datetime_obj', 'option_type'], group_keys=False).apply(lambda g: fn(g, 'IV_simulated'))
    preds = fold.loc[mask, 'predicted_IV']
    truths = fold.loc[mask, 'IV']
    mse = mean_squared_error(truths[~preds.isnull()], preds[~preds.isnull()])
    cv_scores.append(mse)
df_long['predicted_IV'] = df_long['IV']
mask_normal = df_long['date'] < expiry_date
df_long.loc[mask_normal, 'predicted_IV'] = df_long[mask_normal].groupby(['datetime_obj', 'option_type'], group_keys=False).apply(lambda g: apply_normal_spline(g, 'IV'))['predicted_IV']
mask_expiry = df_long['date'] == expiry_date
df_long.loc[mask_expiry, 'predicted_IV'] = df_long[mask_expiry].groupby(['datetime_obj', 'option_type'], group_keys=False).apply(lambda g: apply_expiry_linear(g, 'IV'))['predicted_IV']
df_long = df_long.sort_values(['ticker', 'datetime_obj']).reset_index(drop=True)
df_long['predicted_IV'] = df_long.groupby('ticker')['predicted_IV'].ffill().bfill()
submission = df_long[df_long['is_missing_original']].copy()
submission['id'] = submission['datetime'] + '||' + submission['ticker']
submission[['id', 'predicted_IV']].rename(columns={'predicted_IV': 'value'}).to_csv('submission_6.csv', index=False)
print('Pipeline Complete! Saved: submission_6.csv')
