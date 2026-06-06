import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline, interp1d, Akima1DInterpolator
from sklearn.metrics import mean_squared_error

df = pd.read_csv('dataset.csv')
id_vars = ['datetime', 'underlying_price']
cols = [c for c in df if c not in id_vars]
df_long = df.melt(id_vars=id_vars, value_vars=cols, var_name='ticker', value_name='IV')
df_long['is_missing_original'] = df_long['IV'].isnull()
df_long['strike'] = df_long['ticker'].str.extract(r'(\d{4,5})(?:CE|PE)').astype(float)
df_long['option_type'] = df_long['ticker'].str.extract(r'(CE|PE)')
df_long['datetime_obj'] = pd.to_datetime(df_long['datetime'], format='%d-%m-%Y %H:%M')
df_long['date'] = df_long['datetime_obj'].dt.date
df_long = df_long.sort_values(['datetime_obj', 'strike', 'option_type']).reset_index(drop=True)
near_expiry_date = pd.to_datetime('2026-01-23').date()
expiry_date = pd.to_datetime('2026-01-27').date()

def interpolate_group(group, col, method):
    observed = group.dropna(subset=[col]).sort_values('strike')
    missing = group[group[col].isnull()]
    if len(observed) < 2 or missing.empty:
        return group
    x_obs = observed['strike'].to_numpy()
    y_obs = observed[col].to_numpy()
    if method == 'log':
        y_obs = np.log(y_obs + 1e-8)
    x_miss = missing['strike'].to_numpy()
    if method == 'akima' and len(observed) >= 4:
        akima = Akima1DInterpolator(x_obs, y_obs)
        lin = interp1d(x_obs, y_obs, kind='linear', fill_value='extrapolate')
        min_k, max_k = x_obs[0], x_obs[-1]
        preds = np.array([akima(s) if min_k <= s <= max_k else lin(s) for s in x_miss])
    else:
        model = CubicSpline(x_obs, y_obs, bc_type='natural', extrapolate=True) if len(observed) >= 3 else interp1d(x_obs, y_obs, kind='linear', fill_value='extrapolate')
        preds = model(x_miss)
    if method == 'log':
        preds = np.exp(preds) - 1e-8
    group.loc[missing.index, 'predicted_IV'] = np.clip(preds, 0.0001, 5.0)
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
    method = 'log' if val_date < near_expiry_date else 'linear' if val_date == near_expiry_date else 'akima'
    fold = fold.groupby(['datetime_obj', 'option_type'], group_keys=False).apply(lambda g: interpolate_group(g, 'IV_simulated', method))
    preds = fold.loc[mask, 'predicted_IV']
    truths = fold.loc[mask, 'IV']
    mse = mean_squared_error(truths[~preds.isnull()], preds[~preds.isnull()])
    cv_scores.append(mse)
    regime = 'Far Expiry' if method == 'log' else 'Near Expiry' if method == 'linear' else 'Expiry Day'
    print(f"Fold {val_date} ({regime}): Local MSE = {mse:.6f}")
print(f"=== FINAL LOCAL CV MSE: {np.mean(cv_scores):.6f} ===")

df_long['predicted_IV'] = df_long['IV']
mask_far = df_long['date'] < near_expiry_date
df_long.loc[mask_far, 'predicted_IV'] = df_long[mask_far].groupby(['datetime_obj', 'option_type'], group_keys=False).apply(lambda g: interpolate_group(g, 'IV', 'log'))['predicted_IV']
mask_near = df_long['date'] == near_expiry_date
df_long.loc[mask_near, 'predicted_IV'] = df_long[mask_near].groupby(['datetime_obj', 'option_type'], group_keys=False).apply(lambda g: interpolate_group(g, 'IV', 'linear'))['predicted_IV']
mask_expiry = df_long['date'] == expiry_date
df_long.loc[mask_expiry, 'predicted_IV'] = df_long[mask_expiry].groupby(['datetime_obj', 'option_type'], group_keys=False).apply(lambda g: interpolate_group(g, 'IV', 'akima'))['predicted_IV']
df_long = df_long.sort_values(['ticker', 'datetime_obj']).reset_index(drop=True)
df_long['predicted_IV'] = df_long.groupby('ticker')['predicted_IV'].ffill().bfill()
submission = df_long[df_long['is_missing_original']].copy()
submission['id'] = submission['datetime'] + '||' + submission['ticker']
submission[['id', 'predicted_IV']].rename(columns={'predicted_IV': 'value'}).to_csv('submission_13.csv', index=False)
print('Pipeline complete. Saved as submission_13.csv')