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
df_long['log_IV'] = np.log(df_long['IV'] + 1e-8)

def normal_log_spline(group):
    observed = group.dropna(subset=['log_IV_simulated']).sort_values('strike')
    missing = group[group['log_IV_simulated'].isnull()]
    if len(observed) < 2 or missing.empty:
        return group
    x_obs = observed['strike'].to_numpy()
    y_obs = observed['log_IV_simulated'].to_numpy()
    if len(observed) >= 3:
        preds = CubicSpline(x_obs, y_obs, bc_type='natural', extrapolate=True)(missing['strike'].to_numpy())
    else:
        preds = interp1d(x_obs, y_obs, kind='linear', fill_value='extrapolate')(missing['strike'].to_numpy())
    group.loc[missing.index, 'predicted_log_IV'] = preds
    return group

def expiry_log_linear(group):
    observed = group.dropna(subset=['log_IV_simulated']).sort_values('strike')
    missing = group[group['log_IV_simulated'].isnull()]
    if len(observed) < 2 or missing.empty:
        return group
    preds = interp1d(observed['strike'].to_numpy(), observed['log_IV_simulated'].to_numpy(), kind='linear', fill_value='extrapolate')(missing['strike'].to_numpy())
    group.loc[missing.index, 'predicted_log_IV'] = preds
    return group

unique_dates = sorted(df_long['date'].unique())
val_dates = unique_dates[-5:]
cv_scores = []
np.random.seed(42)
for val_date in val_dates:
    fold = df_long[df_long['date'] == val_date].copy()
    known = fold[~fold['is_missing_original']].index
    mask = np.random.choice(known, size=int(len(known) * 0.2), replace=False)
    fold['log_IV_simulated'] = fold['log_IV']
    fold.loc[mask, 'log_IV_simulated'] = np.nan
    fold['predicted_log_IV'] = fold['log_IV_simulated']
    fn = normal_log_spline if val_date < expiry_date else expiry_log_linear
    fold = fold.groupby(['datetime_obj', 'option_type'], group_keys=False).apply(fn)
    fold['predicted_IV'] = np.exp(fold['predicted_log_IV']) - 1e-8
    preds = fold.loc[mask, 'predicted_IV']
    truths = fold.loc[mask, 'IV']
    mse = mean_squared_error(truths[~preds.isnull()], preds[~preds.isnull()])
    cv_scores.append(mse)
    print(f"Fold {val_date} (Regime: {'Normal' if val_date < expiry_date else 'Expiry'}): Local MSE = {mse:.6f}")
print(f"=== FINAL LOCAL CV MSE: {np.mean(cv_scores):.6f} ===")

df_long['log_IV_simulated'] = df_long['log_IV']
df_long['predicted_log_IV'] = df_long['log_IV']
mask_normal = df_long['date'] < expiry_date
df_long.loc[mask_normal, 'predicted_log_IV'] = df_long[mask_normal].groupby(['datetime_obj', 'option_type'], group_keys=False).apply(normal_log_spline)['predicted_log_IV']
mask_expiry = df_long['date'] == expiry_date
df_long.loc[mask_expiry, 'predicted_log_IV'] = df_long[mask_expiry].groupby(['datetime_obj', 'option_type'], group_keys=False).apply(expiry_log_linear)['predicted_log_IV']
df_long['predicted_IV'] = np.exp(df_long['predicted_log_IV']) - 1e-8
df_long = df_long.sort_values(['ticker', 'datetime_obj']).reset_index(drop=True)
df_long['predicted_IV'] = df_long.groupby('ticker')['predicted_IV'].ffill().bfill()
submission = df_long[df_long['is_missing_original']].copy()
submission['id'] = submission['datetime'] + '||' + submission['ticker']
submission[['id', 'predicted_IV']].rename(columns={'predicted_IV': 'value'}).to_csv('submission_10.csv', index=False)
print('Pipeline complete. Saved as submission_10.csv')