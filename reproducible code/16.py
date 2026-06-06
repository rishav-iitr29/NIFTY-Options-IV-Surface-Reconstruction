import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import warnings
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

df = pd.read_csv('dataset.csv')
id_vars = ['datetime', 'underlying_price']
value_vars = [col for col in df.columns if col not in id_vars]

df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='ticker', value_name='IV')
df_long['is_missing_original'] = df_long['IV'].isnull()

df_long['strike'] = df_long['ticker'].str.extract(r'(\d{4,5})(?:CE|PE)').astype(float)
df_long['option_type'] = df_long['ticker'].str.extract(r'(CE|PE)')
df_long['datetime_obj'] = pd.to_datetime(df_long['datetime'], format='%d-%m-%Y %H:%M')
df_long['date'] = df_long['datetime_obj'].dt.date

df_long = df_long.sort_values(by=['datetime_obj', 'strike', 'option_type']).reset_index(drop=True)
expiry_date = pd.to_datetime('2026-01-27').date()


# 2. Regime 1: Normal Days (Natural Cubic Spline)
def apply_normal_spline(group, target_col):
    observed = group.dropna(subset=[target_col])
    missing = group[group[target_col].isnull()]

    if len(observed) >= 2 and len(missing) > 0:
        observed = observed.sort_values('strike')
        x_obs = observed['strike'].values
        y_obs = observed[target_col].values
        x_miss = missing['strike'].values

        if len(observed) >= 3:
            # Natural spline preserves the U-shape and extrapolates linearly
            cs = CubicSpline(x_obs, y_obs, bc_type='natural', extrapolate=True)
            preds = cs(x_miss)
        else:
            lin = interp1d(x_obs, y_obs, kind='linear', fill_value='extrapolate')
            preds = lin(x_miss)

        # Standard clip for normal days
        group.loc[missing.index, 'predicted_IV'] = np.clip(preds, 0.0001, 5.0)
    return group


# 3. Regime 2: Expiry Day (Linear Interpolation with Expanded Clip)
def apply_expiry_linear(group, target_col):
    observed = group.dropna(subset=[target_col])
    missing = group[group[target_col].isnull()]

    if len(observed) >= 2 and len(missing) > 0:
        observed = observed.sort_values('strike')
        x_obs = observed['strike'].values
        y_obs = observed[target_col].values
        x_miss = missing['strike'].values

        # Pure Linear Interpolation for the V-Shape peak
        lin = interp1d(x_obs, y_obs, kind='linear', fill_value='extrapolate')
        preds = lin(x_miss)

        # OPTIMIZATION: Expanded clip to 100.0 allows valid deep OTM IV spikes
        group.loc[missing.index, 'predicted_IV'] = np.clip(preds, 0.0001, 100.0)
    return group


# 4. Apply Regimes
df_long['predicted_IV'] = df_long['IV']

mask_normal = df_long['date'] < expiry_date
df_normal = df_long[mask_normal].groupby(['datetime_obj', 'option_type'], group_keys=False).apply(
    lambda g: apply_normal_spline(g, 'IV'))
df_long.loc[mask_normal, 'predicted_IV'] = df_normal['predicted_IV']

mask_expiry = df_long['date'] == expiry_date
df_expiry = df_long[mask_expiry].groupby(['datetime_obj', 'option_type'], group_keys=False).apply(
    lambda g: apply_expiry_linear(g, 'IV'))
df_long.loc[mask_expiry, 'predicted_IV'] = df_expiry['predicted_IV']

# 5. Temporal Safety Net (Time-series continuity for total chain failures)
df_long = df_long.sort_values(by=['ticker', 'datetime_obj']).reset_index(drop=True)
df_long['predicted_IV'] = df_long.groupby('ticker')['predicted_IV'].ffill()
df_long['predicted_IV'] = df_long.groupby('ticker')['predicted_IV'].bfill()

# 6. Format Submission
submission_df = df_long[df_long['is_missing_original'] == True].copy()
submission_df['id'] = submission_df['datetime'] + '||' + submission_df['ticker']
submission_df = submission_df[['id', 'predicted_IV']].rename(columns={'predicted_IV': 'value'})

submission_df.to_csv('submission_16.csv', index=False)
print('Pipeline complete. Saved as submission_16.csv')
