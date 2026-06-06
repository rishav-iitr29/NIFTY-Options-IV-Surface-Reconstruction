import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from catboost import CatBoostRegressor

df = pd.read_csv('dataset.csv')
id_vars = ['datetime', 'underlying_price']
value_vars = [c for c in df if c not in id_vars]
df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='ticker', value_name='IV')
df_long['is_missing_original'] = df_long['IV'].isnull()
df_long['strike'] = df_long['ticker'].str.extract(r'(\d{4,5})(?:CE|PE)').astype(float)
df_long['option_type'] = df_long['ticker'].str.extract(r'(CE|PE)')
df_long['is_call'] = (df_long['option_type'] == 'CE').astype(int)
df_long['datetime_obj'] = pd.to_datetime(df_long['datetime'], format='%d-%m-%Y %H:%M')
df_long['date'] = df_long['datetime_obj'].dt.date
df_long['expiry_date_obj'] = (
    pd.to_datetime(df_long['ticker'].str.extract(r'NIFTY(\d{2}[A-Z]{3}\d{2})')[0], format='%d%b%y')
    + pd.Timedelta(hours=15, minutes=30)
)
df_long['minutes_to_expiry'] = (df_long['expiry_date_obj'] - df_long['datetime_obj']).dt.total_seconds() / 60.0
df_long['moneyness'] = df_long['strike'] / df_long['underlying_price']
df_long['is_round_strike'] = (df_long['strike'] % 500 == 0).astype(int)
df_long = df_long.sort_values(['datetime_obj', 'strike', 'option_type']).reset_index(drop=True)
expiry_date = pd.to_datetime('2026-01-27').date()

def baseline(group, col):
    observed = group.dropna(subset=[col]).sort_values('strike')
    missing = group[group[col].isnull()]
    if len(observed) < 2 or missing.empty:
        return group
    x_obs = observed['strike'].to_numpy()
    y_obs = observed[col].to_numpy()
    if len(observed) >= 3 and group['date'].iloc[0] != expiry_date:
        preds = CubicSpline(x_obs, y_obs, bc_type='natural', extrapolate=True)(missing['strike'].to_numpy())
    else:
        preds = interp1d(x_obs, y_obs, kind='linear', fill_value='extrapolate')(missing['strike'].to_numpy())
    group.loc[missing.index, 'Spline_IV'] = np.clip(preds, 0.0001, 5.0)
    return group

np.random.seed(42)
known = ~df_long['is_missing_original']
mask = np.random.choice(df_long[known].index, size=int(known.sum() * 0.2), replace=False)
df_long['is_train_masked'] = False
df_long.loc[mask, 'is_train_masked'] = True
df_long['IV_for_spline'] = df_long['IV']
df_long.loc[mask, 'IV_for_spline'] = np.nan
df_long['Spline_IV'] = df_long['IV_for_spline']
df_long = df_long.groupby(['datetime_obj', 'option_type'], group_keys=False).apply(lambda g: baseline(g, 'IV_for_spline'))
df_long = df_long.sort_values(['ticker', 'datetime_obj']).reset_index(drop=True)
df_long['Spline_IV'] = df_long.groupby('ticker')['Spline_IV'].ffill().bfill()
df_long = df_long.sort_values(['datetime_obj', 'strike', 'option_type']).reset_index(drop=True)
train_df = df_long[df_long['is_train_masked']].copy()
train_df['Target_Residual'] = train_df['IV'] - train_df['Spline_IV']
features = ['moneyness', 'minutes_to_expiry', 'is_call', 'is_round_strike', 'Spline_IV']
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.02,
    depth=5,
    l2_leaf_reg=5,
    loss_function='RMSE',
    random_seed=42,
    verbose=False,
)
model.fit(train_df[features], train_df['Target_Residual'])
df_long['IV_full_spline'] = df_long['IV']
df_long['Spline_IV'] = df_long['IV']
df_long = df_long.groupby(['datetime_obj', 'option_type'], group_keys=False).apply(lambda g: baseline(g, 'IV_full_spline'))
df_long = df_long.sort_values(['ticker', 'datetime_obj']).reset_index(drop=True)
df_long['Spline_IV'] = df_long.groupby('ticker')['Spline_IV'].ffill().bfill()
df_long = df_long.sort_values(['datetime_obj', 'strike', 'option_type']).reset_index(drop=True)
test_df = df_long[df_long['is_missing_original']].copy()
test_df['Predicted_Residual'] = model.predict(test_df[features])
test_df['Final_IV'] = np.clip(test_df['Spline_IV'] + test_df['Predicted_Residual'], 0.0001, 5.0)
test_df['id'] = test_df['datetime'] + '||' + test_df['ticker']
test_df[['id', 'Final_IV']].rename(columns={'Final_IV': 'value'}).to_csv('submission_9.csv', index=False)
print('Pipeline complete. Saved as submission_9.csv')