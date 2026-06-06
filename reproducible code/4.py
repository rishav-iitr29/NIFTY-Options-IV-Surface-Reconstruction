import pandas as pd
import numpy as np
from scipy.interpolate import pchip_interpolate, interp1d


df = pd.read_csv('dataset.csv')
id_vars = ['datetime', 'underlying_price']
value_vars = [c for c in df.columns if c not in id_vars]

df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='ticker', value_name='IV')
df_long['is_missing'] = df_long['IV'].isnull()
df_long['strike'] = df_long['ticker'].str.extract(r'(\d{4,5})(?:CE|PE)').astype(float)
df_long['option_type'] = df_long['ticker'].str.extract(r'(CE|PE)')
df_long['datetime_obj'] = pd.to_datetime(df_long['datetime'], format='%d-%m-%Y %H:%M')
df_long['date'] = df_long['datetime_obj'].dt.date
df_long = df_long.sort_values(['datetime_obj', 'strike', 'option_type']).reset_index(drop=True)
df_long['predicted_IV'] = df_long['IV']

def apply_normal_quant(group):
    observed = group.dropna(subset=['IV']).sort_values('strike')
    missing = group[group['IV'].isnull()]
    if len(observed) < 2 or missing.empty:
        return group
    strikes = observed['strike'].values
    group.loc[missing.index, 'predicted_IV'] = [
        observed.iloc[0]['IV'] if s < strikes[0]
        else observed.iloc[-1]['IV'] if s > strikes[-1]
        else pchip_interpolate(strikes, observed['IV'].values, s)
        for s in missing['strike'].values
    ]
    return group


def apply_expiry_quant(group):
    observed = group.dropna(subset=['IV']).sort_values('strike')
    missing = group[group['IV'].isnull()]
    if len(observed) < 2 or missing.empty:
        return group
    interp = interp1d(observed['strike'].values, observed['IV'].values, kind='linear', fill_value='extrapolate')
    group.loc[missing.index, 'predicted_IV'] = np.clip(interp(missing['strike'].values), 0.0001, 4.0)
    return group

expiry_date = pd.to_datetime('2026-01-27').date()
mask_normal = df_long['date'] < expiry_date
df_long.loc[mask_normal, 'predicted_IV'] = df_long[mask_normal].groupby(['datetime_obj', 'option_type'], group_keys=False).apply(apply_normal_quant)['predicted_IV']

mask_expiry = df_long['date'] == expiry_date
df_long.loc[mask_expiry, 'predicted_IV'] = df_long[mask_expiry].groupby(['datetime_obj', 'option_type'], group_keys=False).apply(apply_expiry_quant)['predicted_IV']

df_long = df_long.sort_values(['ticker', 'datetime_obj']).reset_index(drop=True)
df_long['predicted_IV'] = df_long.groupby('ticker')['predicted_IV'].ffill()
df_long['predicted_IV'] = df_long.groupby('ticker')['predicted_IV'].bfill()

submission_df = df_long[df_long['is_missing']].copy()
submission_df['id'] = submission_df['datetime'] + '||' + submission_df['ticker']
submission_df[['id', 'predicted_IV']].rename(columns={'predicted_IV': 'value'}).to_csv('submission_4.csv', index=False)

print('Pipeline complete. Saved as submission_4.csv')
