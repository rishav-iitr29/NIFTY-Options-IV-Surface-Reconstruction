import pandas as pd
from scipy.interpolate import pchip_interpolate

df = pd.read_csv('dataset.csv')
id_vars = ['datetime', 'underlying_price']
value_vars = [c for c in df.columns if c not in id_vars]

df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='ticker', value_name='IV')
df_long['is_missing'] = df_long['IV'].isnull()
df_long['strike'] = df_long['ticker'].str.extract(r'(\d{4,5})(?:CE|PE)').astype(float)
df_long['option_type'] = df_long['ticker'].str.extract(r'(CE|PE)')
df_long['datetime_obj'] = pd.to_datetime(df_long['datetime'], format='%d-%m-%Y %H:%M')
df_long = df_long.sort_values(['datetime_obj', 'strike', 'option_type']).reset_index(drop=True)

def interpolate_section(group):
    observed = group.dropna(subset=['IV']).sort_values('strike')
    if len(observed) < 2:
        return group
    missing = group[group['IV'].isnull()]
    if missing.empty:
        return group
    strikes = observed['strike'].values
    ivs = observed['IV'].values
    group.loc[missing.index, 'predicted_IV'] = [
        observed.iloc[0]['IV'] if s < strikes[0]
        else observed.iloc[-1]['IV'] if s > strikes[-1]
        else pchip_interpolate(strikes, ivs, s)
        for s in missing['strike'].values
    ]
    return group

df_long['predicted_IV'] = df_long['IV']
df_long = df_long.groupby(['datetime_obj', 'option_type'], group_keys=False).apply(interpolate_section)
df_long = df_long.sort_values(['ticker', 'datetime_obj']).reset_index(drop=True)
df_long['predicted_IV'] = df_long.groupby('ticker')['predicted_IV'].ffill()
df_long['predicted_IV'] = df_long.groupby('ticker')['predicted_IV'].bfill()

submission_df = df_long[df_long['is_missing']].copy()
submission_df['id'] = submission_df['datetime'] + '||' + submission_df['ticker']
submission_df[['id', 'predicted_IV']].rename(columns={'predicted_IV': 'value'}).to_csv('submission_quant_final.csv', index=False)
print('Pipeline complete. Saved as submission_quant_final.csv')
