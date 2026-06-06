import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.interpolate import pchip_interpolate

df = pd.read_csv('dataset.csv')
id_vars = ['datetime', 'underlying_price']
value_vars = [c for c in df if c not in id_vars]

df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='ticker', value_name='IV')
df['is_missing_original'] = df['IV'].isnull()
df['expiry_str'] = df['ticker'].str.extract(r'NIFTY(\d{2}[A-Z]{3}\d{2})')
df['strike'] = df['ticker'].str.extract(r'(\d{4,5})(?:CE|PE)').astype(float)
df['option_type'] = df['ticker'].str.extract(r'(CE|PE)')
df['is_call'] = (df['option_type'] == 'CE').astype(int)

df['datetime_obj'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M')
df['expiry_date'] = pd.to_datetime(df['expiry_str'], format='%d%b%y') + pd.Timedelta(hours=15, minutes=30)
df['minutes_to_expiry'] = (df['expiry_date'] - df['datetime_obj']).dt.total_seconds() / 60

df['moneyness'] = df['strike'] / df['underlying_price']
df['log_moneyness'] = np.log(df['moneyness'])
df = df.sort_values(['datetime_obj', 'strike', 'option_type']).reset_index(drop=True)

def interpolate(group, col):
    observed = group.dropna(subset=[col]).sort_values('strike')
    if len(observed) >= 2:
        missing = group[group[col].isnull()]
        group.loc[missing.index, col] = pchip_interpolate(
            observed['strike'].values,
            observed[col].values,
            missing['strike'].values,
        )
    return group

df['IV_baseline'] = df['IV']
df = df.groupby(['datetime_obj', 'option_type'], group_keys=False).apply(lambda g: interpolate(g, 'IV_baseline'))
df['IV_baseline'] = df.groupby(['strike', 'option_type'])['IV_baseline'].ffill()
df['IV_baseline'] = df['IV_baseline'].fillna(df.groupby(['datetime_obj', 'option_type'])['IV_baseline'].transform('mean'))

np.random.seed(42)
known = df[~df['is_missing_original']].index
mask = np.random.choice(known, size=int(len(known) * 0.2), replace=False)

df['IV_for_training'] = df['IV']
df.loc[mask, 'IV_for_training'] = np.nan

df['IV_baseline_masked'] = df['IV_for_training']
df = df.groupby(['datetime_obj', 'option_type'], group_keys=False).apply(lambda g: interpolate(g, 'IV_baseline_masked'))
df['IV_baseline_masked'] = df.groupby(['strike', 'option_type'])['IV_baseline_masked'].ffill()
df['IV_baseline_masked'] = df['IV_baseline_masked'].fillna(df.groupby(['datetime_obj', 'option_type'])['IV_baseline_masked'].transform('mean'))

train = df.loc[df.index.isin(mask)]
features = ['IV_baseline_masked', 'strike', 'moneyness', 'log_moneyness', 'minutes_to_expiry', 'is_call']

model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=5,
    num_leaves=20,
    random_state=42,
    n_jobs=-1,
)
model.fit(train[features], train['IV'])

test = df.loc[df['is_missing_original']].copy()
test['IV_baseline_masked'] = test['IV_baseline']
test['predicted_IV'] = model.predict(test[features])

submission = test.assign(id=test['datetime'] + '||' + test['ticker'])[['id', 'predicted_IV']].rename(columns={'predicted_IV': 'value'})
submission.to_csv('submission_.csv', index=False)
print('Pipeline complete. Saved as submission_.csv')