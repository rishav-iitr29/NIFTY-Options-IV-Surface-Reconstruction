import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
import lightgbm as lgb

SEED = 42
ALPHA = 0.70
IV_CLIP_LO = 0.005
IV_CLIP_HI = 10.0
EXPIRY_DATE = pd.Timestamp('2026-01-27').date()
EXPIRY_CLOSE = pd.Timestamp('2026-01-27 15:30')

df = pd.read_csv('dataset.csv')
df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
cols = [c for c in df.columns if c.startswith('NIFTY')]
calls = sorted([c for c in cols if c.endswith('CE')], key=lambda x: int(x[12:-2]))
puts = sorted([c for c in cols if c.endswith('PE')], key=lambda x: int(x[12:-2]))
call_strikes = np.array([int(c[12:-2]) for c in calls])
put_strikes = np.array([int(c[12:-2]) for c in puts])
df_s = df.sort_values('datetime').reset_index(drop=True)
is_exp = (df_s['datetime'].dt.date == EXPIRY_DATE).to_numpy()
original_missing = df[cols].isna()

filled_s = df_s.copy()
for i in df_s.index[~is_exp]:
    for group, strikes in [(calls, call_strikes), (puts, put_strikes)]:
        vals = df_s.loc[i, group].to_numpy(dtype=float)
        mask = np.isnan(vals)
        if mask.sum() + 1 < len(vals):
            interp = PchipInterpolator(strikes[~mask], vals[~mask], extrapolate=True)
            filled_s.loc[i, np.array(group)[mask]] = np.clip(interp(strikes[mask]), IV_CLIP_LO, IV_CLIP_HI)
for i in df_s.index[is_exp]:
    for group in [calls, puts]:
        vals = df_s.loc[i, group].to_numpy(dtype=float)
        mask = np.isnan(vals)
        if mask.any():
            filled_s.loc[i, np.array(group)[mask]] = np.nanmedian(vals)
filled_s[cols] = filled_s[cols].ffill().bfill()

loo_spline = pd.DataFrame(np.nan, index=df_s.index, columns=cols)
reg_spline = pd.DataFrame(np.nan, index=df_s.index, columns=cols)
for i in df_s.index[~is_exp]:
    for group, strikes in [(calls, call_strikes), (puts, put_strikes)]:
        vals = df_s.loc[i, group].to_numpy(dtype=float)
        obs = ~np.isnan(vals)
        if obs.sum() < 2:
            continue
        interp_full = PchipInterpolator(strikes[obs], vals[obs], extrapolate=True)
        miss = np.isnan(vals)
        if miss.any():
            reg_spline.loc[i, np.array(group)[miss]] = np.clip(interp_full(strikes[miss]), IV_CLIP_LO, IV_CLIP_HI)
        if obs.sum() < 3:
            continue
        for j in np.where(obs)[0]:
            m = obs.copy(); m[j] = False
            interp_loo = PchipInterpolator(strikes[m], vals[m], extrapolate=True)
            loo_spline.loc[i, group[j]] = float(np.clip(interp_loo(strikes[j]), IV_CLIP_LO, IV_CLIP_HI))

FEATURE_COLS = [
    'moneyness', 'log_moneyness', 'dist_atm', 'ttm_min', 'ttm_days',
    'hour', 'session_progress', 'otype_enc',
    'iv_spline',
    'left_iv', 'right_iv', 'left2_iv', 'right2_iv',
    'slope', 'curvature',
    'cross_mean', 'cross_med', 'cross_std',
    'lag1', 'lag3', 'lag6', 'lag12', 'lag24',
    'roll_mean5', 'roll_std5', 'roll_mean12', 'roll_std12',
    'underlying_ret1',
]

def build_row(i, col, group, k, strikes_arr, spline_pred):
    dt = df_s.loc[i, 'datetime']
    uprice = df_s.loc[i, 'underlying_price']
    ttm_min = (EXPIRY_CLOSE - dt).total_seconds() / 60
    fr = filled_s.loc[i, group].to_numpy(dtype=float)
    ts = filled_s[col].to_numpy(dtype=float)
    left_iv = fr[k - 1] if k > 0 else fr[k]
    right_iv = fr[k + 1] if k < len(group) - 1 else fr[k]
    left2_iv = fr[k - 2] if k > 1 else fr[k]
    right2_iv = fr[k + 2] if k < len(group) - 2 else fr[k]
    slope = (fr[k + 1] - fr[k - 1]) / 200.0 if 0 < k < len(group) - 1 else 0.0
    curvature = (fr[k + 1] - 2 * fr[k] + fr[k - 1]) / 1e4 if 0 < k < len(group) - 1 else 0.0
    lag1 = ts[i - 1] if i >= 1 else ts[i]
    lag3 = ts[i - 3] if i >= 3 else ts[i]
    lag6 = ts[i - 6] if i >= 6 else ts[i]
    lag12 = ts[i - 12] if i >= 12 else ts[i]
    lag24 = ts[i - 24] if i >= 24 else ts[i]
    win5 = ts[max(0, i - 5):i]
    win12 = ts[max(0, i - 12):i]
    uprice_lag1 = df_s.loc[i - 1, 'underlying_price'] if i >= 1 else uprice
    return {
        'moneyness': float(strikes_arr[k]) / uprice,
        'log_moneyness': np.log(float(strikes_arr[k]) / uprice),
        'dist_atm': float(strikes_arr[k]) - uprice,
        'ttm_min': ttm_min,
        'ttm_days': ttm_min / 375,
        'hour': dt.hour + dt.minute / 60,
        'session_progress': (dt.hour * 60 + dt.minute - 555) / 375,
        'otype_enc': 1 if col.endswith('CE') else 0,
        'iv_spline': spline_pred,
        'left_iv': left_iv,
        'right_iv': right_iv,
        'left2_iv': left2_iv,
        'right2_iv': right2_iv,
        'slope': slope,
        'curvature': curvature,
        'cross_mean': fr.mean(),
        'cross_med': np.median(fr),
        'cross_std': fr.std(),
        'lag1': lag1,
        'lag3': lag3,
        'lag6': lag6,
        'lag12': lag12,
        'lag24': lag24,
        'roll_mean5': win5.mean() if len(win5) > 0 else ts[i],
        'roll_std5': win5.std() if len(win5) > 1 else 0.0,
        'roll_mean12': win12.mean() if len(win12) > 0 else ts[i],
        'roll_std12': win12.std() if len(win12) > 1 else 0.0,
        'underlying_ret1': (uprice - uprice_lag1) / uprice_lag1,
    }

train_rows, y_res_list, pred_rows, pred_meta = [], [], [], []
for col in cols:
    group = calls if col.endswith('CE') else puts
    strikes_arr = call_strikes if col.endswith('CE') else put_strikes
    k = list(group).index(col)
    raw = df_s[col].to_numpy(dtype=float)
    for i in df_s.index[~is_exp]:
        actual = raw[i]
        if not np.isnan(actual):
            loo_pred = loo_spline.loc[i, col]
            if pd.isna(loo_pred):
                continue
            row = build_row(i, col, group, k, strikes_arr, float(loo_pred))
            row['iv_spline'] = float(loo_pred)
            train_rows.append(row)
            y_res_list.append(actual - float(loo_pred))
        else:
            reg_pred = reg_spline.loc[i, col]
            if pd.isna(reg_pred):
                continue
            row = build_row(i, col, group, k, strikes_arr, float(reg_pred))
            row['iv_spline'] = float(reg_pred)
            pred_rows.append(row)
            pred_meta.append({'i': i, 'col': col, 'spline': float(reg_pred)})

X_train = pd.DataFrame(train_rows)[FEATURE_COLS]
y_res = np.array(y_res_list)
X_pred = pd.DataFrame(pred_rows)[FEATURE_COLS]

model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.05,
    reg_lambda=0.05,
    random_state=SEED,
    verbose=-1,
)
model.fit(X_train, y_res)

print(pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False).head(15).to_string())

pred_residuals = model.predict(X_pred)
final_filled = df_s.copy()
for record, res_pred in zip(pred_meta, pred_residuals):
    i = record['i']
    col = record['col']
    final_filled.loc[i, col] = np.clip(record['spline'] + ALPHA * res_pred, IV_CLIP_LO, IV_CLIP_HI)
for i in df_s.index[is_exp]:
    for group in [calls, puts]:
        vals = df_s.loc[i, group].to_numpy(dtype=float)
        mask = np.isnan(vals)
        if mask.any():
            final_filled.loc[i, np.array(group)[mask]] = np.nanmedian(vals)
final_filled[cols] = final_filled[cols].ffill().bfill()
assert final_filled[cols].isna().sum().sum() == 0

lookup = final_filled.set_index('datetime')
records = [
    {
        'id': f"{dt.strftime('%d-%m-%Y %H:%M')}||{col}",
        'value': lookup.at[dt, col],
    }
    for col in cols
    for dt in df.loc[original_missing[col], 'datetime']
]
pd.DataFrame(records, columns=['id', 'value']).to_csv('submission_15.csv', index=False)
print('Pipeline complete. Saved as submission_15.csv')