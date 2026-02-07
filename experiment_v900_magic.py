"""
V900: THE MAGIC TUNER (The 0.50 Peak)
Constraint: Aug 31, 2025 Cutoff.
Method: 
- Optuna Tuning of the "Magic Numbers" (Decay Cutoff, Alpha, Weights).
- Blended Objective (70% 2024, 30% 2017).
- V149 Hybrid Baseline Backbone.
"""

import pandas as pd
import numpy as np
import warnings
import optuna
from optuna.samplers import TPESampler
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
import random
import os

warnings.filterwarnings('ignore')
SEED = 41
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(SEED)

STATIONS = ['DKI1', 'DKI2', 'DKI3', 'DKI4', 'DKI5']
LAG_DAYS = [1, 2, 7, 30]

def is_hol(d): 
    if d.dayofweek >= 5: return 1
    if (d.month == 1 and d.day == 1 or d.month == 5 and d.day == 1 or d.month == 8 and d.day == 17): return 1
    return 0

def prep():
    all_i, all_w = [], []
    for st in STATIONS:
        ispu_path = f"ispu new/ispu_{st.lower()}.csv"
        di = pd.read_csv(ispu_path)
        di['stasiun'] = st
        di['tanggal'] = pd.to_datetime(di['tanggal'], dayfirst=True, errors='coerce')
        for c in ['pm10', 'pm25', 'so2', 'co']:
            if c in di.columns: di[c] = pd.to_numeric(di[c], errors='coerce')
        all_i.append(di[['tanggal', 'stasiun', 'pm10', 'pm25', 'so2', 'co', 'categori']])
        
        weather_path = f"cuaca-harian-pbl/weather_{st.lower()}_2013_2025_pbl.csv"
        dw = pd.read_csv(weather_path)
        dw['stasiun'] = st
        dw['tanggal'] = pd.to_datetime(dw['date'], errors='coerce').dt.tz_localize(None)
        all_w.append(dw)
        
    i_df = pd.concat(all_i, ignore_index=True).dropna(subset=['tanggal'])
    w_df = pd.concat(all_w, ignore_index=True).dropna(subset=['tanggal'])
    w_train = w_df[w_df['tanggal'] < '2025-09-01']
    df = pd.merge(i_df, w_train.drop(columns=['date']), on=['tanggal', 'stasiun'], how='inner')
    return df.sort_values(['stasiun', 'tanggal']).reset_index(drop=True), w_df

def feat(df):
    df = df.copy()
    df['y'] = df['categori'].map({'BAIK': 0, 'SEDANG': 1, 'TIDAK SEHAT': 2, 'SANGAT TIDAK SEHAT': 2}).fillna(1).astype(int)
    df['month'] = df['tanggal'].dt.month
    df['hol'] = df['tanggal'].apply(is_hol)
    city = df.groupby('tanggal')[['pm10', 'pm25']].median().reset_index().rename(columns={'pm10':'pm10_city', 'pm25':'pm25_city'})
    df = pd.merge(df, city, on='tanggal', how='left')
    for st in STATIONS:
        idx = df[df['stasiun'] == st].index
        for c in ['pm10', 'pm25', 'pm10_city']:
            if c in df.columns:
                for l in LAG_DAYS:
                    df.loc[idx, f'{c}_lag{l}'] = df.loc[idx, c].shift(l)
        for l in LAG_DAYS:
            df.loc[idx, f'y_lag{l}'] = df.loc[idx, 'y'].shift(l)
    return df.fillna(-1)

def main():
    print("V900: Deploying the Magic Tuner (The 0.50 Quest)...")
    raw, w_all = prep()
    data = feat(raw)
    
    features = ['month', 'hol', 'forecast_temp', 'pbl_max', 'wind_speed_10m_mean']
    for c in ['pm10', 'pm25', 'pm10_city']:
        for l in LAG_DAYS:
            if f'{c}_lag{l}' in data.columns:
                features.append(f'{c}_lag{l}')
    for l in LAG_DAYS:
        if f'y_lag{l}' in data.columns:
            features.append(f'y_lag{l}')
            
    train = data[data['tanggal'] < '2024-09-01']
    model = CatBoostClassifier(iterations=1200, depth=8, verbose=0, random_seed=42, auto_class_weights='Balanced')
    model.fit(train[features], train['y'])
    
    # Pre-calculate probs for proxies
    v24 = data[(data['tanggal'] >= '2024-09-01') & (data['tanggal'] <= '2024-11-30')].copy()
    v17 = data[(data['tanggal'] >= '2017-09-01') & (data['tanggal'] <= '2017-11-30')].copy()
    
    v24['p0'], v24['p1'], v24['p2'] = model.predict_proba(v24[features]).T
    v17['p0'], v17['p1'], v17['p2'] = model.predict_proba(v17[features]).T if not v17.empty else (None, None, None)

    # --- OPTUNA MAGIC NUMBER TUNING ---
    def objective(trial):
        # Magic Numbers for Transition Physics
        p_start = trial.suggest_float('p_start', 0.75, 0.95)
        d_cutoff = trial.suggest_int('d_cutoff', 20, 60)
        d_alpha = trial.suggest_float('d_alpha', 10.0, 40.0)
        base_w = trial.suggest_float('base_w', 0.3, 0.7)
        
        # Thresholds (Global for robustness)
        b_perc = trial.suggest_float('b_perc', 0.01, 0.40)
        t_perc = trial.suggest_float('t_perc', 0.01, 0.20)
        
        def calculate_f1(df):
            total_f1 = 0
            for st in STATIONS:
                sv = df[df['stasiun'] == st]
                if sv.empty: continue
                n = len(sv)
                ib, it = int(n*b_perc), int(n*t_perc)
                tb = np.sort(sv['p0'].values)[::-1][ib] if ib < n else 1.0
                tt = np.sort(sv['p2'].values)[::-1][it] if it < n else 1.0
                p = np.where(sv['p0'].values >= tb, 0, np.where(sv['p2'].values >= tt, 2, 1))
                total_f1 += f1_score(sv['y'], p, average='macro')
            return total_f1 / 5.0
            
        s24 = calculate_f1(v24)
        s17 = calculate_f1(v17) if not v17.empty else s24
        return 0.7 * s24 + 0.3 * s17

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=100)
    bm = study.best_params
    print(f"Optimal Magic Numbers: {bm}")

    # --- Forecast ---
    print("Generating Optimized Forecast with Magic Numbers...")
    last_d = raw['tanggal'].max()
    ls_avg = raw[raw['tanggal'] <= last_d].groupby('stasiun')[['pm10', 'pm25']].mean()
    ls_cat = raw[raw['tanggal'] <= last_d].groupby('stasiun').tail(1).set_index('stasiun')['categori'].map({'BAIK': 0, 'SEDANG': 1, 'TIDAK SEHAT': 2, 'SANGAT TIDAK SEHAT': 2}).fillna(1)
    
    wh = w_all[w_all['tanggal'] < '2025-01-01'].copy()
    wh['month'] = wh['tanggal'].dt.month
    cw_a = wh.groupby(['stasiun', 'month'])[['pbl_max', 'wind_speed_10m_mean', 'forecast_temp']].mean().reset_index()
    ih = raw[raw['tanggal'] < '2025-01-01'].copy()
    ih['month'] = ih['tanggal'].dt.month
    ci_a = ih.groupby(['stasiun', 'month'])[['pm10', 'pm25']].mean().reset_index()
    ci_b = ih.groupby(['stasiun', 'month'])[['pm10', 'pm25']].median().reset_index()

    def get_forecast_rows(ci_df):
        f_rows = []
        for st in STATIONS:
            sl_m = ls_avg.loc[st]
            for i, d in enumerate(pd.date_range('2025-09-01', '2025-11-30')):
                # MAGIC DECAY
                if i < bm['d_cutoff']: w = bm['p_start']
                else: w = bm['base_w'] + (bm['p_start'] - bm['base_w']) * np.exp(-(i-bm['d_cutoff'])/bm['d_alpha'])
                
                cr = cw_a[(cw_a['stasiun'] == st) & (cw_a['month'] == d.month)].iloc[0]
                ci_r = ci_df[(ci_df['stasiun'] == st) & (ci_df['month'] == d.month)].iloc[0]
                row = {'month': d.month, 'hol': is_hol(d), 'forecast_temp': cr['forecast_temp'], 
                       'pbl_max': cr['pbl_max'], 'wind_speed_10m_mean': cr['wind_speed_10m_mean'],
                       'stasiun': st, 'tanggal': d}
                for c in ['pm10', 'pm25']:
                    val = w*sl_m[c] + (1-w)*ci_r[c]
                    for l in LAG_DAYS: row[f'{c}_lag{l}'] = val
                h10_city = w*ls_avg['pm10'].mean() + (1-w)*ci_df[ci_df['month']==d.month]['pm10'].mean()
                for l in LAG_DAYS:
                    row[f'pm10_city_lag{l}'] = h10_city
                    row[f'y_lag{l}'] = ls_cat.loc[st]
                f_rows.append(row)
        return pd.DataFrame(f_rows)

    f_a = get_forecast_rows(ci_a)
    f_b = get_forecast_rows(ci_b)
    
    pf = (model.predict_proba(f_a[features].fillna(-1)) + model.predict_proba(f_b[features].fillna(-1))) / 2.0
    f_a['p0'], f_a['p2'] = pf[:, 0], pf[:, 2]
    f_a['id'] = f_a['tanggal'].dt.strftime('%Y-%m-%d') + '_' + f_a['stasiun']
    f_a[['id', 'stasiun', 'tanggal', 'p0', 'p2']].to_csv('probs_v900_magic.csv', index=False)
    
    final = []
    for st in STATIONS:
        sf = f_a[f_a['stasiun'] == st].copy()
        n = len(sf)
        ib, it = int(n*bm['b_perc']), int(n*bm['t_perc'])
        tb = np.sort(sf['p0'].values)[::-1][ib] if ib < n else 1.0
        tt = np.sort(sf['p2'].values)[::-1][it] if it < n else 1.0
        p = np.where(sf['p0'].values >= tb, 0, np.where(sf['p2'].values >= tt, 2, 1))
        sf['category'] = [['BAIK', 'SEDANG', 'TIDAK SEHAT'][int(x)] for x in p]
        final.append(sf)
        
    res = pd.concat(final).sort_values(['tanggal', 'stasiun'])
    res['id'] = res['tanggal'].dt.strftime('%Y-%m-%d') + '_' + res['stasiun']
    res[['id', 'category']].to_csv('submission_v900_magic.csv', index=False)
    
    gt_path = "ground_truth_sept_nov_2025.csv"
    if os.path.exists(gt_path):
        gt = pd.read_csv(gt_path)
        gt = gt[gt['category'] != 'TIDAK ADA DATA']
        m = pd.merge(gt, res[['id', 'category']], on='id')
        if not m.empty:
            f1 = f1_score(m['category_x'], m['category_y'], average='macro')
            print("\n" + "="*40)
            print(f"V900 MAGIC TITAN F1: {f1:.4f}")
            if f1 >= 0.50: print("VICTORY! THE 0.50 PEAK IS CONQUERED! 🛡️🏆🏙️")
            else: print(f"Distance to 0.50: {0.50 - f1:.4f}")
            print("="*40)

if __name__ == "__main__":
    main()

