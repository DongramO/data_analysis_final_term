# -*- coding: utf-8 -*-
"""
XGBoost + LightGBM + CatBoost  3-class OOF 앙상블
타겟: 분기별 트렌드 상태 변화 (0=하락 / 1=유지 / 2=상승)

━━━ 스켈레톤(kaggle/skeleton) → 3-class 변환 포인트 ━━━
  ① 평가 지표  : binary AUC (기준 0.5)  → Macro F1 (기준 1/3)
  ② OOF 확률  : predict_proba[:, 1]    → predict_proba  shape (n, 3)
  ③ 앙상블    : 스칼라 가중 평균       → 확률 행렬 (n×3) 가중 평균 후 argmax
  ④ XGBoost   : optuna_results.json 재사용 (100 trials 이미 완료)
     LightGBM  : 50 trials 신규 Optuna 튜닝
     CatBoost  : 50 trials 신규 Optuna 튜닝
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
import warnings

import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

N_FOLDS   = 5
N_TRIALS  = 50   # LightGBM, CatBoost 각각
SEED      = 42
N_CLASSES = 3

# ── 1. 데이터 로드 (train_optuna.py와 동일) ────────────────────────
print('데이터 로드 중...')
ml_tr   = pd.read_csv('../data/main_data/ml_dataset_quarterly_train.csv',   encoding='utf-8-sig')
ml_pr   = pd.read_csv('../data/main_data/ml_dataset_quarterly_predict.csv', encoding='utf-8-sig')
profile = pd.read_csv('../data/main_data/area_type_profile.csv',            encoding='utf-8-sig')

cols       = ml_tr.columns.tolist()
TARGET_COL = cols[97]
LABEL_COL  = cols[98]
DONG_COL   = cols[1]
IDX_2030   = 76       # 2030_이동비율 인덱스

profile_slim = profile[['행정동', '주말집중도', '2030주말집중도', '구역유형']].copy()
profile_slim['구역유형_코드'] = profile_slim['구역유형'].map(
    {'오피스형': 0, '혼합형': 1, '주거형': 2, '상권형': 3}).fillna(1)

def add_features(df):
    df = df.merge(profile_slim, left_on=DONG_COL, right_on='행정동', how='left')
    df['주말집중도']     = df['주말집중도'].fillna(1.0)
    df['2030주말집중도'] = df['2030주말집중도'].fillna(1.0)
    df['구역유형_코드']  = df['구역유형_코드'].fillna(2)
    mig = df.iloc[:, IDX_2030].values
    df['보정_2030이동']   = mig * df['주말집중도']
    df['여가형_2030이동'] = mig * df['2030주말집중도']
    df['오피스강도']      = 1.0 / (df['주말집중도'] + 1e-6)
    return df

ml_tr = add_features(ml_tr)
ml_pr = add_features(ml_pr)

EXCLUDE = {cols[0], cols[1], cols[2], cols[3], cols[4], cols[5],
           cols[96], TARGET_COL, LABEL_COL, '행정동', '구역유형'}
FEAT_COLS = [c for c in ml_tr.columns if c not in EXCLUDE]

X_train = ml_tr[FEAT_COLS].fillna(0).values
y_train = ml_tr[TARGET_COL].astype(int).values
X_pred  = ml_pr[FEAT_COLS].fillna(0).values
print(f'피처 수: {len(FEAT_COLS)} | 학습: {X_train.shape} | 클래스: {dict(zip(*np.unique(y_train, return_counts=True)))}')

SKF = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# ── 2. Optuna objective 함수 ────────────────────────────────────────
#
# [binary 스켈레톤 대비 변경점]
# - scoring: roc_auc → f1_macro
# - LightGBM objective: binary → multiclass, num_class=3
# - CatBoost loss_function: Logloss → MultiClass
#
def objective_lgbm(trial):
    params = {
        'objective':          'multiclass',
        'num_class':          N_CLASSES,
        'metric':             'multi_logloss',
        'n_estimators':       trial.suggest_int('n_estimators', 200, 1500),
        'num_leaves':         trial.suggest_int('num_leaves', 20, 200),
        'max_depth':          trial.suggest_int('max_depth', 3, 10),
        'learning_rate':      trial.suggest_float('learning_rate', 0.005, 0.25, log=True),
        'subsample':          trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':   trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'min_child_samples':  trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha':          trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
        'reg_lambda':         trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
        'random_state':       SEED,
        'n_jobs':             -1,
        'verbose':            -1,
    }
    scores = []
    for tr_idx, val_idx in SKF.split(X_train, y_train):
        m = lgb.LGBMClassifier(**params)
        m.fit(X_train[tr_idx], y_train[tr_idx])
        pred = m.predict(X_train[val_idx])
        scores.append(f1_score(y_train[val_idx], pred, average='macro'))
    return np.mean(scores)


def objective_catboost(trial):
    params = {
        'loss_function':   'MultiClass',
        'classes_count':   N_CLASSES,
        'iterations':      trial.suggest_int('iterations', 200, 1200),
        'depth':           trial.suggest_int('depth', 3, 9),
        'learning_rate':   trial.suggest_float('learning_rate', 0.005, 0.25, log=True),
        'l2_leaf_reg':     trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 5.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
        'random_seed':     SEED,
        'verbose':         0,
    }
    scores = []
    for tr_idx, val_idx in SKF.split(X_train, y_train):
        m = CatBoostClassifier(**params)
        m.fit(X_train[tr_idx], y_train[tr_idx])
        pred = m.predict(X_train[val_idx]).ravel()
        scores.append(f1_score(y_train[val_idx], pred, average='macro'))
    return np.mean(scores)


# ── 3. Optuna 실행 ──────────────────────────────────────────────────
print(f'\nLightGBM Optuna ({N_TRIALS} trials)...')
study_lgbm = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
study_lgbm.optimize(
    objective_lgbm, n_trials=N_TRIALS,
    callbacks=[lambda s, t: print(f'  [{t.number:3d}] F1={t.value:.4f} best={s.best_value:.4f}')
               if t.number % 10 == 0 else None])
best_lgbm = dict(study_lgbm.best_params)
best_lgbm.update({'objective': 'multiclass', 'num_class': N_CLASSES,
                  'metric': 'multi_logloss', 'random_state': SEED,
                  'n_jobs': -1, 'verbose': -1})
print(f'LightGBM 최적 F1: {study_lgbm.best_value:.4f}')

print(f'\nCatBoost Optuna ({N_TRIALS} trials)...')
study_cat = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
study_cat.optimize(
    objective_catboost, n_trials=N_TRIALS,
    callbacks=[lambda s, t: print(f'  [{t.number:3d}] F1={t.value:.4f} best={s.best_value:.4f}')
               if t.number % 10 == 0 else None])
best_cat = dict(study_cat.best_params)
best_cat.update({'loss_function': 'MultiClass', 'classes_count': N_CLASSES,
                 'random_seed': SEED, 'verbose': 0})
print(f'CatBoost 최적 F1: {study_cat.best_value:.4f}')

# XGBoost: 이미 100 trials 완료된 결과 재사용
with open('../data/main_data/optuna_results.json', encoding='utf-8') as f:
    saved = json.load(f)
best_xgb = dict(saved['best_params'])
best_xgb.update({'objective': 'multi:softmax', 'num_class': N_CLASSES,
                 'eval_metric': 'merror', 'use_label_encoder': False,
                 'random_state': SEED, 'n_jobs': -1})
print(f'\nXGBoost (재사용): best CV F1 = {saved["best_cv_f1"]:.4f}')

# ── 4. OOF 앙상블 ──────────────────────────────────────────────────
#
# [핵심 변환] binary 스켈레톤은 predict_proba[:, 1]로 1D OOF를 만들어 AUC 측정.
# 3-class에서는 predict_proba가 (n, 3)이므로:
#   - 각 폴드별 확률 행렬을 누적해 OOF 확률 행렬(n, 3)을 만들고
#   - argmax로 클래스 예측 → Macro F1 계산
#   - 가중치 = OOF Macro F1 - 1/3  (랜덤 기준 초과분, 스켈레톤의 AUC - 0.5와 동일 논리)
#
MODELS = {
    'XGBoost':  (xgb.XGBClassifier,   best_xgb),
    'LightGBM': (lgb.LGBMClassifier,  best_lgbm),
    'CatBoost': (CatBoostClassifier,  best_cat),
}

print('\n─── OOF 앙상블 학습 ───')
oof_proba_all  = {}   # 모델별 OOF 확률 행렬 (n_train, 3)
test_proba_all = {}   # 모델별 테스트 확률 행렬 (n_test, 3)
oof_f1_all     = {}

for name, (ModelCls, params) in MODELS.items():
    print(f'\n  [{name}] 5-fold OOF 학습 중...')
    oof_proba  = np.zeros((len(X_train), N_CLASSES))
    test_proba = np.zeros((len(X_pred),  N_CLASSES))

    for fold, (tr_idx, val_idx) in enumerate(SKF.split(X_train, y_train)):
        model = ModelCls(**params)
        model.fit(X_train[tr_idx], y_train[tr_idx])

        # predict_proba: (n, 3) — 스켈레톤의 [:, 1] 대신 전체 행렬 사용
        oof_proba[val_idx] = model.predict_proba(X_train[val_idx])
        test_proba         += model.predict_proba(X_pred) / N_FOLDS

    oof_pred = np.argmax(oof_proba, axis=1)
    f1       = f1_score(y_train, oof_pred, average='macro')
    print(f'  {name} OOF Macro F1: {f1:.4f}')

    oof_proba_all[name]  = oof_proba
    test_proba_all[name] = test_proba
    oof_f1_all[name]     = f1

# 가중치: OOF F1 - 1/3 (랜덤 기준 초과분)
raw_w = {n: max(1e-6, f - 1/3) for n, f in oof_f1_all.items()}
total = sum(raw_w.values())
weights = {n: w / total for n, w in raw_w.items()}
print(f'\n앙상블 가중치: ' + ', '.join(f'{n}={w:.3f}' for n, w in weights.items()))

# 가중 평균 확률 행렬 → argmax
ens_test_proba = sum(weights[n] * test_proba_all[n] for n in MODELS)
ens_oof_proba  = sum(weights[n] * oof_proba_all[n]  for n in MODELS)
ens_pred_train = np.argmax(ens_oof_proba,  axis=1)
ens_pred_test  = np.argmax(ens_test_proba, axis=1)

ens_f1 = f1_score(y_train, ens_pred_train, average='macro')
print(f'\n앙상블 OOF Macro F1: {ens_f1:.4f}')
print(f'(XGBoost 단독 기준: {saved["best_cv_f1"]:.4f})')

print('\n=== 앙상블 최종 성능 ===')
print(f'Macro F1: {ens_f1:.4f}')
print(classification_report(y_train, ens_pred_train, target_names=['하락', '유지', '상승']))

# ── 5. 예측 저장 ────────────────────────────────────────────────────
label_map = {0: '하락', 1: '유지', 2: '상승'}
out = ml_pr[[cols[1], cols[2], cols[3]]].copy()
out.columns = ['행정동', '연도', '분기']
out['예측_상태']  = [label_map[p] for p in ens_pred_test]
out['확률_하락']  = ens_test_proba[:, 0].round(3)
out['확률_유지']  = ens_test_proba[:, 1].round(3)
out['확률_상승']  = ens_test_proba[:, 2].round(3)
out = out.merge(profile_slim[['행정동', '주말집중도', '구역유형']], on='행정동', how='left')
out.to_csv('../data/main_data/model_predictions_v3.csv', encoding='utf-8-sig', index=False)
print('\n예측 저장: data/main_data/model_predictions_v3.csv')

# ── 6. 결과 JSON 저장 ───────────────────────────────────────────────
results = {
    'ensemble_oof_f1': round(ens_f1, 4),
    'individual_oof_f1': {n: round(f, 4) for n, f in oof_f1_all.items()},
    'weights': {n: round(w, 4) for n, w in weights.items()},
    'xgb_params':  {k: v for k, v in best_xgb.items() if k not in ['objective','eval_metric','use_label_encoder']},
    'lgbm_params': {k: v for k, v in best_lgbm.items() if k not in ['objective','metric']},
    'cat_params':  {k: v for k, v in best_cat.items()  if k not in ['loss_function','classes_count']},
}
with open('../data/main_data/ensemble_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print('결과 저장: data/main_data/ensemble_results.json')

# ── 7. 시각화 ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 13))
fig.patch.set_facecolor('#F8F9FA')
gs  = gridspec.GridSpec(2, 3, hspace=0.50, wspace=0.38,
                        left=0.06, right=0.97, top=0.93, bottom=0.06)
fig.suptitle('XGB + LightGBM + CatBoost  3-class OOF 앙상블 결과',
             fontsize=13, fontweight='bold')

COLORS = {'XGBoost': '#E74C3C', 'LightGBM': '#27AE60', 'CatBoost': '#2980B9', '앙상블': '#8E44AD'}

# ① 모델별 OOF F1 비교
ax1 = fig.add_subplot(gs[0, 0])
names  = list(oof_f1_all.keys()) + ['앙상블']
f1vals = list(oof_f1_all.values()) + [ens_f1]
bars   = ax1.bar(names, f1vals,
                 color=[COLORS[n] for n in names], edgecolor='white', linewidth=1.2)
ax1.axhline(saved['best_cv_f1'], color='gray', linestyle='--', linewidth=1.5,
            label=f'XGB 단독 기준 ({saved["best_cv_f1"]:.4f})')
ax1.set_ylim(0.50, max(f1vals) + 0.04)
ax1.set_title('① 모델별 OOF Macro F1', fontsize=10, fontweight='bold')
ax1.set_ylabel('Macro F1')
for bar, v in zip(bars, f1vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.legend(fontsize=8)

# ② 앙상블 가중치 파이차트
ax2 = fig.add_subplot(gs[0, 1])
w_vals  = list(weights.values())
w_names = list(weights.keys())
wedge_colors = [COLORS[n] for n in w_names]
ax2.pie(w_vals, labels=w_names, autopct='%1.1f%%', colors=wedge_colors,
        startangle=90, textprops={'fontsize': 10})
ax2.set_title('② 앙상블 가중치\n(OOF F1 - 1/3 기준)', fontsize=10, fontweight='bold')

# ③ 클래스별 F1
ax3 = fig.add_subplot(gs[0, 2])
classes = ['하락', '유지', '상승']
xgb_report = classification_report(y_train,
    np.argmax(sum(weights[n] * oof_proba_all[n] for n in MODELS), axis=1),
    output_dict=True)
f1_per_class = [xgb_report[str(i)]['f1-score'] for i in range(3)]
ax3.bar(classes, f1_per_class,
        color=['#E74C3C', '#F39C12', '#27AE60'], edgecolor='white', linewidth=1.2)
ax3.set_title('③ 클래스별 F1 (앙상블)', fontsize=10, fontweight='bold')
ax3.set_ylabel('F1-score')
ax3.set_ylim(0, 1.0)
for i, v in enumerate(f1_per_class):
    ax3.text(i, v + 0.015, f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ④ Optuna 수렴 곡선 (LightGBM, CatBoost)
ax4 = fig.add_subplot(gs[1, 0])
for study, name, color in [(study_lgbm, 'LightGBM', '#27AE60'),
                            (study_cat,  'CatBoost', '#2980B9')]:
    t_df = study.trials_dataframe()
    ax4.plot(t_df['number'], t_df['value'], alpha=0.3, color=color, linewidth=1)
    ax4.plot(t_df['number'], t_df['value'].cummax(), color=color, linewidth=2, label=name)
ax4.axhline(saved['best_cv_f1'], color='#E74C3C', linestyle='--', linewidth=1.5,
            label=f'XGB 기준 ({saved["best_cv_f1"]:.4f})')
ax4.set_title('④ Optuna 수렴 곡선', fontsize=10, fontweight='bold')
ax4.set_xlabel('Trial')
ax4.set_ylabel('Macro F1')
ax4.legend(fontsize=8)

# ⑤ 구역유형별 앙상블 상승 확률
ax5 = fig.add_subplot(gs[1, 1])
out_viz = out.copy()
zone_order = ['오피스형', '혼합형', '주거형', '상권형']
zone_mean = out_viz.groupby('구역유형')['확률_상승'].mean().reindex(zone_order)
bar_colors = ['#E74C3C', '#F39C12', '#27AE60', '#9B59B6']
bars5 = ax5.bar(zone_mean.index, zone_mean.values,
                color=bar_colors, edgecolor='white', linewidth=1.2)
ax5.set_title('⑤ 구역유형별 평균 상승 확률', fontsize=10, fontweight='bold')
ax5.set_ylabel('평균 상승 확률')
ax5.set_ylim(0, 0.50)
for bar, v in zip(bars5, zone_mean.values):
    if not np.isnan(v):
        ax5.text(bar.get_x() + bar.get_width()/2, v + 0.008,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=9)

# ⑥ 상승 확률 TOP 15
ax6 = fig.add_subplot(gs[1, 2])
top15 = (out_viz.groupby('행정동')['확률_상승']
         .mean().sort_values(ascending=False).head(15))
ax6.barh(top15.index[::-1], top15.values[::-1], color='#8E44AD', edgecolor='white')
ax6.set_title('⑥ 상승 확률 TOP 15\n(오피스형 제외 보정)', fontsize=10, fontweight='bold')
ax6.set_xlabel('평균 상승 확률')
ax6.set_xlim(0, 1.0)
for i, v in enumerate(top15.values[::-1]):
    ax6.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)

plt.savefig('../image/train_ensemble_result.png', dpi=150, bbox_inches='tight')
print('\n시각화 저장: image/train_ensemble_result.png')
