# -*- coding: utf-8 -*-
"""
XGBoost + Optuna 파라미터 최적화
타겟: 분기별 트렌드 상태 변화 (0=하락 / 1=유지 / 2=상승)

신규 피처:
  - 주말집중도      : 생활인구 기반 주말/평일 비율 (오피스 보정)
  - 2030주말집중도  : 2030 특화 주말집중도 (여가형 2030 판별)
  - 보정_2030이동   : 2030이동비율 × 주말집중도 (오피스형 2030 하향)
  - 여가형_2030이동 : 2030이동비율 × 2030주말집중도
  - 구역유형_코드   : 오피스=0, 혼합=1, 주거=2, 상권=3
"""

import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings, json, os
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 1. 데이터 로드 ──────────────────────────────────────────────
print('데이터 로드 중...')
ml_tr = pd.read_csv('../data/main_data/ml_dataset_quarterly_train.csv', encoding='utf-8-sig')
ml_pr = pd.read_csv('../data/main_data/ml_dataset_quarterly_predict.csv', encoding='utf-8-sig')
profile = pd.read_csv('../data/main_data/area_type_profile.csv', encoding='utf-8-sig')

cols = ml_tr.columns.tolist()
TARGET_COL   = cols[97]   # 타겟
LABEL_COL    = cols[98]   # 타겟_레이블
DONG_COL     = cols[1]    # 행정동
DONG_CODE    = cols[0]    # 행정동_코드
IDX_2030_MIG = 76         # 2030_이동비율

# ── 2. 프로파일 조인 + 신규 피처 생성 ──────────────────────────
profile_slim = profile[['행정동','주말집중도','2030주말집중도','구역유형']].copy()
profile_slim['구역유형_코드'] = profile_slim['구역유형'].map(
    {'오피스형': 0, '혼합형': 1, '주거형': 2, '상권형': 3}).fillna(1)

def add_features(df):
    df = df.merge(profile_slim, left_on=DONG_COL, right_on='행정동', how='left')
    # 주말집중도 미보유 동은 중간값(주거형)으로 채움
    df['주말집중도']     = df['주말집중도'].fillna(1.0)
    df['2030주말집중도'] = df['2030주말집중도'].fillna(1.0)
    df['구역유형_코드']  = df['구역유형_코드'].fillna(2)

    mig = df.iloc[:, IDX_2030_MIG].values
    df['보정_2030이동']   = mig * df['주말집중도']
    df['여가형_2030이동'] = mig * df['2030주말집중도']
    # 순수 오피스 신호: 주말집중도가 낮을수록 높아짐
    df['오피스강도']      = 1.0 / (df['주말집중도'] + 1e-6)
    return df

ml_tr = add_features(ml_tr)
ml_pr = add_features(ml_pr)

# ── 3. 피처 / 타겟 분리 ─────────────────────────────────────────
EXCLUDE = {cols[0], cols[1], cols[2], cols[3],   # 코드·동이름·연도·분기
           cols[4], cols[5],                       # 상권티어·분기티어
           cols[96],                               # 분위변화_next
           TARGET_COL, LABEL_COL,                 # 타겟
           '행정동', '구역유형'}                   # 조인 컬럼

FEAT_COLS = [c for c in ml_tr.columns if c not in EXCLUDE]
print(f'피처 수: {len(FEAT_COLS)}  (기존 90 + 신규 5)')

X_train = ml_tr[FEAT_COLS].fillna(0).values
y_train = ml_tr[TARGET_COL].astype(int).values
X_pred  = ml_pr[FEAT_COLS].fillna(0).values

print(f'학습 데이터: {X_train.shape} | 클래스 분포: {dict(zip(*np.unique(y_train, return_counts=True)))}')

# ── 4. Optuna 목적함수 ──────────────────────────────────────────
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 200, 1200),
        'max_depth':         trial.suggest_int('max_depth', 3, 9),
        'learning_rate':     trial.suggest_float('learning_rate', 0.005, 0.25, log=True),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0),
        'min_child_weight':  trial.suggest_int('min_child_weight', 1, 15),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
        'gamma':             trial.suggest_float('gamma', 1e-8, 2.0, log=True),
        'objective':         'multi:softmax',
        'num_class':         3,
        'eval_metric':       'merror',
        'use_label_encoder': False,
        'random_state':      42,
        'n_jobs':            -1,
    }
    scores = []
    for tr_idx, val_idx in SKF.split(X_train, y_train):
        Xtr, Xval = X_train[tr_idx], X_train[val_idx]
        ytr, yval = y_train[tr_idx], y_train[val_idx]
        model = xgb.XGBClassifier(**params)
        model.fit(Xtr, ytr,
                  eval_set=[(Xval, yval)],
                  verbose=False)
        pred = model.predict(Xval)
        scores.append(f1_score(yval, pred, average='macro'))
    return np.mean(scores)

# ── 5. Optuna 실행 ──────────────────────────────────────────────
print('\nOptuna 최적화 시작 (100회 trial)...')
study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100,
               callbacks=[lambda s, t: print(f'  trial {t.number:3d} | F1={t.value:.4f} | best={s.best_value:.4f}')
                          if t.number % 10 == 0 else None])

best_params = study.best_params
best_params.update({'objective': 'multi:softmax', 'num_class': 3,
                    'eval_metric': 'merror', 'use_label_encoder': False,
                    'random_state': 42, 'n_jobs': -1})

print(f'\n최적 파라미터: {best_params}')
print(f'최적 CV F1:    {study.best_value:.4f}  (이전 기준 0.5383)')

# ── 6. 최종 모델 학습 (전체 학습 데이터) ───────────────────────
print('\n최종 모델 학습 중...')
final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_train, y_train, verbose=False)

# CV 최종 평가 (전체 재현)
cv_preds = np.zeros(len(y_train), dtype=int)
for tr_idx, val_idx in SKF.split(X_train, y_train):
    m = xgb.XGBClassifier(**best_params)
    m.fit(X_train[tr_idx], y_train[tr_idx], verbose=False)
    cv_preds[val_idx] = m.predict(X_train[val_idx])

print('\n=== 최종 모델 성능 (5-Fold CV) ===')
print(f'Macro F1: {f1_score(y_train, cv_preds, average="macro"):.4f}')
print(classification_report(y_train, cv_preds, target_names=['하락','유지','상승']))

# ── 7. 피처 중요도 ──────────────────────────────────────────────
importances = pd.DataFrame({
    '피처': FEAT_COLS,
    '중요도': final_model.feature_importances_
}).sort_values('중요도', ascending=False)

print('\n=== 피처 중요도 TOP 20 ===')
print(importances.head(20).to_string(index=False))

# 신규 피처 순위
new_feats = ['주말집중도','2030주말집중도','보정_2030이동','여가형_2030이동','오피스강도','구역유형_코드']
new_ranks = importances[importances['피처'].isin(new_feats)]
print('\n=== 신규 피처 순위 ===')
print(new_ranks.to_string(index=False))

# ── 8. 2024 예측 ────────────────────────────────────────────────
y_pred_2024 = final_model.predict(X_pred)
y_prob_2024 = final_model.predict_proba(X_pred)

ml_pr_out = ml_pr[[cols[1], cols[2], cols[3]]].copy()
ml_pr_out.columns = ['행정동','연도','분기']
ml_pr_out['예측_상태'] = pd.Categorical(y_pred_2024).rename_categories({0:'하락',1:'유지',2:'상승'})
ml_pr_out['확률_하락'] = y_prob_2024[:, 0].round(3)
ml_pr_out['확률_유지'] = y_prob_2024[:, 1].round(3)
ml_pr_out['확률_상승'] = y_prob_2024[:, 2].round(3)

# 주말집중도, 구역유형 추가
ml_pr_out = ml_pr_out.merge(profile_slim[['행정동','주말집중도','구역유형']], on='행정동', how='left')

ml_pr_out.to_csv('../data/main_data/model_predictions_v2.csv', encoding='utf-8-sig', index=False)

# ── 9. 결과 저장 ────────────────────────────────────────────────
results = {
    'best_cv_f1': study.best_value,
    'best_params': {k: v for k, v in best_params.items()
                    if k not in ['objective','eval_metric','use_label_encoder']},
    'n_features': len(FEAT_COLS),
    'n_trials': 100,
}
with open('../data/main_data/optuna_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

importances.to_csv('../data/main_data/feature_importance_v2.csv', encoding='utf-8-sig', index=False)

# ── 10. 시각화 ──────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#F8F9FA')
gs = gridspec.GridSpec(2, 3, hspace=0.48, wspace=0.38,
                       left=0.06, right=0.97, top=0.93, bottom=0.05)
fig.suptitle('XGBoost + Optuna 최적화 결과\n(주말집중도·오피스보정 신규 피처 포함)',
             fontsize=13, fontweight='bold')

# Optuna 수렴 곡선
ax1 = fig.add_subplot(gs[0, 0])
trials_df = study.trials_dataframe()
ax1.plot(trials_df['number'], trials_df['value'], alpha=0.4, color='#3498DB', linewidth=1, label='각 trial')
best_so_far = trials_df['value'].cummax()
ax1.plot(trials_df['number'], best_so_far, color='#E74C3C', linewidth=2.5, label='최대 F1')
ax1.axhline(0.5383, color='gray', linewidth=1.5, linestyle='--', label='이전 기준 (0.5383)')
ax1.set_title('① Optuna 수렴 곡선', fontsize=10, fontweight='bold')
ax1.set_xlabel('Trial')
ax1.set_ylabel('Macro F1')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# 파라미터 중요도 (Optuna)
ax2 = fig.add_subplot(gs[0, 1])
try:
    param_imp = optuna.importance.get_param_importances(study)
    names = list(param_imp.keys())[:8]
    vals  = [param_imp[n] for n in names]
    ax2.barh(names[::-1], vals[::-1], color='#2ECC71', alpha=0.85)
    ax2.set_title('② 파라미터 중요도 (Optuna)', fontsize=10, fontweight='bold')
    ax2.set_xlabel('중요도')
    ax2.grid(True, alpha=0.3, axis='x')
except:
    ax2.text(0.5, 0.5, '파라미터 중요도\n계산 불가', ha='center', va='center')

# 피처 중요도 TOP 20
ax3 = fig.add_subplot(gs[0, 2])
top20 = importances.head(20)
colors_fi = ['#E74C3C' if f in new_feats else '#3498DB' for f in top20['피처']]
ax3.barh(top20['피처'][::-1], top20['중요도'][::-1], color=colors_fi[::-1], alpha=0.85)
ax3.set_title('③ 피처 중요도 TOP 20\n(빨간색 = 신규 피처)', fontsize=10, fontweight='bold')
ax3.set_xlabel('중요도')
ax3.grid(True, alpha=0.3, axis='x')

# 구역유형별 예측 분포
ax4 = fig.add_subplot(gs[1, 0])
pred_with_type = ml_pr_out.dropna(subset=['구역유형'])
type_order = ['오피스형','혼합형','주거형','상권형']
type_colors_map = {'오피스형':'#E74C3C','혼합형':'#F39C12','주거형':'#3498DB','상권형':'#27AE60'}

x4 = np.arange(len(type_order))
w4 = 0.25
for i, label in enumerate(['하락','유지','상승']):
    counts = [len(pred_with_type[(pred_with_type['구역유형']==t) & (pred_with_type['예측_상태']==label)])
              for t in type_order]
    totals = [len(pred_with_type[pred_with_type['구역유형']==t]) + 1e-6 for t in type_order]
    ratios = [c/t for c, t in zip(counts, totals)]
    ax4.bar(x4 + i*w4, ratios, width=w4,
            label=label, color=['#E74C3C','#BDC3C7','#27AE60'][i], alpha=0.85)

ax4.set_xticks(x4 + w4)
ax4.set_xticklabels(type_order, fontsize=9)
ax4.set_title('④ 구역유형별 2024 예측 분포\n(하락/유지/상승 비율)', fontsize=10, fontweight='bold')
ax4.set_ylabel('비율')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis='y')

# 주말집중도별 상승 확률
ax5 = fig.add_subplot(gs[1, 1])
pred_with_wkc = ml_pr_out.dropna(subset=['주말집중도'])
pred_with_wkc['주말집중도_구간'] = pd.cut(pred_with_wkc['주말집중도'],
    bins=[0, 0.80, 0.95, 1.10, 2.0],
    labels=['오피스\n(<0.80)','혼합\n(0.80~0.95)','주거\n(0.95~1.10)','상권\n(>1.10)'])
rise_by_wkc = pred_with_wkc.groupby('주말집중도_구간')['확률_상승'].mean()
colors_wkc = ['#E74C3C','#F39C12','#3498DB','#27AE60']
ax5.bar(range(len(rise_by_wkc)), rise_by_wkc.values,
        color=colors_wkc[:len(rise_by_wkc)], alpha=0.85)
ax5.set_xticks(range(len(rise_by_wkc)))
ax5.set_xticklabels(rise_by_wkc.index, fontsize=9)
ax5.set_title('⑤ 주말집중도 구간별 평균 상승 확률\n(오피스형 보정 효과)', fontsize=10, fontweight='bold')
ax5.set_ylabel('평균 상승 확률')
ax5.grid(True, alpha=0.3, axis='y')

# 상승 확률 TOP 후보 (주거+상권형)
ax6 = fig.add_subplot(gs[1, 2])
candidate_types = ['주거형','상권형','혼합형']
top_cands = (pred_with_type[pred_with_type['구역유형'].isin(candidate_types)]
             .groupby('행정동')['확률_상승'].mean()
             .sort_values(ascending=False).head(15).reset_index())
top_cands = top_cands.merge(profile_slim[['행정동','구역유형']], on='행정동', how='left')
bar_colors_c = [type_colors_map.get(t, 'gray') for t in top_cands['구역유형']]

ax6.barh(top_cands['행정동'][::-1], top_cands['확률_상승'][::-1],
         color=bar_colors_c[::-1], alpha=0.85)
ax6.set_title('⑥ 상승 확률 TOP 15\n(오피스형 제외 후보)', fontsize=10, fontweight='bold')
ax6.set_xlabel('평균 상승 확률')
ax6.grid(True, alpha=0.3, axis='x')

plt.savefig('../image/train_optuna_result.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print('\n저장 완료: image/train_optuna_result.png')
print('모델 예측: data/main_data/model_predictions_v2.csv')
