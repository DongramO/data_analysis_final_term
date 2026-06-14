"""
ML 모델 학습 및 평가 (분기 단위)
- 데이터: ml_dataset_quarterly_train.csv (raw features only, 복합점수 없음)
- train: 2020Q1~2022Q4  test: 2023Q1~2023Q4
- 평가: Macro F1, 혼동행렬, feature importance
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats.mstats import winsorize

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, f1_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = 'data/main_data/ml_dataset_quarterly_train.csv'
IMG_DIR   = Path('modeling/image')
IMG_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER  = ['상승', '유지', '하락']
LABEL_COLORS = {'상승': '#2166ac', '유지': '#999999', '하락': '#d6604d'}

# ── 1. 데이터 로드 ────────────────────────────────────────────
df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')

# ID 컬럼 + 타겟 컬럼 (feature에서 제외)
ID_COLS = ['행정동_코드', '행정동', '연도', '분기', '상권티어', '분기_티어',
           '분위변화_next', '타겟', '타겟_레이블']
TARGET_COL = '타겟_레이블'
FEAT_COLS  = [c for c in df.columns if c not in ID_COLS]

print(f'Feature 개수: {len(FEAT_COLS)}')
print(f'Feature 목록: {FEAT_COLS}')

# ── 2. 전처리 ─────────────────────────────────────────────────
# 2020행은 직전 연도가 없어 YoY·패널 피처가 NaN → 의미별로 다르게 채움

# 변화량, 편차, 모멘텀 계열: 0 fill (변화 없음 / 편차 없음 / 방향 미결정)
zero_fill_cols = [c for c in FEAT_COLS if '_변화' in c or '_동편차' in c or '_모멘텀' in c]
df[zero_fill_cols] = df[zero_fill_cols].fillna(0)

# lag1 계열: 중앙값 fill (직전 수준을 모를 때 평균적인 수준으로 대체)
lag1_cols = [c for c in FEAT_COLS if '_lag1' in c]
for col in lag1_cols:
    df[col] = df[col].fillna(df[col].median())

# 이상치 winsorize (변화량 계열, 1~99%)
yoy_cols = [c for c in FEAT_COLS if '_변화' in c]
for col in yoy_cols:
    df[col] = winsorize(df[col], limits=[0.01, 0.01])

# 나머지 잔여 결측 → 중앙값 fill
for col in FEAT_COLS:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# ── 3. train / test 분할 ─────────────────────────────────────
# walk-forward split: 시계열 순서 유지 (k-fold 사용 불가)
# train: 2020~2022 (12분기)  test: 2023 (4분기)
train = df[df['연도'] <= 2022].copy()
test  = df[df['연도'] >= 2023].copy()

X_train = train[FEAT_COLS]
y_train = train[TARGET_COL]
X_test  = test[FEAT_COLS]
y_test  = test[TARGET_COL]

print(f'\nTrain: {len(train)}행  ({train["연도"].min()}Q{train["분기"].min()} ~ '
      f'{train["연도"].max()}Q{train["분기"].max()})')
print(f'Test : {len(test)}행   ({test["연도"].min()}Q{test["분기"].min()} ~ '
      f'{test["연도"].max()}Q{test["분기"].max()})')
print()
print('Train 타겟 분포:')
print(y_train.value_counts().to_string())
print()
print('Test 타겟 분포:')
print(y_test.value_counts().to_string())

# ── 4. 모델 정의 ──────────────────────────────────────────────
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42))
    ]),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, class_weight='balanced',
        max_depth=6, min_samples_leaf=5, random_state=42),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', random_state=42, verbosity=0),
}

# XGBoost 숫자 레이블
label_map     = {'상승': 2, '유지': 1, '하락': 0}
label_map_inv = {v: k for k, v in label_map.items()}
y_train_num   = y_train.map(label_map)
y_test_num    = y_test.map(label_map)

# ── 5. 학습 & 평가 ───────────────────────────────────────────
results = {}

for name, model in models.items():
    if name == 'XGBoost':
        model.fit(X_train, y_train_num)
        y_pred_num = model.predict(X_test)
        y_pred = pd.Series(y_pred_num).map(label_map_inv).values
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average='macro', labels=LABEL_ORDER)
    cm       = confusion_matrix(y_test, y_pred, labels=LABEL_ORDER)
    report   = classification_report(y_test, y_pred, labels=LABEL_ORDER,
                                     target_names=LABEL_ORDER, output_dict=True)
    results[name] = {
        'model': model, 'macro_f1': macro_f1,
        'cm': cm, 'report': report, 'y_pred': y_pred
    }
    print(f'\n{"="*50}')
    print(f'{name}  |  Macro F1: {macro_f1:.4f}')
    print(classification_report(y_test, y_pred, labels=LABEL_ORDER, target_names=LABEL_ORDER))

# ── 6. 혼동행렬 + Macro F1 비교 ─────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 6))
fig.suptitle('모델별 성능 비교 (분기 단위 raw features)', fontsize=14, fontweight='bold')

names  = list(results.keys())
f1s    = [results[n]['macro_f1'] for n in names]
colors = ['#9ecae1', '#4393c3', '#08519c']

ax0 = axes[0]
bars = ax0.bar(names, f1s, color=colors, width=0.5, edgecolor='white')
for bar, v in zip(bars, f1s):
    ax0.text(bar.get_x() + bar.get_width()/2, v + 0.005,
             f'{v:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax0.set_ylim(0, min(max(f1s) * 1.2, 1.0))
ax0.set_title('Macro F1 비교', fontsize=12)
ax0.set_ylabel('Macro F1')
ax0.tick_params(axis='x', rotation=15)
ax0.spines[['top', 'right']].set_visible(False)

for ax, name in zip(axes[1:], names):
    cm = results[name]['cm']
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{cm[i,j]}\n({cm_pct[i,j]:.0f}%)',
                    ha='center', va='center', fontsize=10,
                    color='white' if cm_pct[i,j] > 50 else 'black')
    ax.set_xticks([0,1,2])
    ax.set_yticks([0,1,2])
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_yticklabels(LABEL_ORDER)
    ax.set_xlabel('예측')
    ax.set_ylabel('실제')
    ax.set_title(f'{name}\nMacro F1={results[name]["macro_f1"]:.4f}', fontsize=11)

plt.tight_layout()
plt.savefig(IMG_DIR / 'model_01_confusion_matrix.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('\n저장: model_01_confusion_matrix.png')

# ── 7. Feature Importance (XGBoost + RF) ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle('Feature Importance Top 25', fontsize=14, fontweight='bold')

for ax, name in zip(axes, ['XGBoost', 'Random Forest']):
    model = results[name]['model']
    imp   = pd.Series(model.feature_importances_, index=FEAT_COLS)
    top25 = imp.sort_values(ascending=False).head(25)
    bar_colors = ['#e6550d' if '_변화' in c else '#3182bd' for c in top25.index[::-1]]
    ax.barh(top25.index[::-1], top25.values[::-1],
            color=bar_colors, edgecolor='white', height=0.7)
    ax.set_title(f'{name}\n(주황=YoY변화, 파랑=현재수준)', fontsize=11)
    ax.set_xlabel('Feature Importance')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', labelsize=9)

plt.tight_layout()
plt.savefig(IMG_DIR / 'model_02_feature_importance.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: model_02_feature_importance.png')

# ── 8. 클래스별 F1 비교 ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle('모델별 클래스별 F1 Score', fontsize=13, fontweight='bold')

x     = np.arange(len(LABEL_ORDER))
width = 0.25
for i, name in enumerate(names):
    f1s_cls = [results[name]['report'][lbl]['f1-score'] for lbl in LABEL_ORDER]
    bars = ax.bar(x + i*width, f1s_cls, width, label=name,
                  color=colors[i], edgecolor='white')
    for bar, v in zip(bars, f1s_cls):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f'{v:.2f}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x + width)
ax.set_xticklabels(LABEL_ORDER, fontsize=12)
ax.set_ylabel('F1 Score')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(IMG_DIR / 'model_03_class_f1.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: model_03_class_f1.png')

# ── 9. YoY vs 수준 feature importance 비율 ───────────────────
print('\n' + '='*50)
print('YoY 변화 feature vs 현재 수준 feature importance 합계 비율')
print('='*50)
for name in ['XGBoost', 'Random Forest']:
    model     = results[name]['model']
    imp       = pd.Series(model.feature_importances_, index=FEAT_COLS)
    yoy_sum   = imp[[c for c in FEAT_COLS if '_변화' in c]].sum()
    level_sum = imp[[c for c in FEAT_COLS if '_변화' not in c]].sum()
    total     = yoy_sum + level_sum
    print(f'\n{name}')
    print(f'  YoY 변화 feature 합계: {yoy_sum:.4f} ({yoy_sum/total*100:.1f}%)')
    print(f'  현재 수준 feature 합계: {level_sum:.4f} ({level_sum/total*100:.1f}%)')

# ── 10. 2030 관련 feature importance (XGBoost) ───────────────
print('\n' + '='*50)
print('2030 관련 feature importance (XGBoost)')
print('='*50)
xgb_model = results['XGBoost']['model']
imp_xgb   = pd.Series(xgb_model.feature_importances_, index=FEAT_COLS)
feat_2030  = [c for c in FEAT_COLS if '2030' in c or '20대' in c or '30대' in c
              or '연령대20' in c or '연령대30' in c]
imp_2030   = imp_xgb[feat_2030].sort_values(ascending=False)
for feat, val in imp_2030.items():
    rank = (imp_xgb > val).sum() + 1
    print(f'  {feat:<30} {val:.4f}  (전체 {len(FEAT_COLS)}개 중 {rank}위)')
