"""
급부상 레퍼런스 패턴 매칭
- 레퍼런스 4개 동(역삼2동1가, 대현1동, 역삼동, 연화동)의 뜨기 전 피처 시그니처 추출
- 현재(2024) 저분위 행정동 중 그 시그니처와 가장 유사한 곳 발굴

출력:
  pattern_01_reference_signature.png  — 레퍼런스 시그니처 (서울 평균 대비 z-score)
  pattern_02_candidates.png           — 유사도 기반 후보 랭킹
  pattern_03_profile_compare.png      — 상위 후보 vs 레퍼런스 프로파일 비교
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.spatial.distance import cosine
from scipy.stats import zscore as _zscore

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

IMG_DIR    = Path('modeling/image')
TRAIN_PATH = 'data/main_data/ml_dataset_quarterly_train.csv'
PRED_PATH  = 'data/main_data/ml_dataset_quarterly_predict.csv'
MASTER_PATH = 'data/main_data/트렌드매출_마스터.csv'

# ── 레퍼런스 정의 ──────────────────────────────────────────────
# 뜨기 전 연도 기준 (그 해의 분기 데이터를 pre-rise 시그니처로 사용)
REFERENCE = {
    '성수2가1동': [2020, 2021],  # 0.244→0.381→0.797  가장 깔끔한 성수 급부상 패턴
    '신대방1동':  [2020, 2021],  # 0.310→0.449→0.610  4년 연속 꾸준 상승
    '동화동':     [2020],         # 0.417→0.879         2020→2021 초기 폭발 신호
    '여의동':     [2020],         # 0.263→0.544         완만하고 지속적인 상승
}

# ── 매칭에 사용할 피처 ─────────────────────────────────────────
# "상권이 뜨기 전에 먼저 움직이는" 지표 중심으로 선별
MATCH_FEATS = [
    # 업종 구성 — 카페·주류·라이프스타일 집중도
    'FB카페_비중', '주류유흥_비중', '라이프스타일_비중',
    # 시간 패턴 — 금/토/일 저녁 중심 소비
    '트렌드시간대_비중', '야간_비중', '주말_비중', '금요효과',
    # 방문형 강도
    '방문형_강도', '평일_비중',
    # 2030 소비
    '2030소비_비중', '연령대20_비중', '연령대30_비중',
    # 이동인구
    '2030_이동비율', 'HE_비율',
    # YoY 변화 (변화 방향)
    '트렌드시간대_비중_변화', '방문형_강도_변화',
    '2030소비_비중_변화', '2030_이동비율_변화',
]

# ── 데이터 로드 ────────────────────────────────────────────────
train  = pd.read_csv(TRAIN_PATH,  encoding='utf-8-sig')
pred   = pd.read_csv(PRED_PATH,   encoding='utf-8-sig')
master = pd.read_csv(MASTER_PATH, encoding='utf-8-sig')

# 2024 분위 (현재 위치 파악용)
분위_2024 = master[master['연도'] == 2024][['행정동', '복합점수_분위']].set_index('행정동')['복합점수_분위']

# ══════════════════════════════════════════════════════════════
# Step 1: 레퍼런스 시그니처 추출
# 각 분기마다 서울 전체 대비 z-score → 레퍼런스 동의 평균
# ══════════════════════════════════════════════════════════════
feats_avail = [f for f in MATCH_FEATS if f in train.columns]

ref_zscore_list = []

for dong, years in REFERENCE.items():
    for yr in years:
        # 해당 연도의 전체 데이터 (서울 전체 기준)
        period_data = train[train['연도'] == yr].copy()
        period_dong = period_data[period_data['행정동'] == dong]

        if period_dong.empty:
            print(f'[경고] {dong} {yr}년 데이터 없음')
            continue

        # 분기별로 z-score 계산 (분기 내 423개 동 기준)
        for q in sorted(period_data['분기'].unique()):
            q_all  = period_data[period_data['분기'] == q][feats_avail]
            q_dong = period_dong[period_dong['분기'] == q][feats_avail]

            if q_dong.empty:
                continue

            # 분기 전체의 평균/std로 z-score 계산
            q_mean = q_all.mean()
            q_std  = q_all.std().replace(0, 1)
            z = ((q_dong.iloc[0] - q_mean) / q_std)
            ref_zscore_list.append(z.rename(f'{dong}_{yr}Q{q}'))

ref_zscores = pd.DataFrame(ref_zscore_list)
signature   = ref_zscores.mean()  # 4개 동 × 분기 평균 → 대표 시그니처

print('=== 레퍼런스 시그니처 (서울 평균 대비 z-score) ===')
print(signature.sort_values(ascending=False).round(3).to_string())

# ══════════════════════════════════════════════════════════════
# Step 2: 현재(2024) 행정동 z-score 계산
# ══════════════════════════════════════════════════════════════
pred_feat = pred[['행정동', '연도', '분기'] + feats_avail].copy()

# 분기별로 z-score 계산
zscored_rows = []
for q in sorted(pred_feat['분기'].unique()):
    q_data = pred_feat[pred_feat['분기'] == q].copy()
    for col in feats_avail:
        mean_ = q_data[col].mean()
        std_  = q_data[col].std()
        q_data[f'z_{col}'] = (q_data[col] - mean_) / (std_ if std_ > 0 else 1)
    zscored_rows.append(q_data)

pred_z = pd.concat(zscored_rows)
z_cols = [f'z_{f}' for f in feats_avail]

# 행정동별 4분기 평균 z-score
dong_z = pred_z.groupby('행정동')[z_cols].mean()
dong_z.columns = feats_avail  # 이름 복원

# ══════════════════════════════════════════════════════════════
# Step 3: 코사인 유사도 계산
# ══════════════════════════════════════════════════════════════
sig_vec = signature[feats_avail].fillna(0).values

similarities = {}
for dong, row in dong_z.iterrows():
    vec = row.fillna(0).values
    sim = 1 - cosine(sig_vec, vec)  # cosine distance → similarity
    similarities[dong] = sim

sim_series = pd.Series(similarities).sort_values(ascending=False)

# 현재 분위 합치기
result = pd.DataFrame({
    '유사도':   sim_series,
    '현재분위': 분위_2024,
}).dropna()

# 필터: 현재 분위 0.65 미만 (아직 안 뜬 곳)
candidates = result[result['현재분위'] < 0.65].sort_values('유사도', ascending=False)

# 레퍼런스 동 제외
ref_dongs = list(REFERENCE.keys())
candidates = candidates[~candidates.index.isin(ref_dongs)]

print(f'\n=== 후보 행정동 Top 20 (현재분위 < 0.65, 레퍼런스 제외) ===')
print(f'{"순위":>4}  {"행정동":<12}  {"유사도":>8}  {"현재분위(2024)":>14}')
print('-'*48)
for i, (dong, row) in enumerate(candidates.head(20).iterrows(), 1):
    bar = '█' * int(row['유사도'] * 20)
    print(f'{i:>4}  {dong:<12}  {row["유사도"]:>8.4f}  {row["현재분위"]:>14.3f}')

# ══════════════════════════════════════════════════════════════
# Chart 1: 레퍼런스 시그니처 — 어떤 피처가 두드러졌나
# ══════════════════════════════════════════════════════════════
feat_labels = {
    'FB카페_비중': 'FB카페',
    '주류유흥_비중': '주류유흥',
    '라이프스타일_비중': '라이프스타일',
    '트렌드시간대_비중': '트렌드시간대',
    '야간_비중': '야간',
    '주말_비중': '주말',
    '금요효과': '금요효과',
    '방문형_강도': '방문형강도',
    '평일_비중': '평일(역)',
    '2030소비_비중': '2030소비',
    '연령대20_비중': '20대소비',
    '연령대30_비중': '30대소비',
    '2030_이동비율': '2030이동',
    'HE_비율': 'HE이동',
    '트렌드시간대_비중_변화': '트렌드변화',
    '방문형_강도_변화': '방문형변화',
    '2030소비_비중_변화': '2030소비변화',
    '2030_이동비율_변화': '2030이동변화',
}

sig_plot = signature[feats_avail].rename(feat_labels).sort_values()

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#d6604d' if v < 0 else '#2166ac' for v in sig_plot.values]
bars = ax.barh(sig_plot.index, sig_plot.values, color=colors, edgecolor='white', height=0.7)
ax.axvline(0, color='black', linewidth=0.8)
for bar, v in zip(bars, sig_plot.values):
    ax.text(v + (0.02 if v >= 0 else -0.02), bar.get_y() + bar.get_height()/2,
            f'{v:+.2f}', va='center', ha='left' if v >= 0 else 'right', fontsize=8)

ax.set_xlabel('서울 평균 대비 z-score (뜨기 전 레퍼런스 동들의 특성)', fontsize=10)
ax.set_title('급부상 레퍼런스 시그니처\n(성수2가1동·신대방1동·동화동·여의동 뜨기 전 공통 패턴)',
             fontsize=12, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)

handles = [mpatches.Patch(color='#2166ac', label='서울 평균보다 높음'),
           mpatches.Patch(color='#d6604d', label='서울 평균보다 낮음')]
ax.legend(handles=handles, fontsize=9)

plt.tight_layout()
plt.savefig(IMG_DIR / 'pattern_01_reference_signature.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('\n저장: pattern_01_reference_signature.png')

# ══════════════════════════════════════════════════════════════
# Chart 2: 후보 랭킹
# ══════════════════════════════════════════════════════════════
top25 = candidates.head(25).rename_axis('행정동').reset_index()

fig, ax = plt.subplots(figsize=(12, 10))
bar_colors = ['#08519c' if i < 5 else '#4292c6' if i < 10 else '#9ecae1'
              for i in range(len(top25))]
bars = ax.barh(top25['행정동'][::-1], top25['유사도'][::-1],
               color=bar_colors[::-1], edgecolor='white', height=0.75)

# 현재 분위 레이블
for bar, (_, row) in zip(bars[::-1], top25.iterrows()):
    ax.text(row['유사도'] + 0.005, bar.get_y() + bar.get_height()/2,
            f'현재분위 {row["현재분위"]:.2f}',
            va='center', fontsize=7.5, color='#333333')

ax.set_xlabel('레퍼런스 시그니처와의 코사인 유사도')
ax.set_title('다음 급부상 후보 행정동\n(현재 분위 < 0.65 필터, 레퍼런스 패턴 유사도 순)',
             fontsize=12, fontweight='bold')
ax.set_xlim(0, top25['유사도'].max() * 1.18)
ax.tick_params(axis='y', labelsize=9)
ax.spines[['top', 'right']].set_visible(False)

# 구분선
ax.axvline(top25['유사도'].mean(), color='gray', linestyle='--',
           linewidth=0.8, alpha=0.7, label=f'평균 유사도 {top25["유사도"].mean():.3f}')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(IMG_DIR / 'pattern_02_candidates.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: pattern_02_candidates.png')

# ══════════════════════════════════════════════════════════════
# Chart 3: 상위 6개 후보 vs 레퍼런스 프로파일 레이더/비교
# ══════════════════════════════════════════════════════════════
top6 = candidates.head(6).index.tolist()

# 레이더 차트용 피처 선택 (핵심 8개)
radar_feats = [
    '트렌드시간대_비중', '방문형_강도', '2030소비_비중',
    '주말_비중', '금요효과', '야간_비중',
    '2030_이동비율', 'FB카페_비중',
]
radar_labels = ['트렌드\n시간대', '방문형\n강도', '2030\n소비', '주말\n비중',
                '금요효과', '야간\n비중', '2030\n이동', 'FB\n카페']

radar_feats_avail = [f for f in radar_feats if f in feats_avail]
radar_labels_avail = [radar_labels[i] for i, f in enumerate(radar_feats) if f in feats_avail]

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.suptitle('상위 후보 vs 레퍼런스 시그니처 피처 비교\n(z-score 기준, 파란선=레퍼런스, 주황선=후보)',
             fontsize=13, fontweight='bold')

ref_vals = signature[radar_feats_avail].fillna(0).values

for ax, dong in zip(axes.flatten()[:len(top6)], top6):
    if dong not in dong_z.index:
        ax.set_visible(False)
        continue

    cand_vals = dong_z.loc[dong, radar_feats_avail].fillna(0).values
    x = np.arange(len(radar_feats_avail))
    width = 0.35

    ax.bar(x - width/2, ref_vals, width, label='레퍼런스', color='#2166ac', alpha=0.7)
    ax.bar(x + width/2, cand_vals, width, label=dong, color='#e6550d', alpha=0.7)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(radar_labels_avail, fontsize=7)
    ax.set_title(f'{dong}\n유사도: {candidates.loc[dong, "유사도"]:.4f}  현재분위: {candidates.loc[dong, "현재분위"]:.2f}',
                 fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=7)

# 레퍼런스 동 자체 프로파일도 나머지 칸에 표시
ref_items = list(REFERENCE.keys())[:2]
for ax, dong in zip(axes.flatten()[len(top6):len(top6)+2], ref_items):
    ref_dong_z = ref_zscores[[c for c in ref_zscores.columns
                               if c.startswith(dong)]].mean(axis=1) if False else None
    # 레퍼런스 동의 pre-rise z-score (train 데이터에서 직접)
    dong_rows = []
    for yr in REFERENCE[dong]:
        q_data = train[train['연도'] == yr].copy()
        for q in sorted(q_data['분기'].unique()):
            q_all  = q_data[q_data['분기'] == q]
            q_dong = q_all[q_all['행정동'] == dong]
            if q_dong.empty:
                continue
            mean_ = q_all[radar_feats_avail].mean()
            std_  = q_all[radar_feats_avail].std().replace(0, 1)
            z = (q_dong[radar_feats_avail].iloc[0] - mean_) / std_
            dong_rows.append(z)

    if dong_rows:
        ref_dong_vals = pd.DataFrame(dong_rows).mean().fillna(0).values
        x = np.arange(len(radar_feats_avail))
        ax.bar(x - width/2, ref_vals, width, label='레퍼런스 평균', color='#2166ac', alpha=0.7)
        ax.bar(x + width/2, ref_dong_vals, width, label=dong, color='#4dac26', alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(radar_labels_avail, fontsize=7)
        ax.set_title(f'[레퍼런스] {dong}\n(뜨기 전 실제 z-score)', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=7)

for ax in axes.flatten()[len(top6)+2:]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig(IMG_DIR / 'pattern_03_profile_compare.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: pattern_03_profile_compare.png')

# ══════════════════════════════════════════════════════════════
# 최종 요약
# ══════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('레퍼런스 시그니처 핵심 특성 (z-score 상위/하위)')
print('='*60)
sig_sorted = signature[feats_avail].rename(feat_labels).sort_values(ascending=False)
print('  서울 평균보다 높은 지표 (뜨기 전 특징):')
for feat, val in sig_sorted[sig_sorted > 0.1].items():
    print(f'    {feat:<16}  {val:+.3f}')
print('  서울 평균보다 낮은 지표:')
for feat, val in sig_sorted[sig_sorted < -0.1].items():
    print(f'    {feat:<16}  {val:+.3f}')

print('\n' + '='*60)
print('다음 급부상 후보 Top 10')
print('='*60)
for i, (dong, row) in enumerate(candidates.head(10).iterrows(), 1):
    print(f'  {i:>2}위  {dong:<12}  유사도 {row["유사도"]:.4f}  현재분위 {row["현재분위"]:.3f}')
