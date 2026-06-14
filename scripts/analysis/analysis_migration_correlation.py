"""
복합점수 × 20대 이동인구 상관관계 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 ───────────────────────────────────────────────
mapping = pd.read_csv('data/main_data/코드매핑_유입인구_매출.csv', encoding='utf-8-sig')
# migration_코드 → raw_코드 딕셔너리
mig_to_raw = dict(zip(mapping['migration_코드'], mapping['raw_코드']))

# 복합점수 (행정동_코드=8자리, 연도)
master = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig',
                     usecols=['행정동', '행정동_코드', '연도', '복합점수', '복합점수_분위'])

# 20대 이동인구 연간 집계
mig = pd.read_csv('data/main_data/분기별_주말유입인구.csv', encoding='utf-8-sig')
mig['raw_코드'] = mig['도착_행정동코드'].map(mig_to_raw)

annual_mig = (
    mig.groupby(['raw_코드', '연도', '연령대'])['합계_이동인구']
    .sum().reset_index()
)
total = annual_mig.groupby(['raw_코드','연도'])['합계_이동인구'].sum().rename('총_이동인구')
annual_mig = annual_mig.join(total, on=['raw_코드','연도'])
annual_mig['비율'] = annual_mig['합계_이동인구'] / annual_mig['총_이동인구'] * 100

age20 = annual_mig[annual_mig['연령대'] == 20][['raw_코드','연도','비율','합계_이동인구']].copy()
age20.columns = ['행정동_코드','연도','20대_비율','20대_이동인구']

# ── 조인 ─────────────────────────────────────────────────────
df = master.merge(age20, on=['행정동_코드','연도'], how='inner')
print(f'조인 결과: {len(df)}행 ({df["행정동_코드"].nunique()}개 행정동 × {df["연도"].nunique()}개 연도)')

years = sorted(df['연도'].unique())

# ── 그림 1: 연도별 산점도 + 추세선 ───────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=False)
fig.suptitle('복합점수 × 20대 이동인구 비율 상관관계\n(서울 행정동 단위, 금·토·일 트렌드 시간대)',
             fontsize=14, fontweight='bold', y=1.02)

colors_yr = {2020:'#4e79a7', 2021:'#f28e2b', 2022:'#e15759', 2023:'#76b7b2', 2024:'#59a14f'}

for ax, yr in zip(axes, years):
    sub = df[df['연도'] == yr].dropna(subset=['복합점수','20대_비율'])
    x, y = sub['복합점수'].values, sub['20대_비율'].values

    # 산점도
    ax.scatter(x, y, alpha=0.45, s=18, color=colors_yr[yr], edgecolors='none')

    # 추세선
    slope, intercept, r, p, _ = stats.linregress(x, y)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, slope * xline + intercept, color='#222', linewidth=1.5, linestyle='--')

    # 상위 5개 라벨
    sub_sorted = sub.nlargest(5, '20대_비율')
    for _, row in sub_sorted.iterrows():
        ax.annotate(row['행정동'], (row['복합점수'], row['20대_비율']),
                    fontsize=7, ha='left', va='bottom',
                    xytext=(3, 2), textcoords='offset points', color='#333')

    ax.set_title(f'{yr}년\nr={r:.3f}  p={p:.3f}', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('복합점수 (z-score)', fontsize=9)
    ax.set_ylabel('20대 이동비율 (%)', fontsize=9)
    ax.grid(linestyle='--', alpha=0.35)
    ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('image/migration_복합점수_산점도.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: image/migration_복합점수_산점도.png')

# ── 그림 2: 복합점수 분위 구간별 20대 이동비율 박스플롯 ────────
df['분위구간'] = pd.cut(df['복합점수_분위'],
                       bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                       labels=['하위20%','20~40%','40~60%','60~80%','상위20%'])

fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
fig.suptitle('복합점수 분위 구간별 20대 이동비율 분포\n(연도별)',
             fontsize=14, fontweight='bold', y=1.02)

for ax, yr in zip(axes, years):
    sub = df[df['연도'] == yr].dropna(subset=['분위구간','20대_비율'])
    groups = [sub[sub['분위구간']==g]['20대_비율'].values
              for g in ['하위20%','20~40%','40~60%','60~80%','상위20%']]
    bp = ax.boxplot(groups, labels=['하위\n20%','20~\n40%','40~\n60%','60~\n80%','상위\n20%'],
                    patch_artist=True, medianprops=dict(color='black', linewidth=1.5),
                    flierprops=dict(marker='o', markersize=2, alpha=0.4))
    for patch, color in zip(bp['boxes'], ['#d9e8f5','#93c4de','#4a9dc7','#1f6fa3','#0d3d6b']):
        patch.set_facecolor(color)

    # 각 구간 평균 표시
    for i, g in enumerate(['하위20%','20~40%','40~60%','60~80%','상위20%']):
        mean_val = sub[sub['분위구간']==g]['20대_비율'].mean()
        ax.plot(i+1, mean_val, 'D', color='#e05c5c', markersize=5, zorder=5)

    ax.set_title(f'{yr}년', fontsize=11, fontweight='bold', pad=8)
    ax.set_ylabel('20대 이동비율 (%)', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('image/migration_복합점수_분위_박스플롯.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: image/migration_복합점수_분위_박스플롯.png')

# ── 요약 통계 출력 ────────────────────────────────────────────
print('\n=== 연도별 상관계수 (복합점수 × 20대 이동비율) ===')
for yr in years:
    sub = df[df['연도'] == yr].dropna(subset=['복합점수','20대_비율'])
    r, p = stats.pearsonr(sub['복합점수'], sub['20대_비율'])
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    print(f'  {yr}년: r={r:.3f}  p={p:.4f} {sig}  (n={len(sub)})')

print('\n=== 복합점수 분위 상위20% vs 하위20% 20대 이동비율 평균 ===')
top = df[df['분위구간']=='상위20%'].groupby('연도')['20대_비율'].mean()
bot = df[df['분위구간']=='하위20%'].groupby('연도')['20대_비율'].mean()
for yr in years:
    print(f'  {yr}년: 상위20% {top[yr]:.1f}%  vs  하위20% {bot[yr]:.1f}%  (차이 {top[yr]-bot[yr]:.1f}%p)')
