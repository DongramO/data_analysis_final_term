# -*- coding: utf-8 -*-
"""
요일별 매출 패턴으로 트렌드 상권 세분화
- 7일 패턴 기반 클러스터링 (외부유입형 / 오피스형 / 주거지역형 / 혼합형)
- 금요일 효과: 금>목 비율로 주말 시작 특성 측정
- 토/일 비교: 토요일 우세 vs 일요일 우세 상권 구분
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

data_dir = 'data'
image_dir = 'image'

df = pd.read_csv(
    os.path.join(data_dir, '서울시_상권분석서비스(추정매출-행정동)_2020_2024.csv'),
    encoding='utf-8-sig',
)
df['year'] = df['기준_년분기_코드'] // 10

DAY_COLS = ['월요일_매출_금액', '화요일_매출_금액', '수요일_매출_금액',
            '목요일_매출_금액', '금요일_매출_금액', '토요일_매출_금액', '일요일_매출_금액']
DAY_LABELS = ['월', '화', '수', '목', '금', '토', '일']

# ── 행정동별 요일 합계 ────────────────────────────────────────
agg_dict = {col: 'sum' for col in DAY_COLS}
agg_dict.update({'당월_매출_금액': 'sum', '연령대_20_매출_금액': 'sum', '연령대_30_매출_금액': 'sum'})

stats = df.groupby('행정동_코드_명').agg(agg_dict)
stats['전체매출'] = stats['당월_매출_금액']
stats['2030비중'] = (stats['연령대_20_매출_금액'] + stats['연령대_30_매출_금액']) / stats['전체매출']

# 요일별 비중 (7일 합 기준)
for col, label in zip(DAY_COLS, DAY_LABELS):
    stats[f'{label}비중'] = stats[col] / stats[DAY_COLS].sum(axis=1)

# 핵심 파생 지표
stats['주말비중']   = (stats['토요일_매출_금액'] + stats['일요일_매출_금액']) / stats[DAY_COLS].sum(axis=1)
stats['토일비율']   = stats['토요일_매출_금액'] / stats['일요일_매출_금액']   # 토 vs 일
stats['금요효과']   = stats['금요일_매출_금액'] / stats['목요일_매출_금액']   # 금>목 = 주말 시작형
stats['주말가속도'] = stats['주말비중'] / (stats['월비중'] + stats['화비중'] + stats['수비중']) * 3

print("=== 요일별 비중 상위/하위 상권 ===")
print("\n[주말 비중 상위 10]")
print(stats.nlargest(10, '주말비중')[['주말비중','토일비율','금요효과','2030비중']].round(3).to_string())
print("\n[주말 비중 하위 10]")
print(stats.nsmallest(10, '주말비중')[['주말비중','토일비율','금요효과','2030비중']].round(3).to_string())

print("\n[금요일 효과 상위 10 — 주말 시작 상권]")
print(stats.nlargest(10, '금요효과')[['금요효과','주말비중','2030비중']].round(3).to_string())

print("\n[토일비율 — 토요일 우세 상위 10]")
print(stats.nlargest(10, '토일비율')[['토일비율','주말비중','2030비중']].round(3).to_string())


# ════════════════════════════════════════════════════════════
# 1. 주요 트렌드 상권 Top 20 요일별 패턴 히트맵
# ════════════════════════════════════════════════════════════
wk_thr = stats['주말비중'].quantile(0.70)
mz_thr = stats['2030비중'].quantile(0.70)
trend_dongs = stats[(stats['주말비중'] >= wk_thr) & (stats['2030비중'] >= mz_thr)].index
top20 = stats.loc[trend_dongs].nlargest(20, '주말비중')

heatmap_data = top20[[f'{d}비중' for d in DAY_LABELS]]
heatmap_data.columns = DAY_LABELS

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('트렌드 상권 요일별 매출 패턴', fontsize=14, fontweight='bold')

ax = axes[0]
im = ax.imshow(heatmap_data.values, aspect='auto', cmap='RdYlGn', vmin=0.08, vmax=0.22)
ax.set_xticks(range(7))
ax.set_xticklabels(DAY_LABELS, fontsize=11)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20.index, fontsize=9)
for i in range(len(top20)):
    for j in range(7):
        val = heatmap_data.values[i, j]
        ax.text(j, i, f'{val:.1%}', ha='center', va='center', fontsize=7.5,
                color='black' if 0.10 < val < 0.18 else 'white')
plt.colorbar(im, ax=ax, label='요일별 매출 비중')
ax.set_title(f'트렌드 상권 Top 20 요일별 매출 비중\n(주말비중≥{wk_thr:.1%} & 2030비중≥{mz_thr:.1%})', fontsize=11)
ax.set_xlabel('요일')


# ════════════════════════════════════════════════════════════
# 2. 금요일 효과 + 주말비중 산점도 (트렌드 상권 하이라이트)
# ════════════════════════════════════════════════════════════
ax = axes[1]
is_trend = stats.index.isin(trend_dongs)

ax.scatter(stats.loc[~is_trend, '주말비중'] * 100,
           stats.loc[~is_trend, '금요효과'],
           alpha=0.35, color='#aec7e8', s=40, label='일반 상권', edgecolors='none')
ax.scatter(stats.loc[is_trend, '주말비중'] * 100,
           stats.loc[is_trend, '금요효과'],
           alpha=0.75, color='#e15759', s=60, label='트렌드 상권', edgecolors='white', linewidths=0.5)

highlight_dongs = stats.loc[is_trend].nlargest(12, '주말비중').index
for dong in highlight_dongs:
    row = stats.loc[dong]
    ax.annotate(dong, (row['주말비중'] * 100, row['금요효과']),
                fontsize=7, xytext=(3, 2), textcoords='offset points', color='#c0392b')

ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='금=목 (효과 없음)')
ax.set_xlabel('주말 매출 비중 (%) — 7일 기준', fontsize=11)
ax.set_ylabel('금요일 효과 (금요일/목요일 매출)', fontsize=11)
ax.set_title('주말비중 vs 금요일 효과\n(금요효과>1.0 = 주말 유동 시작)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'dayofweek_heatmap.png'), dpi=150, bbox_inches='tight')
print("\n저장: image/dayofweek_heatmap.png")
plt.show()


# ════════════════════════════════════════════════════════════
# 3. 요일 프로파일 4분류 클러스터링
#    토일비중 vs 금효과 기준 4분면 분류
# ════════════════════════════════════════════════════════════
wk_med = stats['주말비중'].median()
fri_med = stats['금요효과'].median()

def classify(row):
    if row['주말비중'] >= wk_med and row['금요효과'] >= fri_med:
        return '주말집중+금요선행'
    elif row['주말비중'] >= wk_med and row['금요효과'] < fri_med:
        return '주말집중+주중평탄'
    elif row['주말비중'] < wk_med and row['금요효과'] >= fri_med:
        return '평일집중+금요약상승'
    else:
        return '평일집중+주중평탄'

stats['패턴유형'] = stats.apply(classify, axis=1)
print("\n=== 요일 패턴 4분류 ===")
print(stats['패턴유형'].value_counts())

CLUSTER_COLORS = {
    '주말집중+금요선행': '#e15759',
    '주말집중+주중평탄': '#f28e2b',
    '평일집중+금요약상승': '#4e79a7',
    '평일집중+주중평탄': '#76b7b2',
}

fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))
fig2.suptitle('요일 패턴 유형별 분류 및 특성', fontsize=14, fontweight='bold')

ax = axes2[0]
for ptype, color in CLUSTER_COLORS.items():
    sub = stats[stats['패턴유형'] == ptype]
    ax.scatter(sub['주말비중'] * 100, sub['금요효과'],
               color=color, alpha=0.6, s=45, label=f'{ptype} ({len(sub)}개)',
               edgecolors='white', linewidths=0.4)

# 트렌드 상권 테두리 강조
for dong in trend_dongs[:30]:
    if dong in stats.index:
        row = stats.loc[dong]
        ax.scatter(row['주말비중'] * 100, row['금요효과'],
                   edgecolors='black', facecolors='none', s=70, linewidths=1.2)

ax.axhline(fri_med, color='gray', linestyle=':', linewidth=1)
ax.axvline(wk_med * 100, color='gray', linestyle=':', linewidth=1)
ax.set_xlabel('주말 매출 비중 (%) — 7일 기준', fontsize=11)
ax.set_ylabel('금요일 효과 (금요일/목요일)', fontsize=11)
ax.set_title('4분면 패턴 분류\n(검은 테두리 = 트렌드 상권)', fontsize=11)
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)


# ════════════════════════════════════════════════════════════
# 4. 트렌드 상권 Top 15 토요일 vs 일요일 비교
# ════════════════════════════════════════════════════════════
ax = axes2[1]
top15 = stats.loc[trend_dongs].nlargest(15, '주말비중').sort_values('토일비율')

sat_share = top15['토비중'] * 100
sun_share = top15['일비중'] * 100
y = range(len(top15))

ax.barh(y, sat_share, color='#e15759', alpha=0.8, label='토요일', height=0.4, align='center')
ax.barh([i + 0.42 for i in y], sun_share, color='#4e79a7', alpha=0.8, label='일요일', height=0.4, align='center')

for i, (dong, row) in enumerate(top15.iterrows()):
    ax.text(row['토비중'] * 100 + 0.1, i, f'{row["토비중"]:.1%}', va='center', fontsize=7.5, color='#c0392b')
    ax.text(row['일비중'] * 100 + 0.1, i + 0.42, f'{row["일비중"]:.1%}', va='center', fontsize=7.5, color='#2c5f8a')

ax.set_yticks([i + 0.21 for i in y])
ax.set_yticklabels(top15.index, fontsize=9)
ax.set_xlabel('요일별 매출 비중 (%)', fontsize=11)
ax.set_title('트렌드 상권 Top 15\n토요일 vs 일요일 매출 비중', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, axis='x', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'dayofweek_cluster.png'), dpi=150, bbox_inches='tight')
print("저장: image/dayofweek_cluster.png")
plt.show()


# ════════════════════════════════════════════════════════════
# 5. 연도별 요일 패턴 변화 — 트렌드 상권 Top 8
# ════════════════════════════════════════════════════════════
top8 = stats.loc[trend_dongs].nlargest(8, '주말비중').index.tolist()

annual_day = (
    df[df['행정동_코드_명'].isin(top8)]
    .groupby(['year', '행정동_코드_명'])[DAY_COLS]
    .sum()
    .reset_index()
)
for col, label in zip(DAY_COLS, DAY_LABELS):
    annual_day[f'{label}비중'] = annual_day[col] / annual_day[DAY_COLS].sum(axis=1)
annual_day['주말비중_일별'] = annual_day['토비중'] + annual_day['일비중']

fig3, axes3 = plt.subplots(2, 4, figsize=(20, 10))
fig3.suptitle('트렌드 상권 Top 8: 연도별 요일 패턴 변화 (2020→2024)', fontsize=14, fontweight='bold')
axes3 = axes3.flatten()

cmap = plt.colormaps.get_cmap('viridis').resampled(5)
years = sorted(annual_day['year'].unique())

for idx, dong in enumerate(top8):
    ax = axes3[idx]
    sub = annual_day[annual_day['행정동_코드_명'] == dong].sort_values('year')
    for yi, yr in enumerate(years):
        row = sub[sub['year'] == yr]
        if len(row) == 0:
            continue
        vals = [row[f'{d}비중'].values[0] * 100 for d in DAY_LABELS]
        ax.plot(DAY_LABELS, vals, marker='o', markersize=5,
                color=cmap(yi), linewidth=1.8, label=str(yr), alpha=0.85)

    ax.axvspan(4.5, 6.5, alpha=0.08, color='red')
    ax.set_title(dong, fontsize=11, fontweight='bold')
    ax.set_ylabel('매출 비중 (%)', fontsize=8)
    ax.set_ylim(0, None)
    ax.legend(fontsize=6.5, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'dayofweek_trend_change.png'), dpi=150, bbox_inches='tight')
print("저장: image/dayofweek_trend_change.png")
plt.show()

print("\n완료!")
