# -*- coding: utf-8 -*-
"""
20대 이동 선행 → 매출 반응 시각화
출력: image/20대_선행매출_분석.png
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT     = Path(__file__).resolve().parent
SUM_PATH = ROOT / 'data/20대_선행매출_분석.xlsx'
IMG_DIR  = ROOT / 'image'

summary = pd.read_excel(SUM_PATH, sheet_name='①요약_가설부합행정동')
yoy_df  = pd.read_excel(SUM_PATH, sheet_name='③YoY비교_분기별')

TOP_N   = 15
top     = summary.nlargest(TOP_N, 'CCF_최고r').reset_index(drop=True)

LAG_COLOR = {1: '#E05C5C', 2: '#F5A623', 3: '#27AE60', 4: '#3A5CA9'}

# ══════════════════════════════════════════════
# 레이아웃: 2행 × 2열
# ══════════════════════════════════════════════
fig = plt.figure(figsize=(22, 18))
fig.suptitle('20대 이동인구 선행 → 매출 반응 분석 (2020–2024, 423개 행정동)',
             fontsize=15, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig,
                       hspace=0.40, wspace=0.32,
                       left=0.07, right=0.97, top=0.93, bottom=0.06)

ax1 = fig.add_subplot(gs[0, 0])   # 패널1: Top15 CCF r 막대
ax2 = fig.add_subplot(gs[0, 1])   # 패널2: 선행분기 분포 + 산포도
ax3 = fig.add_subplot(gs[1, 0])   # 패널3: Top6 YoY 시계열
ax4 = fig.add_subplot(gs[1, 1])   # 패널4: 세대별 매출 비중 변화

# ──────────────────────────────────────────────
# 패널 1: Top15 행정동 CCF r 가로 막대
# ──────────────────────────────────────────────
colors1 = [LAG_COLOR.get(lag, '#999') for lag in top['CCF_선행분기']]
bars = ax1.barh(top['행정동'][::-1], top['CCF_최고r'][::-1],
                color=colors1[::-1], edgecolor='white', height=0.7)

for bar, (_, row) in zip(bars, top[::-1].iterrows()):
    ax1.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height()/2,
             f"lag={int(row['CCF_선행분기'])}Q  +{row['매출성장률_2020→2024(%)']:.0f}%",
             va='center', fontsize=8, color='#444')

ax1.axvline(0.35, color='#AAAAAA', lw=1.2, ls='--', label='임계값 r=0.35')
ax1.set_xlim(0, 1.15)
ax1.set_xlabel('CCF 최고 상관계수 (r)', fontsize=10)
ax1.set_title(f'Top {TOP_N} 행정동 — CCF r값 (색상=선행분기)', fontsize=11)
ax1.set_facecolor('#F8F8F8')
ax1.grid(axis='x', alpha=0.2)
ax1.tick_params(axis='y', labelsize=9)

legend_h = [mpatches.Patch(color=c, label=f'선행 {l}분기')
            for l, c in LAG_COLOR.items()]
ax1.legend(handles=legend_h, fontsize=8.5, loc='lower right')

# ──────────────────────────────────────────────
# 패널 2: 20대 이동 변화율 vs 매출 성장률 산포도
# ──────────────────────────────────────────────
sc_colors = [LAG_COLOR.get(lag, '#999') for lag in summary['CCF_선행분기']]
sc = ax2.scatter(summary['20대이동_변화(%)'], summary['매출성장률_2020→2024(%)'],
                 c=sc_colors, alpha=0.75, s=60,
                 edgecolors='white', linewidths=0.8, zorder=3)

# Top10 레이블
for _, row in summary.nlargest(10, 'CCF_최고r').iterrows():
    ax2.annotate(row['행정동'],
                 (row['20대이동_변화(%)'], row['매출성장률_2020→2024(%)']),
                 xytext=(4, 3), textcoords='offset points',
                 fontsize=7.5, color='#333')

ax2.axhline(0, color='#CCCCCC', lw=1, ls='--')
ax2.axvline(0, color='#CCCCCC', lw=1, ls='--')
ax2.set_xlabel('20대 이동인구 변화율 2020→2024 (%)', fontsize=10)
ax2.set_ylabel('총매출 성장률 2020→2024 (%)', fontsize=10)
ax2.set_title('20대 이동 변화 vs 매출 성장 (가설 부합 67개 동)', fontsize=11)
ax2.set_facecolor('#F8F8F8')
ax2.grid(alpha=0.15)
ax2.legend(handles=legend_h, fontsize=8.5, loc='upper left')

# outlier 클리핑
p95x = np.nanpercentile(summary['20대이동_변화(%)'].dropna(), 95)
p95y = np.nanpercentile(summary['매출성장률_2020→2024(%)'].dropna(), 95)
ax2.set_xlim(summary['20대이동_변화(%)'].min() - 10, min(p95x + 30, 300))
ax2.set_ylim(-20, min(p95y + 50, 700))

# ──────────────────────────────────────────────
# 패널 3: Top6 행정동 20대 이동 YoY vs 매출 YoY 시계열
# 개포1동 등 이상치 제외 (매출 성장률 200% 이하만)
# ──────────────────────────────────────────────
top6_cand = summary[summary['매출성장률_2020→2024(%)'] <= 200].nlargest(6, 'CCF_최고r')['행정동'].tolist()
cmap6 = plt.cm.get_cmap('tab10', 6)
xl = []

for i, dong in enumerate(top6_cand):
    sub = yoy_df[yoy_df['행정동'] == dong].sort_values('연도분기')
    if sub.empty:
        continue
    xl = sub['연도분기'].tolist()
    x  = range(len(sub))
    c  = cmap6(i)

    ax3.plot(x, sub['20대이동_YoY(%)'], color=c, lw=1.8, ls='--',
             marker='o', markersize=4, alpha=0.85)
    ax3.plot(x, sub['매출_YoY(%)'], color=c, lw=2.2, ls='-',
             marker='s', markersize=4, alpha=0.85, label=dong)

ax3.axhline(0, color='#AAAAAA', lw=1, ls='-')
if xl:
    ax3.set_xticks(range(len(xl)))
    ax3.set_xticklabels(xl, rotation=45, fontsize=7.5, ha='right')

# y축 클리핑
yvals = yoy_df[yoy_df['행정동'].isin(top6_cand)][['20대이동_YoY(%)','매출_YoY(%)']].stack().dropna()
ax3.set_ylim(max(yvals.min() - 10, -120), min(yvals.max() + 20, 200))

ax3.set_ylabel('YoY 증감률 (%)', fontsize=10)
ax3.set_title('Top6 행정동 — 20대 이동 YoY(점선) vs 매출 YoY(실선)\n(매출 성장률 200% 이하 기준)', fontsize=11)
ax3.set_facecolor('#F8F8F8')
ax3.grid(alpha=0.15)

# 범례: 행정동명
dong_legend = ax3.legend(fontsize=8, loc='upper left', ncol=2, title='행정동', title_fontsize=8)
ax3.add_artist(dong_legend)

# 범례: 선 종류
line_handles = [
    Line2D([0],[0], color='gray', lw=1.8, ls='--', marker='o', markersize=4, label='20대 이동 YoY'),
    Line2D([0],[0], color='gray', lw=2.2, ls='-',  marker='s', markersize=4, label='매출 YoY'),
]
ax3.legend(handles=line_handles, fontsize=8, loc='lower left')

# ──────────────────────────────────────────────
# 패널 4: Top12 세대별 매출 비중 변화 (2020 → 2024)
# ──────────────────────────────────────────────
top12 = summary.nlargest(12, 'CCF_최고r').reset_index(drop=True)

ages  = ['20대','30대','40대','50대']
cols20 = [f'{a}매출비중_2020(%)' for a in ages]
cols24 = [f'{a}매출비중_2024(%)' for a in ages]
age_colors = ['#E05C5C','#F5A623','#3A5CA9','#27AE60']

x_pos = np.arange(len(top12))
width = 0.35

for j, (age, c20, c24, col) in enumerate(zip(ages, cols20, cols24, age_colors)):
    delta = top12[c24].values - top12[c20].values
    bars4 = ax4.bar(x_pos + j*0.18 - 0.27, delta, width=0.16,
                    color=col, alpha=0.8, label=age)

ax4.axhline(0, color='#888', lw=1)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(top12['행정동'], rotation=45, fontsize=8, ha='right')
ax4.set_ylabel('매출 비중 변화 (%p, 2024-2020)', fontsize=10)
ax4.set_title('Top12 행정동 — 세대별 매출 비중 변화 (2024−2020)', fontsize=11)
ax4.set_facecolor('#F8F8F8')
ax4.grid(axis='y', alpha=0.2)
ax4.legend(fontsize=9, loc='upper right')

# ──────────────────────────────────────────────
# 저장
# ──────────────────────────────────────────────
out = IMG_DIR / '20대_선행매출_분석.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[저장] {out.name}")
