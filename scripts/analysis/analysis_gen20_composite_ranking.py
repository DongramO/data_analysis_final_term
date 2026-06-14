# -*- coding: utf-8 -*-
"""
20대 매출 z-score 복합점수 TOP20 시각화
성장률: 2020~2024 전체 YoY(4개) z-score 평균 사용
출력: image/gen20_composite_ranking.png
"""
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import zscore as sp_zscore
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path('.')
df = pd.read_csv(ROOT / 'data/main_data/분기별_원본피처.csv', encoding='utf-8-sig')

# ── 연도별 집계 ───────────────────────────────────────────────
ann = (df.groupby(['행정동', '연도'])[['연령대_20_매출_금액', '총_매출금액']]
         .sum().reset_index())

# 연도별 피벗 (행정동 × 연도)
piv = ann.pivot(index='행정동', columns='연도', values='연령대_20_매출_금액')
piv_tot = ann.pivot(index='행정동', columns='연도', values='총_매출금액')

# 모든 연도가 존재하는 행정동만 유지
years = [2020, 2021, 2022, 2023, 2024]
piv     = piv.dropna(subset=years)
piv_tot = piv_tot.reindex(piv.index)

# ── 지표 계산 ─────────────────────────────────────────────────
base = pd.DataFrame(index=piv.index)

# 규모: 2024년 20대 매출 절대값
base['20대_매출_억']  = piv[2024] / 1e8
base['총매출_억']     = piv_tot[2024] / 1e8

# 비중: 2024년 기준 (전체 주 기준 — 연령×요일 교차 데이터 없음)
base['20대_비중_%']   = base['20대_매출_억'] / base['총매출_억'] * 100

# YoY 성장률 4개
YOY_PAIRS = [(2020,2021), (2021,2022), (2022,2023), (2023,2024)]
for y0, y1 in YOY_PAIRS:
    col = f'성장률_{y0}v{y1}%'
    base[col] = (piv[y1] - piv[y0]) / piv[y0].abs() * 100

# ── z-score 변환 ─────────────────────────────────────────────
base['z_규모'] = sp_zscore(base['20대_매출_억'], ddof=0)
base['z_비중'] = sp_zscore(base['20대_비중_%'],  ddof=0)

# YoY별 z-score (각각 0.5% 클리핑 후 표준화)
yoy_z_cols = []
for y0, y1 in YOY_PAIRS:
    raw_col = f'성장률_{y0}v{y1}%'
    z_col   = f'z_성장률_{y0}v{y1}'
    g = base[raw_col]
    lo, hi = g.quantile(0.005), g.quantile(0.995)
    base[z_col] = sp_zscore(g.clip(lo, hi), ddof=0)
    yoy_z_cols.append(z_col)

# 성장률 컴포넌트: 4개 YoY z-score 평균
base['z_성장률_avg'] = base[yoy_z_cols].mean(axis=1)

# ── 복합점수 ─────────────────────────────────────────────────
base['복합점수'] = (base['z_규모'] + base['z_비중'] + base['z_성장률_avg']) / 3
base = base.sort_values('복합점수', ascending=False)

top20 = base.head(20).copy()

# ── 시각화 ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 10))
fig.suptitle(
    "20대 매출 복합점수 TOP 20\n"
    "(z_규모 + z_비중 + z_성장률_avg) ÷ 3  |  규모·비중: 2024년 / 성장률: 2020→2024 YoY 4개 평균",
    fontsize=12, fontweight='bold', y=1.01
)

dongs  = top20.index[::-1]
scores = top20['복합점수'].values[::-1]
z_r    = top20['z_규모'].values[::-1]
z_b    = top20['z_비중'].values[::-1]
z_g    = top20['z_성장률_avg'].values[::-1]

COLOR_HI = '#4A4ABF'
COLOR_LO = '#A8AADE'
colors = [COLOR_HI if s >= 1.0 else COLOR_LO for s in scores]

ax.barh(range(len(dongs)), scores, color=colors,
        edgecolor='white', linewidth=0.6, height=0.7)

for i, (dong, sc, zr, zb, zg) in enumerate(zip(dongs, scores, z_r, z_b, z_g)):
    rank = len(dongs) - i
    ax.text(-0.05, i, f'{rank}위  {dong}',
            ha='right', va='center', fontsize=10, color='#333333')
    ax.text(sc + 0.02, i,
            f'{sc:.3f}   (규모 {zr:+.2f} / 비중 {zb:+.2f} / 성장avg {zg:+.2f})',
            ha='left', va='center', fontsize=8.5, color='#555555')

ax.axvline(0,   color='#AAAAAA', lw=0.8)
ax.axvline(1.0, color='#CCCCCC', lw=0.8, ls='--')
ax.text(1.0, len(dongs) - 0.4, '복합=1.0', fontsize=7.5, color='#AAAAAA', ha='center')

ax.set_xlim(-0.3, scores.max() * 1.55)
ax.set_yticks([])
ax.set_xlabel('복합점수 (z-score 평균)', fontsize=11)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.grid(axis='x', alpha=0.2)
ax.legend(handles=[
    mpatches.Patch(color=COLOR_HI, label='복합점수 ≥ 1.0'),
    mpatches.Patch(color=COLOR_LO, label='복합점수 < 1.0'),
], fontsize=9, loc='lower right')

plt.tight_layout()
fig.savefig(ROOT / 'image/gen20_composite_ranking.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('[저장] gen20_composite_ranking.png')
