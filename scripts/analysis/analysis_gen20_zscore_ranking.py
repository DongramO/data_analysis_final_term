# -*- coding: utf-8 -*-
"""
20대 매출 z-score 3개 지표 TOP20 시각화
출력: image/gen20_zscore_ranking.png
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
df = pd.read_csv(ROOT/'data/main_data/분기별_원본피처.csv', encoding='utf-8-sig')

# ── 연도별 집계 ───────────────────────────────────────────────
ann = df.groupby(['행정동','연도'])[['연령대_20_매출_금액','총_매출금액']].sum().reset_index()

y23 = ann[ann['연도']==2023].set_index('행정동')
y24 = ann[ann['연도']==2024].set_index('행정동')

base = pd.DataFrame({
    '20대_매출_억':   y24['연령대_20_매출_금액'] / 1e8,
    '총매출_억':      y24['총_매출금액'] / 1e8,
    '20대_매출_23억': y23['연령대_20_매출_금액'] / 1e8,
}).dropna()

base['20대_비중_%']  = base['20대_매출_억'] / base['총매출_억'] * 100
base['성장률_%']     = (base['20대_매출_억'] - base['20대_매출_23억']) / base['20대_매출_23억'].abs() * 100

# ── z-score 변환 ─────────────────────────────────────────────
# 규모·비중: 그대로
base['z_규모'] = sp_zscore(base['20대_매출_억'], ddof=0)
base['z_비중'] = sp_zscore(base['20대_비중_%'],  ddof=0)

# 성장률: 클리핑 없이 원값 z-score (단, 상하위 0.5% 클리핑으로 극단 이상치만 제거)
g = base['성장률_%']
lo, hi = g.quantile(0.005), g.quantile(0.995)
base['z_성장률'] = sp_zscore(g.clip(lo, hi), ddof=0)

# ── TOP20 추출 ────────────────────────────────────────────────
top_규모   = base[['z_규모','20대_매출_억']].sort_values('z_규모', ascending=False).head(20)
top_비중   = base[['z_비중','20대_비중_%']].sort_values('z_비중', ascending=False).head(20)
top_성장률 = base[['z_성장률','성장률_%','20대_매출_억']].sort_values('z_성장률', ascending=False).head(20)

# ── 시각화 ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 10))
fig.suptitle("20대 매출 z-score 기준 TOP 20  (2024년 기준 / 성장률: 2023→2024)",
             fontsize=14, fontweight='bold', y=1.01)

PALETTES = {
    'z_규모':   ('#2E86AB', '#A8DADC'),
    'z_비중':   ('#E05C5C', '#F4BFBF'),
    'z_성장률': ('#2E8B57', '#A8D5B5'),
}

def draw_bar(ax, data, z_col, label_col, title, unit, color_hi, color_lo, rank_col=None):
    dongs = data.index[::-1]          # 낮은 순위가 아래
    zvals = data[z_col].values[::-1]
    lvals = data[label_col].values[::-1]
    ranks = list(range(len(dongs), 0, -1))  # 20→1

    colors = [color_hi if z >= 2.0 else color_lo for z in zvals]
    bars = ax.barh(range(len(dongs)), zvals, color=colors, edgecolor='white',
                   linewidth=0.5, height=0.7)

    for i, (dong, z, lv, rk) in enumerate(zip(dongs, zvals, lvals, ranks)):
        # 왼쪽: 순위 + 행정동명
        ax.text(-0.08, i, f'{len(dongs)-i}위  {dong}',
                ha='right', va='center', fontsize=9.5, color='#333333')
        # 오른쪽: 실제 수치
        ax.text(z + 0.05, i, f'{lv:.1f}{unit}',
                ha='left', va='center', fontsize=8.5, color='#555555')

    ax.set_xlim(-0.5, max(zvals) * 1.22)
    ax.axvline(0, color='#AAAAAA', lw=0.8)
    ax.axvline(2.0, color='#CCCCCC', lw=0.8, ls='--')
    ax.text(2.0, len(dongs) - 0.3, 'z=2', fontsize=7.5, color='#AAAAAA', ha='center')
    ax.set_yticks([])
    ax.set_xlabel('z-score', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.spines[['top','right','left']].set_visible(False)
    ax.grid(axis='x', alpha=0.2)

draw_bar(axes[0], top_규모,   'z_규모',   '20대_매출_억',
         '① 20대 매출 규모\n(2024년 절대 매출액)',
         '억', *PALETTES['z_규모'])

draw_bar(axes[1], top_비중,   'z_비중',   '20대_비중_%',
         '② 20대 매출 비중\n(전체 세대 中 20대 점유율)',
         '%', *PALETTES['z_비중'])

draw_bar(axes[2], top_성장률, 'z_성장률', '성장률_%',
         '③ 20대 매출 성장률\n(2023→2024 YoY)',
         '%', *PALETTES['z_성장률'])

# 범례
for ax, (c_hi, c_lo) in zip(axes, PALETTES.values()):
    ax.legend(handles=[
        mpatches.Patch(color=c_hi, label='z ≥ 2.0 (상위 ~2.3%)'),
        mpatches.Patch(color=c_lo, label='z < 2.0'),
    ], fontsize=8, loc='lower right')

plt.tight_layout()
out = ROOT / 'image/gen20_zscore_ranking.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'[저장] {out.name}')
