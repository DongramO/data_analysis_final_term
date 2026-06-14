# -*- coding: utf-8 -*-
"""
성수2가1동 분기별 상위 5개 업종 매출 막대그래프
출력: image/성수2가1동_업종비율.png
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT  = Path(__file__).resolve().parent
SRC   = ROOT / 'data/archive/서울시_상권분석서비스(추정매출-행정동)_2020_2024.csv'
DONG  = '성수2가1동'

# ── 데이터 로드 (성수2가1동만) ────────────────────────────────
df = pd.read_csv(SRC, encoding='utf-8-sig',
                 usecols=['기준_년분기_코드','행정동_코드_명',
                          '서비스_업종_코드_명','당월_매출_금액'])
df.columns = ['yrq','dong','업종','매출']
sub = df[df['dong'] == DONG].copy()

# 분기 순서
QUARTERS = sorted(sub['yrq'].unique())
NQ = len(QUARTERS)

# ── 분기별 총매출 및 비율 ─────────────────────────────────────
agg = sub.groupby(['yrq','업종'])['매출'].sum().reset_index()
total_q = agg.groupby('yrq')['매출'].sum().rename('total')
agg = agg.merge(total_q, on='yrq')
agg['비율'] = agg['매출'] / agg['total'] * 100

# ── 분기별 상위 5개 업종 추출 ────────────────────────────────
top5_per_q = (agg.sort_values(['yrq','비율'], ascending=[True, False])
                 .groupby('yrq')
                 .head(5)
                 .reset_index(drop=True))

# 전체 등장 업종 (색상 고정용) — 전 분기 통합 상위 빈도순
업종_freq = top5_per_q.groupby('업종')['비율'].mean().sort_values(ascending=False)
ALL_업종  = 업종_freq.index.tolist()

COLORS = [
    '#E05C5C','#F5A623','#3A5CA9','#2E8B57','#7B2D8B',
    '#5B9BD5','#ED7D31','#A5A5A5','#4472C4','#FF6699',
    '#00B0F0','#CC3300','#7030A0','#92D050','#00B050',
]
COLOR_MAP = {업종: COLORS[i % len(COLORS)] for i, 업종 in enumerate(ALL_업종)}

# ── X축 레이블 ───────────────────────────────────────────────
def q_labels(quarters):
    labels, prev_yr = [], None
    for q in quarters:
        yr = str(q)[:4]
        label = f"'{yr[2:]}Q{str(q)[-1]}" if yr != prev_yr else f"Q{str(q)[-1]}"
        labels.append(label)
        prev_yr = yr
    return labels

XTICK = q_labels(QUARTERS)

# 연도 경계
def yr_bounds(quarters):
    bounds, prev_yr = [], None
    for i, q in enumerate(quarters):
        yr = str(q)[:4]
        if yr != prev_yr and i > 0:
            bounds.append(i - 0.5)
        prev_yr = yr
    return bounds

# ── 시각화 ───────────────────────────────────────────────────
fig, (ax_main, ax_pct) = plt.subplots(2, 1, figsize=(22, 12),
                                       gridspec_kw={'height_ratios': [3, 2]})
fig.suptitle(f'{DONG}  분기별 상위 5개 업종  매출 비교  (2020Q1 ~ 2024Q4)',
             fontsize=14, fontweight='bold')

BAR_W   = 0.15
OFFSETS = np.linspace(-2, 2, 5) * BAR_W

# ── 패널1: 절대 매출 (억 원) 묶음 막대 ──────────────────────
for qi, q in enumerate(QUARTERS):
    rows = top5_per_q[top5_per_q['yrq'] == q].sort_values('비율', ascending=False)
    for rank, (_, row) in enumerate(rows.iterrows()):
        color = COLOR_MAP[row['업종']]
        bar = ax_main.bar(qi + OFFSETS[rank], row['매출'] / 1e8,
                          width=BAR_W * 0.9, color=color,
                          edgecolor='white', linewidth=0.4)

for xb in yr_bounds(QUARTERS):
    ax_main.axvline(xb, color='#BBBBBB', lw=0.8, ls='--')

ax_main.set_xticks(range(NQ))
ax_main.set_xticklabels(XTICK, fontsize=8, rotation=45, ha='right')
ax_main.set_ylabel('매출 (억 원)', fontsize=11)
ax_main.set_title('① 분기별 상위 5개 업종 매출 (절대값)', fontsize=11, fontweight='bold')
ax_main.grid(axis='y', alpha=0.25)
ax_main.set_xlim(-0.6, NQ - 0.4)

# ── 패널2: 비율 (%) 묶음 막대 ────────────────────────────────
for qi, q in enumerate(QUARTERS):
    rows = top5_per_q[top5_per_q['yrq'] == q].sort_values('비율', ascending=False)
    for rank, (_, row) in enumerate(rows.iterrows()):
        color = COLOR_MAP[row['업종']]
        ax_pct.bar(qi + OFFSETS[rank], row['비율'],
                   width=BAR_W * 0.9, color=color,
                   edgecolor='white', linewidth=0.4)

for xb in yr_bounds(QUARTERS):
    ax_pct.axvline(xb, color='#BBBBBB', lw=0.8, ls='--')

ax_pct.set_xticks(range(NQ))
ax_pct.set_xticklabels(XTICK, fontsize=8, rotation=45, ha='right')
ax_pct.set_ylabel('매출 비율 (%)', fontsize=11)
ax_pct.set_title('② 분기별 상위 5개 업종 매출 비율', fontsize=11, fontweight='bold')
ax_pct.grid(axis='y', alpha=0.25)
ax_pct.set_xlim(-0.6, NQ - 0.4)

# ── 공통 범례 ────────────────────────────────────────────────
from matplotlib.patches import Patch
handles = [Patch(facecolor=COLOR_MAP[u], label=u) for u in ALL_업종]
fig.legend(handles=handles, loc='center right', fontsize=9,
           bbox_to_anchor=(1.13, 0.5), framealpha=0.9,
           title='업종', title_fontsize=10)

plt.tight_layout()
out = ROOT / 'image/성수2가1동_업종비율.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'[저장] {out.name}')
