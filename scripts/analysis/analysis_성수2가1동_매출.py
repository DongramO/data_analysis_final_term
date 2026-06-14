# -*- coding: utf-8 -*-
"""
성수2가1동 매출 변화 시각화
출력: image/성수2가1동_매출변화.png
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT  = Path(__file__).resolve().parent
SRC_Q = ROOT / 'data/main_data/분기별_원본피처.csv'
DONG  = '성수2가1동'

# ── 데이터 로드 ───────────────────────────────────────────────
df = pd.read_csv(SRC_Q, encoding='utf-8-sig')
df.columns = [c.strip() for c in df.columns]
sub = df[df['행정동'] == DONG].sort_values('연도분기').reset_index(drop=True)

QUARTERS = sub['연도분기'].tolist()
NQ = len(QUARTERS)
X  = np.arange(NQ)

def q_labels(quarters):
    labels, prev_yr = [], None
    for q in quarters:
        yr = str(q)[:4]
        label = f"'{yr[2:]}Q{str(q)[-1]}" if yr != prev_yr else f"Q{str(q)[-1]}"
        labels.append(label)
        prev_yr = yr
    return labels

def yr_bounds(quarters):
    bounds, prev_yr = [], None
    for i, q in enumerate(quarters):
        yr = str(q)[:4]
        if yr != prev_yr and i > 0:
            bounds.append(i - 0.5)
        prev_yr = yr
    return bounds

XTICK   = q_labels(QUARTERS)
YBOUNDS = yr_bounds(QUARTERS)

# ── 총매출 ────────────────────────────────────────────────────
total = sub['총_매출금액'].values
total_억 = total / 1e8

qoq = np.full(NQ, np.nan)
qoq[1:] = (total_억[1:] - total_억[:-1]) / total_억[:-1] * 100

# ── 연령대별 ─────────────────────────────────────────────────
GENS = ['10이하', '20대', '30대', '40대', '50대', '60이상']
GEN_COLORS = ['#DDDDDD', '#E05C5C', '#F5A623', '#3A5CA9', '#7B2D8B', '#2E8B57']
AGE_COLS = ['연령대_10_매출_금액', '연령대_20_매출_금액', '연령대_30_매출_금액',
            '연령대_40_매출_금액', '연령대_50_매출_금액', '연령대_60_이상_매출_금액']

gen_wide     = {g: sub[col].values / 1e8       for g, col in zip(GENS, AGE_COLS)}
gen_pct_wide = {g: sub[col].values / total * 100 for g, col in zip(GENS, AGE_COLS)}

# ── 업종 그룹 ────────────────────────────────────────────────
GRP_COLS   = ['FB카페_매출금액', 'FB식사_매출금액', '주류유흥_매출금액',
              '라이프스타일_매출금액', '기타_매출금액']
GRP_LABELS = ['FB카페', 'FB식사', '주류·유흥', '라이프스타일', '기타']
GRP_COLORS = ['#F5A623', '#E05C5C', '#7B2D8B', '#3A5CA9', '#AAAAAA']
grp_pct = {lbl: sub[col].values / total * 100
           for col, lbl in zip(GRP_COLS, GRP_LABELS)}

# ── 시각화 ───────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.suptitle(f'{DONG}  매출 변화 분석  (2020Q1 ~ 2024Q4)',
             fontsize=15, fontweight='bold')

gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.38)
ax1 = fig.add_subplot(gs[0, :])   # 총매출 + QoQ (전체 span)
ax2 = fig.add_subplot(gs[1, 0])   # 연령대별 절대 stacked bar
ax3 = fig.add_subplot(gs[1, 1])   # 연령대별 비율 stacked area
ax4 = fig.add_subplot(gs[2, 0])   # 전 연령대 비중 라인 + 총매출 바
ax5 = fig.add_subplot(gs[2, 1])   # 업종 그룹 비율 stacked bar

# ── ax1: 총매출 바 + QoQ 라인 ────────────────────────────────
ax1.bar(X, total_억, color='#5B9BD5', alpha=0.6, width=0.6, label='총매출 (억 원)')
for xb in YBOUNDS:
    ax1.axvline(xb, color='#BBBBBB', lw=0.8, ls='--')
ax1r = ax1.twinx()
ax1r.plot(X, qoq, color='#E05C5C', lw=2, marker='o', markersize=5, label='QoQ 성장률 (%)')
ax1r.axhline(0, color='gray', lw=0.7, ls=':')
ax1r.set_ylabel('QoQ 성장률 (%)', fontsize=10, color='#E05C5C')
ax1r.tick_params(axis='y', colors='#E05C5C', labelsize=9)
ax1.set_xticks(X); ax1.set_xticklabels(XTICK, fontsize=8, rotation=45, ha='right')
ax1.set_ylabel('총매출 (억 원)', fontsize=11)
ax1.set_title('① 분기별 총매출 및 QoQ 성장률', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.2)
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax1r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc='upper left')
for i in [0, NQ - 1]:
    v = total_억[i]
    ax1.text(i, v + 8, f'{v:.0f}억', ha='center', fontsize=9,
             color='#2255AA', fontweight='bold')

# ── ax2: 연령대별 절대 stacked bar ───────────────────────────
bottom = np.zeros(NQ)
for g, color in zip(GENS, GEN_COLORS):
    vals = gen_wide[g]
    ax2.bar(X, vals, bottom=bottom, color=color, width=0.6, label=g, alpha=0.85)
    bottom += np.nan_to_num(vals)
for xb in YBOUNDS:
    ax2.axvline(xb, color='#888888', lw=0.7, ls='--')
ax2.set_xticks(X); ax2.set_xticklabels(XTICK, fontsize=7, rotation=45, ha='right')
ax2.set_ylabel('매출 (억 원)', fontsize=10)
ax2.set_title('② 연령대별 매출 (절대값)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8, loc='upper left', ncol=2)
ax2.grid(axis='y', alpha=0.2)

# ── ax3: 연령대별 비율 stacked area ──────────────────────────
bottom = np.zeros(NQ)
patches3 = []
for g, color in zip(GENS, GEN_COLORS):
    pct = gen_pct_wide[g]
    ax3.fill_between(X, bottom, bottom + pct, color=color, alpha=0.5)
    ax3.plot(X, bottom + pct, color=color, lw=0.8, alpha=0.7)
    patches3.append(mpatches.Patch(facecolor=color, alpha=0.7, label=g))
    bottom += pct
for xb in YBOUNDS:
    ax3.axvline(xb, color='#888888', lw=0.7, ls='--')
ax3.set_xlim(-0.5, NQ - 0.5)
ax3.set_ylim(0, 105)
ax3.set_yticks([0, 25, 50, 75, 100])
ax3.set_xticks(X); ax3.set_xticklabels(XTICK, fontsize=7, rotation=45, ha='right')
ax3.set_ylabel('비율 (%)', fontsize=10)
ax3.set_title('③ 연령대별 매출 비율 추이', fontsize=11, fontweight='bold')
ax3.legend(handles=patches3, fontsize=8, loc='upper right', ncol=2)
ax3.grid(axis='y', alpha=0.2)

# ── ax4: 전 연령대 매출 비중 라인 + 총매출 바 ────────────────
ax4r = ax4.twinx()
ax4r.bar(X, total_억, color='#5B9BD5', alpha=0.25, width=0.65, zorder=1)
ax4r.set_ylabel('총매출 (억 원)', fontsize=9, color='#2255AA')
ax4r.tick_params(axis='y', colors='#2255AA', labelsize=8)
ax4r.set_ylim(0, total_억.max() * 2.8)
ax4r.set_zorder(1)

ax4.set_zorder(2)
ax4.patch.set_visible(False)
MARKERS = ['o', 's', '^', 'D', 'v', 'P']
for g, color, marker in zip(GENS, GEN_COLORS, MARKERS):
    pct = gen_pct_wide[g]
    ax4.plot(X, pct, color=color, lw=2, marker=marker, markersize=4, label=g, zorder=3)
    ax4.text(-0.3,      pct[0],  f'{pct[0]:.1f}%',  fontsize=7, color=color, ha='right', va='center')
    ax4.text(NQ - 0.7,  pct[-1], f'{pct[-1]:.1f}%', fontsize=7, color=color, ha='left',  va='center')

for xb in YBOUNDS:
    ax4.axvline(xb, color='#BBBBBB', lw=0.7, ls='--', zorder=0)
ax4.set_xticks(X); ax4.set_xticklabels(XTICK, fontsize=7, rotation=45, ha='right')
ax4.set_ylabel('매출 비중 (%)', fontsize=10)
ax4.set_title('④ 전 연령대 매출 비중 추이  +  총매출 (바)', fontsize=11, fontweight='bold')
ax4.legend(fontsize=8, ncol=2, loc='upper right')
ax4.grid(alpha=0.15, zorder=0)

# ── ax5: 업종 그룹 비율 stacked bar ──────────────────────────
bottom = np.zeros(NQ)
patches5 = []
for lbl, color in zip(GRP_LABELS, GRP_COLORS):
    pct = grp_pct[lbl]
    ax5.bar(X, pct, bottom=bottom, color=color, width=0.6, alpha=0.85)
    patches5.append(mpatches.Patch(facecolor=color, alpha=0.85, label=lbl))
    bottom += pct
for xb in YBOUNDS:
    ax5.axvline(xb, color='#888888', lw=0.7, ls='--')
ax5.set_xlim(-0.5, NQ - 0.5)
ax5.set_ylim(0, 105)
ax5.set_yticks([0, 25, 50, 75, 100])
ax5.set_xticks(X); ax5.set_xticklabels(XTICK, fontsize=7, rotation=45, ha='right')
ax5.set_ylabel('비율 (%)', fontsize=10)
ax5.set_title('⑤ 업종 그룹별 매출 비율 추이', fontsize=11, fontweight='bold')
ax5.legend(handles=patches5, fontsize=8, loc='upper right')
ax5.grid(axis='y', alpha=0.2)

fig.text(0.5, -0.01,
         '※ 연령대별 매출은 전체 주(평일 포함) 기준  |  업종 그룹: FB카페·식사, 주류·유흥, 라이프스타일, 기타',
         ha='center', fontsize=9, color='#555555')

out = ROOT / 'image/성수2가1동_매출변화.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'[저장] {out.name}')
