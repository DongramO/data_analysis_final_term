# -*- coding: utf-8 -*-
"""
성수2가1동 유입 인구 변화 시각화
출력: image/성수2가1동_유입인구.png
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT     = Path(__file__).resolve().parent
MIG_PATH = ROOT / 'data/main_data/분기별_주말유입인구.csv'
MAP_PATH = ROOT / 'data/main_data/코드매핑_유입인구_매출.csv'
RF_PATH  = ROOT / 'data/main_data/분기별_원본피처.csv'
DONG     = '성수2가1동'

# ── 코드 → 동 이름 매핑 ──────────────────────────────────────
_map = pd.read_csv(MAP_PATH, encoding='utf-8-sig')
_map.columns = ['mig_code','raw_code','gu','dong_map']
code2dong = (
    pd.read_csv(RF_PATH, encoding='utf-8-sig', usecols=['행정동_코드','행정동'])
    .drop_duplicates()
    .rename(columns={'행정동_코드':'raw_code','행정동':'dong'})
    .set_index('raw_code')['dong'].to_dict()
)

mig = pd.read_csv(MIG_PATH, encoding='utf-8-sig')
mig.columns = ['mig_code','yr','q','yrq','age','HE','WE','EE','mig_total']
mig['raw_code'] = mig['mig_code'].map(_map.set_index('mig_code')['raw_code'].to_dict())
mig = mig.dropna(subset=['raw_code'])
mig['raw_code'] = mig['raw_code'].astype(int)
mig['dong'] = mig['raw_code'].map(code2dong)
mig = mig[mig['dong'] == DONG].copy()

QUARTERS = sorted(mig['yrq'].unique())
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

XTICK  = q_labels(QUARTERS)
YBOUNDS = yr_bounds(QUARTERS)

# ── 집계 ─────────────────────────────────────────────────────
# 전체 합계 (age==0)
total_mig = mig[mig['age'] == 0].groupby('yrq')['mig_total'].sum().reindex(QUARTERS)

# 연령대별 (age>0)
GENS = ['10이하','20대','30대','40대','50대','60이상']
GEN_COLORS = ['#DDDDDD','#E05C5C','#F5A623','#3A5CA9','#7B2D8B','#2E8B57']

mig_a = mig[mig['age'] > 0].copy()
mig_a['gen'] = mig_a['age'].apply(
    lambda a: '10이하' if a<=10 else ('20대' if a==20 else ('30대' if a==30
              else ('40대' if a==40 else ('50대' if a==50 else '60이상')))))
gen_agg = mig_a.groupby(['yrq','gen'])['mig_total'].sum().reset_index()
gen_tot = mig_a.groupby('yrq')['mig_total'].sum().rename('tot').reset_index()
gen_agg = gen_agg.merge(gen_tot, on='yrq')
gen_agg['pct'] = gen_agg['mig_total'] / gen_agg['tot'] * 100

gen_wide     = {g: gen_agg[gen_agg['gen']==g].set_index('yrq')['mig_total'].reindex(QUARTERS).values / 1e4
                for g in GENS}
gen_pct_wide = {g: gen_agg[gen_agg['gen']==g].set_index('yrq')['pct'].reindex(QUARTERS).fillna(0).values
                for g in GENS}

# 시간대별
time_agg = mig[mig['age'] == 0].groupby('yrq')[['HE','WE','EE']].sum().reindex(QUARTERS)
time_tot  = time_agg.sum(axis=1)
time_pct  = time_agg.div(time_tot, axis=0) * 100

# QoQ 성장률
total_arr = total_mig.values
qoq = np.full(NQ, np.nan)
qoq[1:] = (total_arr[1:] - total_arr[:-1]) / total_arr[:-1] * 100

# ── 시각화 ───────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 16))
fig.suptitle(f'{DONG}  주말 유입 인구 변화 분석  (2020Q1 ~ 2024Q4)',
             fontsize=15, fontweight='bold')

gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.38)
ax1 = fig.add_subplot(gs[0, :])   # 총 유입 + QoQ (전체 span)
ax2 = fig.add_subplot(gs[1, 0])   # 연령대별 절대 stacked bar
ax3 = fig.add_subplot(gs[1, 1])   # 연령대별 비율 stacked area
ax4 = fig.add_subplot(gs[2, 0])   # 2030 비중 추이
ax5 = fig.add_subplot(gs[2, 1])   # 시간대별 비율

# ── ax1: 총 유입 바 + QoQ 라인 ───────────────────────────────
ax1.bar(X, total_mig.values / 1e4, color='#5B9BD5', alpha=0.6, width=0.6, label='총 유입인구 (만 명)')
for xb in YBOUNDS:
    ax1.axvline(xb, color='#BBBBBB', lw=0.8, ls='--')
ax1r = ax1.twinx()
ax1r.plot(X, qoq, color='#E05C5C', lw=2, marker='o', markersize=5, label='QoQ 성장률 (%)')
ax1r.axhline(0, color='gray', lw=0.7, ls=':')
ax1r.set_ylabel('QoQ 성장률 (%)', fontsize=10, color='#E05C5C')
ax1r.tick_params(axis='y', colors='#E05C5C', labelsize=9)
ax1.set_xticks(X); ax1.set_xticklabels(XTICK, fontsize=8, rotation=45, ha='right')
ax1.set_ylabel('유입 인구 (만 명)', fontsize=11)
ax1.set_title('① 분기별 총 유입인구 및 QoQ 성장률', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.2)
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax1r.get_legend_handles_labels()
ax1.legend(lines1+lines2, labs1+labs2, fontsize=9, loc='upper left')
# 2020Q1 / 2024Q4 수치 표기
for i in [0, NQ-1]:
    v = total_mig.values[i] / 1e4
    ax1.text(i, v + 0.3, f'{v:.1f}만', ha='center', fontsize=9, color='#2255AA', fontweight='bold')

# ── ax2: 연령대별 절대 stacked bar ──────────────────────────
bottom = np.zeros(NQ)
for g, color in zip(GENS, GEN_COLORS):
    vals = gen_wide[g]
    ax2.bar(X, vals, bottom=bottom, color=color, width=0.6, label=g, alpha=0.85)
    bottom += np.nan_to_num(vals)
for xb in YBOUNDS:
    ax2.axvline(xb, color='#888888', lw=0.7, ls='--')
ax2.set_xticks(X); ax2.set_xticklabels(XTICK, fontsize=7, rotation=45, ha='right')
ax2.set_ylabel('유입 인구 (만 명)', fontsize=10)
ax2.set_title('② 연령대별 유입 인구 (절대값)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8, loc='upper left', ncol=2)
ax2.grid(axis='y', alpha=0.2)

# ── ax3: 연령대별 비율 stacked area ─────────────────────────
bottom = np.zeros(NQ)
patches = []
for g, color in zip(GENS, GEN_COLORS):
    pct = gen_pct_wide[g]
    ax3.fill_between(X, bottom, bottom + pct, color=color, alpha=0.5)
    ax3.plot(X, bottom + pct, color=color, lw=0.8, alpha=0.7)
    patches.append(mpatches.Patch(facecolor=color, alpha=0.7, label=g))
    bottom += pct
for xb in YBOUNDS:
    ax3.axvline(xb, color='#888888', lw=0.7, ls='--')
ax3.set_xlim(-0.5, NQ-0.5)
ax3.set_ylim(0, 105)
ax3.set_yticks([0, 25, 50, 75, 100])
ax3.set_xticks(X); ax3.set_xticklabels(XTICK, fontsize=7, rotation=45, ha='right')
ax3.set_ylabel('비율 (%)', fontsize=10)
ax3.set_title('③ 연령대별 유입 비율 추이', fontsize=11, fontweight='bold')
ax3.legend(handles=patches, fontsize=8, loc='upper right', ncol=2)
ax3.grid(axis='y', alpha=0.2)

# ── ax4: 전 세대 유입 비중 라인 + 총 유입인구 바 ────────────
# 우축: 총 유입인구 바
ax4r = ax4.twinx()
ax4r.bar(X, total_mig.values / 1e4, color='#5B9BD5', alpha=0.25, width=0.65, zorder=1)
ax4r.set_ylabel('총 유입인구 (만 명)', fontsize=9, color='#2255AA')
ax4r.tick_params(axis='y', colors='#2255AA', labelsize=8)
ax4r.set_ylim(0, total_mig.values.max() / 1e4 * 2.8)
ax4r.set_zorder(1)

# 좌축: 전 세대 비중 라인
ax4.set_zorder(2)
ax4.patch.set_visible(False)
MARKERS = ['o','s','^','D','v','P']
for g, color, marker in zip(GENS, GEN_COLORS, MARKERS):
    pct = gen_pct_wide[g]
    ax4.plot(X, pct, color=color, lw=2, marker=marker, markersize=4, label=g, zorder=3)
    ax4.text(-0.3,     pct[0],  f'{pct[0]:.1f}%',  fontsize=7, color=color, ha='right', va='center')
    ax4.text(NQ - 0.7, pct[-1], f'{pct[-1]:.1f}%', fontsize=7, color=color, ha='left',  va='center')

for xb in YBOUNDS:
    ax4.axvline(xb, color='#BBBBBB', lw=0.7, ls='--', zorder=0)
ax4.set_xticks(X); ax4.set_xticklabels(XTICK, fontsize=7, rotation=45, ha='right')
ax4.set_ylabel('유입 비중 (%)', fontsize=10)
ax4.set_title('④ 전 세대 유입 비중 추이  +  총 유입인구 (바)', fontsize=11, fontweight='bold')
ax4.legend(fontsize=8, ncol=2, loc='upper right')
ax4.grid(alpha=0.15, zorder=0)

# ── ax5: 시간대별 비율 (HE야간 / WE낮 / EE아침) ─────────────
TIME_COLORS = ['#7B2D8B','#5B9BD5','#F5A623']
TIME_LABELS = ['야간(HE)','낮(WE)','아침(EE)']
bottom = np.zeros(NQ)
t_patches = []
for col, color, label in zip(['HE','WE','EE'], TIME_COLORS, TIME_LABELS):
    pct = time_pct[col].values
    ax5.bar(X, pct, bottom=bottom, color=color, width=0.6, alpha=0.8, label=label)
    t_patches.append(mpatches.Patch(facecolor=color, alpha=0.8, label=label))
    bottom += pct
for xb in YBOUNDS:
    ax5.axvline(xb, color='#888888', lw=0.7, ls='--')
ax5.set_xticks(X); ax5.set_xticklabels(XTICK, fontsize=7, rotation=45, ha='right')
ax5.set_ylim(0, 105)
ax5.set_yticks([0, 25, 50, 75, 100])
ax5.set_ylabel('비율 (%)', fontsize=10)
ax5.set_title('⑤ 시간대별 유입 비율 (야간·낮·아침)', fontsize=11, fontweight='bold')
ax5.legend(handles=t_patches, fontsize=9)
ax5.grid(axis='y', alpha=0.2)

fig.text(0.5, -0.01,
         '※ 유입 데이터 기준: 금요일 야간(17~24시) + 토요일 전체 + 일요일 전체  |  HE=야간, WE=낮, EE=아침',
         ha='center', fontsize=9, color='#555555')

out = ROOT / 'image/성수2가1동_유입인구.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'[저장] {out.name}')
