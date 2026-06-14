# -*- coding: utf-8 -*-
"""
성수동 상세 분석: 4개 개별 행정동 + 성수동 전체 집계 (분기별)
- 연령대별 매출 비중 (좌축)
- 총매출 바 (우축 1, 억 원)
- 연령별 유입 비중 stack (우축 2)

출력: image/성수동_상세_분석.png
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
RF_PATH  = ROOT / 'data/main_data/분기별_원본피처.csv'
MIG_PATH = ROOT / 'data/main_data/분기별_주말유입인구.csv'
MAP_PATH = ROOT / 'data/main_data/코드매핑_유입인구_매출.csv'

SEONGSU = ['성수1가1동','성수1가2동','성수2가1동','성수2가3동']
PANELS  = SEONGSU + ['성수동 전체']

C20  = '#E05C5C'
C30  = '#F5A623'
C40  = '#3A5CA9'
C50  = '#7B2D8B'
C60  = '#2E8B57'
CBAR = '#5B9BD5'
CMIG = '#CC6600'

# ══ 1. 매출 데이터 (분기별) ═══════════════════════════════════
rf = pd.read_csv(RF_PATH, encoding='utf-8-sig')
rf.columns = [c.strip() for c in rf.columns]
rf = rf.rename(columns={
    '행정동': 'dong', '연도': 'yr', '분기': 'q', '연도분기': 'yrq',
    '총_매출금액': 'total',
    '연령대_20_매출_금액': 'age20', '연령대_30_매출_금액': 'age30',
    '연령대_40_매출_금액': 'age40', '연령대_50_매출_금액': 'age50',
    '연령대_60_이상_매출_금액': 'age60',
})
agg = rf.groupby(['dong','yrq','yr','q'])[['total','age20','age30','age40','age50','age60']].sum().reset_index()
for g in ['20','30','40','50','60']:
    agg[f'r{g}'] = agg[f'age{g}'] / agg['total'] * 100

# 분기 레이블 순서 확정
QUARTERS = sorted(agg['yrq'].unique())
Q_LABELS = [f"'{str(q)[2:4]}Q{str(q)[-1]}" for q in QUARTERS]  # e.g. '20Q1

# 성수동 전체 집계
ss_total = (
    agg[agg['dong'].isin(SEONGSU)]
    .groupby(['yrq','yr','q'])[['total','age20','age30','age40','age50','age60']].sum()
    .reset_index()
)
ss_total['dong'] = '성수동 전체'
for g in ['20','30','40','50','60']:
    ss_total[f'r{g}'] = ss_total[f'age{g}'] / ss_total['total'] * 100
agg = pd.concat([agg, ss_total], ignore_index=True)

# ══ 2. 유입 데이터 (분기별) ═══════════════════════════════════
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
mig = mig.dropna(subset=['dong'])

mig_agg = mig.groupby(['dong','yrq'])['mig_total'].sum().reset_index()

# 연령대별 유입 비중 (stacked용)
mig_a = mig[mig['age'] > 0].copy()
mig_a['gen'] = mig_a['age'].apply(
    lambda a: '10이하' if a<=10 else ('20대' if a==20 else ('30대' if a==30
              else ('40대' if a==40 else ('50대' if a==50 else '60이상')))))
mig_gen = mig_a.groupby(['dong','yrq','gen'])['mig_total'].sum().reset_index()
mig_gen_tot = mig_a.groupby(['dong','yrq'])['mig_total'].sum().rename('tot').reset_index()
mig_gen = mig_gen.merge(mig_gen_tot, on=['dong','yrq'])
mig_gen['pct'] = mig_gen['mig_total'] / mig_gen['tot'] * 100
GENS       = ['10이하','20대','30대','40대','50대','60이상']
GEN_COLORS = ['#DDDDDD', C20, C30, C40, C50, C60]
mig_gen_wide = {g: mig_gen[mig_gen['gen']==g].pivot(index='dong', columns='yrq', values='pct')
                for g in GENS}

# 성수동 전체 유입 집계
ss_mig = (
    mig_agg[mig_agg['dong'].isin(SEONGSU)]
    .groupby('yrq')['mig_total'].sum()
    .reset_index()
)
ss_mig['dong'] = '성수동 전체'
mig_agg = pd.concat([mig_agg, ss_mig], ignore_index=True)

sales_wide = agg.pivot_table(index='dong', columns='yrq', values='total') / 1e8
mig_wide   = mig_agg.pivot(index='dong', columns='yrq', values='mig_total') / 1e4

X = np.arange(len(QUARTERS))

# ── X축 레이블: 연도 첫 분기만 표시 ──────────────────────────
def make_xtick_labels(quarters):
    labels = []
    prev_yr = None
    for q in quarters:
        yr = str(q)[:4]
        if yr != prev_yr:
            labels.append(f"'{yr[2:]}Q{str(q)[-1]}")
            prev_yr = yr
        else:
            labels.append(f"Q{str(q)[-1]}")
    return labels

XTICK_LABELS = make_xtick_labels(QUARTERS)

# ══ 3. 시각화 ═════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 20))
fig.suptitle(
    '성수동 상세 분석  │  4개 행정동 개별 + 성수동 전체 집계  (분기별)\n'
    '좌축(선): 연령대별 매출 비중(%)  /  우축1(파랑 바): 총매출(억 원)  /  우축2(stack): 연령별 유입 비중(%)',
    fontsize=13, fontweight='bold', y=1.01)

gs = fig.add_gridspec(3, 2, hspace=0.6, wspace=0.45)
axes_top = [
    fig.add_subplot(gs[0, 0]),
    fig.add_subplot(gs[0, 1]),
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
]
ax_full = fig.add_subplot(gs[2, :])


def draw_panel(ax, dong, is_full=False):
    sub = agg[agg['dong']==dong].set_index('yrq')
    r20 = sub['r20'].reindex(QUARTERS).values
    r30 = sub['r30'].reindex(QUARTERS).values
    r40 = sub['r40'].reindex(QUARTERS).values
    r50 = sub['r50'].reindex(QUARTERS).values
    r60 = sub['r60'].reindex(QUARTERS).values
    sales = sales_wide.loc[dong].reindex(QUARTERS).values if dong in sales_wide.index else np.full(len(QUARTERS), np.nan)
    mig_v = mig_wide.loc[dong].reindex(QUARTERS).values  if dong in mig_wide.index  else np.full(len(QUARTERS), np.nan)

    lw_main  = 2.0 if is_full else 1.5
    ms_main  = 5   if is_full else 4
    fs_label = 7   if is_full else 6
    fs_val   = 7   if is_full else 5.5

    # ── 우축1: 총매출 바 ────────────────────────────────────
    ax2 = ax.twinx()
    max_s = np.nanmax(sales) if not np.all(np.isnan(sales)) else 1
    ax2.bar(X, sales, width=0.55, color=CBAR, alpha=0.55, zorder=1)
    ax2.set_ylim(0, max_s * 1.65)
    ax2.set_ylabel('총매출 (억 원)', fontsize=fs_label, color='#2255AA')
    ax2.tick_params(axis='y', labelsize=fs_label - 0.5, colors='#2255AA')
    ax2.spines['right'].set_edgecolor(CBAR)
    # 연도 첫 분기(Q1)와 마지막 분기(Q4)만 수치 표기
    for i, (q, v) in enumerate(zip(QUARTERS, sales)):
        if not np.isnan(v) and str(q).endswith(('1','4')):
            ax2.text(i, v + max_s * 0.04, f'{v:.0f}억',
                     ha='center', fontsize=fs_val - 1.5, color='#2255AA')
    ax2.set_zorder(1)

    # ── 우축2: 연령 유입 비중 stack ─────────────────────────
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 42))
    ax3.spines['right'].set_edgecolor('#AAAAAA')
    stacks = []
    for g in GENS:
        df = mig_gen_wide[g]
        pct = df.loc[dong].reindex(QUARTERS).fillna(0).values if dong in df.index else np.zeros(len(QUARTERS))
        stacks.append(pct)
    ax3.stackplot(X, *stacks, colors=GEN_COLORS, alpha=0.35, zorder=2)
    ax3.set_ylim(0, 115)
    ax3.set_ylabel('유입 연령\n비중 (%)', fontsize=fs_label, color='#555555')
    ax3.tick_params(axis='y', labelsize=fs_label - 0.5, colors='#555555')
    ax3.set_yticks([0, 25, 50, 75, 100])
    # 연도 첫 분기만 유입 총량 표기
    for i, (q, v) in enumerate(zip(QUARTERS, mig_v)):
        if not np.isnan(v) and str(q).endswith('1'):
            ax3.text(i, 107, f'{v:.0f}만', ha='center',
                     fontsize=fs_val - 2, color=CMIG, fontweight='bold')
    ax3.set_zorder(1)

    # ── 좌축: 세대별 비중 라인 ──────────────────────────────
    ax.set_zorder(3)
    ax.patch.set_visible(False)
    ax.plot(X, r20, color=C20, lw=lw_main,       marker='o', markersize=ms_main, zorder=5)
    ax.plot(X, r30, color=C30, lw=lw_main,       marker='s', markersize=ms_main, zorder=5)
    ax.plot(X, r40, color=C40, lw=lw_main - 0.5, marker='^', markersize=ms_main, zorder=5)
    ax.plot(X, r50, color=C50, lw=lw_main - 0.5, marker='D', markersize=ms_main, zorder=5)
    ax.plot(X, r60, color=C60, lw=lw_main - 0.5, marker='v', markersize=ms_main, zorder=5)

    # 2024Q4 끝값 표기
    for vals, c in [(r20,C20),(r30,C30),(r40,C40),(r50,C50),(r60,C60)]:
        if is_full:
            # full 패널: Q1만 값 표기
            for i, (q, v) in enumerate(zip(QUARTERS, vals)):
                if not np.isnan(v) and str(q).endswith('1'):
                    ax.text(i, v + 0.4, f'{v:.1f}%', ha='center',
                            fontsize=6, color=c, zorder=6)
        else:
            if not np.isnan(vals[-1]):
                ax.text(len(QUARTERS) - 0.7, vals[-1], f'{vals[-1]:.1f}%',
                        fontsize=5, color=c, va='center')

    d20 = r20[-1] - r20[0] if not (np.isnan(r20[-1]) or np.isnan(r20[0])) else np.nan
    d_s = sales[-1] - sales[0] if not (np.isnan(sales[-1]) or np.isnan(sales[0])) else np.nan
    d_m = mig_v[-1] - mig_v[0] if not (np.isnan(mig_v[-1]) or np.isnan(mig_v[0])) else np.nan
    title_fs = 10 if is_full else 9
    ax.set_title(
        f'{"▶ " if is_full else ""}{dong}\n'
        f'20대 {r20[-1]:.1f}% ({d20:+.1f}%p)  │  매출 {d_s:+.0f}억  │  유입 {d_m:+.0f}만명',
        fontsize=title_fs, fontweight='bold' if is_full else 'normal', pad=4)

    ax.set_xticks(X)
    ax.set_xticklabels(XTICK_LABELS, fontsize=7 if is_full else 6, rotation=45, ha='right')

    # 연도 경계 점선
    for i, q in enumerate(QUARTERS):
        if str(q).endswith('1') and i > 0:
            ax.axvline(i - 0.5, color='#CCCCCC', lw=0.7, ls='--', zorder=0)

    valid = np.concatenate([v[~np.isnan(v)] for v in [r20,r30,r40,r50,r60]])
    if len(valid):
        ax.set_ylim(max(0, valid.min()-3), min(65, valid.max()+5))
    ax.set_ylabel('매출 비중 (%)', fontsize=fs_label)
    ax.tick_params(axis='y', labelsize=fs_label)
    ax.grid(alpha=0.15, zorder=0)


for ax, dong in zip(axes_top, SEONGSU):
    draw_panel(ax, dong, is_full=False)
draw_panel(ax_full, '성수동 전체', is_full=True)

# 공통 범례
fig.legend(handles=[
    Line2D([0],[0], color=C20,  lw=2,   marker='o', ms=6, label='매출 20대'),
    Line2D([0],[0], color=C30,  lw=2,   marker='s', ms=6, label='매출 30대'),
    Line2D([0],[0], color=C40,  lw=1.5, marker='^', ms=6, label='매출 40대'),
    Line2D([0],[0], color=C50,  lw=1.5, marker='D', ms=6, label='매출 50대'),
    Line2D([0],[0], color=C60,  lw=1.5, marker='v', ms=6, label='매출 60이상'),
    mpatches.Patch(facecolor=CBAR, alpha=0.7, label='총매출 (억 원)'),
    mpatches.Patch(facecolor='#DDDDDD', alpha=0.6, label='유입 10대이하'),
    mpatches.Patch(facecolor=C20, alpha=0.45, label='유입 20대'),
    mpatches.Patch(facecolor=C30, alpha=0.45, label='유입 30대'),
    mpatches.Patch(facecolor=C40, alpha=0.45, label='유입 40대'),
    mpatches.Patch(facecolor=C50, alpha=0.45, label='유입 50대'),
    mpatches.Patch(facecolor=C60, alpha=0.45, label='유입 60이상'),
    Line2D([0],[0], color=CMIG, lw=0, marker='$↑$', ms=8, label='총유입(만명,Q1표기)'),
], loc='lower center', ncol=7, fontsize=9,
   bbox_to_anchor=(0.5, -0.02), frameon=True)

out = ROOT / 'image/성수동_상세_분석.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'[저장] {out.name}')
