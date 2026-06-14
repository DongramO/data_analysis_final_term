# -*- coding: utf-8 -*-
"""
20대 매출 비중 상위 10% & 서울 매출 중앙값 이상 행정동 (33개)
- 연령대별 매출 비중 (좌축)
- 총매출 바 (우축 1, 억 원)
- 주말 총 유입인원 라인 (우축 2, 만 명)

데이터: 분기별_원본피처.csv + 분기별_주말유입인구.csv
출력: image/상위10_세대별_분석.png
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

YEARS = [2020, 2021, 2022, 2023, 2024]
C20   = '#E05C5C'
C30   = '#F5A623'
C40   = '#3A5CA9'
C50   = '#7B2D8B'
C60   = '#2E8B57'
CBAR  = '#5B9BD5'
CMIG  = '#CC6600'

# ══ 1. 매출 데이터 ════════════════════════════════════════
rf = pd.read_csv(RF_PATH, encoding='utf-8-sig')
rf.columns = [c.strip() for c in rf.columns]
rf = rf.rename(columns={
    '행정동': 'dong', '연도': 'yr', '총_매출금액': 'total',
    '연령대_20_매출_금액': 'age20', '연령대_30_매출_금액': 'age30',
    '연령대_40_매출_금액': 'age40', '연령대_50_매출_금액': 'age50',
    '연령대_60_이상_매출_금액': 'age60',
})
agg = rf.groupby(['dong','yr'])[['total','age20','age30','age40','age50','age60']].sum().reset_index()
for g in ['20','30','40','50','60']:
    agg[f'r{g}'] = agg[f'age{g}'] / agg['total'] * 100

# ══ 2. 유입(이동) 데이터 ══════════════════════════════════
_map = pd.read_csv(MAP_PATH, encoding='utf-8-sig')
_map.columns = ['mig_code','raw_code','gu','dong_map']
code2dong = (
    pd.read_csv(RF_PATH, encoding='utf-8-sig', usecols=['행정동_코드','행정동'])
    .drop_duplicates()
    .rename(columns={'행정동_코드':'raw_code','행정동':'dong'})
    .set_index('raw_code')['dong']
    .to_dict()
)
mig = pd.read_csv(MIG_PATH, encoding='utf-8-sig')
mig.columns = ['mig_code','yr','q','yrq','age','HE','WE','EE','mig_total']
mig['raw_code'] = mig['mig_code'].map(_map.set_index('mig_code')['raw_code'].to_dict())
mig = mig.dropna(subset=['raw_code'])
mig['raw_code'] = mig['raw_code'].astype(int)
mig['dong'] = mig['raw_code'].map(code2dong)
mig = mig.dropna(subset=['dong'])

mig_agg = mig.groupby(['dong','yr'])['mig_total'].sum().reset_index()

# ── 연령대별 유입 비중 (stacked용) ────────────────────────
mig_a = mig[mig['age'] > 0].copy()
mig_a['gen'] = mig_a['age'].apply(
    lambda a: '10이하' if a<=10 else ('20대' if a==20 else ('30대' if a==30
              else ('40대' if a==40 else ('50대' if a==50 else '60이상')))))
mig_gen = mig_a.groupby(['dong','yr','gen'])['mig_total'].sum().reset_index()
mig_gen_tot = mig_a.groupby(['dong','yr'])['mig_total'].sum().rename('tot').reset_index()
mig_gen = mig_gen.merge(mig_gen_tot, on=['dong','yr'])
mig_gen['pct'] = mig_gen['mig_total'] / mig_gen['tot'] * 100
GENS       = ['10이하','20대','30대','40대','50대','60이상']
GEN_COLORS = ['#DDDDDD', C20, C30, C40, C50, C60]
mig_gen_wide = {g: mig_gen[mig_gen['gen']==g].pivot(index='dong', columns='yr', values='pct')
                for g in GENS}

# ══ 3. 행정동 선정 ════════════════════════════════════════
비중_2024   = agg[agg['yr']==2024].set_index('dong')['r20']
threshold  = 비중_2024.quantile(0.90)
top10_set  = set(비중_2024[비중_2024 >= threshold].index)
dong_avg_s = agg.groupby('dong')['total'].mean()
seoul_med  = dong_avg_s.median()
above_med  = set(dong_avg_s[dong_avg_s >= seoul_med].index)
target_set = top10_set & above_med

sales_2024   = agg[agg['yr']==2024].set_index('dong')['total']
target_dongs = sales_2024.reindex(sorted(target_set)).sort_values(ascending=False).index.tolist()

print(f'20대 비중 기준 (상위 10%): ≥ {threshold:.1f}%')
print(f'서울 평균매출 중앙값: {seoul_med/1e8:.0f}억')
print(f'최종 대상: {len(target_dongs)}개')

# 데이터 wide
sales_wide = agg.pivot(index='dong', columns='yr', values='total') / 1e8
mig_wide   = mig_agg.pivot(index='dong', columns='yr', values='mig_total') / 1e4  # 만 명

# ══ 4. 시각화 ═════════════════════════════════════════════
NCOLS = 5
NROWS = (len(target_dongs) + NCOLS - 1) // NCOLS

fig, axes = plt.subplots(NROWS, NCOLS, figsize=(NCOLS * 4.4, NROWS * 4.0))
fig.suptitle(
    f'20대 매출 비중 상위 10% & 서울 매출 중앙값 이상 행정동 ({len(target_dongs)}개)\n'
    f'좌축(선): 연령대별 매출 비중(%)  /  우축1(초록 바): 총매출(억 원)  /  우축2(stack): 연령별 유입 비중(%)\n'
    f'기준: 20대 비중 ≥ {threshold:.1f}%  &  평균매출 ≥ {seoul_med/1e8:.0f}억  |  2024 총매출 규모 큰 순 정렬',
    fontsize=11, fontweight='bold', y=1.01)

for idx, dong in enumerate(target_dongs):
    row, col = divmod(idx, NCOLS)
    ax = axes[row][col]

    sub  = agg[agg['dong']==dong].set_index('yr')
    r20  = sub['r20'].reindex(YEARS).values
    r30  = sub['r30'].reindex(YEARS).values
    r40  = sub['r40'].reindex(YEARS).values
    r50  = sub['r50'].reindex(YEARS).values
    r60  = sub['r60'].reindex(YEARS).values
    sales = sales_wide.loc[dong].reindex(YEARS).values if dong in sales_wide.index else np.full(5, np.nan)
    mig_v = mig_wide.loc[dong].reindex(YEARS).values  if dong in mig_wide.index  else np.full(5, np.nan)

    # ── 우축 1: 총매출 바 (초록) ──────────────────────────
    ax2 = ax.twinx()
    max_s = np.nanmax(sales) if not np.all(np.isnan(sales)) else 1
    ax2.bar(YEARS, sales, width=0.55, color=CBAR, alpha=0.60, zorder=1)
    ax2.set_ylim(0, max_s * 1.7)
    ax2.set_ylabel('총매출 (억 원)', fontsize=6, color='#3a7a3a')
    ax2.tick_params(axis='y', labelsize=5.5, colors='#3a7a3a')
    ax2.spines['right'].set_edgecolor(CBAR)
    for yr, v in zip([2020, 2024], [sales[0], sales[-1]]):
        if not np.isnan(v):
            ax2.text(yr, v + max_s * 0.04, f'{v:.0f}억', ha='center', fontsize=5, color='#3a7a3a')
    ax2.set_zorder(1)

    # ── 우축 2: 연령 유입 비중 stack ──────────────────────
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 38))
    ax3.spines['right'].set_edgecolor('#AAAAAA')
    stacks = []
    for g in GENS:
        df = mig_gen_wide[g]
        pct = df.loc[dong].reindex(YEARS).fillna(0).values if dong in df.index else np.zeros(len(YEARS))
        stacks.append(pct)
    ax3.stackplot(YEARS, *stacks, colors=GEN_COLORS, alpha=0.35, zorder=2)
    ax3.set_ylim(0, 112)
    ax3.set_ylabel('유입 연령\n비중 (%)', fontsize=6, color='#555555')
    ax3.tick_params(axis='y', labelsize=5.5, colors='#555555')
    ax3.set_yticks([0, 25, 50, 75, 100])
    for yr, v in zip([2020, 2024], [mig_v[0], mig_v[-1]]):
        if not np.isnan(v):
            ax3.text(yr, 104, f'{v:.0f}만명', ha='center', fontsize=4.8, color=CMIG,
                     fontweight='bold')
    ax3.set_zorder(1)

    # ── 좌축: 세대별 비중 라인 ────────────────────────────
    ax.set_zorder(3)
    ax.patch.set_visible(False)
    ax.plot(YEARS, r20, color=C20, lw=2,   marker='o', markersize=4, zorder=5)
    ax.plot(YEARS, r30, color=C30, lw=2,   marker='s', markersize=4, zorder=5)
    ax.plot(YEARS, r40, color=C40, lw=1.5, marker='^', markersize=4, zorder=5)
    ax.plot(YEARS, r50, color=C50, lw=1.5, marker='D', markersize=4, zorder=5)
    ax.plot(YEARS, r60, color=C60, lw=1.5, marker='v', markersize=4, zorder=5)

    for v, c in [(r20[-1], C20),(r30[-1], C30),(r40[-1], C40),(r50[-1], C50),(r60[-1], C60)]:
        if not np.isnan(v):
            ax.text(2024.12, v, f'{v:.1f}%', fontsize=4.8, color=c, va='center')

    r20_24 = r20[-1] if not np.isnan(r20[-1]) else 0
    d20    = r20[-1] - r20[0] if not (np.isnan(r20[-1]) or np.isnan(r20[0])) else np.nan
    ax.set_title(f'{dong}\n20대 {r20_24:.1f}% ({d20:+.1f}%p)', fontsize=8, pad=3)
    ax.set_xticks(YEARS)
    ax.set_xticklabels(["'20","'21","'22","'23","'24"], fontsize=7)
    valid = np.concatenate([v[~np.isnan(v)] for v in [r20,r30,r40,r50,r60]])
    if len(valid):
        ax.set_ylim(max(0, valid.min()-3), min(75, valid.max()+4))
    ax.set_ylabel('매출 비중 (%)', fontsize=6.5)
    ax.tick_params(axis='y', labelsize=6.5)
    ax.grid(alpha=0.15, zorder=0)

# 공통 범례
fig.legend(handles=[
    Line2D([0],[0], color=C20,  lw=2,   marker='o', ms=5, label='매출 20대'),
    Line2D([0],[0], color=C30,  lw=2,   marker='s', ms=5, label='매출 30대'),
    Line2D([0],[0], color=C40,  lw=1.5, marker='^', ms=5, label='매출 40대'),
    Line2D([0],[0], color=C50,  lw=1.5, marker='D', ms=5, label='매출 50대'),
    Line2D([0],[0], color=C60,  lw=1.5, marker='v', ms=5, label='매출 60대이상'),
    mpatches.Patch(facecolor=CBAR, alpha=0.8, label='총매출 (억 원)'),
    mpatches.Patch(facecolor='#DDDDDD', alpha=0.6, label='유입 10대이하'),
    mpatches.Patch(facecolor=C20, alpha=0.45, label='유입 20대'),
    mpatches.Patch(facecolor=C30, alpha=0.45, label='유입 30대'),
    mpatches.Patch(facecolor=C40, alpha=0.45, label='유입 40대'),
    mpatches.Patch(facecolor=C50, alpha=0.45, label='유입 50대'),
    mpatches.Patch(facecolor=C60, alpha=0.45, label='유입 60이상'),
    Line2D([0],[0], color=CMIG, lw=0, marker='$↑$', ms=7, label='총 유입(만명, 숫자)'),
], loc='lower center', ncol=7, fontsize=8,
   bbox_to_anchor=(0.5, -0.02), frameon=True)

for idx in range(len(target_dongs), NROWS * NCOLS):
    row, col = divmod(idx, NCOLS)
    axes[row][col].axis('off')

plt.tight_layout(h_pad=2.5, w_pad=1.5)
out = ROOT / 'image/상위10_세대별_분석.png'
fig.savefig(out, dpi=130, bbox_inches='tight')
plt.close(fig)
print(f'[저장] {out.name}')
