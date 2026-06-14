# -*- coding: utf-8 -*-
"""
기준① 트렌드 시간대 업종 성향 — 개별 행정동 버전
- 배경: 서울 전체 423개 동 산포 (회색)
- 강조: 레퍼런스(4) / Tier1(2) / 신규 선별(9) 개별 레이블
- Panel 1: 트렌드업종비중 × 트렌드시간비중 산포도
- Panel 2: 개별 행정동 업종 구성 스택바 (REF+Tier1+신규선별)
출력: image/gen20_precursor_01_industry.png (덮어쓰기)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT     = Path(__file__).resolve().parent
RAW_PATH = ROOT / 'data/main_data/분기별_원본피처.csv'
AREA_PATH= ROOT / 'data/main_data/area_type_profile.csv'
IMG_DIR  = ROOT / 'image'

REF   = ['방배2동', '청운효자동', '합정동', '성수2가1동']
TIER1 = ['숭인2동', '신당5동']
# analysis_gen20_precursor.py 에서 선별된 9개
CANDIDATES = ['잠실6동','자양3동','신대방1동','용산2가동','동화동',
              '문정1동','등촌2동','개포1동','사당3동']

FOCUS = REF + TIER1 + CANDIDATES

GROUP_COLOR = {
    '레퍼런스(4)': '#E05C5C',
    'Tier1(2)':   '#F5A623',
    '신규 선별':  '#27AE60',
}
GROUP_MARKER = {'레퍼런스(4)': 'D', 'Tier1(2)': 's', '신규 선별': 'o'}

def label(d):
    if d in REF:        return '레퍼런스(4)'
    if d in TIER1:      return 'Tier1(2)'
    if d in CANDIDATES: return '신규 선별'
    return '서울 전체'

# ── 데이터 ────────────────────────────────────────────────────────────────
raw  = pd.read_csv(RAW_PATH,  encoding='utf-8-sig')
area = pd.read_csv(AREA_PATH, encoding='utf-8-sig')

raw['야간비중'] = (
    raw['시간대_17~21_매출_금액'] + raw['시간대_21~24_매출_금액']
) / raw['총_매출금액'].replace(0, np.nan)

raw['트렌드시간_매출'] = (
    raw['토요일_매출_금액'] + raw['일요일_매출_금액']
    + raw['금요일_매출_금액'] * raw['야간비중']
)
raw['트렌드시간_비중'] = raw['트렌드시간_매출'] / raw['총_매출금액'].replace(0, np.nan)

raw['트렌드업종_매출'] = (
    raw['FB카페_매출금액'] + raw['FB식사_매출금액']
    + raw['주류유흥_매출금액'] + raw['라이프스타일_매출금액']
)
raw['트렌드업종_비중'] = raw['트렌드업종_매출'] / raw['총_매출금액'].replace(0, np.nan)

IND_COLS  = ['FB카페_매출금액','FB식사_매출금액','주류유흥_매출금액','라이프스타일_매출금액','기타_매출금액']
IND_NAMES = ['FB카페','FB식사','주류·유흥','라이프스타일','기타']
IND_COLORS= ['#5B9BD5','#ED7D31','#A5A5A5','#FFC000','#E0E0E0']

# 2024년 연간 평균 (행정동별)
y24 = raw[raw['연도'] == 2024].groupby('행정동').agg(
    트렌드업종_비중 =('트렌드업종_비중', 'mean'),
    트렌드시간_비중 =('트렌드시간_비중', 'mean'),
    야간비중        =('야간비중', 'mean'),
    **{c: (c, 'mean') for c in IND_COLS},
    총_매출금액     =('총_매출금액', 'mean'),
).reset_index()

# 업종 비중 정규화
for c in IND_COLS:
    y24[c+'_비중'] = y24[c] / y24[[cc for cc in IND_COLS]].sum(axis=1)

y24['그룹'] = y24['행정동'].map(label)

# 전체 / 포커스 분리
all_dong = y24.copy()
focus    = y24[y24['행정동'].isin(FOCUS)].copy()

# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('기준①  트렌드 시간대 업종 성향 — 개별 행정동 비교 (2024)',
             fontsize=14, fontweight='bold')

# ── Panel 1: 트렌드업종비중 × 트렌드시간비중 산포도 ──────────────────────
ax1 = axes[0]

# 배경: 서울 전체
bg = all_dong[~all_dong['행정동'].isin(FOCUS)]
ax1.scatter(bg['트렌드업종_비중'], bg['트렌드시간_비중'],
            s=18, color='#CCCCCC', alpha=0.5, zorder=1, label='서울 전체')

# 서울 전체 중앙값 십자선
mx = all_dong['트렌드업종_비중'].median()
my = all_dong['트렌드시간_비중'].median()
ax1.axvline(mx, color='#AAAAAA', lw=0.9, ls='--', zorder=2)
ax1.axhline(my, color='#AAAAAA', lw=0.9, ls='--', zorder=2)
ax1.text(mx + 0.003, my + 0.003, f'서울중앙값\n({mx:.2f}, {my:.2f})',
         fontsize=7.5, color='#888888', va='bottom')

# 그룹별 중앙값 타원 배경
for grp, color in GROUP_COLOR.items():
    sub = focus[focus['그룹'] == grp]
    if len(sub) == 0: continue
    cx = sub['트렌드업종_비중'].mean()
    cy = sub['트렌드시간_비중'].mean()
    ax1.scatter(cx, cy, s=200, color=color, alpha=0.15, zorder=2)

# 포커스 행정동
for grp in ['신규 선별', 'Tier1(2)', '레퍼런스(4)']:
    sub = focus[focus['그룹'] == grp]
    color  = GROUP_COLOR[grp]
    marker = GROUP_MARKER[grp]
    ax1.scatter(sub['트렌드업종_비중'], sub['트렌드시간_비중'],
                s=90, color=color, marker=marker, zorder=5,
                edgecolors='white', linewidths=0.8, label=grp)
    for _, row in sub.iterrows():
        # 레이블 위치 조정 (겹침 방지)
        va = 'bottom'; ha = 'left'; dy = 0.004; dx = 0.003
        ax1.annotate(row['행정동'],
                     (row['트렌드업종_비중'], row['트렌드시간_비중']),
                     fontsize=8, color=color, fontweight='bold',
                     xytext=(dx, dy), textcoords='offset points',
                     zorder=6)

ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax1.set_xlabel('트렌드업종 비중 (FB카페+FB식사+주류유흥+라이프스타일)', fontsize=10)
ax1.set_ylabel('트렌드 시간대 비중 (금야간+토+일)', fontsize=10)
ax1.set_title('트렌드업종 비중 × 트렌드시간 비중\n(배경: 서울 전체 423개 동)', fontsize=11)
ax1.legend(fontsize=9, loc='upper left')
ax1.set_facecolor('#F8F8F8')
ax1.grid(alpha=0.2)

# ── Panel 2: 개별 행정동 업종 구성 스택바 ──────────────────────────────
ax2 = axes[1]

# 행정동 정렬: 레퍼런스 → Tier1 → 신규선별, 각 그룹 내 트렌드업종비중 내림차순
order_list = []
for grp_name in ['레퍼런스(4)', 'Tier1(2)', '신규 선별']:
    sub = focus[focus['그룹'] == grp_name].sort_values('트렌드업종_비중', ascending=False)
    order_list.append(sub)
plot_df = pd.concat(order_list).reset_index(drop=True)
plot_df['x'] = range(len(plot_df))

# 구분선 위치
n_ref   = len(plot_df[plot_df['그룹']=='레퍼런스(4)'])
n_tier1 = len(plot_df[plot_df['그룹']=='Tier1(2)'])

bottom = np.zeros(len(plot_df))
for idx, (col, name, color) in enumerate(zip(IND_COLS, IND_NAMES, IND_COLORS)):
    vals = plot_df[col+'_비중'].values
    bars = ax2.bar(plot_df['x'], vals, bottom=bottom, color=color,
                   label=name, width=0.75)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 0.07:
            ax2.text(i, b + v/2, f'{v*100:.0f}%',
                     ha='center', va='center', fontsize=7.5,
                     fontweight='bold', color='white')
    bottom += vals

# 그룹 구분선
sep1 = n_ref - 0.5
sep2 = n_ref + n_tier1 - 0.5
ax2.axvline(sep1, color='#666666', lw=1.5, ls='-')
ax2.axvline(sep2, color='#666666', lw=1.5, ls='-')

# 그룹 레이블 (상단)
ax2.text((sep1)/2, 1.04, '레퍼런스(4)', ha='center', fontsize=9,
         color=GROUP_COLOR['레퍼런스(4)'], fontweight='bold',
         transform=ax2.get_xaxis_transform())
ax2.text((sep1 + sep2)/2, 1.04, 'Tier1(2)', ha='center', fontsize=9,
         color=GROUP_COLOR['Tier1(2)'], fontweight='bold',
         transform=ax2.get_xaxis_transform())
ax2.text((sep2 + len(plot_df))/2, 1.04, '신규 선별(9)', ha='center', fontsize=9,
         color=GROUP_COLOR['신규 선별'], fontweight='bold',
         transform=ax2.get_xaxis_transform())

# 서울 전체 트렌드업종 평균 기준선
seoul_avg = all_dong['트렌드업종_비중'].median()
# 트렌드업종 = FB카페+FB식사+주류유흥+라이프스타일 = 아래 4개 업종의 합 → y값으로
ind_sum = plot_df[[c+'_비중' for c in IND_COLS[:4]]].sum(axis=1)
# 서울 전체 트렌드업종 비중 선 (점선)
seoul_trend = all_dong['트렌드업종_비중'].mean()
ax2.axhline(seoul_trend, color='#888888', lw=1.2, ls='--', label=f'서울전체 평균({seoul_trend:.2f})')

# x축 레이블 (행정동 이름 + 트렌드업종비중)
xlabels = [f"{r['행정동']}\n({r['트렌드업종_비중']:.2f})" for _, r in plot_df.iterrows()]
ax2.set_xticks(range(len(plot_df)))
ax2.set_xticklabels(xlabels, fontsize=8, rotation=0)

# 그룹별 테두리 색상 표시 (x축 레이블 색상)
tick_colors = [GROUP_COLOR[plot_df.iloc[i]['그룹']] for i in range(len(plot_df))]
for tick, color in zip(ax2.get_xticklabels(), tick_colors):
    tick.set_color(color)
    tick.set_fontweight('bold')

ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax2.set_ylim(0, 1.10)
ax2.set_ylabel('업종 구성 비중')
ax2.set_title('개별 행정동 업종 구성 (트렌드업종비중 내림차순)', fontsize=11)
ax2.legend(loc='upper right', fontsize=8.5, ncol=2)
ax2.set_facecolor('#F8F8F8')

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = IMG_DIR / 'gen20_precursor_01_industry.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[저장] {out.name}")
