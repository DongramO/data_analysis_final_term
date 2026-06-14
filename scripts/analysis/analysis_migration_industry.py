# -*- coding: utf-8 -*-
"""
분析 2: 트렌드 업종 매출이 늘어나는 동네에 20대가 더 많이 오는가
분析 3: 이동 증가 이후 어떤 업종이 먼저 변화하는가

출력: image/migration_industry_01_corr.png  (분析 2)
      image/migration_industry_02_lag.png   (분析 3)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import pearsonr
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT      = Path(__file__).resolve().parent
RAW_PATH  = ROOT / 'data/main_data/분기별_원본피처.csv'
MIG_PATH  = ROOT / 'data/main_data/분기별_주말유입인구.csv'
CMAP_PATH = ROOT / 'data/main_data/코드매핑_유입인구_매출.csv'
IMG_DIR   = ROOT / 'image'

REF   = ['방배2동', '청운효자동', '합정동', '성수2가1동']
TIER1 = ['숭인2동', '신당5동']
FOCUS = REF + TIER1

GROUP_COLOR = {
    '성수2가1동': '#E05C5C',
    '합정동':     '#E8836B',
    '방배2동':    '#C0392B',
    '청운효자동': '#F1948A',
    '숭인2동':    '#F5A623',
    '신당5동':    '#F0C27F',
}
DEF_COLOR = '#AAAAAA'

IND_COLS  = ['FB카페_매출금액','FB식사_매출금액','주류유흥_매출금액','라이프스타일_매출금액','기타_매출금액']
IND_NAMES = ['FB카페','FB식사','주류·유흥','라이프스타일','기타']
IND_COLORS= ['#5B9BD5','#ED7D31','#A5A5A5','#FFC000','#D0D0D0']

# ── 데이터 로드 & 병합 ────────────────────────────────────────────────────
raw  = pd.read_csv(RAW_PATH,  encoding='utf-8-sig')
mig  = pd.read_csv(MIG_PATH,  encoding='utf-8-sig')
cmap = pd.read_csv(CMAP_PATH, encoding='utf-8-sig')

mig = mig.merge(cmap[['migration_코드','행정동명']],
                left_on='도착_행정동코드', right_on='migration_코드', how='left')
mig = mig.rename(columns={'행정동명': '행정동'})

# 전체 이동 (모든 연령 합산, 연령대≠0 → 10~80)
total_mig = (mig[mig['연령대'] >= 10]
             .groupby(['행정동','연도','분기'])['합계_이동인구']
             .sum().reset_index().rename(columns={'합계_이동인구':'전체_이동'}))

age20_mig = (mig[mig['연령대'] == 20]
             .groupby(['행정동','연도','분기'])['합계_이동인구']
             .sum().reset_index().rename(columns={'합계_이동인구':'이십대_이동'}))

mig_q = total_mig.merge(age20_mig, on=['행정동','연도','분기'], how='left')
mig_q['이십대_이동비율'] = mig_q['이십대_이동'] / mig_q['전체_이동'].replace(0, np.nan)

# 매출 데이터 업종 비중
raw['트렌드업종_매출'] = sum(raw[c] for c in IND_COLS[:4])
raw['트렌드업종_비중'] = raw['트렌드업종_매출'] / raw['총_매출금액'].replace(0, np.nan)
for c in IND_COLS:
    raw[c+'_비중'] = raw[c] / raw['총_매출금액'].replace(0, np.nan)

# 병합
df = raw.merge(mig_q[['행정동','연도','분기','이십대_이동','이십대_이동비율','전체_이동']],
               on=['행정동','연도','분기'], how='left')
df['분기순번'] = (df['연도'] - 2020) * 4 + df['분기']

# 전년동기 대비 변화율
df = df.sort_values(['행정동','연도','분기'])
df['이십대_이동_YoY'] = df.groupby(['행정동','분기'])['이십대_이동'].pct_change() * 100
df['트렌드업종_비중_YoY'] = df.groupby(['행정동','분기'])['트렌드업종_비중'].diff()

# ══════════════════════════════════════════════════════════════════════════
# 분析 2 — 트렌드 업종 증가 동네에 20대가 더 많이 오는가
# ══════════════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(2, 3, figsize=(20, 12))
fig1.suptitle('분析②  트렌드 업종 매출이 늘어나는 동네에 20대가 더 많이 오는가',
              fontsize=14, fontweight='bold')

# ── 상단: 레퍼런스+Tier1 각 동 이중축 시계열 ─────────────────────────
YEARS = [2020, 2021, 2022, 2023, 2024]

for idx, dong in enumerate(FOCUS):
    ax = axes1[0][idx] if idx < 3 else axes1[1][idx-3]
    sub = df[df['행정동']==dong].groupby('연도').agg(
        트렌드업종_비중=('트렌드업종_비중','mean'),
        이십대_이동비율=('이십대_이동비율','mean'),
        이십대_이동=('이십대_이동','sum'),
    ).reindex(YEARS)

    color = GROUP_COLOR.get(dong, DEF_COLOR)
    ax2 = ax.twinx()

    l1, = ax.plot(YEARS, sub['트렌드업종_비중']*100, 'o-', color=color,
                  lw=2.5, label='트렌드업종 비중(%)')
    l2, = ax2.plot(YEARS, sub['이십대_이동비율']*100, 's--', color='#3A5CA9',
                   lw=2, label='20대 이동비율(%)')

    # 업종 전환 시점 음영
    ax.set_facecolor('#F8F8F8')
    ax.set_xticks(YEARS)
    ax.set_xticklabels([str(y)[2:] + '년' for y in YEARS], fontsize=8)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))
    ax.set_ylabel('트렌드업종 비중', fontsize=8, color=color)
    ax2.set_ylabel('20대 이동비율', fontsize=8, color='#3A5CA9')

    # 상관계수
    valid = sub.dropna(subset=['트렌드업종_비중','이십대_이동비율'])
    if len(valid) >= 3:
        r, p = pearsonr(valid['트렌드업종_비중'], valid['이십대_이동비율'])
        ax.set_title(f'{dong}  (r={r:.3f}{"*" if p<0.05 else ""})', fontsize=10, fontweight='bold', color=color)
    else:
        ax.set_title(dong, fontsize=10, fontweight='bold', color=color)

    lines = [l1, l2]
    labs  = [l.get_label() for l in lines]
    ax.legend(lines, labs, fontsize=7.5, loc='upper left')

# 나머지 칸: 서울 전체 산포도
ax_scatter = axes1[1][2]
# 423개 동 × 5년 패널 — 트렌드업종비중 vs 이십대이동비율
panel = df[df['연도']==2024].groupby('행정동').agg(
    트렌드업종_비중=('트렌드업종_비중','mean'),
    이십대_이동비율=('이십대_이동비율','mean'),
).dropna()

ax_scatter.scatter(panel['트렌드업종_비중']*100, panel['이십대_이동비율']*100,
                   s=15, color='#CCCCCC', alpha=0.5, zorder=1)

for dong in FOCUS:
    row = panel.loc[dong] if dong in panel.index else None
    if row is not None:
        color = GROUP_COLOR.get(dong, DEF_COLOR)
        ax_scatter.scatter(row['트렌드업종_비중']*100, row['이십대_이동비율']*100,
                           s=80, color=color, zorder=3, edgecolors='white', lw=0.8)
        ax_scatter.annotate(dong, (row['트렌드업종_비중']*100, row['이십대_이동비율']*100),
                            fontsize=7.5, color=color, fontweight='bold',
                            xytext=(3, 3), textcoords='offset points')

r_all, p_all = pearsonr(panel['트렌드업종_비중'], panel['이십대_이동비율'])
ax_scatter.set_xlabel('트렌드업종 비중 (%)', fontsize=9)
ax_scatter.set_ylabel('20대 이동비율 (%)', fontsize=9)
ax_scatter.set_title(f'서울 전체 423동 산포 (2024)\nr={r_all:.3f}{"***" if p_all<0.001 else ""}', fontsize=10)
ax_scatter.set_facecolor('#F8F8F8')
ax_scatter.grid(alpha=0.2)

# 추세선
z = np.polyfit(panel['트렌드업종_비중']*100, panel['이십대_이동비율']*100, 1)
xline = np.linspace(panel['트렌드업종_비중'].min()*100, panel['트렌드업종_비중'].max()*100, 100)
ax_scatter.plot(xline, np.polyval(z, xline), 'r--', lw=1.2, alpha=0.7)

plt.tight_layout()
out1 = IMG_DIR / 'migration_industry_01_corr.png'
fig1.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f"[저장] {out1.name}")


# ══════════════════════════════════════════════════════════════════════════
# 분析 3 — 이동 증가 이후 어떤 업종이 먼저 변화하는가 (성수2가1동 집중)
# ══════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 3, figsize=(20, 12))
fig2.suptitle('분析③  20대 이동 증가 이후 어떤 업종이 먼저 변화하는가\n(성수2가1동 케이스 + 전체 교차상관)',
              fontsize=13, fontweight='bold')

# ── 상단: 성수2가1동 20대 이동 vs 각 업종 비중 시계열 ─────────────────
s2 = df[df['행정동']=='성수2가1동'].sort_values(['연도','분기'])
annual_s2 = s2.groupby('연도').agg(
    이십대_이동비율=('이십대_이동비율','mean'),
    이십대_이동=('이십대_이동','sum'),
    **{c+'_비중': (c+'_비중','mean') for c in IND_COLS}
).reindex(YEARS)

for idx, (col, name, color) in enumerate(zip(IND_COLS, IND_NAMES, IND_COLORS)):
    ax = axes2[0][idx] if idx < 3 else axes2[1][idx-3]
    ratio_col = col+'_비중'

    ax2t = ax.twinx()
    l1, = ax.plot(YEARS, annual_s2[ratio_col]*100, 'o-', color=color,
                  lw=2.5, label=f'{name} 비중(%)')
    l2, = ax2t.plot(YEARS, annual_s2['이십대_이동비율']*100, 's--',
                    color='#3A5CA9', lw=2, alpha=0.7, label='20대 이동비율(%)')

    valid = annual_s2[[ratio_col,'이십대_이동비율']].dropna()
    if len(valid) >= 3:
        r, p = pearsonr(valid[ratio_col], valid['이십대_이동비율'])
        ax.set_title(f'{name}\nr={r:.3f}{"*" if p<0.05 else ""}', fontsize=10, fontweight='bold')
    else:
        ax.set_title(name, fontsize=10)

    ax.set_facecolor('#F8F8F8')
    ax.set_xticks(YEARS)
    ax.set_xticklabels([str(y)[2:]+'년' for y in YEARS], fontsize=8)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))
    ax2t.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))
    ax.set_ylabel(f'{name} 비중', fontsize=8, color=color)
    ax2t.set_ylabel('20대 이동비율', fontsize=8, color='#3A5CA9')
    ax.legend([l1, l2], [l1.get_label(), l2.get_label()], fontsize=7.5, loc='upper left')

# ── 하단 우측: 교차상관 (CCF) — lag별 r ──────────────────────────────
ax_ccf = axes2[1][2]

# 분기 단위 CCF (lag -3 ~ +3)
s2q = df[df['행정동']=='성수2가1동'].sort_values('분기순번').copy()
s2q = s2q[s2q['분기순번'] >= 5].copy()  # 2021Q1부터 (YoY 계산 후)

LAGS = range(-3, 4)
ccf_results = {}

for col, name in zip(IND_COLS, IND_NAMES):
    biz_col = col + '_비중'
    rs = []
    for lag in LAGS:
        mig_ser = s2q['이십대_이동'].values
        ind_ser = s2q[biz_col].values
        n = len(mig_ser)
        if lag >= 0:
            x = mig_ser[:n-lag]
            y = ind_ser[lag:]
        else:
            x = mig_ser[-lag:]
            y = ind_ser[:n+lag]
        if len(x) >= 4:
            r, _ = pearsonr(x, y)
        else:
            r = np.nan
        rs.append(r)
    ccf_results[name] = rs

x = list(LAGS)
for name, color in zip(IND_NAMES, IND_COLORS):
    rs = ccf_results[name]
    ax_ccf.plot(x, rs, 'o-', color=color, lw=2, label=name, markersize=6)

ax_ccf.axvline(0, color='black', lw=0.8, ls=':')
ax_ccf.axhline(0, color='black', lw=0.5, ls=':')
ax_ccf.set_xticks(list(LAGS))
ax_ccf.set_xticklabels([f'lag{l:+d}\n{"이동先" if l<0 else ("동시" if l==0 else "업종先")}' for l in LAGS], fontsize=8)
ax_ccf.set_ylabel('Pearson r (이동 vs 업종비중)')
ax_ccf.set_xlabel('← 이동 선행   |   업종 선행 →')
ax_ccf.set_title('교차상관 (CCF)\n20대 이동 vs 각 업종 비중 (성수2가1동, 분기단위)', fontsize=10)
ax_ccf.legend(fontsize=8.5, loc='upper right')
ax_ccf.set_facecolor('#F8F8F8')
ax_ccf.grid(alpha=0.2)
ax_ccf.set_ylim(-1, 1)

plt.tight_layout()
out2 = IMG_DIR / 'migration_industry_02_lag.png'
fig2.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"[저장] {out2.name}")
print("[완료]")
