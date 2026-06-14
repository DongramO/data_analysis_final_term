# -*- coding: utf-8 -*-
"""
20대 유입(이동인구) 증가 후 매출이 얼마나 증가하는가
- Panel 1: 20대 이동비율 vs 총매출 YoY 산포 (423개 동 전체)
- Panel 2: 케이스 스터디 이중축 (성수2가1동 외 레퍼런스)
- Panel 3: CCF — 이동 증가가 몇 분기 후 매출로 이어지는가
- Panel 4: 그룹별 이동↑ 분기의 매출 증가율 분포 비교
출력: image/inflow_sales.png
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

REF        = ['방배2동', '청운효자동', '합정동', '성수2가1동']
TIER1      = ['숭인2동', '신당5동']
CANDIDATES = ['잠실6동','자양3동','신대방1동','용산2가동','동화동',
              '문정1동','등촌2동','개포1동','사당3동']

GROUP_COLOR = {'레퍼런스': '#E05C5C', 'Tier1': '#F5A623',
               '신규선별': '#27AE60', '서울전체': '#CCCCCC'}
DONG_COLOR  = {
    '성수2가1동': '#E05C5C', '합정동': '#C0392B',
    '방배2동':    '#E8836B', '청운효자동': '#F1948A',
    '숭인2동':    '#F5A623', '신당5동':    '#F0C27F',
}

def group_label(d):
    if d in REF:        return '레퍼런스'
    if d in TIER1:      return 'Tier1'
    if d in CANDIDATES: return '신규선별'
    return '서울전체'

# ── 데이터 로드 & 병합 ─────────────────────────────────────────────────
raw  = pd.read_csv(RAW_PATH,  encoding='utf-8-sig')
mig  = pd.read_csv(MIG_PATH,  encoding='utf-8-sig')
cmap = pd.read_csv(CMAP_PATH, encoding='utf-8-sig')

mig = mig.merge(cmap[['migration_코드','행정동명']],
                left_on='도착_행정동코드', right_on='migration_코드', how='left')
mig.rename(columns={'행정동명': '행정동'}, inplace=True)

# 20대 이동 / 전체 이동
age20 = (mig[mig['연령대']==20]
         .groupby(['행정동','연도','분기'])['합계_이동인구']
         .sum().reset_index().rename(columns={'합계_이동인구':'이십대_이동'}))
total = (mig[mig['연령대']>=10]
         .groupby(['행정동','연도','분기'])['합계_이동인구']
         .sum().reset_index().rename(columns={'합계_이동인구':'전체_이동'}))

mig_q = age20.merge(total, on=['행정동','연도','분기'])
mig_q['이십대_이동비율'] = mig_q['이십대_이동'] / mig_q['전체_이동'].replace(0, np.nan)

# 매출 + 이동 병합
df = raw.merge(mig_q, on=['행정동','연도','분기'], how='left')
df['분기순번'] = (df['연도'] - 2020) * 4 + df['분기']
df['그룹']     = df['행정동'].map(group_label)

# 전년동기 YoY
df = df.sort_values(['행정동','연도','분기'])
df['매출_YoY']      = df.groupby(['행정동','분기'])['총_매출금액'].pct_change() * 100
df['이동_YoY']      = df.groupby(['행정동','분기'])['이십대_이동'].pct_change() * 100
df['이동비율_diff']  = df.groupby(['행정동','분기'])['이십대_이동비율'].diff()

# ── 그림 ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('20대 유입 증가 후 매출 증가 분析', fontsize=15, fontweight='bold')

YEARS = [2020, 2021, 2022, 2023, 2024]

# ════════════════════════════════════════════════════════════════════════
# Panel 1 (상단 좌): 20대 이동비율 vs 총매출 YoY — 423동 산포
# ════════════════════════════════════════════════════════════════════════
ax1 = axes[0][0]

# 2021 이후 데이터 (YoY 계산 가능한 분기)
panel = df[(df['연도'] >= 2021) & df['매출_YoY'].notna() & df['이동_YoY'].notna()].copy()

# 서울 전체 배경
bg = panel[panel['그룹']=='서울전체']
ax1.scatter(bg['이동_YoY'], bg['매출_YoY'],
            s=10, color='#CCCCCC', alpha=0.3, zorder=1)

# 그룹별 강조
for grp, color in [('레퍼런스','#E05C5C'),('Tier1','#F5A623'),('신규선별','#27AE60')]:
    sub = panel[panel['그룹']==grp]
    ax1.scatter(sub['이동_YoY'], sub['매출_YoY'],
                s=40, color=color, alpha=0.7, zorder=3, label=grp)

# 추세선 (서울 전체)
finite = panel[panel['이동_YoY'].between(-200,200) & panel['매출_YoY'].between(-80,150)]
z = np.polyfit(finite['이동_YoY'], finite['매출_YoY'], 1)
xr = np.linspace(-100, 200, 200)
ax1.plot(xr, np.polyval(z, xr), 'r--', lw=1.5, alpha=0.7, label='추세선')

r, p = pearsonr(finite['이동_YoY'], finite['매출_YoY'])
ax1.axvline(0, color='gray', lw=0.8, ls=':')
ax1.axhline(0, color='gray', lw=0.8, ls=':')
ax1.set_xlabel('20대 이동인구 YoY 변화율 (%)')
ax1.set_ylabel('총매출 YoY 변화율 (%)')
ax1.set_title(f'20대 이동 증가율 vs 매출 증가율\n(전체 분기, r={r:.3f}{"***" if p<0.001 else ""})',
              fontsize=11)
ax1.set_xlim(-150, 300)
ax1.set_ylim(-60, 100)
ax1.legend(fontsize=9)
ax1.set_facecolor('#F8F8F8')
ax1.grid(alpha=0.15)

# ════════════════════════════════════════════════════════════════════════
# Panel 2 (상단 우): 케이스 스터디 — 성수2가1동 이중축
# ════════════════════════════════════════════════════════════════════════
ax2 = axes[0][1]
ax2r = ax2.twinx()

for dong in REF + TIER1:
    sub = df[df['행정동']==dong].groupby('연도').agg(
        총_매출금액=('총_매출금액','sum'),
        이십대_이동=('이십대_이동','sum'),
    ).reindex(YEARS)

    # 2020 기준 인덱스 (100)
    매출_idx = sub['총_매출금액'] / sub['총_매출금액'].iloc[0] * 100
    이동_idx = sub['이십대_이동'] / sub['이십대_이동'].iloc[0] * 100

    color = DONG_COLOR.get(dong, '#AAAAAA')
    lw = 2.5 if dong == '성수2가1동' else 1.5
    ls = '-' if dong in REF else '--'

    ax2.plot(YEARS, 매출_idx, 'o-', color=color, lw=lw, ls=ls, label=dong)
    ax2r.plot(YEARS, 이동_idx, 's:', color=color, lw=lw, alpha=0.5)

ax2.axhline(100, color='gray', lw=0.8, ls=':')
ax2.set_xlabel('연도')
ax2.set_ylabel('총매출 (2020=100)', fontsize=9)
ax2r.set_ylabel('20대 이동인구 (2020=100)', fontsize=9, color='#555555')
ax2r.axhline(100, color='gray', lw=0.8, ls=':')
ax2.set_title('2020년 기준 매출 & 20대이동 지수 비교\n(실선=매출, 점선=이동, 동일 색상)',
              fontsize=11)
ax2.set_xticks(YEARS)
ax2.legend(fontsize=8.5, loc='upper left', ncol=2)
ax2.set_facecolor('#F8F8F8')
ax2.grid(alpha=0.2)

# ════════════════════════════════════════════════════════════════════════
# Panel 3 (하단 좌): CCF — 이동 증가가 몇 분기 후 매출로 이어지는가
# ════════════════════════════════════════════════════════════════════════
ax3 = axes[1][0]

LAGS = range(-4, 5)

for dong, color in [('성수2가1동','#E05C5C'),('방배2동','#C0392B'),
                    ('숭인2동','#F5A623'),('신당5동','#F0C27F')]:
    sub = df[df['행정동']==dong].sort_values('분기순번')
    sub = sub[(sub['연도']>=2021) & sub['이동_YoY'].notna() & sub['매출_YoY'].notna()]
    mig_s  = sub['이동_YoY'].values
    sale_s = sub['매출_YoY'].values
    n = len(mig_s)

    rs = []
    for lag in LAGS:
        if lag >= 0:
            x, y = mig_s[:n-lag] if lag > 0 else mig_s, sale_s[lag:] if lag > 0 else sale_s
        else:
            x, y = mig_s[-lag:], sale_s[:n+lag]
        if len(x) >= 4:
            r, _ = pearsonr(x, y)
        else:
            r = np.nan
        rs.append(r)

    lw = 2.5 if dong in ['성수2가1동'] else 1.8
    ax3.plot(list(LAGS), rs, 'o-', color=color, lw=lw, label=dong, markersize=5)

ax3.axvline(0, color='black', lw=0.8, ls=':')
ax3.axhline(0, color='black', lw=0.5, ls=':')
ax3.fill_between([0, 4], -1, 1, alpha=0.05, color='orange', label='이동 선행 구간')
ax3.set_xticks(list(LAGS))
ax3.set_xticklabels([f'lag{l:+d}' for l in LAGS], fontsize=8.5)
ax3.set_xlabel('← 매출 선행          이동 선행 →', fontsize=9)
ax3.set_ylabel('Pearson r (이동 YoY vs 매출 YoY)')
ax3.set_title('교차상관 (CCF): 20대 이동 증가 → 매출 증가\n양의 lag = 이동 선행 (이동 후 n분기 뒤 매출)', fontsize=11)
ax3.legend(fontsize=9)
ax3.set_facecolor('#F8F8F8')
ax3.grid(alpha=0.2)
ax3.set_ylim(-1, 1)

# ════════════════════════════════════════════════════════════════════════
# Panel 4 (하단 우): 이동 증가 분기 vs 감소 분기의 매출 증가율 분포
# ════════════════════════════════════════════════════════════════════════
ax4 = axes[1][1]

# 서울 전체에서 "이동 증가 분기"와 "이동 감소 분기"로 나눠 매출 증가율 분포 비교
valid = df[(df['연도']>=2021) & df['이동_YoY'].notna() & df['매출_YoY'].notna()].copy()
valid['이동방향'] = valid['이동_YoY'].apply(lambda x: '20대 이동 증가(+)' if x > 0 else '20대 이동 감소(-)')

# 그룹별 박스플롯
import matplotlib.patches as mpatches

groups_to_plot = ['레퍼런스', 'Tier1', '신규선별', '서울전체']
positions = []
colors_bp = ['#E05C5C','#F5A623','#27AE60','#AAAAAA']
offset = 0

for gi, grp in enumerate(groups_to_plot):
    sub_grp = valid[valid['그룹']==grp]
    up   = sub_grp[sub_grp['이동방향']=='20대 이동 증가(+)']['매출_YoY'].dropna()
    down = sub_grp[sub_grp['이동방향']=='20대 이동 감소(-)']['매출_YoY'].dropna()

    x_up   = gi * 3 + 0.5
    x_down = gi * 3 + 1.5
    color  = colors_bp[gi]

    bp_up = ax4.boxplot(up.clip(-60, 100), positions=[x_up], widths=0.7,
                        patch_artist=True, showfliers=False,
                        boxprops=dict(facecolor=color, alpha=0.8),
                        medianprops=dict(color='white', lw=2))
    bp_down = ax4.boxplot(down.clip(-60, 100), positions=[x_down], widths=0.7,
                          patch_artist=True, showfliers=False,
                          boxprops=dict(facecolor=color, alpha=0.3),
                          medianprops=dict(color='black', lw=2))

    # 중앙값 표시
    ax4.text(x_up,   np.median(up.clip(-60,100))   + 2, f'{np.median(up.clip(-60,100)):.1f}%',
             ha='center', fontsize=7.5, fontweight='bold', color=color)
    ax4.text(x_down, np.median(down.clip(-60,100)) + 2, f'{np.median(down.clip(-60,100)):.1f}%',
             ha='center', fontsize=7.5, color=color)

ax4.axhline(0, color='black', lw=0.8, ls=':')
ax4.set_xticks([1, 4, 7, 10])
ax4.set_xticklabels(groups_to_plot, fontsize=9)
ax4.set_ylabel('총매출 YoY 변화율 (%)')
ax4.set_title('20대 이동 증가 분기 vs 감소 분기의 매출 증가율 분포\n(진한 박스=이동↑, 연한 박스=이동↓, 흰숫자=중앙값)', fontsize=10)
ax4.set_facecolor('#F8F8F8')
ax4.grid(axis='y', alpha=0.2)

# 범례
legend_patches = [
    mpatches.Patch(color='gray', alpha=0.8, label='이동 증가(+) 분기'),
    mpatches.Patch(color='gray', alpha=0.3, label='이동 감소(-) 분기'),
]
ax4.legend(handles=legend_patches, fontsize=9)

plt.tight_layout()
out = IMG_DIR / 'inflow_sales.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[저장] {out.name}")
