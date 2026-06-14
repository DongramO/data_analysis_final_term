# -*- coding: utf-8 -*-
"""
20대 트렌드 반응 민감도 검증
: 급부상 상권에서 20대 유입 증가율이 다른 세대보다 일관되게 높은가?
출력: image/age_reactivity_분석.png
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import wilcoxon

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT     = Path(__file__).resolve().parent
MASTER   = ROOT / 'data/main_data/트렌드매출_마스터.csv'
MIG_PATH = ROOT / 'data/main_data/분기별_주말유입인구.csv'
MAP_PATH = ROOT / 'data/main_data/코드매핑_유입인구_매출.csv'
IMG_DIR  = ROOT / 'image'

# ── 그룹 분류 ─────────────────────────────────────────────
master = pd.read_csv(MASTER, encoding='utf-8-sig')
master.columns = ['dong','code','yr','trend_sales','total_sales','trend_ratio','biz_score',
                  'ratio_2030','fri_effect','weekend_ratio','night_ratio',
                  'composite','composite_qtile','tier','tier_label']
p20 = master[master['yr']==2020][['code','dong','composite_qtile','total_sales']].copy()
p24 = master[master['yr']==2024][['code','composite_qtile','total_sales']].copy()
p20.columns = ['code','dong','qtile_2020','sales_2020']
p24.columns = ['code','qtile_2024','sales_2024']
pvt = p20.merge(p24, on='code')
pvt['momentum']     = pvt['qtile_2024'] - pvt['qtile_2020']
pvt['sales_growth'] = (pvt['sales_2024'] - pvt['sales_2020']) / pvt['sales_2020'] * 100
mom_thr = pvt['momentum'].quantile(0.85)
emrg_codes = pvt[(pvt['qtile_2020'] < 0.60) & (pvt['momentum'] >= mom_thr) & (pvt['sales_growth'] > 0)]['code'].tolist()
lgnd_codes = pvt[pvt['qtile_2020'] >= 0.80]['code'].tolist()
norm_codes = [c for c in pvt['code'] if c not in emrg_codes and c not in lgnd_codes]

# ── 이동 데이터 로드 ──────────────────────────────────────
_map = pd.read_csv(MAP_PATH, encoding='utf-8-sig')
_map.columns = ['mig_code','raw_code','gu','dong_map']
_map['raw_code'] = _map['raw_code'].astype('Int64')
code_mig2raw = _map.set_index('mig_code')['raw_code'].to_dict()

_mig = pd.read_csv(MIG_PATH, encoding='utf-8-sig')
_mig.columns = ['mig_code','yr','q','yrq','age','HE','WE','EE','mig_total']
_mig['raw_code'] = _mig['mig_code'].map(code_mig2raw)
_mig = _mig.dropna(subset=['raw_code'])
_mig['raw_code'] = _mig['raw_code'].astype(int)

_mig['gen'] = _mig['age'].apply(
    lambda a: '20대' if a==20 else ('30대' if a==30 else ('40+' if a>=40 else 'other')))

yr_gen = (_mig.groupby(['raw_code','yr','gen'])['mig_total'].sum().reset_index())

def growth_rate(codes, gen, yr_from, yr_to):
    """각 동별 특정 세대 유입 증가율 리스트 반환"""
    sub = yr_gen[yr_gen['raw_code'].isin(codes) & (yr_gen['gen']==gen)]
    piv = sub.pivot_table(index='raw_code', columns='yr', values='mig_total', aggfunc='sum')
    if yr_from not in piv.columns or yr_to not in piv.columns:
        return np.array([])
    valid = piv[(piv[yr_from] > 0) & (piv[yr_to] > 0)]
    return ((valid[yr_to] - valid[yr_from]) / valid[yr_from] * 100).values

def median_growth(codes, gen, yr_from, yr_to):
    g = growth_rate(codes, gen, yr_from, yr_to)
    return np.median(g) if len(g) else np.nan

# ── 전체 구간(2020→2024) + 전반기(2020→2022) + 후반기(2022→2024) ──
PERIODS = [('전체 (2020→2024)', 2020, 2024),
           ('전반기 (2020→2022)', 2020, 2022),
           ('후반기 (2022→2024)', 2022, 2024)]
GENS   = ['20대','30대','40+']
GROUPS = [('급부상', emrg_codes, '#E05C5C'),
          ('기성',   lgnd_codes, '#3A5CA9'),
          ('일반',   norm_codes, '#95A5A6')]

# ── 동별 증가율 계산 (급부상 전체 구간) ─────────────────────
rows = []
for code in emrg_codes:
    dong_name = pvt[pvt['code']==code]['dong'].values
    dong_name = dong_name[0] if len(dong_name) else str(code)
    r = {'code': code, 'dong': dong_name}
    for gen in GENS:
        sub = yr_gen[(yr_gen['raw_code']==code) & (yr_gen['gen']==gen)]
        piv = sub.set_index('yr')['mig_total']
        v20 = piv.get(2020, 0)
        v24 = piv.get(2024, 0)
        r[f'{gen}_2020'] = v20
        r[f'{gen}_2024'] = v24
        r[f'{gen}_growth'] = (v24-v20)/v20*100 if v20 > 0 else np.nan
    rows.append(r)
df_emrg = pd.DataFrame(rows).dropna(subset=['20대_growth','30대_growth','40+_growth'])

# 20대 > 30대 > 40+ 인 비율
n_20gt30 = (df_emrg['20대_growth'] > df_emrg['30대_growth']).sum()
n_20gt40 = (df_emrg['20대_growth'] > df_emrg['40+_growth']).sum()
n_all    = (df_emrg['20대_growth'] > df_emrg['30대_growth']) & (df_emrg['20대_growth'] > df_emrg['40+_growth'])
n_all    = n_all.sum()
n_tot    = len(df_emrg)

# Wilcoxon 검정 (20대 vs 40+)
_, p_val = wilcoxon(df_emrg['20대_growth'], df_emrg['40+_growth'])

# ── 레이아웃 ─────────────────────────────────────────────
fig = plt.figure(figsize=(22, 18))
fig.suptitle('20대 트렌드 반응 민감도 검증: 급부상 상권에서 20대 유입 증가율이 다른 세대보다 높은가?',
             fontsize=13, fontweight='bold', y=0.99)

gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.48, wspace=0.35,
                       left=0.07, right=0.97, top=0.94, bottom=0.06)

ax1 = fig.add_subplot(gs[0, :2])   # 세대별 중앙값 증가율 비교 (기간×그룹)
ax2 = fig.add_subplot(gs[0, 2])    # 20대 > 40+ 비율 요약
ax3 = fig.add_subplot(gs[1, :2])   # 산점도: 급부상 동별 20대 vs 40+ 증가율
ax4 = fig.add_subplot(gs[1, 2])    # 증가율 분포 박스플롯 (급부상만)
ax5 = fig.add_subplot(gs[2, :])    # 전반기 vs 후반기 비교 (그룹별)

C20  = '#E05C5C'
C30  = '#F5A623'
C40P = '#3A5CA9'

# ══ 패널 1: 기간×그룹별 세대 중앙값 증가율 ═══════════════
period_labels = ['전체\n(2020→24)', '전반기\n(2020→22)', '후반기\n(2022→24)']
x = np.arange(len(PERIODS))
w = 0.12
offsets = {'급부상': -0.25, '기성': 0, '일반': 0.25}
gen_colors = {'20대': C20, '30대': C30, '40+': C40P}
gen_markers = {'20대': 'o', '30대': 's', '40+': '^'}

for gi, (grp, codes, gc) in enumerate(GROUPS):
    for gen in GENS:
        vals = [median_growth(codes, gen, yf, yt) for _, yf, yt in PERIODS]
        xpos = x + offsets[grp] + (GENS.index(gen) - 1) * w
        ax1.bar(xpos, vals, width=w*0.85,
                color=gen_colors[gen], alpha=(0.9 if grp=='급부상' else 0.45),
                edgecolor='white', linewidth=0.5,
                label=f'{grp}-{gen}' if gi==0 else None)

ax1.axhline(0, color='#888', lw=1)
ax1.set_xticks(x)
ax1.set_xticklabels(period_labels, fontsize=10)
ax1.set_ylabel('주말 유입 증가율 중앙값 (%)', fontsize=10)
ax1.set_title('기간별·그룹별·세대별 주말 유입 증가율 (중앙값)\n'
              '진한 색=급부상 / 연한 색=기성·일반',
              fontsize=11, pad=8)
ax1.set_facecolor('#F8F8F8')
ax1.grid(axis='y', alpha=0.2)

# 범례 직접 구성
handles = []
for gen, c in gen_colors.items():
    handles.append(mpatches.Patch(color=c, label=gen))
handles += [mpatches.Patch(color='#888', alpha=0.9, label='급부상'),
            mpatches.Patch(color='#888', alpha=0.4, label='기성/일반')]
ax1.legend(handles=handles, fontsize=8.5, loc='upper right', ncol=2)

# ══ 패널 2: 20대 > 40+ 비율 요약 ════════════════════════
categories = [
    f'20대 > 40+\n({n_20gt40}/{n_tot})',
    f'20대 > 30대\n({n_20gt30}/{n_tot})',
    f'20대 > 30대 & 40+\n({n_all}/{n_tot})',
]
pcts = [n_20gt40/n_tot*100, n_20gt30/n_tot*100, n_all/n_tot*100]
colors_bar = [C40P, C30, C20]
bars = ax2.barh(categories, pcts, color=colors_bar, alpha=0.80, edgecolor='white', height=0.5)
for bar, pct in zip(bars, pcts):
    ax2.text(pct + 1, bar.get_y() + bar.get_height()/2,
             f'{pct:.0f}%', va='center', fontsize=11, fontweight='bold')
ax2.axvline(50, color='#888', lw=1.2, ls='--', alpha=0.6)
ax2.text(51, -0.5, '50%', fontsize=8, color='#888')
ax2.set_xlim(0, 105)
ax2.set_xlabel('해당 조건 충족 동 비율 (%)', fontsize=10)
ax2.set_title(f'급부상 {n_tot}개 동 중\n20대 증가율이 타 세대보다 높은 비율\n(Wilcoxon 20대 vs 40+: p={p_val:.4f})',
              fontsize=10, pad=8)
ax2.set_facecolor('#F8F8F8')
ax2.grid(axis='x', alpha=0.2)

# ══ 패널 3: 산점도 20대 vs 40+ (급부상 동별) ════════════
ax3.scatter(df_emrg['40+_growth'], df_emrg['20대_growth'],
            color=C20, alpha=0.7, s=60, edgecolors='white', linewidth=0.5, zorder=3)

lim_max = max(df_emrg['40+_growth'].max(), df_emrg['20대_growth'].max()) * 1.05
lim_min = min(df_emrg['40+_growth'].min(), df_emrg['20대_growth'].min()) - 10
ax3.plot([lim_min, lim_max], [lim_min, lim_max], color='#888', lw=1.2, ls='--', alpha=0.7, label='동일 증가율 기준선')
ax3.fill_between([lim_min, lim_max], [lim_min, lim_max], lim_max,
                 alpha=0.05, color=C20)
ax3.text(lim_max*0.3, lim_max*0.85, '↑ 20대 증가율 > 40+ 증가율', fontsize=9, color=C20, alpha=0.8)

# 성수동 강조
ss_codes = [11200650, 11200660, 11200670, 11200690]
ss_df = df_emrg[df_emrg['code'].isin(ss_codes)]
ax3.scatter(ss_df['40+_growth'], ss_df['20대_growth'],
            color='#8E44AD', s=100, edgecolors='white', linewidth=1, zorder=5, label='성수동')
for _, r in ss_df.iterrows():
    ax3.annotate(r['dong'], (r['40+_growth'], r['20대_growth']),
                 fontsize=7.5, color='#8E44AD',
                 xytext=(4, 4), textcoords='offset points')

n_above = (df_emrg['20대_growth'] > df_emrg['40+_growth']).sum()
ax3.set_xlabel('40+ 주말 유입 증가율 (%)', fontsize=10)
ax3.set_ylabel('20대 주말 유입 증가율 (%)', fontsize=10)
ax3.set_title(f'급부상 상권 동별 20대 vs 40+ 주말 유입 증가율\n'
              f'기준선 위(20대 우세): {n_above}/{n_tot}개 동 ({n_above/n_tot*100:.0f}%)',
              fontsize=11, pad=8)
ax3.set_facecolor('#F8F8F8')
ax3.grid(alpha=0.2)
ax3.legend(fontsize=9)

# ══ 패널 4: 박스플롯 (급부상 세대별 증가율 분포) ══════════
bp_data = [df_emrg['20대_growth'].values,
           df_emrg['30대_growth'].values,
           df_emrg['40+_growth'].values]
bp = ax4.boxplot(bp_data, patch_artist=True, widths=0.5,
                 medianprops=dict(color='white', linewidth=2.5))
for patch, c in zip(bp['boxes'], [C20, C30, C40P]):
    patch.set_facecolor(c)
    patch.set_alpha(0.75)
for i, d in enumerate(bp_data):
    med = np.median(d)
    ax4.text(i+1, med + 2, f'{med:.0f}%', ha='center', fontsize=9, fontweight='bold')
ax4.set_xticklabels(['20대','30대','40+'], fontsize=11)
ax4.set_ylabel('주말 유입 증가율 (%)', fontsize=10)
ax4.set_title('급부상 상권 세대별\n증가율 분포 (2020→2024)', fontsize=11, pad=8)
ax4.set_facecolor('#F8F8F8')
ax4.grid(axis='y', alpha=0.2)

# ══ 패널 5: 전반기 vs 후반기 비교 ═══════════════════════
period_pairs = [('전반기\n(2020→22)', 2020, 2022), ('후반기\n(2022→24)', 2022, 2024)]
x5 = np.arange(len(GROUPS))
w5 = 0.22
for pi, (plabel, yf, yt) in enumerate(period_pairs):
    for gi, (grp, codes, gc) in enumerate(GROUPS):
        medians = [median_growth(codes, gen, yf, yt) for gen in GENS]
        x_pos = gi * (len(GENS) * w5 + 0.3)
        for geni, (gen, med) in enumerate(zip(GENS, medians)):
            bar_x = x_pos + geni * w5 + pi * 0.08
            alpha = 0.9 if pi == 0 else 0.55
            hatch = None if pi == 0 else '//'
            ax5.bar(bar_x, med, width=w5*0.85,
                    color=gen_colors[gen], alpha=alpha, hatch=hatch,
                    edgecolor='white', linewidth=0.5)
            if abs(med) > 3:
                ax5.text(bar_x, med + (1 if med >= 0 else -4),
                         f'{med:.0f}%', ha='center', fontsize=7, color='#333')

# 그룹 레이블
group_x_centers = []
for gi in range(len(GROUPS)):
    cx = gi * (len(GENS) * w5 + 0.3) + (len(GENS)-1) * w5 / 2 + 0.04
    group_x_centers.append(cx)
    grp = GROUPS[gi][0]
    ax5.text(cx, ax5.get_ylim()[0] - 5 if hasattr(ax5, '_ylim') else -20,
             grp, ha='center', fontsize=10, fontweight='bold', color=GROUPS[gi][2])

ax5.axhline(0, color='#888', lw=1)
ax5.set_ylabel('주말 유입 증가율 중앙값 (%)', fontsize=10)
ax5.set_title('전반기(진한색) vs 후반기(연한+빗금) 세대별 증가율 비교\n'
              '→ 급부상 상권에서 전반기에 20대 반응이 먼저 강했는가?',
              fontsize=11, pad=8)
ax5.set_facecolor('#F8F8F8')
ax5.grid(axis='y', alpha=0.2)

handles5 = []
for gen, c in gen_colors.items():
    handles5.append(mpatches.Patch(color=c, alpha=0.85, label=gen))
handles5 += [mpatches.Patch(color='#888', alpha=0.9, label='전반기(2020→22)'),
             mpatches.Patch(color='#888', alpha=0.5, hatch='//', label='후반기(2022→24)')]
for gi, (grp, _, gc) in enumerate(GROUPS):
    ax5.text(group_x_centers[gi], -30, grp, ha='center', fontsize=10,
             fontweight='bold', color=gc)
ax5.legend(handles=handles5, fontsize=8.5, loc='upper right', ncol=3)

# ── 저장 ─────────────────────────────────────────────────
out = IMG_DIR / 'age_reactivity_분석.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[저장] {out.name}")
