# -*- coding: utf-8 -*-
"""
2030 vs 40+ 소비 비중 비교: 급부상 / 기성 / 일반 상권
매출 기반(ratio_2030, trend_sales_master) + 이동 기반 보조 비교
출력: image/2030vs40_분석.png
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT     = Path(__file__).resolve().parent
MASTER   = ROOT / 'data/main_data/트렌드매출_마스터.csv'
MIG_PATH = ROOT / 'data/main_data/분기별_주말유입인구.csv'
MAP_PATH = ROOT / 'data/main_data/코드매핑_유입인구_매출.csv'
IMG_DIR  = ROOT / 'image'

C_EMRG = '#E05C5C'
C_LGND = '#3A5CA9'
C_NORM = '#95A5A6'

# ── 그룹 분류 ─────────────────────────────────────────────
master = pd.read_csv(MASTER, encoding='utf-8-sig')
master.columns = ['dong','code','yr','trend_sales','total_sales','trend_ratio','biz_score',
                  'ratio_2030','fri_effect','weekend_ratio','night_ratio',
                  'composite','composite_qtile','tier','tier_label']
p20 = master[master['yr']==2020][['code','composite_qtile','total_sales']].copy()
p24 = master[master['yr']==2024][['code','composite_qtile','total_sales']].copy()
p20.columns = ['code','qtile_2020','sales_2020']
p24.columns = ['code','qtile_2024','sales_2024']
pvt = p20.merge(p24, on='code')
pvt['momentum']     = pvt['qtile_2024'] - pvt['qtile_2020']
pvt['sales_growth'] = (pvt['sales_2024'] - pvt['sales_2020']) / pvt['sales_2020'] * 100
mom_thr = pvt['momentum'].quantile(0.85)
emrg_codes = pvt[(pvt['qtile_2020'] < 0.60) & (pvt['momentum'] >= mom_thr) & (pvt['sales_growth'] > 0)]['code'].tolist()
lgnd_codes = pvt[pvt['qtile_2020'] >= 0.80]['code'].tolist()
norm_codes = [c for c in pvt['code'].tolist() if c not in emrg_codes and c not in lgnd_codes]

YEARS = [2020, 2021, 2022, 2023, 2024]

def sales_2030_trend(codes):
    sub = master[master['code'].isin(codes)]
    return sub.groupby('yr')['ratio_2030'].median() * 100

s_emrg = sales_2030_trend(emrg_codes)
s_lgnd = sales_2030_trend(lgnd_codes)
s_norm = sales_2030_trend(norm_codes)

# ── 이동 기반 2030비중 (보조) ─────────────────────────────
_map = pd.read_csv(MAP_PATH, encoding='utf-8-sig')
_map.columns = ['mig_code','raw_code','gu','dong_map']
_map['raw_code'] = _map['raw_code'].astype('Int64')
code_mig2raw = _map.set_index('mig_code')['raw_code'].to_dict()

_mig = pd.read_csv(MIG_PATH, encoding='utf-8-sig')
_mig.columns = ['mig_code','yr','q','yrq','age','HE','WE','EE','mig_total']
_mig['raw_code'] = _mig['mig_code'].map(code_mig2raw)
_mig = _mig.dropna(subset=['raw_code'])
_mig['raw_code'] = _mig['raw_code'].astype(int)

# 이동: 20대/30대/40+/10대이하 비중
_mig['gen'] = _mig['age'].apply(
    lambda a: '20대' if a==20 else ('30대' if a==30 else ('40+' if a>=40 else '10대이하')))
yr_gen = _mig.groupby(['raw_code','yr','gen'])['mig_total'].sum().reset_index()
yr_tot = yr_gen.groupby(['raw_code','yr'])['mig_total'].sum().rename('tot').reset_index()
yr_gen = yr_gen.merge(yr_tot, on=['raw_code','yr'])
yr_gen['share'] = yr_gen['mig_total'] / yr_gen['tot'] * 100

def mig_trend(codes, gen):
    sub = yr_gen[yr_gen['raw_code'].isin(codes) & (yr_gen['gen'] == gen)]
    return sub.groupby('yr')['share'].median().reindex(YEARS)

# ── 레이아웃 ─────────────────────────────────────────────
fig = plt.figure(figsize=(22, 18))
fig.suptitle('2030 소비·유입 비중 비교: 급부상 / 기성 / 일반 상권 (2020-2024)',
             fontsize=14, fontweight='bold', y=0.99)

gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.50, wspace=0.35,
                       left=0.07, right=0.97, top=0.95, bottom=0.06)

ax1 = fig.add_subplot(gs[0, :])   # ① 매출 기반 2030비중 추이 (핵심)
ax2 = fig.add_subplot(gs[1, :2])  # ② 이동 기반 20대·30대·40+ 비중 추이 (급부상)
ax3 = fig.add_subplot(gs[1, 2])   # ③ 2020→2024 변화량 bar
ax4 = fig.add_subplot(gs[2, :2])  # ④ 이동 기반 2030 비중 - 두 기준 직접 비교
ax5 = fig.add_subplot(gs[2, 2])   # ⑤ 해석

# ══ 패널 1: 매출 기반 2030비중 추이 ════════════════════════
for s, grp, c, ls, lw, mk in [
    (s_emrg, f'급부상 (n={len(emrg_codes)})', C_EMRG, '-',  3.0, 'o'),
    (s_lgnd, f'기성   (n={len(lgnd_codes)})', C_LGND, '--', 2.2, 's'),
    (s_norm, f'일반   (n={len(norm_codes)})', C_NORM, ':',  1.8, '^'),
]:
    ax1.plot(YEARS, s[YEARS].values, color=c, lw=lw, ls=ls, marker=mk, markersize=8, label=grp)
    for yr, v in zip(YEARS, s[YEARS].values):
        offset = 0.4 if grp.startswith('급부상') else (-0.9 if grp.startswith('기성') else 0.4)
        ax1.text(yr, v + offset, f'{v:.1f}%', ha='center', fontsize=9, color=c, fontweight='bold')

ax1.set_xticks(YEARS)
ax1.set_ylabel('2030 소비 비중 중앙값 (%)', fontsize=11)
ax1.set_title('【매출 기반】 2030 소비 비중 연도별 추이\n'
              '→ 기성 상권이 2030 소비 비중 압도적으로 높음 / 급부상 상권은 상승 아닌 하락',
              fontsize=12, pad=8)
ax1.set_facecolor('#FFF9F0')
ax1.grid(alpha=0.2)
ax1.legend(fontsize=10, loc='upper right')
ax1.set_ylim(17, 43)
ax1.axhspan(30, 43, alpha=0.04, color=C_LGND)
ax1.text(2020.05, 41.5, '기성 상권 구간', fontsize=9, color=C_LGND, alpha=0.7)

# ══ 패널 2: 이동 기반 세대별 추이 (급부상) ═════════════════
age_cfg_mig = [
    ('20대', '#E05C5C', '-',  2.5),
    ('30대', '#F5A623', '--', 2.0),
    ('40+',  '#3A5CA9', ':',  1.8),
]
for gen, c, ls, lw in age_cfg_mig:
    s = mig_trend(emrg_codes, gen)
    ax2.plot(YEARS, s.values, color=c, lw=lw, ls=ls, marker='o', markersize=6, label=gen)
    for yr, v in zip(YEARS, s.values):
        ax2.text(yr, v + 0.3, f'{v:.1f}%', ha='center', fontsize=7.5, color=c)

# 기성도 점선으로 추가 (비교용)
for gen, c, ls, lw in [('20대','#E05C5C','-',1.2), ('30대','#F5A623','--',1.2)]:
    s = mig_trend(lgnd_codes, gen)
    ax2.plot(YEARS, s.values, color=c, lw=lw, ls=':', marker='s', markersize=4,
             alpha=0.5, label=f'{gen}(기성)')

ax2.set_xticks(YEARS)
ax2.set_ylabel('이동 비중 중앙값 (%)', fontsize=10)
ax2.set_title('【이동 기반】 급부상 상권 세대별 이동 비중 (보조)\n'
              '(10대이하 14~16% 분모 포함 → 이동비중은 매출비중보다 낮게 측정됨)',
              fontsize=11, pad=8)
ax2.set_facecolor('#F8F8F8')
ax2.grid(alpha=0.2)
ax2.legend(fontsize=8.5, loc='upper right', ncol=2)

# ══ 패널 3: 2020→2024 변화량 ═══════════════════════════════
grps = ['급부상', '기성', '일반']
d_sales = [s_emrg[2024]-s_emrg[2020], s_lgnd[2024]-s_lgnd[2020], s_norm[2024]-s_norm[2020]]
d_mig   = [mig_trend(emrg_codes,'20대')[2024]-mig_trend(emrg_codes,'20대')[2020],
           mig_trend(lgnd_codes,'20대')[2024]-mig_trend(lgnd_codes,'20대')[2020],
           mig_trend(norm_codes,'20대')[2024]-mig_trend(norm_codes,'20대')[2020]]

x = np.arange(3)
w = 0.35
ax3.bar(x - w/2, d_sales, width=w, color='#E67E22', alpha=0.85, edgecolor='white', label='매출 기반 2030비중')
ax3.bar(x + w/2, d_mig,   width=w, color='#27AE60', alpha=0.85, edgecolor='white', label='이동 기반 20대비중')

for i, (vs, vm) in enumerate(zip(d_sales, d_mig)):
    ax3.text(i - w/2, vs - 0.15, f'{vs:.1f}%p', ha='center', va='top', fontsize=8.5,
             color='white', fontweight='bold')
    ax3.text(i + w/2, vm - 0.10, f'{vm:.1f}%p', ha='center', va='top', fontsize=8.5,
             color='white', fontweight='bold')

ax3.axhline(0, color='#888', lw=1)
ax3.set_xticks(x)
ax3.set_xticklabels(grps, fontsize=10)
ax3.set_ylabel('변화량 (%p)', fontsize=10)
ax3.set_title('2020→2024 2030 비중 변화량\n(두 기준 비교)', fontsize=11, pad=8)
ax3.set_facecolor('#F8F8F8')
ax3.grid(axis='y', alpha=0.2)
ax3.legend(fontsize=8.5)

# ══ 패널 4: 두 기준 직접 비교 (급부상 집중) ════════════════
x_pos = np.arange(5)
w2 = 0.3

for codes, grp, c, off in [(emrg_codes, '급부상', C_EMRG, -w2/2),
                             (lgnd_codes, '기성',   C_LGND,  w2/2)]:
    s_s = sales_2030_trend(codes)
    s_m_2030 = mig_trend(codes, '20대') + mig_trend(codes, '30대')
    ax4.plot(YEARS, s_s[YEARS].values, color=c, lw=2.5, marker='o', markersize=7,
             label=f'{grp} 매출기반 2030')
    ax4.plot(YEARS, s_m_2030.values, color=c, lw=1.5, marker='s', markersize=5,
             ls='--', alpha=0.7, label=f'{grp} 이동기반 2030')

ax4.axhspan(0, 15, alpha=0.04, color='gray')
ax4.text(2020.05, 14, '10대이하 분모 포함 → 이동비중 하방 편향', fontsize=8, color='gray')
ax4.set_xticks(YEARS)
ax4.set_ylabel('2030 비중 (%)', fontsize=10)
ax4.set_title('매출 기반 vs 이동 기반 2030비중 직접 비교\n(실선=매출기반, 점선=이동기반 / 매출기반이 실질 소비력 반영)',
              fontsize=11, pad=8)
ax4.set_facecolor('#F8F8FF')
ax4.grid(alpha=0.2)
ax4.legend(fontsize=8.5, loc='upper right')
ax4.set_ylim(10, 45)

# ══ 패널 5: 해석 ═════════════════════════════════════════
ax5.axis('off')
findings = [
    ('핵심 수정', '이동비중 17% → 오해 원인'),
    ('',          '10대이하 14~16%가 분모 포함'),
    ('',          '이동 카운트 ≠ 소비 금액 기준'),
    ('', ''),
    ('매출 기반', '급부상: 29.4% → 23.5%'),
    ('2030비중',  '기성:   38.5% → 31.2%'),
    ('',          '일반:   25.7% → 20.3%'),
    ('', ''),
    ('① 절대 수준', '기성>급부상>일반 (모두 동일)'),
    ('', ''),
    ('② 공통 추세', '전 그룹에서 2030비중 하락'),
    ('',            '→ 전반적 고령화/인구구조 변화'),
    ('', ''),
    ('③ 급부상',   '기성보다 2030 비중 낮지만'),
    ('특이점',      '일반보다는 9%p 이상 높음'),
    ('',            '→ 2030 친화적 환경이 상승의'),
    ('',            '   배경이 될 수는 있으나'),
    ('',            '   2030이 주도하는 구조 아님'),
]

y = 0.97
for label, text in findings:
    if label:
        ax5.text(0.03, y, label, fontsize=9, fontweight='bold',
                 color='#333', transform=ax5.transAxes)
        ax5.text(0.36, y, text, fontsize=9, color='#333', transform=ax5.transAxes)
    else:
        ax5.text(0.36, y, text, fontsize=8.5, color='#555', transform=ax5.transAxes)
    y -= 0.058 if label else 0.052

ax5.set_facecolor('#FAFAFA')
for spine in ax5.spines.values():
    spine.set_edgecolor('#DDDDDD')

# ── 저장 ─────────────────────────────────────────────────
out = IMG_DIR / '2030vs40_분석.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[저장] {out.name}")
