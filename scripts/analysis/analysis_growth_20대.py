# -*- coding: utf-8 -*-
"""
성장 행정동에서 20대 이동의 선행 여부 & 비중 분석
출력: image/growth_20대_분석.png
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.stats import mannwhitneyu
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT      = Path(__file__).resolve().parent
BASE_PATH = ROOT / 'data/_tmp_growth_base.csv'
CCF_PATH  = ROOT / 'data/_tmp_age_ccf.csv'
MIG_PATH  = ROOT / 'data/main_data/분기별_주말유입인구.csv'
MAP_PATH  = ROOT / 'data/main_data/코드매핑_유입인구_매출.csv'
IMG_DIR   = ROOT / 'image'

C_GROW  = '#E05C5C'   # 성장 — 빨강
C_OTHER = '#3A5CA9'   # 정체/감소 — 파랑
C_20대  = '#27AE60'   # 20대 강조 — 초록

# ── 데이터 로드 ──────────────────────────────────────────
base    = pd.read_csv(BASE_PATH, encoding='utf-8-sig')
age_ccf = pd.read_csv(CCF_PATH, encoding='utf-8-sig')

_map = pd.read_csv(MAP_PATH, encoding='utf-8-sig')
_map.columns = ['mig_code','raw_code','gu','dong_map']
_map['raw_code'] = _map['raw_code'].astype('Int64')
code_mig2raw = _map.set_index('mig_code')['raw_code'].to_dict()

_mig = pd.read_csv(MIG_PATH, encoding='utf-8-sig')
_mig.columns = ['mig_code','yr','q','yrq','age','HE','WE','EE','mig_total']
_mig['raw_code'] = _mig['mig_code'].map(code_mig2raw)
_mig = _mig.dropna(subset=['raw_code'])
_mig['raw_code'] = _mig['raw_code'].astype(int)

# ── 그룹 매핑 ─────────────────────────────────────────────
age_ccf = age_ccf.merge(base[['행정동코드','성장여부']],
                         left_on='code', right_on='행정동코드', how='left')
grow  = age_ccf[age_ccf['성장여부'] == '성장']
other = age_ccf[age_ccf['성장여부'] == '정체/감소']

# ── 연도별 세대별 비중 시계열 ──────────────────────────────
yr_age = _mig.groupby(['raw_code','yr','age'])['mig_total'].sum().reset_index()
yr_tot = yr_age.groupby(['raw_code','yr'])['mig_total'].sum().rename('tot').reset_index()
yr_age = yr_age.merge(yr_tot, on=['raw_code','yr'])
yr_age['share'] = yr_age['mig_total'] / yr_age['tot'] * 100

# 20대 비중
share20 = yr_age[yr_age['age'] == 20][['raw_code','yr','share']].copy()
share20 = share20.merge(base[['행정동코드','성장여부']], left_on='raw_code', right_on='행정동코드', how='left')

# 연도×그룹별 중앙값
trend = share20.groupby(['yr','성장여부'])['share'].median().reset_index()

# ── 세대별 비중 분포 (2020 vs 2024, 성장 그룹만) ────────────
AGE_MAP = {10:'10대', 20:'20대', 30:'30대', 40:'40대', 50:'50대', 60:'60이상', 70:'60이상', 80:'60이상'}
yr_age['age_label'] = yr_age['age'].map(AGE_MAP)
age_grp = yr_age.groupby(['raw_code','yr','age_label'])['share'].sum().reset_index()

# ══════════════════════════════════════════════════════════
# 레이아웃: 3행 × 2열
# ══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 18))
fig.suptitle('성장 행정동에서 20대 이동의 선행 여부 및 비중 분석 (2020–2024)',
             fontsize=15, fontweight='bold', y=0.99)

gs = gridspec.GridSpec(3, 2, figure=fig,
                       hspace=0.50, wspace=0.32,
                       left=0.07, right=0.97, top=0.94, bottom=0.05)

ax1 = fig.add_subplot(gs[0, 0])   # lag 분포 비교
ax2 = fig.add_subplot(gs[0, 1])   # lag별 평균 r 프로파일
ax3 = fig.add_subplot(gs[1, 0])   # 20대 비중 박스플롯
ax4 = fig.add_subplot(gs[1, 1])   # 20대 비중 연도별 추이
ax5 = fig.add_subplot(gs[2, 0])   # 세대별 비중 2020 vs 2024 (성장 그룹)
ax6 = fig.add_subplot(gs[2, 1])   # 세대별 비중 2020 vs 2024 (정체 그룹)

# ══ 패널 1: peak_lag 분포 막대 비교 ══════════════════════
LAGS = range(-3, 5)
lag_cnt = {}
for grp, df_g in [('성장(n=379)', grow), ('정체/감소(n=44)', other)]:
    cnt = df_g['peak_lag_age'].value_counts().reindex(LAGS, fill_value=0)
    lag_cnt[grp] = (cnt / len(df_g) * 100).round(1)

x  = np.arange(len(LAGS))
w  = 0.35
ax1.bar(x - w/2, lag_cnt['성장(n=379)'],     width=w, color=C_GROW,  alpha=0.85, label='성장', edgecolor='white')
ax1.bar(x + w/2, lag_cnt['정체/감소(n=44)'], width=w, color=C_OTHER, alpha=0.85, label='정체/감소', edgecolor='white')

ax1.axvline(np.where(list(LAGS) == np.array(0))[0][0], color='#888', lw=1, ls='--', alpha=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels([f'lag{l:+d}' for l in LAGS], fontsize=9)
ax1.set_ylabel('해당 lag가 peak인 행정동 비율 (%)', fontsize=10)
ax1.set_title('20대↔30대+ 선행 패턴 분포\n(lag>0=20대 선행 / lag<0=30대 선행 / lag0=동시)', fontsize=11, pad=8)
ax1.set_facecolor('#F8F8F8')
ax1.grid(axis='y', alpha=0.2)
ax1.legend(fontsize=9)

# lag=0 경계선 강조
for lag_i, lag_v in enumerate(LAGS):
    if lag_v == 1:
        ax1.axvspan(lag_i - 0.5, len(list(LAGS)) - 0.5,
                    alpha=0.06, color=C_20대, zorder=0)
        ax1.text(lag_i + 0.2, ax1.get_ylim()[1] * 0.92, '← 20대 선행 구간',
                 fontsize=8, color=C_20대)

# ══ 패널 2: lag별 평균 r 프로파일 ══════════════════════════
LAGS_LIST = list(LAGS)
r_grow  = [grow[f'r_lag{l:+d}'].mean()  for l in LAGS_LIST]
r_other = [other[f'r_lag{l:+d}'].mean() for l in LAGS_LIST]

ax2.plot(LAGS_LIST, r_grow,  color=C_GROW,  lw=2.5, marker='o', markersize=6, label='성장')
ax2.plot(LAGS_LIST, r_other, color=C_OTHER, lw=2.5, marker='s', markersize=6, label='정체/감소', ls='--')
ax2.axvline(0, color='#888', lw=1, ls='--', alpha=0.6)
ax2.axhline(0, color='#CCC', lw=1)
ax2.fill_between([0, 4], -0.1, 1.0, alpha=0.07, color=C_20대, label='20대 선행 구간')

ax2.set_xticks(LAGS_LIST)
ax2.set_xticklabels([f'{l:+d}Q' for l in LAGS_LIST], fontsize=9)
ax2.set_xlabel('lag (분기)', fontsize=10)
ax2.set_ylabel('평균 CCF r', fontsize=10)
ax2.set_ylim(-0.2, 1.0)
ax2.set_title('lag별 평균 상관계수 (20대 YoY vs 30대+ YoY)\n→ lag+1 이후 r이 급락 = 20대가 30대+를 끌지 못함', fontsize=11, pad=8)
ax2.set_facecolor('#F8F8F8')
ax2.grid(alpha=0.2)
ax2.legend(fontsize=9)

# ══ 패널 3: 20대 비중 박스플롯 ════════════════════════════
bp_data = [
    share20[(share20['yr']==2020)&(share20['성장여부']=='성장')]['share'].dropna(),
    share20[(share20['yr']==2024)&(share20['성장여부']=='성장')]['share'].dropna(),
    share20[(share20['yr']==2020)&(share20['성장여부']=='정체/감소')]['share'].dropna(),
    share20[(share20['yr']==2024)&(share20['성장여부']=='정체/감소')]['share'].dropna(),
]
bp_labels = ['성장\n2020', '성장\n2024', '정체/감소\n2020', '정체/감소\n2024']
bp_colors = [C_GROW, C_GROW, C_OTHER, C_OTHER]

bp = ax3.boxplot(bp_data, patch_artist=True, widths=0.5,
                 medianprops=dict(color='white', linewidth=2.5))
for patch, c in zip(bp['boxes'], bp_colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.75)

# 중앙값 텍스트
for i, d in enumerate(bp_data):
    med = np.median(d)
    ax3.text(i+1, med + 0.3, f'{med:.1f}%', ha='center', fontsize=8.5, fontweight='bold')

# 통계 검정 (성장2024 vs 정체2024)
stat, pval = mannwhitneyu(bp_data[1], bp_data[3], alternative='greater')
ax3.set_xticklabels(bp_labels, fontsize=9)
ax3.set_ylabel('20대 이동인구 비중 (%)', fontsize=10)
ax3.set_title(f'20대 이동 비중 분포 비교\n(성장2024 > 정체2024 Mann-Whitney p={pval:.3f})', fontsize=11, pad=8)
ax3.set_facecolor('#F8F8F8')
ax3.grid(axis='y', alpha=0.2)

# ══ 패널 4: 20대 비중 연도별 추이 ══════════════════════════
yr_list = sorted(trend['yr'].unique())
for grp, c, ls in [('성장', C_GROW, '-'), ('정체/감소', C_OTHER, '--')]:
    sub = trend[trend['성장여부'] == grp].sort_values('yr')
    ax4.plot(sub['yr'], sub['share'],
             color=c, lw=2.5, ls=ls, marker='o', markersize=7, label=grp)
    for _, r in sub.iterrows():
        ax4.text(r['yr'], r['share'] + 0.15, f"{r['share']:.1f}%",
                 ha='center', fontsize=8.5, color=c)

ax4.set_xticks(yr_list)
ax4.set_xlabel('연도', fontsize=10)
ax4.set_ylabel('20대 이동 비중 중앙값 (%)', fontsize=10)
ax4.set_title('20대 이동 비중 연도별 추이\n(중앙값 기준)', fontsize=11, pad=8)
ax4.set_facecolor('#F8F8F8')
ax4.grid(alpha=0.2)
ax4.legend(fontsize=9)

# ══ 패널 5·6: 세대별 비중 변화 (2020 vs 2024) ══════════════
AGE_ORDER  = ['10대','20대','30대','40대','50대','60이상']
AGE_COLORS = ['#95A5A6','#E05C5C','#F5A623','#3A5CA9','#27AE60','#8E44AD']

for ax_b, grp_name, bg in [(ax5, '성장', '#FFF5F5'), (ax6, '정체/감소', '#F5F7FF')]:
    codes = base[base['성장여부'] == grp_name]['행정동코드'].tolist()
    sub   = age_grp[age_grp['raw_code'].isin(codes)]
    grp_m = sub.groupby(['yr','age_label'])['share'].median().reset_index()

    x2  = np.arange(len(AGE_ORDER))
    w2  = 0.35
    v20 = [grp_m[(grp_m['yr']==2020)&(grp_m['age_label']==a)]['share'].values[0]
           if len(grp_m[(grp_m['yr']==2020)&(grp_m['age_label']==a)]) else 0 for a in AGE_ORDER]
    v24 = [grp_m[(grp_m['yr']==2024)&(grp_m['age_label']==a)]['share'].values[0]
           if len(grp_m[(grp_m['yr']==2024)&(grp_m['age_label']==a)]) else 0 for a in AGE_ORDER]

    bars20 = ax_b.bar(x2 - w2/2, v20, width=w2, color=AGE_COLORS, alpha=0.55,
                      edgecolor='white', linewidth=1.5, label='2020')
    bars24 = ax_b.bar(x2 + w2/2, v24, width=w2, color=AGE_COLORS, alpha=0.9,
                      edgecolor='white', linewidth=1.5, label='2024')

    for i, (a20, a24) in enumerate(zip(v20, v24)):
        diff = a24 - a20
        sign = '+' if diff >= 0 else ''
        color = '#E05C5C' if diff < 0 else '#27AE60'
        ax_b.text(i, max(a20, a24) + 0.3, f'{sign}{diff:.1f}%p',
                  ha='center', fontsize=8, color=color, fontweight='bold')

    ax_b.set_xticks(x2)
    ax_b.set_xticklabels(AGE_ORDER, fontsize=10)
    ax_b.set_ylabel('이동 비중 중앙값 (%)', fontsize=10)
    ax_b.set_title(f'【{grp_name}】 세대별 이동 비중 변화 (2020 → 2024)', fontsize=11,
                   color=C_GROW if grp_name=='성장' else C_OTHER, fontweight='bold', pad=8)
    ax_b.set_facecolor(bg)
    ax_b.grid(axis='y', alpha=0.2)
    ax_b.legend(fontsize=9, loc='upper right')

# ── 저장 ─────────────────────────────────────────────────
out = IMG_DIR / 'growth_20대_분석.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[저장] {out.name}")
