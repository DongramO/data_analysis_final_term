# -*- coding: utf-8 -*-
"""
보고서 시각화 재생성 (신규 레퍼런스 4개 기준)
출력: report/images/
  S3_01  레퍼런스 4개 궤적 (업종·트렌드·2030비중·복합점수)
  S3_05  레퍼런스 4개 매출 시계열 (전체 vs 20대)
  S3_06  레퍼런스 4개 이동 시계열 (전체·20대·비율)
  S3_07  레퍼런스 vs 서울 지표 비교 (bar)
  S3_08  성수2가1동 케이스 스터디 (이동→매출→임대료)
  S4a_01 Phase35 차세대 후보 v3 (유사도 bar + scatter)
  S4b_01 Phase41 사분면 + Tier 분류
  S5_04  레퍼런스 vs Tier 그룹 비교 (이동·매출·Tier 지표)
  S5_05  시계열 3차원 연관 (레퍼런스 + Tier1)
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.makedirs('report/images', exist_ok=True)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 색상 ──────────────────────────────────────────────────────────
C_REF   = ['#2E86AB', '#E84855', '#F18F01', '#3BB273']  # 레퍼런스 4개
C_T1    = '#9B2226'   # Tier1
C_T2    = '#AE2012'
C_T3    = '#CA6702'
C_SEOUL = '#888888'   # 서울 전체
C_FILL  = '#DDDDDD'   # 배경 fill

REF_DONGS = ["방배2동", "청운효자동", "합정동", "성수2가1동"]
REF_LABEL = {"방배2동": "방배2동", "청운효자동": "청운효자동",
             "합정동": "합정동", "성수2가1동": "성수2가1동"}
YEARS = [2020, 2021, 2022, 2023, 2024]

# ── 데이터 로드 ───────────────────────────────────────────────────
print("데이터 로드 중...")
sales   = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig')
area    = pd.read_csv('data/main_data/area_type_profile.csv', encoding='utf-8-sig')
cands   = pd.read_csv('data/main_data/trend_candidates_v3.csv', encoding='utf-8-sig')
quad    = pd.read_csv('data/main_data/quadrant_20_result.csv', encoding='utf-8-sig')
mig_q   = pd.read_csv('data/main_data/분기별_주말유입인구.csv', encoding='utf-8-sig')
codemap = pd.read_csv('data/main_data/코드매핑_유입인구_매출.csv', encoding='utf-8-sig')
rental  = pd.read_csv('data/main_data/임대료_공실률.csv', encoding='utf-8-sig')

# 이동인구: 연간 합산
code_dict = dict(zip(codemap['행정동명'].str.strip(), codemap['migration_코드']))
mig20 = mig_q[mig_q['연령대'] == 20].groupby(['도착_행정동코드','연도'])['합계_이동인구'].sum().reset_index()
mig_all = mig_q.groupby(['도착_행정동코드','연도'])['합계_이동인구'].sum().reset_index()
mig20.columns  = ['코드','연도','20대이동']
mig_all.columns= ['코드','연도','전체이동']
mig_ann = mig20.merge(mig_all, on=['코드','연도'])
mig_ann['20대비율'] = mig_ann['20대이동'] / mig_ann['전체이동'] * 100

def get_mig(dong, year):
    code = code_dict.get(dong)
    if code is None: return np.nan, np.nan, np.nan
    row = mig_ann[(mig_ann['코드']==code)&(mig_ann['연도']==year)]
    if row.empty: return np.nan, np.nan, np.nan
    r = row.iloc[0]
    return r['20대이동']/10000, r['전체이동']/10000, r['20대비율']

# 임대료 딕셔너리
rent_map = {'성수2가1동':'뚝섬','방배2동':'방배역/내방역','합정동':'홍대/합정','청운효자동':'광화문'}
rent_dict= {(r['상권명'],int(r['연도'])):(r['임대료'],r['공실률']) for _,r in rental.iterrows()}

def get_rent(dong, year):
    sang = rent_map.get(dong)
    if sang is None: return np.nan, np.nan
    return rent_dict.get((sang, year), (np.nan, np.nan))

# 서울 전체 연도별 평균
seoul_avg = sales.groupby('연도')[['업종점수','트렌드매출_비중','2030비중','복합점수_분위']].mean()

# Tier 분류
tier1 = cands[cands['tier'].str.contains('Tier1',na=False)]['행정동'].tolist()
tier2 = cands[cands['tier'].str.contains('Tier2',na=False)]['행정동'].tolist()
tier3 = cands[cands['tier'].str.contains('Tier3',na=False)]['행정동'].tolist()

# ═══════════════════════════════════════════════════════════════════
# S3_01  레퍼런스 4개 궤적 (2×3 grid)
# ═══════════════════════════════════════════════════════════════════
print("[1/8] S3_01 레퍼런스 궤적...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('레퍼런스 4개 동 — 트렌드 상권 전환 궤적 (2020~2024)', fontsize=15, fontweight='bold', y=0.98)

METRICS = [
    ('업종점수', '업종점수 (%)', lambda v: v*100),
    ('트렌드매출_비중', '트렌드매출 비중 (%)', lambda v: v*100),
    ('2030비중', '2030 소비 비중 (%)', lambda v: v*100),
    ('복합점수_분위', '복합점수 분위 (0~1)', lambda v: v),
]

for idx, (col, ylabel, fn) in enumerate(METRICS):
    ax = axes[idx // 3][idx % 3]
    seoul_y = [fn(seoul_avg.loc[y, col]) for y in YEARS]
    ax.fill_between(YEARS, [v*0.85 for v in seoul_y], [v*1.15 for v in seoul_y],
                    alpha=0.15, color=C_SEOUL)
    ax.plot(YEARS, seoul_y, '--', color=C_SEOUL, lw=1.2, label='서울 전체')
    for i, dong in enumerate(REF_DONGS):
        s = sales[sales['행정동']==dong].set_index('연도')
        vals = [fn(s.loc[y, col]) if y in s.index else np.nan for y in YEARS]
        ax.plot(YEARS, vals, 'o-', color=C_REF[i], lw=2.2, ms=6, label=dong)
    ax.set_title(ylabel, fontsize=11, fontweight='bold')
    ax.set_xticks(YEARS); ax.set_xticklabels([str(y) for y in YEARS], fontsize=9)
    ax.grid(axis='y', alpha=0.3); ax.spines[['top','right']].set_visible(False)
    if idx == 0: ax.legend(fontsize=8, loc='upper left')

# 5번째: 2024 지표 bar 비교
ax5 = axes[1][1]
metrics_bar = ['업종점수','트렌드매출_비중','2030비중']
labels_bar  = ['업종점수','트렌드매출','2030비중']
x = np.arange(len(metrics_bar)); w = 0.18
for i, dong in enumerate(REF_DONGS):
    s24 = sales[(sales['행정동']==dong)&(sales['연도']==2024)]
    if s24.empty: continue
    vals = [s24[m].values[0]*100 for m in metrics_bar]
    ax5.bar(x + i*w - 1.5*w, vals, w, color=C_REF[i], label=dong, alpha=0.85)
seoul24_vals = [seoul_avg.loc[2024, m]*100 for m in metrics_bar]
ax5.bar(x + 4*w - 1.5*w, seoul24_vals, w, color=C_SEOUL, label='서울전체', alpha=0.6)
ax5.set_title('2024년 지표 비교', fontsize=11, fontweight='bold')
ax5.set_xticks(x); ax5.set_xticklabels(labels_bar, fontsize=9)
ax5.legend(fontsize=7); ax5.set_ylabel('%'); ax5.grid(axis='y', alpha=0.3)
ax5.spines[['top','right']].set_visible(False)

# 6번째: 레퍼런스 선정 기준 텍스트 요약
ax6 = axes[1][2]
ax6.axis('off')
summary = (
    "레퍼런스 선정 기준 (2026-05-26 확정)\n\n"
    "① 주말집중도 ≥ 0.80   (오피스형 제거)\n"
    "② 업종점수 전환        (≥ 0.50, 2020 미충족)\n"
    "③ 2030비중 전환        (≥ 0.35)\n"
    "④ 트렌드매출_비중 ≥ 0.30\n"
    "⑤ 대학가 제외\n\n"
    "확정 레퍼런스 4개\n"
    "  ▸ 방배2동    혼합형  0.937\n"
    "  ▸ 청운효자동  주거형  0.975\n"
    "  ▸ 합정동     주거형  0.993\n"
    "  ▸ 성수2가1동  혼합형  0.921"
)
ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=9,
         va='top', ha='left', family='Malgun Gothic',
         bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))

plt.tight_layout()
plt.savefig('report/images/S3_01_reference_trajectory.png', dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════
# S3_05  레퍼런스 4개 매출 시계열 (2×2)
# ═══════════════════════════════════════════════════════════════════
print("[2/8] S3_05 레퍼런스 매출 시계열...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('레퍼런스 4개 동 — 전체 매출 vs 20대 매출 시계열 (2020~2024)', fontsize=14, fontweight='bold')

raw_q = pd.read_csv('data/main_data/분기별_원본피처.csv', encoding='utf-8-sig')
raw_q['YQ'] = raw_q['연도'].astype(str) + 'Q' + raw_q['분기'].astype(str)
yq_labels = sorted(raw_q['YQ'].unique())

for i, dong in enumerate(REF_DONGS):
    ax = axes[i//2][i%2]
    sub = raw_q[raw_q['행정동']==dong].sort_values(['연도','분기'])
    yq_idx = range(len(sub))
    tot = sub['총_매출금액'].values / 1e8
    s20 = (sub['연령대_20_매출_금액'] + sub['연령대_30_매출_금액']).values / 1e8

    ax2 = ax.twinx()
    ax.bar(yq_idx, tot, color='#BDC3C7', alpha=0.7, label='전체 매출')
    ax.bar(yq_idx, s20, color=C_REF[i], alpha=0.85, label='2030 매출')

    ratio = (sub['연령대_20_매출_금액']+sub['연령대_30_매출_금액']) / sub['총_매출금액'] * 100
    ax2.plot(yq_idx, ratio.values, 'o-', color='#2C3E50', lw=1.8, ms=4, label='2030 비중(%)')
    ax2.set_ylabel('2030 비중 (%)', fontsize=8, color='#2C3E50')
    ax2.tick_params(axis='y', labelcolor='#2C3E50', labelsize=8)

    # x축 연도 구분선
    year_starts = [0]
    for yr in [2021,2022,2023,2024]:
        idx_yr = sub[sub['연도']==yr].index
        if len(idx_yr): year_starts.append(list(sub.index).index(idx_yr[0]))
    for xs in year_starts[1:]:
        ax.axvline(xs-0.5, color='gray', lw=0.7, ls='--', alpha=0.5)

    tick_pos  = [0,4,8,12,16]
    tick_labs = ['2020','2021','2022','2023','2024']
    ax.set_xticks([p for p in tick_pos if p < len(sub)])
    ax.set_xticklabels([tick_labs[j] for j,p in enumerate(tick_pos) if p < len(sub)], fontsize=9)
    ax.set_title(f'{dong}', fontsize=12, fontweight='bold', color=C_REF[i])
    ax.set_ylabel('매출 (억원)', fontsize=9)
    ax.grid(axis='y', alpha=0.3); ax.spines[['top','right']].set_visible(False)

    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labs1+labs2, fontsize=7, loc='upper left')

plt.tight_layout()
plt.savefig('report/images/S3_05_ref_sales_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════
# S3_06  레퍼런스 4개 이동 시계열 (2×2)
# ═══════════════════════════════════════════════════════════════════
print("[3/8] S3_06 레퍼런스 이동 시계열...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('레퍼런스 4개 동 — 전체 이동인구 vs 20대 이동인구 시계열 (2020~2024 연간)', fontsize=14, fontweight='bold')

for i, dong in enumerate(REF_DONGS):
    ax = axes[i//2][i%2]
    code = code_dict.get(dong)
    if code is None: continue

    sub_all = mig_ann[mig_ann['코드']==code].sort_values('연도')
    tot_y = sub_all['전체이동'].values/10000
    s20_y = sub_all['20대이동'].values/10000
    rat_y = sub_all['20대비율'].values
    yrs   = sub_all['연도'].values

    ax2 = ax.twinx()
    ax.fill_between(yrs, tot_y, color='#BDC3C7', alpha=0.5, label='전체 이동인구')
    ax.fill_between(yrs, s20_y, color=C_REF[i], alpha=0.8, label='20대 이동인구')
    ax.plot(yrs, s20_y, 'o-', color=C_REF[i], lw=2, ms=6)
    ax2.plot(yrs, rat_y, 's--', color='#2C3E50', lw=1.8, ms=5, label='20대 비율(%)')
    ax2.set_ylabel('20대 비율 (%)', fontsize=8, color='#2C3E50')
    ax2.tick_params(axis='y', labelcolor='#2C3E50', labelsize=8)

    ax.set_title(f'{dong}', fontsize=12, fontweight='bold', color=C_REF[i])
    ax.set_ylabel('이동인구 (만 명)', fontsize=9)
    ax.set_xticks(yrs); ax.set_xticklabels([str(y) for y in yrs], fontsize=9)
    ax.grid(axis='y', alpha=0.3); ax.spines[['top','right']].set_visible(False)

    # 값 표기
    for y, v20, vr in zip(yrs, s20_y, rat_y):
        ax.annotate(f'{v20:.1f}만', (y, v20), textcoords='offset points',
                    xytext=(0,6), ha='center', fontsize=7.5, color=C_REF[i])

    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labs1+labs2, fontsize=7, loc='upper left')

plt.tight_layout()
plt.savefig('report/images/S3_06_ref_migration_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════
# S3_07  구역유형별 지표 비교 bar (§2-1, §2-2)
# ═══════════════════════════════════════════════════════════════════
print("[4/8] S3_07 구역유형 비교...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('구역유형 분류 — 주말집중도 기준 및 소비 특성 비교', fontsize=14, fontweight='bold')

TYPE_ORDER  = ['오피스형','혼합형','주거형','상권형']
TYPE_COLORS = ['#E84855','#F18F01','#3BB273','#2E86AB']

area_s = area.copy()
# 주말집중도 분포 박스플롯
ax0 = axes[0]
data_box = [area_s[area_s['구역유형']==t]['주말집중도'].values for t in TYPE_ORDER]
bp = ax0.boxplot(data_box, patch_artist=True, medianprops={'color':'black','lw':2})
for patch, color in zip(bp['boxes'], TYPE_COLORS):
    patch.set_facecolor(color); patch.set_alpha(0.75)
# 임계선
ax0.axhline(0.80, ls='--', color='red', lw=1.2, alpha=0.7, label='오피스형 임계(0.80)')
ax0.axhline(0.95, ls='--', color='orange', lw=1.2, alpha=0.7, label='혼합형 임계(0.95)')
ax0.axhline(1.10, ls='--', color='green', lw=1.2, alpha=0.7, label='주거→상권 임계(1.10)')
ax0.set_xticklabels(TYPE_ORDER, fontsize=9)
ax0.set_ylabel('주말집중도', fontsize=10)
ax0.set_title('주말집중도 분포\n(분류 기준)', fontsize=11, fontweight='bold')
ax0.legend(fontsize=7); ax0.grid(axis='y', alpha=0.3)
ax0.spines[['top','right']].set_visible(False)

# 동 수 + 평균 주말집중도
ax1 = axes[1]
cnts = [len(area_s[area_s['구역유형']==t]) for t in TYPE_ORDER]
avgs = [area_s[area_s['구역유형']==t]['주말집중도'].mean() for t in TYPE_ORDER]
bars = ax1.bar(TYPE_ORDER, cnts, color=TYPE_COLORS, alpha=0.8, edgecolor='white')
for bar, cnt, avg in zip(bars, cnts, avgs):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
             f'{cnt}개\n평균 {avg:.3f}', ha='center', va='bottom', fontsize=8.5)
ax1.set_ylabel('행정동 수', fontsize=10)
ax1.set_title('구역유형별 동 수\n(서울 423개)', fontsize=11, fontweight='bold')
ax1.set_ylim(0, 360); ax1.grid(axis='y', alpha=0.3)
ax1.spines[['top','right']].set_visible(False)

# 업종점수 vs 트렌드매출 scatter
ax2 = axes[2]
for t, color in zip(TYPE_ORDER, TYPE_COLORS):
    sub = area_s[area_s['구역유형']==t]
    ax2.scatter(sub['업종점수']*100, sub['주말집중도'], c=color, alpha=0.5, s=25, label=t)

# 레퍼런스 강조
for i, dong in enumerate(REF_DONGS):
    row = area_s[area_s['행정동']==dong]
    if row.empty: continue
    ax2.scatter(row['업종점수'].values[0]*100, row['주말집중도'].values[0],
                c=C_REF[i], s=180, marker='D', zorder=5, edgecolors='black', lw=0.8)
    ax2.annotate(dong, (row['업종점수'].values[0]*100, row['주말집중도'].values[0]),
                 xytext=(4,4), textcoords='offset points', fontsize=7.5, color=C_REF[i])

ax2.axhline(0.80, ls='--', color='red', lw=1, alpha=0.6)
ax2.set_xlabel('업종점수 (%)', fontsize=10)
ax2.set_ylabel('주말집중도', fontsize=10)
ax2.set_title('업종점수 × 주말집중도\n(◆ = 레퍼런스)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8); ax2.grid(alpha=0.2)
ax2.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('report/images/S3_07_area_type_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════
# S3_08  성수2가1동 케이스 스터디 (이동→매출→임대료)
# ═══════════════════════════════════════════════════════════════════
print("[5/8] S3_08 성수2가1동 케이스...")
fig = plt.figure(figsize=(18, 10))
fig.suptitle('성수2가1동 — 이동 선행 → 소비 전환 → 임대료 반응 케이스 스터디',
             fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

dong_s = '성수2가1동'
s_sub  = sales[sales['행정동']==dong_s].set_index('연도')
code_s = code_dict.get(dong_s)
mig_s  = mig_ann[mig_ann['코드']==code_s].set_index('연도') if code_s else pd.DataFrame()

# (1) 20대 이동인구 + 비율
ax1 = fig.add_subplot(gs[0,0])
m20  = [mig_s.loc[y,'20대이동']/10000 if y in mig_s.index else np.nan for y in YEARS]
mrat = [mig_s.loc[y,'20대비율'] if y in mig_s.index else np.nan for y in YEARS]
ax1b = ax1.twinx()
ax1.bar(YEARS, m20, color='#2E86AB', alpha=0.7, label='20대 이동인구(만)')
ax1b.plot(YEARS, mrat, 's--', color='#E84855', lw=2, ms=7, label='20대 비율(%)')
ax1b.set_ylabel('20대 비율 (%)', color='#E84855', fontsize=9)
ax1b.tick_params(axis='y', labelcolor='#E84855')
ax1.set_title('① 20대 이동인구', fontsize=11, fontweight='bold')
ax1.set_ylabel('만 명')
ax1.axvspan(2020.5, 2021.5, alpha=0.12, color='yellow', label='선행 구간')
ax1.set_xticks(YEARS)
for y, v in zip(YEARS, m20):
    if not np.isnan(v): ax1.text(y, v+0.3, f'{v:.1f}', ha='center', fontsize=8, color='#2E86AB')
ax1.grid(axis='y', alpha=0.3); ax1.spines[['top','right']].set_visible(False)

# (2) 업종점수 + 트렌드매출
ax2 = fig.add_subplot(gs[0,1])
bs = [s_sub.loc[y,'업종점수']*100 if y in s_sub.index else np.nan for y in YEARS]
tm = [s_sub.loc[y,'트렌드매출_비중']*100 if y in s_sub.index else np.nan for y in YEARS]
ax2.plot(YEARS, bs, 'o-', color='#3BB273', lw=2.5, ms=8, label='업종점수(%)')
ax2.plot(YEARS, tm, 's-', color='#F18F01', lw=2.5, ms=8, label='트렌드매출(%)')
ax2.axvspan(2021.5, 2022.5, alpha=0.12, color='#3BB273', label='전환 구간')
for y, v1, v2 in zip(YEARS, bs, tm):
    if not np.isnan(v1): ax2.annotate(f'{v1:.0f}%', (y,v1), xytext=(0,6), textcoords='offset points',
                                       ha='center', fontsize=7.5, color='#3BB273')
    if not np.isnan(v2): ax2.annotate(f'{v2:.0f}%', (y,v2), xytext=(0,-12), textcoords='offset points',
                                       ha='center', fontsize=7.5, color='#F18F01')
ax2.set_title('② 소비 지표 전환', fontsize=11, fontweight='bold')
ax2.set_ylabel('%'); ax2.set_ylim(15, 80)
ax2.set_xticks(YEARS); ax2.legend(fontsize=8)
ax2.grid(axis='y', alpha=0.3); ax2.spines[['top','right']].set_visible(False)

# (3) 임대료 (뚝섬) + 공실률
ax3 = fig.add_subplot(gs[0,2])
rent_y = [get_rent(dong_s,y)[0] for y in YEARS]
vac_y  = [get_rent(dong_s,y)[1] for y in YEARS]
ax3b = ax3.twinx()
ax3.plot(YEARS, rent_y, 'D-', color='#9B2226', lw=2.5, ms=8, label='임대료(천원/㎡)')
ax3b.bar(YEARS, vac_y, color='#CA6702', alpha=0.5, label='공실률(%)', width=0.4)
ax3b.set_ylabel('공실률 (%)', color='#CA6702', fontsize=9)
ax3b.tick_params(axis='y', labelcolor='#CA6702')
for y, v in zip(YEARS, rent_y):
    if not np.isnan(v): ax3.annotate(f'{v:.1f}', (y,v), xytext=(0,7), textcoords='offset points',
                                      ha='center', fontsize=8, color='#9B2226')
ax3.set_title('③ 임대료 반응 (뚝섬 상권)', fontsize=11, fontweight='bold')
ax3.set_ylabel('임대료'); ax3.set_xticks(YEARS)
ax3.grid(axis='y', alpha=0.3); ax3.spines[['top','right']].set_visible(False)

# (4) 20대 매출 절대액
ax4 = fig.add_subplot(gs[1,0])
raw_q_s = pd.read_csv('data/main_data/분기별_원본피처.csv', encoding='utf-8-sig')
ann_s = raw_q_s[raw_q_s['행정동']==dong_s].groupby('연도').agg(
    총매출=('총_매출금액','sum'), 매출20대=('연령대_20_매출_금액','sum')).reset_index()
ann_s['총억'] = ann_s['총매출']/1e8; ann_s['20대억'] = ann_s['매출20대']/1e8
x = np.arange(len(ann_s)); w=0.35
ax4.bar(x-w/2, ann_s['총억'], w, color='#BDC3C7', label='전체 매출')
ax4.bar(x+w/2, ann_s['20대억'], w, color='#2E86AB', label='20대 매출')
ax4.set_xticks(x); ax4.set_xticklabels(ann_s['연도'].astype(str), fontsize=9)
ax4.set_title('④ 전체 vs 20대 매출 (억원)', fontsize=11, fontweight='bold')
ax4.set_ylabel('억원'); ax4.legend(fontsize=8)
ax4.grid(axis='y', alpha=0.3); ax4.spines[['top','right']].set_visible(False)

# (5) 전환 타임라인 텍스트 박스
ax5 = fig.add_subplot(gs[1,1])
ax5.axis('off')
timeline = (
    "전환 타임라인\n\n"
    "2020  20대 이동 13.1만 (10.7%)\n"
    "        업종점수 33.7%, 트렌드매출 24.8%\n"
    "        ─────────────────────────────\n"
    "2021  ▲ 20대 이동 17.5만 (+34%)\n"
    "        이동 먼저 증가, 소비 미전환\n"
    "        ─────────────────────────────\n"
    "2022  ▲▲ 이동 29.7만 + 소비 급전환\n"
    "        업종점수 47→61% (+13.8%p)\n"
    "        트렌드매출 24→31% (+7.2%p)\n"
    "        임대료 37.5→43.8 (뚝섬)\n"
    "        ─────────────────────────────\n"
    "2024  이동 44.1만 (26.0%)\n"
    "        임대료 54.7 → +51.1% (2020比)\n"
    "        공실률 3.0% (초저위 유지)"
)
ax5.text(0.05, 0.95, timeline, transform=ax5.transAxes, fontsize=9,
         va='top', family='Malgun Gothic',
         bbox=dict(boxstyle='round', facecolor='#fff9e6', alpha=0.9))
ax5.set_title('전환 단계 요약', fontsize=11, fontweight='bold')

# (6) 합정동 vs 성수 비교 (포화 vs 전환)
ax6 = fig.add_subplot(gs[1,2])
compare_dongs = {'성수2가1동':'전환 완료', '합정동':'포화(이탈)'}
compare_colors= {'성수2가1동':'#2E86AB', '합정동':'#E84855'}
mets = ['업종점수','트렌드매출_비중','2030비중','복합점수_분위']
met_labs = ['업종점수','트렌드매출','2030비중','복합점수']
for dong_c, label in compare_dongs.items():
    s24c = sales[(sales['행정동']==dong_c)&(sales['연도']==2024)]
    if s24c.empty: continue
    vals = [s24c[m].values[0]*100 if m != '복합점수_분위' else s24c[m].values[0]*100 for m in mets]
    ax6.plot(met_labs, vals, 'o-', color=compare_colors[dong_c], lw=2.2, ms=7, label=f'{dong_c} ({label})')

# 서울 평균
s_avg = [seoul_avg.loc[2024, m]*100 for m in mets]
ax6.plot(met_labs, s_avg, 's--', color=C_SEOUL, lw=1.5, ms=6, label='서울 평균')
ax6.set_title('⑥ 성수 vs 합정동 vs 서울 (2024)', fontsize=11, fontweight='bold')
ax6.legend(fontsize=8); ax6.grid(alpha=0.25)
ax6.spines[['top','right']].set_visible(False)

plt.savefig('report/images/S3_08_seongsu_case_study.png', dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════
# S4a_01  Phase35 차세대 후보 v3 (코사인 유사도 + scatter)
# ═══════════════════════════════════════════════════════════════════
print("[6/8] S4a_01 Phase35 후보 v3...")
fig, axes = plt.subplots(1, 3, figsize=(20, 8))
fig.suptitle('Phase35 차세대 상권 후보 — 레퍼런스 궤적 코사인 유사도 (v3, 신규 레퍼런스 4개 기준)',
             fontsize=13, fontweight='bold')

p35 = cands[cands['tier'].str.contains('Tier1|Tier2', na=False)].sort_values('코사인유사도', ascending=False).head(15)

TIER_COLOR_MAP = {'Tier1_교집합': C_T1, 'Tier2_업종전환': '#AE6C00'}
tier_colors = [TIER_COLOR_MAP.get(t, '#888') for t in p35['tier']]

# (1) 유사도 bar
ax = axes[0]
bars = ax.barh(p35['행정동'][::-1], p35['코사인유사도'][::-1],
               color=[TIER_COLOR_MAP.get(t,'#888') for t in p35['tier'][::-1]], alpha=0.85)
for bar, val in zip(bars, p35['코사인유사도'][::-1]):
    ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=8)
ax.set_xlabel('레퍼런스 궤적 코사인 유사도'); ax.set_title('코사인 유사도 TOP15', fontsize=11, fontweight='bold')
ax.set_xlim(0.85, 0.97); ax.grid(axis='x', alpha=0.3)
ax.axvline(0.93, ls='--', color='red', lw=1, alpha=0.6, label='0.93 기준선')
patches = [Patch(color=C_T1, label='Tier1 교집합'), Patch(color='#AE6C00', label='Tier2 업종전환')]
ax.legend(handles=patches, fontsize=8, loc='lower right')
ax.spines[['top','right']].set_visible(False)

# (2) scatter: 업종점수 × 트렌드매출
ax2 = axes[1]
for _, row in cands.iterrows():
    t = row['tier']
    if 'Tier3' in t: c, mk, ms = C_T3, 'o', 30
    elif 'Tier2' in t: c, mk, ms = '#AE6C00', 's', 50
    elif 'Tier1' in t: c, mk, ms = C_T1, '*', 140
    else: continue
    ax2.scatter(row['업종점수']*100, row['트렌드매출_비중']*100, c=c, marker=mk, s=ms, alpha=0.8)

# 레퍼런스
for i, dong in enumerate(REF_DONGS):
    s24 = sales[(sales['행정동']==dong)&(sales['연도']==2024)]
    if s24.empty: continue
    ax2.scatter(s24['업종점수'].values[0]*100, s24['트렌드매출_비중'].values[0]*100,
                c=C_REF[i], marker='D', s=100, edgecolors='black', lw=1, zorder=5)
    ax2.annotate(dong, (s24['업종점수'].values[0]*100, s24['트렌드매출_비중'].values[0]*100),
                 xytext=(3,3), textcoords='offset points', fontsize=7, color=C_REF[i])

# Tier1 라벨
for _, row in cands[cands['tier'].str.contains('Tier1',na=False)].iterrows():
    ax2.annotate(row['행정동'], (row['업종점수']*100, row['트렌드매출_비중']*100),
                 xytext=(4,4), textcoords='offset points', fontsize=8, color=C_T1, fontweight='bold')

ax2.set_xlabel('업종점수 (%)', fontsize=10)
ax2.set_ylabel('트렌드매출 비중 (%)', fontsize=10)
ax2.set_title('업종점수 × 트렌드매출 분포\n(2024 기준)', fontsize=11, fontweight='bold')
patches = [Patch(color=C_T1,label='Tier1 교집합'), Patch(color='#AE6C00',label='Tier2 업종전환'),
           Patch(color=C_T3,label='Tier3 이동선행'), Patch(color='gray',label='레퍼런스')]
ax2.legend(handles=patches, fontsize=8); ax2.grid(alpha=0.2)
ax2.spines[['top','right']].set_visible(False)

# (3) 그룹 평균 비교 bar
ax3 = axes[2]
grp_labels = ['서울전체', 'Tier3\n이동선행', 'Tier2\n업종전환', 'Tier1\n교집합', '레퍼런스(4)']
grp_colors = [C_SEOUL, C_T3, '#AE6C00', C_T1, '#2E86AB']
metrics_grp = ['업종점수', '트렌드매출_비중', '복합점수_분위']
met_labs_g = ['업종점수', '트렌드매출', '복합점수분위']

seoul24 = sales[sales['연도']==2024][metrics_grp].mean()
ref24   = sales[(sales['행정동'].isin(REF_DONGS))&(sales['연도']==2024)][metrics_grp].mean()

grp_vals = []
grp_vals.append([seoul24[m]*100 for m in metrics_grp])
for tier_dongs_list, tier_name in [(tier3,'Tier3_이동선행'),(tier2,'Tier2_업종전환'),(tier1,'Tier1_교집합')]:
    sub = sales[(sales['행정동'].isin(tier_dongs_list))&(sales['연도']==2024)][metrics_grp].mean()
    grp_vals.append([sub[m]*100 for m in metrics_grp])
grp_vals.append([ref24[m]*100 for m in metrics_grp])

x = np.arange(len(met_labs_g)); w = 0.14
for gi, (grp, color) in enumerate(zip(grp_labels, grp_colors)):
    ax3.bar(x + gi*w - 2*w, grp_vals[gi], w, color=color, label=grp, alpha=0.85)
ax3.set_xticks(x); ax3.set_xticklabels(met_labs_g, fontsize=9)
ax3.set_ylabel('%'); ax3.legend(fontsize=7, loc='upper right')
ax3.set_title('그룹별 평균 지표 비교\n(2024)', fontsize=11, fontweight='bold')
ax3.grid(axis='y', alpha=0.3); ax3.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('report/images/S4a_01_phase35_candidates.png', dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════
# S4b_01  Phase41 사분면 + Tier 분류
# ═══════════════════════════════════════════════════════════════════
print("[7/8] S4b_01 Phase41 사분면...")
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Phase41 — 20대 이동 성장세 × 매출 활성도 사분면 분석', fontsize=14, fontweight='bold')

p41 = quad.copy()
p41 = p41.merge(area[['행정동','구역유형','주말집중도']], on='행정동', how='left')
p41_valid = p41[p41['주말집중도']>=0.80].copy()

ax = axes[0]
# 전체 배경 (주말집중도 기준 통과)
mask_q2   = p41_valid['사분면'] == '2사분면_유망'
mask_other= ~mask_q2

ax.scatter(p41_valid[mask_other]['매출활성도점수'], p41_valid[mask_other]['이동성장세점수'],
           c='#CCCCCC', alpha=0.4, s=20, zorder=1)
ax.scatter(p41_valid[mask_q2]['매출활성도점수'], p41_valid[mask_q2]['이동성장세점수'],
           c=C_T3, alpha=0.7, s=60, zorder=3, label='2사분면 유망 (Tier3)')

# Tier1 강조
for dong in tier1:
    row = p41_valid[p41_valid['행정동']==dong]
    if row.empty: continue
    ax.scatter(row['매출활성도점수'], row['이동성장세점수'], c=C_T1, s=150, marker='*',
               edgecolors='black', lw=1, zorder=5)
    ax.annotate(dong, (row['매출활성도점수'].values[0], row['이동성장세점수'].values[0]),
                xytext=(5,3), textcoords='offset points', fontsize=9, color=C_T1, fontweight='bold')

# 상위 Tier3 라벨
p41_q2_sorted = p41_valid[mask_q2].nlargest(8,'이동성장세점수')
for _, row in p41_q2_sorted.iterrows():
    if row['행정동'] in tier1: continue
    ax.annotate(row['행정동'], (row['매출활성도점수'], row['이동성장세점수']),
                xytext=(4,3), textcoords='offset points', fontsize=7.5, color=C_T3)

ax.axhline(0, ls='--', color='black', lw=0.8, alpha=0.5)
ax.axvline(0, ls='--', color='black', lw=0.8, alpha=0.5)
ax.text(0.25, 0.97, '① 고활성도·고성장\n(이미 전환)', transform=ax.transAxes,
        fontsize=8, va='top', color='gray', ha='center')
ax.text(0.25, 0.55, '② 유망\n이동↑소비↓', transform=ax.transAxes,
        fontsize=9, va='top', color=C_T3, ha='center', fontweight='bold')
ax.set_xlabel('매출 활성도 점수', fontsize=10)
ax.set_ylabel('20대 이동 성장세 점수', fontsize=10)
ax.set_title('사분면 분석 (주말집중도 ≥ 0.80)', fontsize=11, fontweight='bold')
patches = [Patch(color=C_T1,label='Tier1 교집합(★)'),
           Patch(color=C_T3,label='Tier3 이동선행(2사분면)'),
           Patch(color='#CCC',label='기타 동')]
ax.legend(handles=patches, fontsize=8); ax.grid(alpha=0.2)
ax.spines[['top','right']].set_visible(False)

# Phase41 TOP15 bar
ax2 = axes[1]
p41_top = p41_valid[mask_q2].nlargest(15,'이동성장세점수')
tier_c_bar = [C_T1 if r['행정동'] in tier1 else C_T3 for _, r in p41_top.iterrows()]
ax2.barh(p41_top['행정동'][::-1], p41_top['이동성장세점수'][::-1],
         color=tier_c_bar[::-1], alpha=0.85)
for _, row in p41_top.iterrows():
    ax2.text(row['이동성장세점수']+0.01,
             list(p41_top['행정동'][::-1]).index(row['행정동']),
             f"{row['이동성장세점수']:.3f}", va='center', fontsize=8)
ax2.set_xlabel('이동 성장세 점수', fontsize=10)
ax2.set_title('Phase41 유망 후보 TOP15\n(이동 성장세 순)', fontsize=11, fontweight='bold')
patches2 = [Patch(color=C_T1,label='Tier1 교집합'), Patch(color=C_T3,label='Tier3 이동선행')]
ax2.legend(handles=patches2, fontsize=8, loc='lower right')
ax2.grid(axis='x', alpha=0.3); ax2.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('report/images/S4b_01_quadrant.png', dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════
# S5_04  레퍼런스 vs Tier 그룹 시계열 비교
# ═══════════════════════════════════════════════════════════════════
print("[8/8] S5_04 레퍼런스 vs 후보 비교...")
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
fig.suptitle('레퍼런스 vs 차세대 후보 — 시계열 지표 비교 (2020~2024)', fontsize=14, fontweight='bold')

COMPARE_GROUPS = {
    '레퍼런스(4)': (REF_DONGS, '#2E86AB', '-'),
    'Tier1 교집합(2)': (tier1, C_T1, '-'),
    'Tier2 업종전환(13)': (tier2[:], '#AE6C00', '--'),
    'Tier3 이동선행(13)': (tier3[:], C_T3, ':'),
    '서울 전체': (None, C_SEOUL, '--'),
}

metrics_ts = [
    ('업종점수', lambda s: s*100, '업종점수 (%)'),
    ('트렌드매출_비중', lambda s: s*100, '트렌드매출 비중 (%)'),
    ('2030비중', lambda s: s*100, '2030 소비 비중 (%)'),
    ('복합점수_분위', lambda s: s, '복합점수 분위 (0~1)'),
]

# 이동비율 연도별 그룹 평균 계산
def get_group_mig_ratio(dong_list, years=YEARS):
    vals = []
    for y in years:
        ratios = []
        for d in dong_list:
            _, _, rat = get_mig(d, y)
            if not np.isnan(rat): ratios.append(rat)
        vals.append(np.mean(ratios) if ratios else np.nan)
    return vals

for mi, (col, fn, ylabel) in enumerate(metrics_ts):
    ax = axes[mi//3][mi%3]
    for grp_name, (dong_list, color, ls) in COMPARE_GROUPS.items():
        if dong_list is None:
            vals = [fn(seoul_avg.loc[y, col]) for y in YEARS]
        else:
            vals = []
            for y in YEARS:
                sub = sales[(sales['행정동'].isin(dong_list))&(sales['연도']==y)]
                vals.append(fn(sub[col].mean()) if not sub.empty else np.nan)
        lw = 2.5 if '레퍼런스' in grp_name or 'Tier1' in grp_name else 1.8
        ax.plot(YEARS, vals, ls, color=color, lw=lw, ms=6, marker='o' if '레퍼런스' in grp_name else 's',
                label=grp_name)
    ax.set_title(ylabel, fontsize=11, fontweight='bold')
    ax.set_xticks(YEARS); ax.grid(axis='y', alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    if mi == 0: ax.legend(fontsize=7.5, loc='upper left')

# (5) 20대 이동비율 그룹 비교
ax5 = axes[1][1]
for grp_name, (dong_list, color, ls) in COMPARE_GROUPS.items():
    if dong_list is None:
        # 서울 전체 이동비율
        vals = []
        for y in YEARS:
            sub = mig_ann[mig_ann['연도']==y]
            vals.append((sub['20대이동'].sum()/sub['전체이동'].sum()*100) if not sub.empty else np.nan)
    else:
        vals = get_group_mig_ratio(dong_list)
    lw = 2.5 if '레퍼런스' in grp_name or 'Tier1' in grp_name else 1.8
    ax5.plot(YEARS, vals, ls, color=color, lw=lw, marker='o' if '레퍼런스' in grp_name else 's',
             ms=6, label=grp_name)
ax5.set_title('20대 이동비율 (%)', fontsize=11, fontweight='bold')
ax5.set_xticks(YEARS); ax5.legend(fontsize=7.5); ax5.grid(axis='y', alpha=0.3)
ax5.spines[['top','right']].set_visible(False)

# (6) 임대료 레퍼런스별 시계열
ax6 = axes[1][2]
for i, dong in enumerate(REF_DONGS):
    sang = rent_map.get(dong, None)
    if sang is None: continue
    rent_sub = rental[rental['상권명']==sang].sort_values('연도')
    rent_sub_yr = rent_sub[rent_sub['연도'].isin(YEARS)]
    if rent_sub_yr.empty: continue
    ax6.plot(rent_sub_yr['연도'], rent_sub_yr['임대료'], 'o-',
             color=C_REF[i], lw=2, ms=6, label=f'{dong}({sang})')
    for _, r in rent_sub_yr.iterrows():
        ax6.annotate(f"{r['임대료']:.0f}", (r['연도'], r['임대료']),
                     xytext=(0,6), textcoords='offset points', ha='center', fontsize=7.5, color=C_REF[i])
ax6.set_title('레퍼런스 상권 임대료\n(천원/㎡, 대응 상권 기준)', fontsize=11, fontweight='bold')
ax6.set_xticks(YEARS); ax6.legend(fontsize=7); ax6.grid(axis='y', alpha=0.3)
ax6.set_ylabel('임대료 (천원/㎡)'); ax6.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('report/images/S5_04_ref_vs_candidates.png', dpi=150, bbox_inches='tight')
plt.close()

# ═══════════════════════════════════════════════════════════════════
# S5_05  3차원 시계열 연관 (레퍼런스 + Tier1 통합 대시보드)
# ═══════════════════════════════════════════════════════════════════
TARGET_REGIONS = {
    '성수2가1동': (C_REF[3], '레퍼런스', '-'),
    '합정동':    (C_REF[2], '레퍼런스', '-'),
    '방배2동':   (C_REF[0], '레퍼런스', '-'),
    '청운효자동': (C_REF[1], '레퍼런스', '-'),
    '숭인2동':   (C_T1, 'Tier1', '--'),
    '신당5동':   ('#CA6702', 'Tier1', '--'),
}

fig, axes = plt.subplots(3, 2, figsize=(18, 14))
fig.suptitle('시계열 3차원 연관 분석 — 레퍼런스 vs Tier1\n이동인구 → 매출 전환 → 임대료 반응',
             fontsize=14, fontweight='bold', y=0.99)

# 행 0: 20대 이동비율
ax = axes[0][0]
for dong, (color, tier, ls) in TARGET_REGIONS.items():
    vals = [get_mig(dong, y)[2] for y in YEARS]
    lw = 2.5 if tier == '레퍼런스' else 2.0
    ax.plot(YEARS, vals, ls, color=color, lw=lw, ms=6 if tier=='레퍼런스' else 5,
            marker='o', label=f'{dong}')
ax.set_title('① 20대 이동비율 (%)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
ax.set_xticks(YEARS); ax.spines[['top','right']].set_visible(False)

# 20대 이동 절대량
ax = axes[0][1]
for dong, (color, tier, ls) in TARGET_REGIONS.items():
    vals = [get_mig(dong, y)[0] for y in YEARS]
    lw = 2.5 if tier == '레퍼런스' else 2.0
    ax.plot(YEARS, vals, ls, color=color, lw=lw, marker='o', ms=6 if tier=='레퍼런스' else 5, label=dong)
ax.set_title('② 20대 이동인구 (만 명)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
ax.set_xticks(YEARS); ax.spines[['top','right']].set_visible(False)

# 행 1: 업종점수 / 트렌드매출
ax = axes[1][0]
for dong, (color, tier, ls) in TARGET_REGIONS.items():
    s = sales[sales['행정동']==dong].set_index('연도')
    vals = [s.loc[y,'업종점수']*100 if y in s.index else np.nan for y in YEARS]
    lw = 2.5 if tier == '레퍼런스' else 2.0
    ax.plot(YEARS, vals, ls, color=color, lw=lw, ms=6 if tier=='레퍼런스' else 5,
            marker='s', label=dong)
ax.set_title('③ 업종점수 (%)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
ax.set_xticks(YEARS); ax.spines[['top','right']].set_visible(False)

ax = axes[1][1]
for dong, (color, tier, ls) in TARGET_REGIONS.items():
    s = sales[sales['행정동']==dong].set_index('연도')
    vals = [s.loc[y,'트렌드매출_비중']*100 if y in s.index else np.nan for y in YEARS]
    lw = 2.5 if tier == '레퍼런스' else 2.0
    ax.plot(YEARS, vals, ls, color=color, lw=lw, ms=6 if tier=='레퍼런스' else 5,
            marker='s', label=dong)
ax.set_title('④ 트렌드매출 비중 (%)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
ax.set_xticks(YEARS); ax.spines[['top','right']].set_visible(False)

# 행 2: 임대료 / 전환 단계 요약표
ax = axes[2][0]
for dong, (color, tier, ls) in TARGET_REGIONS.items():
    sang = rent_map.get(dong, None)
    if sang is None: continue
    rent_y = [get_rent(dong, y)[0] for y in YEARS]
    vac_y  = [get_rent(dong, y)[1] for y in YEARS]
    valid_y = [y for y, v in zip(YEARS, rent_y) if not np.isnan(v)]
    valid_r = [v for v in rent_y if not np.isnan(v)]
    if not valid_y: continue
    lw = 2.5 if tier == '레퍼런스' else 2.0
    ax.plot(valid_y, valid_r, ls, color=color, lw=lw, ms=8, marker='D', label=f'{dong}({sang})')
ax.set_title('⑤ 임대료 (천원/㎡, 대응 상권 기준)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_xticks(YEARS); ax.spines[['top','right']].set_visible(False)

# 전환 단계 요약표
ax6 = axes[2][1]
ax6.axis('off')
col_labels = ['지역', '구분', '20대이동\nΔ%', '이동비율\nΔ%p', '업종점수\nΔ%p', '임대료\nΔ%']
stage_data = []
for dong, (color, tier, _) in TARGET_REGIONS.items():
    s = sales[sales['행정동']==dong].set_index('연도')
    m20_0, _, rat0 = get_mig(dong, 2020)
    m20_4, _, rat4 = get_mig(dong, 2024)
    bs0 = s.loc[2020,'업종점수']*100 if 2020 in s.index else np.nan
    bs4 = s.loc[2024,'업종점수']*100 if 2024 in s.index else np.nan
    rent0 = get_rent(dong,2020)[0]; rent4 = get_rent(dong,2024)[0]
    dm = f'+{(m20_4/m20_0-1)*100:.0f}%' if pd.notna(m20_0) and m20_0>0 else '-'
    dr = f'{rat4-rat0:+.1f}' if pd.notna(rat0) and pd.notna(rat4) else '-'
    db = f'{bs4-bs0:+.1f}' if pd.notna(bs0) and pd.notna(bs4) else '-'
    drent = f'+{(rent4/rent0-1)*100:.0f}%' if pd.notna(rent0) and pd.notna(rent4) and rent0>0 else '-'
    stage_data.append([dong, tier, dm, dr, db, drent])

table = ax6.table(cellText=stage_data, colLabels=col_labels,
                  loc='center', cellLoc='center')
table.auto_set_font_size(False); table.set_fontsize(9)
table.scale(1, 1.8)
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#2C3E50'); table[0, j].set_text_props(color='white', fontweight='bold')
for i, (_, (color, _, __)) in enumerate(TARGET_REGIONS.items()):
    for j in range(len(col_labels)):
        table[i+1, j].set_facecolor(color + '22')

ax6.set_title('⑥ 지역별 2020→2024 변화 요약', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('report/images/S5_05_timeseries_3d_chain.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n완료! 생성된 이미지:")
for f in sorted(os.listdir('report/images')):
    if f.endswith('.png'):
        print(f"  report/images/{f}")
