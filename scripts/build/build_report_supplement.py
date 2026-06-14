# -*- coding: utf-8 -*-
"""
보고서 보완 분석 자료 생성
  S2_05 — 행정동 용도별 업종 구성 비교 (오피스/혼합/주거/상권형)
  S3_05 — 레퍼런스 동 매출 시계열 (전체 vs 20대, QoQ)
  S3_06 — 레퍼런스 동 이동 시계열 (전체 + 20대)
  S5_04 — 레퍼런스 vs 차세대 후보 비교 (20대 매출/이동)

  report/data/reference_sales_qoq.csv
  report/data/reference_migration_timeseries.csv
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

sys.stdout.reconfigure(encoding='utf-8')
os.makedirs('report/data', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

from report_config import (
    REFERENCE_DONGS as REF_DONGS,
    load_v3,
    tier1_dongs,
    tier2_dongs,
    tier3_dongs,
)

_v3 = load_v3()
CAND_TIER1 = tier1_dongs(_v3)
CAND_TIER3 = tier3_dongs(_v3)[:8]
CAND_DONGS = CAND_TIER1 + CAND_TIER3

# ────────────────────────────────────────────────────────────────
# 데이터 로드
# ────────────────────────────────────────────────────────────────
print('[1/5] 데이터 로드...')
raw = pd.read_csv('data/archive/서울시_상권분석서비스(추정매출-행정동)_2020_2024.csv', encoding='utf-8-sig')
raw['연도'] = raw['기준_년분기_코드'] // 10
raw['분기'] = raw['기준_년분기_코드'] % 10
raw['YQ']  = (raw['연도']-2020)*4 + raw['분기']
raw.rename(columns={'행정동_코드_명':'행정동'}, inplace=True)

tsm     = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig')
at      = pd.read_csv('data/main_data/area_type_profile.csv', encoding='utf-8-sig')
mig_raw = pd.read_csv('data/main_data/분기별_주말유입인구.csv', encoding='utf-8-sig')
kt      = pd.read_excel('data/archive/KT_서울생활이동데이터_행정동코드_20210907.xlsx')
kt.columns = ['시도','시군구','행정동코드','행정동명','전체명']
kt_map  = dict(zip(kt['행정동코드'], kt['행정동명']))

# 매출 분기별 집계
TREND_CATS = {
    'FB카페': ['커피','음료','제과'],
    'FB식사': ['한식','일식','양식','중식','분식','패스트푸드','치킨'],
    '주류유흥': ['호프','노래방'],
    '라이프스타일': ['일반의류','편의점','애완동물','화초','인테리어'],
}

def classify_cat(name):
    if any(k in str(name) for k in ['커피','음료','제과','음료']):       return 'FB카페'
    if any(k in str(name) for k in ['한식','일식','양식','중식','분식','패스트','치킨']): return 'FB식사'
    if any(k in str(name) for k in ['호프','간이','노래']):              return '주류유흥'
    if any(k in str(name) for k in ['의류','편의점','애완','화초','인테리어']): return '라이프스타일'
    return '기타'

raw['업종대분류'] = raw['서비스_업종_코드_명'].apply(classify_cat)

# 행정동 × 분기 × 업종 집계
sq_cat = raw.groupby(['행정동','YQ','연도','분기','업종대분류'])['당월_매출_금액'].sum().reset_index()
sq_tot = raw.groupby(['행정동','YQ','연도','분기'])[['당월_매출_금액','연령대_20_매출_금액']].sum().reset_index()
sq_tot.columns = ['행정동','YQ','연도','분기','총매출','20대매출']
sq_tot['20대비중'] = sq_tot['20대매출'] / sq_tot['총매출'].replace(0, np.nan)
sq_tot['QoQ_총매출'] = sq_tot.groupby('행정동')['총매출'].pct_change()
sq_tot['QoQ_20대매출'] = sq_tot.groupby('행정동')['20대매출'].pct_change()

# 이동 데이터
mig20  = mig_raw[mig_raw['연령대']==20].groupby(['도착_행정동코드','연도','분기'])['합계_이동인구'].sum().reset_index()
mall   = mig_raw.groupby(['도착_행정동코드','연도','분기'])['합계_이동인구'].sum().reset_index()
mq     = mig20.merge(mall, on=['도착_행정동코드','연도','분기'], suffixes=('_20','_전체'))
mq['이동비율_20대'] = mq['합계_이동인구_20'] / mq['합계_이동인구_전체'].replace(0, np.nan)
mq['행정동'] = mq['도착_행정동코드'].map(kt_map)
mq['YQ'] = (mq['연도']-2020)*4 + mq['분기']
mq = mq.dropna(subset=['행정동'])

# ────────────────────────────────────────────────────────────────
# [S2_05] 행정동 용도별 업종 구성 비교
# ────────────────────────────────────────────────────────────────
print('[2/5] S2_05 용도별 업종 구성 시각화...')
at_map = dict(zip(at['행정동'], at['구역유형']))
sq_2024 = sq_cat[(sq_cat['연도']==2024)]
sq_2024_tot = raw[(raw['연도']==2024)].groupby('행정동')['당월_매출_금액'].sum().reset_index()
sq_2024_tot.columns = ['행정동','총매출_2024']
sq_2024 = sq_2024.merge(sq_2024_tot, on='행정동', how='left')
sq_2024['비중'] = sq_2024['당월_매출_금액'] / sq_2024['총매출_2024'].replace(0, np.nan)
sq_2024['구역유형'] = sq_2024['행정동'].map(at_map)
sq_2024 = sq_2024.dropna(subset=['구역유형'])

type_order = ['오피스형','혼합형','주거형','상권형']
cat_order  = ['FB카페','FB식사','주류유흥','라이프스타일','기타']
cat_colors = {'FB카페':'#42A5F5','FB식사':'#1565C0','주류유흥':'#9B5DE5','라이프스타일':'#00B4D8','기타':'#BDBDBD'}

agg = sq_2024.groupby(['구역유형','업종대분류'])['비중'].mean().reset_index()
agg = agg[agg['구역유형'].isin(type_order)]

fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=False)
fig.suptitle('행정동 용도별 업종 매출 구성 비교 (2024년 기준)', fontsize=13, fontweight='bold')

for ax, atype in zip(axes, type_order):
    sub = agg[agg['구역유형']==atype]
    sub = sub.set_index('업종대분류').reindex(cat_order).fillna(0)
    n   = (at['구역유형']==atype).sum()
    ax.bar(sub.index, sub['비중']*100,
           color=[cat_colors[c] for c in sub.index], edgecolor='white', linewidth=0.5)
    for i, (cat, val) in enumerate(zip(sub.index, sub['비중']*100)):
        ax.text(i, val+0.3, f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    ax.set_title(f'{atype}\n(n={n}개 동)', fontsize=10, fontweight='bold')
    ax.set_ylabel('매출 비중 (%)' if ax == axes[0] else '')
    ax.set_ylim(0, max(sub['비중']*100)*1.35 + 2)
    ax.tick_params(axis='x', rotation=25)
    ax.grid(axis='y', alpha=0.3)

legend_els = [Patch(facecolor=cat_colors[c], label=c) for c in cat_order]
fig.legend(handles=legend_els, loc='lower center', ncol=5, fontsize=9, bbox_to_anchor=(0.5,-0.04))
plt.tight_layout()
plt.savefig('report/images/S2_05_area_type_industry.png', dpi=150, bbox_inches='tight')
plt.close()
print('  → S2_05_area_type_industry.png')

# ────────────────────────────────────────────────────────────────
# [S3_05] 레퍼런스 동 매출 시계열 (전체/20대/QoQ)
# ────────────────────────────────────────────────────────────────
print('[3/5] S3_05 레퍼런스 매출 시계열...')

# CSV 저장
ref_sales = sq_tot[sq_tot['행정동'].isin(REF_DONGS)].copy()
ref_sales.to_csv('report/data/reference_sales_qoq.csv', encoding='utf-8-sig', index=False)

fig = plt.figure(figsize=(14, 10))
fig.suptitle('레퍼런스 4개 동 — 전체 매출 vs 20대 매출 시계열 (2020~2024 분기)', fontsize=13, fontweight='bold')

for idx, dong in enumerate(REF_DONGS):
    ax = fig.add_subplot(2, 2, idx + 1)
    sub = ref_sales[ref_sales['행정동']==dong].sort_values('YQ')
    if len(sub) == 0: continue
    ax2 = ax.twinx()
    ax.bar(sub['YQ'], sub['총매출']/1e8, color='#CFD8DC', alpha=0.7, label='총매출(억)')
    ax.bar(sub['YQ'], sub['20대매출']/1e8, color='#1565C0', alpha=0.85, label='20대매출(억)')
    ax2.plot(sub['YQ'], sub['20대비중']*100, 'o-', color='#00838F', markersize=3, linewidth=1.5, label='20대비중(%)')
    ax.set_title(f'{dong}', fontsize=8.5, fontweight='bold')
    ax.set_xticks([1,5,9,13,17,20])
    ax.set_xticklabels(['2020Q1','2021Q1','2022Q1','2023Q1','2024Q1','2024Q4'], fontsize=5.5, rotation=30)
    ax.tick_params(axis='y', labelsize=6)
    ax2.tick_params(axis='y', labelsize=6, colors='#00838F')
    ax2.yaxis.set_label_position('right')
    ax2.set_ylabel('20대비중(%)', fontsize=6, color='#00838F')

fig.text(0.01, 0.5, '매출(억원)', va='center', rotation='vertical', fontsize=9)
handles = [Patch(color='#CFD8DC',label='총매출'), Patch(color='#1565C0',label='20대매출'),
           plt.Line2D([0],[0],color='#00838F',marker='o',markersize=4,label='20대비중(%)')]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8, bbox_to_anchor=(0.5,-0.01))
plt.tight_layout()
plt.savefig('report/images/S3_05_ref_sales_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print('  → S3_05_ref_sales_timeseries.png')

# ────────────────────────────────────────────────────────────────
# [S3_06] 레퍼런스 동 이동 시계열
# ────────────────────────────────────────────────────────────────
print('[4/5] S3_06 레퍼런스 이동 시계열...')

ref_mig = mq[mq['행정동'].isin(REF_DONGS)][
    ['행정동','YQ','연도','분기','합계_이동인구_전체','합계_이동인구_20','이동비율_20대']].copy()
ref_mig.to_csv('report/data/reference_migration_timeseries.csv', encoding='utf-8-sig', index=False)

fig = plt.figure(figsize=(14, 10))
fig.suptitle('레퍼런스 4개 동 — 전체 이동인구 vs 20대 이동인구 시계열 (2020~2024 분기)', fontsize=13, fontweight='bold')

for idx, dong in enumerate(REF_DONGS):
    ax = fig.add_subplot(2, 2, idx + 1)
    sub = ref_mig[ref_mig['행정동']==dong].sort_values('YQ')
    if len(sub) == 0: continue
    ax2 = ax.twinx()
    ax.fill_between(sub['YQ'], sub['합계_이동인구_전체']/1e4, color='#CFD8DC', alpha=0.7, label='전체(만명)')
    ax.fill_between(sub['YQ'], sub['합계_이동인구_20']/1e4,   color='#1565C0', alpha=0.85, label='20대(만명)')
    ax2.plot(sub['YQ'], sub['이동비율_20대']*100, 'o-', color='#00838F', markersize=3, linewidth=1.5, label='20대비율(%)')
    ax.set_title(f'{dong}', fontsize=8.5, fontweight='bold')
    ax.set_xticks([1,5,9,13,17,20])
    ax.set_xticklabels(['2020Q1','2021Q1','2022Q1','2023Q1','2024Q1','2024Q4'], fontsize=5.5, rotation=30)
    ax.tick_params(axis='y', labelsize=6)
    ax2.tick_params(axis='y', labelsize=6, colors='#00838F')
    ax2.set_ylabel('20대비율(%)', fontsize=6, color='#00838F')

fig.text(0.01, 0.5, '이동인구(만명)', va='center', rotation='vertical', fontsize=9)
handles = [Patch(color='#CFD8DC',label='전체 이동인구'), Patch(color='#1565C0',label='20대 이동인구'),
           plt.Line2D([0],[0],color='#00838F',marker='o',markersize=4,label='20대 이동비율(%)')]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=8, bbox_to_anchor=(0.5,-0.01))
plt.tight_layout()
plt.savefig('report/images/S3_06_ref_migration_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print('  → S3_06_ref_migration_timeseries.png')

# ────────────────────────────────────────────────────────────────
# [S5_04] 레퍼런스 vs 차세대 후보 비교
# ────────────────────────────────────────────────────────────────
print('[5/5] S5_04 레퍼런스 vs 후보 비교...')

# 연도별 집계
sq_yr = sq_tot.groupby(['행정동','연도'])[['총매출','20대매출','20대비중']].mean().reset_index()
mig_yr = mq.groupby(['행정동','연도'])[['합계_이동인구_전체','합계_이동인구_20','이동비율_20대']].mean().reset_index()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('레퍼런스(4) vs 차세대 후보 Tier: 20대 매출·이동 지표 (2020~2024)', fontsize=13, fontweight='bold')

YEARS = list(range(2020, 2025))

for dong_list, label, color in [
    (REF_DONGS, '레퍼런스(4)', '#0D47A1'),
    (CAND_TIER1, 'Tier1 교집합', '#2E7D32'),
    (tier2_dongs(_v3)[:8], 'Tier2 업종전환', '#1565C0'),
    (CAND_TIER3[:6], 'Tier3 이동선행', '#00838F'),
]:
    sub_s = sq_yr[sq_yr['행정동'].isin(dong_list)].groupby('연도')[['20대비중','총매출','20대매출']].mean()
    sub_m = mig_yr[mig_yr['행정동'].isin(dong_list)].groupby('연도')['이동비율_20대'].mean()
    sub_s = sub_s.reindex(YEARS)
    sub_m = sub_m.reindex(YEARS)
    ls = '-' if '확정' in label else ('--' if '후보' in label else ':')
    marker = 'o' if '확정' in label else ('s' if '후보' in label else '^')
    axes[0,0].plot(YEARS, sub_s['20대비중']*100, color=color, ls=ls, marker=marker, lw=2, label=label)
    axes[0,1].plot(YEARS, sub_s['총매출']/1e9,   color=color, ls=ls, marker=marker, lw=2, label=label)
    axes[1,0].plot(YEARS, sub_m*100,              color=color, ls=ls, marker=marker, lw=2, label=label)
    axes[1,1].plot(YEARS, sub_s['20대매출']/1e9,  color=color, ls=ls, marker=marker, lw=2, label=label)

titles = ['① 20대 매출 비중 (%)', '② 총 매출 (십억원)', '③ 20대 이동비율 (%)', '④ 20대 매출 절대액 (십억원)']
ylabels = ['20대비중 (%)', '총매출 (십억원)', '20대이동비율 (%)', '20대매출 (십억원)']
for ax, title, ylabel in zip(axes.flat, titles, ylabels):
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_xticks(YEARS)
    ax.set_xticklabels([str(y) for y in YEARS])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('report/images/S5_04_ref_vs_candidates.png', dpi=150, bbox_inches='tight')
plt.close()
print('  → S5_04_ref_vs_candidates.png')

print('\n=== 완료 ===')
import os
for f in sorted(os.listdir('report/images')):
    print(f'  {f}')
