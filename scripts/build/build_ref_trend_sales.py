# -*- coding: utf-8 -*-
"""
레퍼런스 상권 — 트렌드 업종 & 트렌드 시간대 전체 vs 20대 매출 비교

[트렌드 업종]
  카테고리: FB카페, FB식사, 주류유흥, 라이프스타일
  행별로 연령대_20_매출_금액 존재 → 전체 vs 20대 직접 비교 가능

[트렌드 시간대]
  = 금요일 × 야간비중 + 토요일 + 일요일  (trend_sales_master 기준)
  요일/시간대 컬럼에 연령대 분리 없음 → 전체 트렌드 매출 + 20대 전체비중 병기

출력:
  report/data/reference_trend_industry_sales.csv
  report/data/reference_trend_time_sales.csv
  report/images/S3_07_ref_trend_industry.png
  report/images/S3_08_ref_trend_timesales.png
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

from report_config import REFERENCE_DONGS as REF_DONGS

REF_LABELS = {d: '레퍼런스' for d in REF_DONGS}

TREND_INDUSTRY_KEYWORDS = {
    'FB카페':       ['커피','음료','제과'],
    'FB식사':       ['한식','일식','양식','중식','분식','패스트푸드','치킨'],
    '주류유흥':     ['호프','간이주점','노래'],
    '라이프스타일': ['일반의류','편의점','애완동물','화초','인테리어'],
}

def classify_trend(name):
    for cat, kws in TREND_INDUSTRY_KEYWORDS.items():
        if any(k in str(name) for k in kws):
            return cat
    return None  # 트렌드 업종 아님

# ────────────────────────────────────────────────────────────────
# 1. 원본 데이터 로드
# ────────────────────────────────────────────────────────────────
print('[1/4] 데이터 로드...')
raw = pd.read_csv('data/archive/서울시_상권분석서비스(추정매출-행정동)_2020_2024.csv', encoding='utf-8-sig')
raw.rename(columns={'행정동_코드_명':'행정동'}, inplace=True)
raw['연도'] = raw['기준_년분기_코드'] // 10
raw['분기'] = raw['기준_년분기_코드'] % 10
raw['트렌드업종'] = raw['서비스_업종_코드_명'].apply(classify_trend)

tsm = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig')

# ────────────────────────────────────────────────────────────────
# 2. 트렌드 업종 × 레퍼런스: 전체 vs 20대 매출
# ────────────────────────────────────────────────────────────────
print('[2/4] 트렌드 업종 매출 집계...')

ref_raw = raw[raw['행정동'].isin(REF_DONGS)].copy()

# 트렌드 업종만 필터
trend_raw = ref_raw[ref_raw['트렌드업종'].notna()].copy()

# 행정동 × 연도 × 트렌드업종 집계
agg_cat = trend_raw.groupby(['행정동','연도','트렌드업종'])[
    ['당월_매출_금액','연령대_20_매출_금액']].sum().reset_index()
agg_cat.columns = ['행정동','연도','트렌드업종','전체매출','20대매출']
agg_cat['20대비중'] = agg_cat['20대매출'] / agg_cat['전체매출'].replace(0, np.nan)

# 전체 트렌드 업종 합산 (업종 구분 없이)
agg_total = trend_raw.groupby(['행정동','연도'])[
    ['당월_매출_금액','연령대_20_매출_금액']].sum().reset_index()
agg_total.columns = ['행정동','연도','트렌드업종_전체매출','트렌드업종_20대매출']
agg_total['트렌드업종_20대비중'] = agg_total['트렌드업종_20대매출'] / agg_total['트렌드업종_전체매출'].replace(0, np.nan)

# 동 전체 매출 (분모)
agg_dong = ref_raw.groupby(['행정동','연도'])[['당월_매출_금액','연령대_20_매출_금액']].sum().reset_index()
agg_dong.columns = ['행정동','연도','동_전체매출','동_20대매출']
agg_total = agg_total.merge(agg_dong, on=['행정동','연도'])
agg_total['트렌드업종_매출비중'] = agg_total['트렌드업종_전체매출'] / agg_total['동_전체매출'].replace(0, np.nan)

agg_total.to_csv('report/data/reference_trend_industry_sales.csv', encoding='utf-8-sig', index=False)
print(f'  → reference_trend_industry_sales.csv ({len(agg_total)}행)')

# ────────────────────────────────────────────────────────────────
# 3. 트렌드 시간대 (trend_sales_master 기반)
# ────────────────────────────────────────────────────────────────
print('[3/4] 트렌드 시간대 매출 집계...')

ref_tsm = tsm[tsm['행정동'].isin(REF_DONGS)][
    ['행정동','연도','트렌드_대표_매출','총_매출','트렌드매출_비중','2030비중','업종점수']].copy()

# 야간비중 재산출 (원본 기준 확인용)
raw_yr = ref_raw.groupby(['행정동','연도'])[[
    '당월_매출_금액','연령대_20_매출_금액',
    '시간대_17~21_매출_금액','시간대_21~24_매출_금액',
    '금요일_매출_금액','토요일_매출_금액','일요일_매출_금액',
    '시간대_00~06_매출_금액','시간대_06~11_매출_금액',
    '시간대_11~14_매출_금액','시간대_14~17_매출_금액',
]].sum().reset_index()

raw_yr['야간매출합'] = raw_yr['시간대_17~21_매출_금액'] + raw_yr['시간대_21~24_매출_금액']
raw_yr['전체시간합'] = (raw_yr['시간대_00~06_매출_금액'] + raw_yr['시간대_06~11_매출_금액'] +
                        raw_yr['시간대_11~14_매출_금액'] + raw_yr['시간대_14~17_매출_금액'] +
                        raw_yr['시간대_17~21_매출_금액'] + raw_yr['시간대_21~24_매출_금액'])
raw_yr['야간비중'] = raw_yr['야간매출합'] / raw_yr['전체시간합'].replace(0, np.nan)
raw_yr['트렌드시간_매출'] = raw_yr['금요일_매출_금액'] * raw_yr['야간비중'] + \
                            raw_yr['토요일_매출_금액'] + raw_yr['일요일_매출_금액']
raw_yr['트렌드시간_비중'] = raw_yr['트렌드시간_매출'] / raw_yr['당월_매출_금액'].replace(0, np.nan)
raw_yr['20대_전체비중'] = raw_yr['연령대_20_매출_금액'] / raw_yr['당월_매출_금액'].replace(0, np.nan)

ref_time = raw_yr[['행정동','연도','당월_매출_금액','연령대_20_매출_금액',
                   '트렌드시간_매출','트렌드시간_비중','20대_전체비중',
                   '금요일_매출_금액','토요일_매출_금액','일요일_매출_금액','야간비중']].copy()
ref_time.columns = ['행정동','연도','총매출','20대매출_전체',
                    '트렌드시간_매출','트렌드시간_매출비중','20대_전체비중',
                    '금요일매출','토요일매출','일요일매출','야간비중']
ref_time.to_csv('report/data/reference_trend_time_sales.csv', encoding='utf-8-sig', index=False)
print(f'  → reference_trend_time_sales.csv ({len(ref_time)}행)')

# ────────────────────────────────────────────────────────────────
# 4. 시각화
# ────────────────────────────────────────────────────────────────
print('[4/4] 시각화...')
YEARS = [2020, 2021, 2022, 2023, 2024]
CAT_COLORS = {'FB카페':'#F4A261','FB식사':'#E76F51','주류유흥':'#9B5DE5','라이프스타일':'#00B4D8'}

# ── S3_07: 트렌드 업종 전체 vs 20대 매출 ─────────────────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle('레퍼런스 4개 동 — 트렌드 업종 전체 매출 vs 20대 매출\n(FB카페·식사·주류유흥·라이프스타일 합산)', fontsize=13, fontweight='bold')

for idx, dong in enumerate(REF_DONGS):
    ax = fig.add_subplot(2, 2, idx + 1)
    sub = agg_total[agg_total['행정동']==dong].sort_values('연도')
    ax2 = ax.twinx()

    x = np.arange(len(YEARS))
    w = 0.35
    sub_yr = sub[sub['연도'].isin(YEARS)].set_index('연도').reindex(YEARS)

    ax.bar(x - w/2, sub_yr['트렌드업종_전체매출'].fillna(0)/1e8, w,
           color='#B0BEC5', label='전체(억)', alpha=0.85)
    ax.bar(x + w/2, sub_yr['트렌드업종_20대매출'].fillna(0)/1e8, w,
           color='#1565C0', label='20대(억)', alpha=0.9)
    ax2.plot(x, sub_yr['트렌드업종_20대비중'].fillna(np.nan)*100,
             'o-', color='#00838F', markersize=4, linewidth=1.5, label='20대비중(%)')

    grade = REF_LABELS[dong]
    ax.set_title(f'{dong} [{grade}]', fontsize=8.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in YEARS], fontsize=6, rotation=30)
    ax.tick_params(axis='y', labelsize=6)
    ax2.tick_params(axis='y', labelsize=6, colors='#00838F')
    ax2.set_ylabel('20대비중(%)', fontsize=5.5, color='#00838F')
    ax.grid(axis='y', alpha=0.2)

legend_els = [Patch(color='#B0BEC5', label='트렌드업종 전체 매출(억원)'),
              Patch(color='#1565C0', label='트렌드업종 20대 매출(억원)'),
              plt.Line2D([0],[0], color='#00838F', marker='o', markersize=4, label='20대 매출비중(%)')]
fig.legend(handles=legend_els, loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(0.5,-0.01))
fig.text(0.01, 0.5, '매출(억원)', va='center', rotation='vertical', fontsize=9)

plt.tight_layout()
plt.savefig('report/images/S3_07_ref_trend_industry.png', dpi=150, bbox_inches='tight')
plt.close()
print('  → S3_07_ref_trend_industry.png')

# ── S3_08: 트렌드 시간대 매출 (전체 + 20대비중 병기) ─────────────
fig = plt.figure(figsize=(14, 10))
fig.suptitle('레퍼런스 4개 동 — 트렌드 시간대 매출 (금요야간+토+일)\n'
             '※ 시간대별 연령 분리 불가 → 트렌드 시간대 전체 매출 + 20대 전체 비중 병기', fontsize=12, fontweight='bold')

for idx, dong in enumerate(REF_DONGS):
    ax = fig.add_subplot(2, 2, idx + 1)
    sub = ref_time[ref_time['행정동']==dong].sort_values('연도')
    ax2 = ax.twinx()

    sub_yr = sub[sub['연도'].isin(YEARS)].set_index('연도').reindex(YEARS)
    x = np.arange(len(YEARS))
    w = 0.35

    ax.bar(x - w/2, sub_yr['총매출'].fillna(0)/1e8, w, color='#CFD8DC', label='총매출(억)', alpha=0.7)
    ax.bar(x + w/2, sub_yr['트렌드시간_매출'].fillna(0)/1e8, w, color='#00838F', label='트렌드시간(억)', alpha=0.85)
    ax2.plot(x, sub_yr['트렌드시간_매출비중'].fillna(np.nan)*100,
             'b-^', markersize=4, linewidth=1.5, label='트렌드시간비중(%)')
    ax2.plot(x, sub_yr['20대_전체비중'].fillna(np.nan)*100,
             'o--', color='#00838F', markersize=3, linewidth=1.2, label='20대전체비중(%)')

    grade = REF_LABELS[dong]
    ax.set_title(f'{dong} [{grade}]', fontsize=8.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in YEARS], fontsize=6, rotation=30)
    ax.tick_params(axis='y', labelsize=6)
    ax2.tick_params(axis='y', labelsize=6)
    ax.grid(axis='y', alpha=0.2)

legend_els = [
    Patch(color='#CFD8DC', label='총 매출(억원)'),
    Patch(color='#00838F', label='트렌드 시간대 매출(억원)'),
    plt.Line2D([0],[0], color='blue', marker='^', markersize=4, label='트렌드시간 매출비중(%)'),
    plt.Line2D([0],[0], color='#00838F', marker='o', linestyle='--', markersize=4, label='20대 전체 매출비중(%)'),
]
fig.legend(handles=legend_els, loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5,-0.01))
fig.text(0.01, 0.5, '매출(억원)', va='center', rotation='vertical', fontsize=9)

plt.tight_layout()
plt.savefig('report/images/S3_08_ref_trend_timesales.png', dpi=150, bbox_inches='tight')
plt.close()
print('  → S3_08_ref_trend_timesales.png')

# ── 수치 요약 출력 ────────────────────────────────────────────────
print('\n=== 레퍼런스 동별 2024년 핵심 수치 요약 ===')
print(f'{"동명":10s}  {"등급":4s}  {"트렌드업종_전체(억)":>16s}  {"트렌드업종_20대(억)":>16s}  {"20대비중":>8s}  {"트렌드시간비중":>12s}')
for dong in REF_DONGS:
    s = agg_total[(agg_total['행정동']==dong)&(agg_total['연도']==2024)]
    t = ref_time[(ref_time['행정동']==dong)&(ref_time['연도']==2024)]
    if len(s)==0 or len(t)==0: continue
    s, t = s.iloc[0], t.iloc[0]
    print(f'{dong:10s}  {REF_LABELS[dong]:4s}  '
          f'{s["트렌드업종_전체매출"]/1e8:>16.1f}  '
          f'{s["트렌드업종_20대매출"]/1e8:>16.1f}  '
          f'{s["트렌드업종_20대비중"]*100:>7.1f}%  '
          f'{t["트렌드시간_매출비중"]*100:>11.1f}%')
