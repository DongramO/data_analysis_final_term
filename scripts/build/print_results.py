# -*- coding: utf-8 -*-
"""결과 출력 (UTF-8 강제)"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import unicodedata
import warnings
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings('ignore')

def nfc(s):
    return unicodedata.normalize('NFC', s)

DATA_DIR = r'C:\Users\30032\OneDrive\바탕 화면\workspace\data_analytics_final\data'

# 데이터 로드
dong_file = next(f for f in os.listdir(DATA_DIR) if '2020_2024' in nfc(f) and f.endswith('.csv'))
df = pd.read_csv(os.path.join(DATA_DIR, dong_file), encoding='utf-8-sig')
df['year']    = df['기준_년분기_코드'] // 10
df['quarter'] = df['기준_년분기_코드'] % 10

naver_adj = pd.read_csv(os.path.join(DATA_DIR, 'naver_trend_행정동보정_20260510.csv'), encoding='utf-8-sig')
blog = pd.read_csv(os.path.join(DATA_DIR, '블로그트렌드_상권.csv'), encoding='utf-8-sig')

# 분기 집계
dong_qtr = (
    df.groupby(['기준_년분기_코드', 'year', 'quarter', '행정동_코드_명'])
    .agg(
        매출=('당월_매출_금액', 'sum'),
        gen20=('연령대_20_매출_금액', 'sum'),
        gen30=('연령대_30_매출_금액', 'sum'),
    )
    .reset_index()
)
dong_qtr['gen2030'] = dong_qtr['gen20'] + dong_qtr['gen30']
dong_qtr['gen2030_ratio'] = dong_qtr['gen2030'] / dong_qtr['매출'].replace(0, np.nan)

naver = naver_adj.copy()
naver['period'] = pd.to_datetime(naver['period'])
naver['year'] = naver['period'].dt.year
naver['quarter'] = naver['period'].dt.quarter
naver_qtr = (
    naver.groupby(['group', 'year', 'quarter'])
    .agg(naver_adj=('adjusted_ratio', 'mean'))
    .reset_index()
)
naver_qtr['기준_년분기_코드'] = naver_qtr['year'] * 10 + naver_qtr['quarter']

dong_names = sorted(df['행정동_코드_명'].unique())
naver_groups = sorted(naver_qtr['group'].unique())

mapping = {}
for g in naver_groups:
    matches = [d for d in dong_names if g in d or d in g]
    if matches:
        mapping[g] = sorted(matches, key=len)[0]

# 상관관계 분석
results = []
for naver_g, dong_name in mapping.items():
    sales_ts = dong_qtr[dong_qtr['행정동_코드_명'] == dong_name].sort_values('기준_년분기_코드')
    naver_ts = naver_qtr[naver_qtr['group'] == naver_g].sort_values('기준_년분기_코드')
    merged = pd.merge(sales_ts, naver_ts, on='기준_년분기_코드', how='inner')
    if len(merged) < 6:
        continue
    r_sync, p_sync = stats.pearsonr(merged['naver_adj'], merged['매출'])
    r_2030, _ = stats.pearsonr(merged['naver_adj'], merged['gen2030_ratio'].fillna(merged['gen2030_ratio'].mean()))
    x = (merged['naver_adj'] - merged['naver_adj'].mean()) / merged['naver_adj'].std()
    y = (merged['매출'] - merged['매출'].mean()) / merged['매출'].std()
    n = len(x)
    best_lag, best_r = 0, r_sync
    for lag in range(-3, 4):
        if lag == 0:
            continue
        if lag > 0:
            xi, yi = x.values[:n-lag], y.values[lag:]
        else:
            xi, yi = x.values[-lag:], y.values[:n+lag]
        if len(xi) < 4:
            continue
        r_lag, _ = stats.pearsonr(xi, yi)
        if abs(r_lag) > abs(best_r):
            best_lag, best_r = lag, r_lag
    results.append({
        'dong': naver_g,
        'dong_full': dong_name,
        'r_sync': round(r_sync, 3),
        'p_sync': round(p_sync, 3),
        'best_lag': best_lag,
        'best_r': round(best_r, 3),
        'r_2030': round(r_2030, 3),
        'n_obs': len(merged),
    })

results_df = pd.DataFrame(results).sort_values('r_sync', ascending=False)

# 블로그 분석
dong_annual = df.groupby(['year', '행정동_코드_명'])['당월_매출_금액'].sum().reset_index()
dong_annual_22 = dong_annual[dong_annual['year'] == 2022].set_index('행정동_코드_명')['당월_매출_금액']
dong_annual_24 = dong_annual[dong_annual['year'] == 2024].set_index('행정동_코드_명')['당월_매출_금액']

blog_mapping = {}
for area in blog['area'].unique():
    area_nfc = nfc(str(area))
    matches = [d for d in dong_names if area_nfc in d or d in area_nfc]
    if matches:
        blog_mapping[area_nfc] = sorted(matches, key=len)[0]

blog_merged = []
for _, row in blog.iterrows():
    area_nfc = nfc(str(row['area']))
    if area_nfc not in blog_mapping:
        continue
    dong = blog_mapping[area_nfc]
    s22 = dong_annual_22.get(dong, np.nan)
    s24 = dong_annual_24.get(dong, np.nan)
    if pd.notna(s22) and pd.notna(s24) and s22 > 0:
        sales_growth = (s24 - s22) / s22 * 100
    else:
        sales_growth = np.nan
    blog_merged.append({
        'area': area_nfc, 'dong': dong,
        'blog_growth_22_24': (row['2024'] - row['2022']) / row['2022'] * 100,
        'blog_growth_22_25': row['growth_22_25'],
        'sales_growth_22_24': round(sales_growth, 2),
        'total_vol': row['total_vol'],
    })

blog_df = pd.DataFrame(blog_merged).dropna(subset=['sales_growth_22_24'])
if len(blog_df) >= 3:
    r_blog, p_blog = stats.pearsonr(blog_df['blog_growth_22_24'], blog_df['sales_growth_22_24'])

# 2030 집중도
gen2030_top = (
    dong_qtr[dong_qtr['year'].between(2022, 2024)]
    .groupby('행정동_코드_명')
    .agg(총매출=('매출', 'sum'), gen2030합계=('gen2030', 'sum'))
    .reset_index()
)
gen2030_top['gen2030_비율'] = gen2030_top['gen2030합계'] / gen2030_top['총매출']
gen2030_top = gen2030_top[gen2030_top['총매출'] > 1e10]

# ═══ 출력 ═══
print("=" * 70)
print("네이버 트렌드 vs 매출 상관관계 분석 결과")
print("=" * 70)

print(f"\n분석 행정동 수: {len(results_df)}개 (매핑 성공 {len(mapping)}개 중 관측치 6개 이상)")
print(f"분석 기간: 2022Q1 ~ 2024Q4 (12분기)")

print("\n[네이버-매출 행정동 매핑]")
for k, v in sorted(mapping.items()):
    print(f"  {k:10s} -> {v}")

print("\n[동시점 상관계수 전체 순위]")
print(f"{'행정동':12s} {'r_sync':>8s} {'p값':>8s} {'최적lag':>8s} {'최적r':>8s}")
print("-" * 50)
for _, r in results_df.iterrows():
    sig = "**" if r['p_sync'] < 0.05 else ("*" if r['p_sync'] < 0.1 else "")
    print(f"  {r['dong']:10s} {r['r_sync']:+8.3f} {r['p_sync']:8.3f}{sig:2s} {r['best_lag']:+8d} {r['best_r']:+8.3f}")

print("\n[선행 유형 분류]")
lead_count = sum(1 for _, r in results_df.iterrows() if r['best_lag'] > 0)
lag_count  = sum(1 for _, r in results_df.iterrows() if r['best_lag'] < 0)
sync_count = sum(1 for _, r in results_df.iterrows() if r['best_lag'] == 0)
print(f"  선행(네이버가 앞섬): {lead_count}개 행정동")
print(f"  동행(동시):          {sync_count}개 행정동")
print(f"  후행(매출이 앞섬):   {lag_count}개 행정동")

print(f"\n[블로그 트렌드 vs 매출 성장률 상관관계]")
print(f"  Pearson r = {r_blog:.3f}, p = {p_blog:.3f}")
if p_blog < 0.05:
    print("  → 통계적으로 유의미한 양의 상관관계")
elif p_blog < 0.1:
    print("  → 약한 통계적 유의성 (10% 수준)")
else:
    print("  → 통계적으로 유의미하지 않음 (p>0.1)")

print("\n[블로그 성장 vs 매출 성장 (2022→2024)]")
print(f"{'상권':12s} {'블로그성장(%)':>14s} {'매출성장(%)':>12s}")
print("-" * 42)
for _, r in blog_df.sort_values('blog_growth_22_24', ascending=False).iterrows():
    print(f"  {r['area']:10s} {r['blog_growth_22_24']:+13.1f}% {r['sales_growth_22_24']:+11.1f}%")

print(f"\n[2030세대 매출 비중 상위 행정동 (2022~2024)]")
print(f"{'행정동':20s} {'2030비중':>10s} {'총매출':>12s}")
print("-" * 46)
for _, r in gen2030_top.nlargest(15, 'gen2030_비율').iterrows():
    print(f"  {r['행정동_코드_명']:18s} {r['gen2030_비율']*100:9.1f}% {r['총매출']/1e8:10,.0f}억")

print("\n완료")
