# -*- coding: utf-8 -*-
"""
네이버/블로그 트렌드 vs 실제 매출 상관관계 분석
+ 추가 고부가가치 분석 제안 시각화
"""
import os
import sys
import unicodedata
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.signal import correlate
warnings.filterwarnings('ignore')

# ── 한글 폰트 설정 ─────────────────────────────────────────────
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def nfc(s):
    return unicodedata.normalize('NFC', s)

DATA_DIR = r'C:\Users\30032\OneDrive\바탕 화면\workspace\data_analytics_final\data'

# ══════════════════════════════════════════════════════════════
# 0. 데이터 로드
# ══════════════════════════════════════════════════════════════
print("데이터 로딩 중...")

# 행정동 매출 데이터
dong_file = next(f for f in os.listdir(DATA_DIR) if '2020_2024' in nfc(f) and f.endswith('.csv'))
df = pd.read_csv(os.path.join(DATA_DIR, dong_file), encoding='utf-8-sig')
df['year']    = df['기준_년분기_코드'] // 10
df['quarter'] = df['기준_년분기_코드'] % 10
print(f"  행정동 매출: {df.shape}")

# 네이버 트렌드 (보정본 우선)
naver_adj = pd.read_csv(os.path.join(DATA_DIR, 'naver_trend_행정동보정_20260510.csv'), encoding='utf-8-sig')
naver_raw = pd.read_csv(os.path.join(DATA_DIR, 'naver_trend_행정동_20260510.csv'), encoding='utf-8-sig')
print(f"  네이버 트렌드 보정: {naver_adj.shape}")
print(f"  네이버 트렌드 원본: {naver_raw.shape}")
print(f"  행정동 목록: {sorted(naver_adj['group'].unique())}")

# 블로그 트렌드
blog = pd.read_csv(os.path.join(DATA_DIR, '블로그트렌드_상권.csv'), encoding='utf-8-sig')
print(f"  블로그 트렌드: {blog.shape}")

# ══════════════════════════════════════════════════════════════
# 1. 행정동-분기별 매출 집계 (2030세대 포함)
# ══════════════════════════════════════════════════════════════
age_cols = [
    '연령대_10_매출_금액', '연령대_20_매출_금액', '연령대_30_매출_금액',
    '연령대_40_매출_금액', '연령대_50_매출_금액', '연령대_60_이상_매출_금액',
]

# 분기별 집계
dong_qtr = (
    df.groupby(['기준_년분기_코드', 'year', 'quarter', '행정동_코드_명'])
    .agg(
        매출=('당월_매출_금액', 'sum'),
        gen20=('연령대_20_매출_금액', 'sum'),
        gen30=('연령대_30_매출_금액', 'sum'),
        gen40=('연령대_40_매출_금액', 'sum'),
        gen50=('연령대_50_매출_금액', 'sum'),
        gen60=('연령대_60_이상_매출_금액', 'sum'),
        건수=('당월_매출_건수', 'sum'),
    )
    .reset_index()
)
dong_qtr['gen2030'] = dong_qtr['gen20'] + dong_qtr['gen30']
dong_qtr['gen2030_ratio'] = dong_qtr['gen2030'] / dong_qtr['매출'].replace(0, np.nan)

# 분기 코드 → 월(중간월) 매핑
qtr_to_month = {1: 2, 2: 5, 3: 8, 4: 11}
dong_qtr['month'] = dong_qtr['quarter'].map(qtr_to_month)
dong_qtr['date'] = pd.to_datetime(
    dong_qtr['year'].astype(str) + '-' +
    dong_qtr['month'].astype(str).str.zfill(2) + '-01'
)

print(f"  행정동-분기 집계: {dong_qtr.shape}")

# ══════════════════════════════════════════════════════════════
# 2. 네이버 트렌드 행정동명 → 매출 데이터 행정동 매핑
# ══════════════════════════════════════════════════════════════
# 네이버 트렌드 기간: 2022-01 ~ 2024-12 (월별)
# 매출 데이터: 2020Q1 ~ 2024Q4 (분기별)
# → 겹치는 기간 2022Q1 ~ 2024Q4 사용 (분기별로 리샘플)

naver = naver_adj.copy()
naver['period'] = pd.to_datetime(naver['period'])

# 분기별 평균으로 리샘플
naver['year'] = naver['period'].dt.year
naver['quarter'] = naver['period'].dt.quarter
naver_qtr = (
    naver.groupby(['group', 'year', 'quarter'])
    .agg(
        naver_ratio=('ratio', 'mean'),
        naver_adj=('adjusted_ratio', 'mean'),
    )
    .reset_index()
)
naver_qtr['기준_년분기_코드'] = naver_qtr['year'] * 10 + naver_qtr['quarter']
print(f"  네이버 분기 집계: {naver_qtr.shape}")
print(f"  네이버 행정동 목록: {sorted(naver_qtr['group'].unique())}")

# ── 네이버 행정동명 → 매출 행정동명 매핑 딕셔너리 ──
# 매출 데이터의 '행정동_코드_명' 고유값 확인
dong_names = sorted(df['행정동_코드_명'].unique())
print(f"\n  매출 데이터 행정동 수: {len(dong_names)}")

# 네이버 그룹명이 행정동명에 포함되는지 확인
naver_groups = sorted(naver_qtr['group'].unique())
print(f"  네이버 그룹: {naver_groups}")

# 매핑 시도: 포함 기준
mapping = {}
for g in naver_groups:
    matches = [d for d in dong_names if g in d or d in g]
    if matches:
        # 가장 짧은 매칭 우선 (정확 매칭 우선)
        matches_sorted = sorted(matches, key=len)
        mapping[g] = matches_sorted[0]

print(f"\n  매핑 성공: {len(mapping)} / {len(naver_groups)}")
for k, v in sorted(mapping.items()):
    print(f"    {k} -> {v}")

# ══════════════════════════════════════════════════════════════
# 3. 네이버 트렌드 vs 매출 상관관계 분석
# ══════════════════════════════════════════════════════════════
print("\n상관관계 분석 시작...")

results = []
lag_results = []

for naver_g, dong_name in mapping.items():
    # 매출 시계열
    sales_ts = (
        dong_qtr[dong_qtr['행정동_코드_명'] == dong_name]
        [['기준_년분기_코드', '매출', 'gen2030', 'gen2030_ratio']]
        .sort_values('기준_년분기_코드')
    )
    # 네이버 시계열
    naver_ts = (
        naver_qtr[naver_qtr['group'] == naver_g]
        [['기준_년분기_코드', 'naver_adj']]
        .sort_values('기준_년분기_코드')
    )
    # 겹치는 기간
    merged = pd.merge(sales_ts, naver_ts, on='기준_년분기_코드', how='inner')
    if len(merged) < 6:
        continue

    # 전분기 대비 성장률
    merged['sales_yoy'] = merged['매출'].pct_change() * 100
    merged['naver_chg'] = merged['naver_adj'].pct_change() * 100
    merged = merged.dropna()

    if len(merged) < 5:
        continue

    # 동시점 상관
    r_sync, p_sync = stats.pearsonr(merged['naver_adj'], merged['매출'])
    r_chg, p_chg   = stats.pearsonr(merged['naver_chg'].dropna(), merged['sales_yoy'].dropna()) if len(merged.dropna()) >= 3 else (np.nan, np.nan)

    # 선행-후행 Cross-Correlation (lag = -3 ~ +3 분기)
    x = (merged['naver_adj'] - merged['naver_adj'].mean()) / merged['naver_adj'].std()
    y = (merged['매출'] - merged['매출'].mean()) / merged['매출'].std()
    n = len(x)

    best_lag, best_r = 0, r_sync
    for lag in range(-3, 4):
        if lag == 0:
            continue
        if lag > 0:  # naver leads sales
            xi, yi = x.values[:n-lag], y.values[lag:]
        else:       # naver lags sales
            xi, yi = x.values[-lag:], y.values[:n+lag]
        if len(xi) < 4:
            continue
        r_lag, _ = stats.pearsonr(xi, yi)
        lag_results.append({
            'dong': naver_g, 'lag': lag, 'r': r_lag,
        })
        if abs(r_lag) > abs(best_r):
            best_lag, best_r = lag, r_lag

    results.append({
        'dong': naver_g,
        'dong_full': dong_name,
        'r_sync': round(r_sync, 3),
        'p_sync': round(p_sync, 3),
        'best_lag': best_lag,
        'best_r': round(best_r, 3),
        'n_obs': len(merged),
        'relation': '선행' if best_lag > 0 else ('후행' if best_lag < 0 else '동행'),
    })

results_df = pd.DataFrame(results).sort_values('r_sync', ascending=False)
lag_df = pd.DataFrame(lag_results)

print("\n=== 동시점 상관계수 (네이버 트렌드 vs 매출) ===")
print(results_df[['dong', 'r_sync', 'p_sync', 'best_lag', 'best_r', 'relation']].to_string(index=False))

# ══════════════════════════════════════════════════════════════
# 4. 블로그 트렌드 vs 매출 성장률 분석
# ══════════════════════════════════════════════════════════════
print("\n블로그 트렌드 분석...")

# 연도별 매출 (행정동명 기준) - 2022~2024
dong_annual = (
    df.groupby(['year', '행정동_코드_명'])['당월_매출_금액']
    .sum()
    .reset_index()
    .rename(columns={'당월_매출_금액': '매출'})
)
dong_annual_22 = dong_annual[dong_annual['year'] == 2022].set_index('행정동_코드_명')['매출']
dong_annual_24 = dong_annual[dong_annual['year'] == 2024].set_index('행정동_코드_명')['매출']

# 블로그 상권명 → 행정동명 매핑
blog_mapping = {}
for area in blog['area'].unique():
    area_clean = nfc(str(area))
    matches = [d for d in dong_names if area_clean in d or d in area_clean]
    if matches:
        blog_mapping[area_clean] = sorted(matches, key=len)[0]

print(f"  블로그-행정동 매핑: {len(blog_mapping)} / {len(blog['area'].unique())}")
for k, v in sorted(blog_mapping.items()):
    print(f"    {k} -> {v}")

# 블로그 데이터에 매출 성장률 결합
blog_merged = []
for _, row in blog.iterrows():
    area_nfc = nfc(str(row['area']))
    if area_nfc not in blog_mapping:
        continue
    dong = blog_mapping[area_nfc]
    s22 = dong_annual_22.get(dong, np.nan)
    s24 = dong_annual_24.get(dong, np.nan)
    if pd.notna(s22) and pd.notna(s24) and s22 > 0:
        sales_growth_22_24 = (s24 - s22) / s22 * 100
    else:
        sales_growth_22_24 = np.nan

    blog_merged.append({
        'area': area_nfc,
        'dong': dong,
        'blog_2022': row['2022'],
        'blog_2024': row['2024'],
        'blog_growth_22_24': (row['2024'] - row['2022']) / row['2022'] * 100,
        'blog_growth_22_25': row['growth_22_25'],
        'total_vol': row['total_vol'],
        'sales_growth_22_24': round(sales_growth_22_24, 2),
        'sales_2022': s22,
        'sales_2024': s24,
    })

blog_df = pd.DataFrame(blog_merged).dropna(subset=['sales_growth_22_24'])
print(f"\n  블로그-매출 결합: {len(blog_df)}개 상권")
print(blog_df[['area', 'blog_growth_22_24', 'sales_growth_22_24']].sort_values('blog_growth_22_24', ascending=False).to_string(index=False))

if len(blog_df) >= 3:
    r_blog, p_blog = stats.pearsonr(blog_df['blog_growth_22_24'], blog_df['sales_growth_22_24'])
    print(f"\n  블로그 성장률 vs 매출 성장률 Pearson r = {r_blog:.3f}, p = {p_blog:.3f}")

# ══════════════════════════════════════════════════════════════
# 5. 시각화 - Figure 1: 네이버 트렌드 vs 매출 상관관계
# ══════════════════════════════════════════════════════════════
print("\n시각화 생성 중...")

# 색상 설정
BLUE  = '#4e79a7'
ORANGE= '#f28e2b'
RED   = '#e15759'
GREEN = '#59a14f'
TEAL  = '#76b7b2'
GRAY  = '#9c9c9c'

fig1, axes = plt.subplots(2, 3, figsize=(22, 14))
fig1.suptitle('네이버 검색 트렌드 vs 실제 매출 상관관계 분석 (2022Q1~2024Q4)', fontsize=15, fontweight='bold')

# ── 5-1. 상관계수 랭킹 막대 ──
ax = axes[0, 0]
rd = results_df.sort_values('r_sync', ascending=True)
colors_bar = [GREEN if r >= 0.5 else (BLUE if r >= 0.3 else ORANGE) for r in rd['r_sync']]
bars = ax.barh(rd['dong'], rd['r_sync'], color=colors_bar, alpha=0.85, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.axvline(0.5, color='green', linewidth=1, linestyle='--', alpha=0.6, label='r=0.5')
ax.axvline(0.3, color='orange', linewidth=1, linestyle=':', alpha=0.6, label='r=0.3')
for bar, r in zip(bars, rd['r_sync']):
    ax.text(r + 0.01 if r >= 0 else r - 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{r:.2f}', va='center', ha='left' if r >= 0 else 'right', fontsize=8)
ax.set_xlabel('Pearson r (동시점)', fontsize=10)
ax.set_title('행정동별 네이버 트렌드-매출 상관계수', fontsize=11)
ax.legend(fontsize=8)
ax.set_xlim(-0.1, 1.05)
ax.grid(True, axis='x', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

# ── 5-2. 선행-후행 최적 lag 분포 ──
ax = axes[0, 1]
lead_lag_counts = results_df['relation'].value_counts()
rel_colors = {'선행': GREEN, '동행': BLUE, '후행': ORANGE}
bars2 = ax.bar(lead_lag_counts.index, lead_lag_counts.values,
               color=[rel_colors.get(r, GRAY) for r in lead_lag_counts.index],
               alpha=0.85, edgecolor='white', width=0.5)
for bar, v in zip(bars2, lead_lag_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            str(v), ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_ylabel('행정동 수', fontsize=10)
ax.set_title('네이버 트렌드의 매출 대비 시차 유형', fontsize=11)
ax.set_ylim(0, lead_lag_counts.max() * 1.3)
ax.grid(True, axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

# ── 5-3. 최적 lag 막대 (행정동별) ──
ax = axes[0, 2]
lag_rd = results_df.sort_values('best_lag', ascending=True)
lag_colors = [GREEN if l > 0 else (ORANGE if l < 0 else BLUE) for l in lag_rd['best_lag']]
bars3 = ax.barh(lag_rd['dong'], lag_rd['best_lag'],
                color=lag_colors, alpha=0.85, edgecolor='white')
ax.axvline(0, color='black', linewidth=1)
for bar, lag in zip(bars3, lag_rd['best_lag']):
    ax.text(lag + 0.05 if lag >= 0 else lag - 0.05,
            bar.get_y() + bar.get_height()/2,
            f'{lag:+d}분기', va='center',
            ha='left' if lag >= 0 else 'right', fontsize=8)
ax.set_xlabel('최적 선행 시차 (분기, 양수=트렌드 선행)', fontsize=10)
ax.set_title('행정동별 최적 선행-후행 시차 (quarters)', fontsize=11)
ax.grid(True, axis='x', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

# ── 5-4. 상위 4개 행정동 시계열 비교 ──
ax = axes[1, 0]
# 상관관계 상위 4개 행정동 선택
top4 = results_df.nlargest(4, 'r_sync')
line_colors = [BLUE, ORANGE, RED, GREEN]
ax2_twin = ax.twinx()

for i, (_, row) in enumerate(top4.iterrows()):
    naver_g = row['dong']
    dong_n  = row['dong_full']

    sales_ts = dong_qtr[dong_qtr['행정동_코드_명'] == dong_n].sort_values('기준_년분기_코드')
    naver_ts = naver_qtr[naver_qtr['group'] == naver_g].sort_values('기준_년분기_코드')
    merged2  = pd.merge(sales_ts, naver_ts, on='기준_년분기_코드', how='inner')

    if len(merged2) < 3:
        continue

    # 정규화
    s_norm = (merged2['매출'] - merged2['매출'].mean()) / merged2['매출'].std()
    n_norm = (merged2['naver_adj'] - merged2['naver_adj'].mean()) / merged2['naver_adj'].std()

    x_ticks = range(len(merged2))
    ax.plot(x_ticks, s_norm, '-', color=line_colors[i], linewidth=2,
            label=f'{naver_g} 매출 (r={row["r_sync"]:.2f})', alpha=0.9)
    ax.plot(x_ticks, n_norm, '--', color=line_colors[i], linewidth=1.5,
            label=f'{naver_g} 네이버', alpha=0.6)

    if i == 0 and len(merged2) > 0:
        labels = [str(c) for c in merged2['기준_년분기_코드']]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)

ax.set_ylabel('표준화 값 (Z-score)', fontsize=9)
ax.set_title('상관계수 상위 행정동: 매출(실선) vs 네이버 트렌드(점선)', fontsize=10)
ax.legend(fontsize=7, loc='upper left', ncol=2)
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

# ── 5-5. 블로그 성장률 vs 매출 성장률 산점도 ──
ax = axes[1, 1]
if len(blog_df) >= 3:
    sc = ax.scatter(blog_df['blog_growth_22_24'], blog_df['sales_growth_22_24'],
                    c=blog_df['total_vol'], cmap='YlOrRd', s=80, alpha=0.8,
                    edgecolors='white', linewidth=0.5)
    plt.colorbar(sc, ax=ax, label='블로그 총 언급량')

    # 회귀선
    xf = blog_df['blog_growth_22_24']
    yf = blog_df['sales_growth_22_24']
    m, b = np.polyfit(xf, yf, 1)
    xline = np.linspace(xf.min(), xf.max(), 50)
    ax.plot(xline, m * xline + b, 'r--', linewidth=1.5, alpha=0.7,
            label=f'회귀선 (r={r_blog:.2f}, p={p_blog:.3f})')

    for _, row in blog_df.iterrows():
        ax.annotate(row['area'],
                    (row['blog_growth_22_24'], row['sales_growth_22_24']),
                    fontsize=7, ha='center', va='bottom')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.set_xlabel('블로그 언급량 성장률 (2022→2024, %)', fontsize=10)
    ax.set_ylabel('실제 매출 성장률 (2022→2024, %)', fontsize=10)
    ax.set_title('블로그 트렌드 성장 vs 실제 매출 성장', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

# ── 5-6. 2030세대 소비비중 vs 네이버 트렌드 상관관계 ──
ax = axes[1, 2]
gen2030_corr = []
for naver_g, dong_name in mapping.items():
    sales_ts = dong_qtr[dong_qtr['행정동_코드_명'] == dong_name].sort_values('기준_년분기_코드')
    naver_ts = naver_qtr[naver_qtr['group'] == naver_g].sort_values('기준_년분기_코드')
    merged3  = pd.merge(sales_ts, naver_ts, on='기준_년분기_코드', how='inner')
    if len(merged3) < 6:
        continue
    r_2030, p_2030 = stats.pearsonr(merged3['naver_adj'], merged3['gen2030_ratio'])
    r_total, _     = stats.pearsonr(merged3['naver_adj'], merged3['매출'])
    gen2030_corr.append({
        'dong': naver_g,
        'r_2030ratio': round(r_2030, 3),
        'r_total': round(r_total, 3),
    })

gc_df = pd.DataFrame(gen2030_corr).sort_values('r_2030ratio', ascending=False)

x_gc = range(len(gc_df))
w = 0.35
ax.bar([xi - w/2 for xi in x_gc], gc_df['r_total'], w,
       color=BLUE, alpha=0.8, label='전체매출-네이버 r', edgecolor='white')
ax.bar([xi + w/2 for xi in x_gc], gc_df['r_2030ratio'], w,
       color=ORANGE, alpha=0.8, label='2030비중-네이버 r', edgecolor='white')
ax.set_xticks(x_gc)
ax.set_xticklabels(gc_df['dong'], rotation=45, ha='right', fontsize=8)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Pearson r', fontsize=10)
ax.set_title('네이버 트렌드와 전체매출/2030비중 상관계수 비교', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
out1 = os.path.join(DATA_DIR, 'trend_correlation_analysis.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
print(f"저장: {out1}")
plt.close()

# ══════════════════════════════════════════════════════════════
# 6. Figure 2: 행정동별 상세 시계열 비교 (그리드)
# ══════════════════════════════════════════════════════════════
# 상관관계 데이터 충분한 행정동 (n>=8) 선택 → 최대 12개
sel_dongs = results_df[results_df['n_obs'] >= 8].head(12)

n_dongs = len(sel_dongs)
ncols = 4
nrows = int(np.ceil(n_dongs / ncols))

fig2, axes2 = plt.subplots(nrows, ncols, figsize=(22, nrows * 4.5))
fig2.suptitle('행정동별 네이버 트렌드(보정) vs 실제 매출 추이 비교', fontsize=15, fontweight='bold')

axes2_flat = axes2.flatten() if nrows > 1 else axes2

for idx, (_, row) in enumerate(sel_dongs.iterrows()):
    ax = axes2_flat[idx]
    naver_g = row['dong']
    dong_n  = row['dong_full']
    r_val   = row['r_sync']
    lag_val = row['best_lag']

    sales_ts = dong_qtr[dong_qtr['행정동_코드_명'] == dong_n].sort_values('기준_년분기_코드')
    naver_ts = naver_qtr[naver_qtr['group'] == naver_g].sort_values('기준_년분기_코드')
    merged_d = pd.merge(sales_ts, naver_ts, on='기준_년분기_코드', how='inner')

    if len(merged_d) < 3:
        ax.set_visible(False)
        continue

    x_idx = range(len(merged_d))
    labels = [str(c) for c in merged_d['기준_년분기_코드']]

    ax_twin = ax.twinx()

    # 매출 (억원)
    ax.fill_between(x_idx, merged_d['매출'] / 1e8, alpha=0.25, color=BLUE)
    ax.plot(x_idx, merged_d['매출'] / 1e8, '-o', color=BLUE, linewidth=2,
            markersize=5, label='매출(억원)')
    ax.set_ylabel('매출 (억원)', color=BLUE, fontsize=8)
    ax.tick_params(axis='y', labelcolor=BLUE, labelsize=7)

    # 네이버 트렌드
    ax_twin.plot(x_idx, merged_d['naver_adj'], '--s', color=ORANGE, linewidth=2,
                 markersize=4, label='네이버 트렌드', alpha=0.85)
    ax_twin.set_ylabel('네이버 트렌드 지수', color=ORANGE, fontsize=8)
    ax_twin.tick_params(axis='y', labelcolor=ORANGE, labelsize=7)

    # 제목
    rel_str = f"선행 {lag_val}분기" if lag_val > 0 else (f"후행 {abs(lag_val)}분기" if lag_val < 0 else "동행")
    color_r = GREEN if r_val >= 0.5 else (ORANGE if r_val >= 0.3 else RED)
    ax.set_title(f'{naver_g}  r={r_val:.2f}  {rel_str}', fontsize=10, color=color_r, fontweight='bold')

    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6.5)
    ax.grid(True, alpha=0.3)
    ax.spines[['top']].set_visible(False)
    ax_twin.spines[['top']].set_visible(False)

# 남은 서브플롯 숨기기
for idx in range(n_dongs, len(axes2_flat)):
    axes2_flat[idx].set_visible(False)

plt.tight_layout()
out2 = os.path.join(DATA_DIR, 'trend_timeseries_grid.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
print(f"저장: {out2}")
plt.close()

# ══════════════════════════════════════════════════════════════
# 7. Figure 3: 추가 고부가가치 분석 제안 미리보기 (2030세대 집중도)
# ══════════════════════════════════════════════════════════════
# 2030 소비 집중도 상위 행정동 트렌드
gen2030_top = (
    dong_qtr[dong_qtr['year'].between(2022, 2024)]
    .groupby('행정동_코드_명')
    .agg(
        총매출=('매출', 'sum'),
        gen2030_합계=('gen2030', 'sum'),
    )
    .reset_index()
)
gen2030_top['gen2030_비율'] = gen2030_top['gen2030_합계'] / gen2030_top['총매출']
gen2030_top = gen2030_top[gen2030_top['총매출'] > 1e10]  # 최소 매출 필터

top20_2030 = gen2030_top.nlargest(20, 'gen2030_비율')
high_total = gen2030_top.nlargest(20, '총매출')

fig3, axes3 = plt.subplots(1, 2, figsize=(22, 9))
fig3.suptitle('2030세대 소비 집중도 분석 (2022~2024년)', fontsize=15, fontweight='bold')

ax = axes3[0]
sorted_top = top20_2030.sort_values('gen2030_비율', ascending=True)
col_map = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_top)))
bars_h = ax.barh(sorted_top['행정동_코드_명'], sorted_top['gen2030_비율'] * 100,
                 color=col_map, alpha=0.88, edgecolor='white')
for bar in bars_h:
    w = bar.get_width()
    ax.text(w + 0.2, bar.get_y() + bar.get_height()/2,
            f'{w:.1f}%', va='center', ha='left', fontsize=8.5)
ax.set_xlabel('2030세대 매출 비중 (%)', fontsize=11)
ax.set_title('2030세대 매출 비중 상위 20 행정동', fontsize=12)
ax.set_xlim(0, sorted_top['gen2030_비율'].max() * 100 * 1.18)
ax.grid(True, axis='x', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

ax = axes3[1]
# 2030 비율 변화 추이 (2020→2024): 상위 10개 행정동 선택
top10_2030_dongs = top20_2030.nlargest(10, 'gen2030_비율')['행정동_코드_명'].tolist()
gen2030_trend = (
    dong_qtr[dong_qtr['행정동_코드_명'].isin(top10_2030_dongs)]
    .groupby(['year', '행정동_코드_명'])
    .agg(
        gen2030=('gen2030', 'sum'),
        매출=('매출', 'sum'),
    )
    .reset_index()
)
gen2030_trend['비율'] = gen2030_trend['gen2030'] / gen2030_trend['매출'] * 100

pivot_2030 = gen2030_trend.pivot(index='year', columns='행정동_코드_명', values='비율')
cmap_l = plt.colormaps.get_cmap('tab10').resampled(len(pivot_2030.columns))
for i, dong in enumerate(pivot_2030.columns):
    vals = pivot_2030[dong].dropna()
    ax.plot(vals.index.astype(str), vals.values, '-o', linewidth=2,
            color=cmap_l(i), label=dong, markersize=6)
    if len(vals) > 0:
        ax.annotate(f"{vals.iloc[-1]:.1f}%",
                    xy=(vals.index[-1].astype(str) if hasattr(vals.index[-1], 'astype') else str(vals.index[-1]),
                        vals.iloc[-1]),
                    xytext=(6, 0), textcoords='offset points', fontsize=7.5,
                    color=cmap_l(i), va='center')
ax.set_xlabel('연도', fontsize=11)
ax.set_ylabel('2030세대 매출 비중 (%)', fontsize=11)
ax.set_title('2030세대 매출 비중 추이 (2020~2024)', fontsize=12)
ax.legend(title='행정동', fontsize=8, loc='upper left', ncol=2)
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
out3 = os.path.join(DATA_DIR, 'gen2030_concentration.png')
plt.savefig(out3, dpi=150, bbox_inches='tight')
print(f"저장: {out3}")
plt.close()

# ══════════════════════════════════════════════════════════════
# 8. 텍스트 인사이트 출력
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("분석 결과 요약")
print("="*70)

print("\n[1] 네이버 트렌드 vs 매출 상관관계 TOP 5")
for _, r in results_df.head(5).iterrows():
    lag_val2 = r['best_lag']
    if lag_val2 > 0:
        rel = f"네이버가 {lag_val2}분기 선행"
    elif lag_val2 == 0:
        rel = "동행"
    else:
        rel = f"네이버가 {abs(lag_val2)}분기 후행"
    print(f"  {r['dong']:10s}  r={r['r_sync']:+.3f}  {rel}")

print("\n[2] 네이버 트렌드 vs 매출 상관관계 BOTTOM 5 (약한 지역)")
for _, r in results_df.tail(5).iterrows():
    print(f"  {r['dong']:10s}  r={r['r_sync']:+.3f}")

if len(blog_df) >= 3:
    print(f"\n[3] 블로그 트렌드 성장률 vs 매출 성장률 상관계수: r={r_blog:.3f} (p={p_blog:.3f})")
    print("    블로그 성장률 상위 5개 상권:")
    for _, r in blog_df.nlargest(5, 'blog_growth_22_24').iterrows():
        print(f"    {r['area']:10s}  블로그성장={r['blog_growth_22_24']:.1f}%  매출성장={r['sales_growth_22_24']:.1f}%")

print("\n[4] 2030세대 매출 비중 최고 행정동 TOP 10")
for _, r in top20_2030.head(10).iterrows():
    print(f"  {r['행정동_코드_명']:15s}  2030비중={r['gen2030_비율']*100:.1f}%  총매출={r['총매출']/1e8:,.0f}억")

print("\n분석 완료!")
print(f"저장된 파일:")
for out in [out1, out2, out3]:
    print(f"  {out}")
