# -*- coding: utf-8 -*-
"""
네이버 트렌드 vs 매출 기반 트렌드 분석 비교
============================================
[비교 구조]
  A. 네이버 트렌드 점수로 25개 지역 순위
  B. 매출 기반 복합점수(v3)로 동일 지역 순위
  C. 두 순위의 스피어만 상관 (같은 지역을 비슷하게 보는가?)
  D. 시계열 비교: 연도별 네이버 ratio 추이 vs 2030비중 추이
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

data_dir = 'data'
image_dir = 'image'

# ── 네이버 그룹 → 행정동 매핑 ────────────────────────────────
NAVER_TO_DONG = {
    '을지로':  ['을지로동'],
    '대학로':  ['혜화동'],
    '후암동':  ['후암동'],
    '독립문':  ['교남동'],
    '청구':    ['청구동'],
    '왕십리':  ['왕십리2동', '왕십리도선동'],
    '회기':    ['회기동'],
    '중계':    ['중계본동', '중계1동', '중계2·3동', '중계4동'],
    '도봉':    ['도봉1동', '도봉2동'],
    '청량리':  ['청량리동', '전농1동', '전농2동'],
    '금호':    ['금호1가동', '금호2·3가동', '금호4가동'],
    '노량진':  ['노량진1동', '노량진2동'],
    '신림':    ['신림동'],
    '보라매':  ['보라매동', '신대방1동', '신대방2동'],
    '난곡':    ['난곡동'],
    '개포':    ['개포1동', '개포2동', '개포4동'],
    '연희동':  ['연희동'],
    '가락':    ['가락본동', '가락1동', '가락2동'],
    '명일':    ['명일1동', '명일2동'],
    '둔촌':    ['둔촌2동'],
    '당산':    ['당산1동', '당산2동'],
    '신길':    ['신길1동', '신길3동', '신길4동', '신길5동', '신길6동', '신길7동'],
    '방화':    ['방화1동', '방화2동', '방화3동'],
    '신월':    ['신월1동', '신월2동', '신월3동', '신월4동', '신월5동', '신월6동', '신월7동'],
    '화곡':    ['화곡1동', '화곡2동', '화곡3동', '화곡4동', '화곡본동', '화곡6동', '화곡8동'],
}
# 역방향: 행정동 → 네이버그룹
DONG_TO_NAVER = {d: g for g, dongs in NAVER_TO_DONG.items() for d in dongs}


# ════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ════════════════════════════════════════════════════════════
df_sales = pd.read_csv(
    os.path.join(data_dir, '서울시_상권분석서비스(추정매출-행정동)_2020_2024.csv'),
    encoding='utf-8-sig',
)
df_sales['year'] = df_sales['기준_년분기_코드'] // 10

# 네이버 트렌드 (2020~2024 소비키워드 파일 우선, ratio 사용)
nv = pd.read_csv(
    os.path.join(data_dir, '네이버트렌드_소비키워드_20260517.csv'),
    encoding='utf-8-sig',
)
nv['period'] = pd.to_datetime(nv['period'])
nv['year'] = nv['period'].dt.year


# ════════════════════════════════════════════════════════════
# 2. 매출 기반 복합점수 재계산 (v3 방식, 25개 그룹 집계)
# ════════════════════════════════════════════════════════════
CATEG_MAP = {
    'F&B (카페/디저트)': ['커피-음료', '제과점'],
    'F&B (식사)':       ['한식음식점', '일식음식점', '양식음식점', '중식음식점',
                         '분식전문점', '패스트푸드점', '치킨전문점'],
    '주류/유흥':        ['호프-간이주점', '노래방'],
    '라이프스타일':     ['일반의류', '편의점', '애완동물', '화초', '인테리어'],
}
TREND_CATEGS = list(CATEG_MAP.keys())

df_sales['category'] = df_sales['서비스_업종_코드_명'].map(
    lambda x: next((c for c, items in CATEG_MAP.items() if x in items), '기타')
)
df_sales['naver_group'] = df_sales['행정동_코드_명'].map(DONG_TO_NAVER)

DAY_COLS  = ['월요일_매출_금액','화요일_매출_금액','수요일_매출_금액',
             '목요일_매출_금액','금요일_매출_금액','토요일_매출_금액','일요일_매출_금액']
TIME_COLS = ['시간대_00~06_매출_금액','시간대_06~11_매출_금액','시간대_11~14_매출_금액',
             '시간대_14~17_매출_금액','시간대_17~21_매출_금액','시간대_21~24_매출_금액']

# 네이버 그룹 단위로 집계 (매핑된 25개 그룹만)
df_mapped = df_sales[df_sales['naver_group'].notna()].copy()

# 업종 점수
categ_sum = (
    df_mapped.groupby(['naver_group', 'category'])['당월_매출_금액']
    .sum().unstack(fill_value=0)
)
total_g = categ_sum.sum(axis=1)
categ_share_g = categ_sum.div(total_g, axis=0)
industry_score_g = categ_share_g[TREND_CATEGS].sum(axis=1)

# 시간 점수
base_g = df_mapped.groupby('naver_group')[DAY_COLS + TIME_COLS].sum()
day_total_g  = base_g[DAY_COLS].sum(axis=1)
time_total_g = base_g[TIME_COLS].sum(axis=1)

금요효과_g = base_g['금요일_매출_금액'] / base_g['목요일_매출_금액']
주말비중_g  = (base_g['토요일_매출_금액'] + base_g['일요일_매출_금액']) / day_total_g
야간비중_g  = (base_g['시간대_17~21_매출_금액'] + base_g['시간대_21~24_매출_금액']) / time_total_g

def zscore(s):
    return (s - s.mean()) / s.std()

temporal_score_g  = zscore(금요효과_g) + zscore(주말비중_g) + zscore(야간비중_g)
composite_g       = zscore(industry_score_g) + zscore(temporal_score_g)

# 2030비중
gen2030_g = (
    df_mapped.groupby('naver_group')
    .agg(g20=('연령대_20_매출_금액','sum'),
         g30=('연령대_30_매출_금액','sum'),
         tot=('당월_매출_금액','sum'))
)
gen2030_g['비중'] = (gen2030_g['g20'] + gen2030_g['g30']) / gen2030_g['tot']


# ════════════════════════════════════════════════════════════
# 3. 네이버 점수: 그룹별 ratio 평균 (전체 기간)
# ════════════════════════════════════════════════════════════
naver_score = nv.groupby('group')['ratio'].mean()
naver_score.name = '네이버점수'

# 통합 비교 테이블
compare = pd.DataFrame({
    '네이버점수':  naver_score,
    '매출복합점수': composite_g,
    '업종점수':   industry_score_g,
    '시간점수':   temporal_score_g,
    '금요효과':   금요효과_g,
    '주말비중':   주말비중_g,
    '야간비중':   야간비중_g,
    '2030비중':  gen2030_g['비중'],
}).dropna()

# 스피어만 순위 상관
rho, pval = sp_stats.spearmanr(compare['네이버점수'], compare['매출복합점수'])
rho2, pval2 = sp_stats.spearmanr(compare['네이버점수'], compare['2030비중'])

print("=== 25개 지역 비교 테이블 ===")
print(compare[['네이버점수','매출복합점수','2030비중','금요효과','주말비중']].sort_values('네이버점수', ascending=False).round(3).to_string())
print(f"\n스피어만 상관 (네이버점수 vs 매출복합점수): ρ={rho:.3f}, p={pval:.3f}")
print(f"스피어만 상관 (네이버점수 vs 2030비중):   ρ={rho2:.3f}, p={pval2:.3f}")

# 네이버 순위 vs 매출 순위
compare['네이버순위']  = compare['네이버점수'].rank(ascending=False).astype(int)
compare['매출순위']    = compare['매출복합점수'].rank(ascending=False).astype(int)
compare['순위차이']    = (compare['네이버순위'] - compare['매출순위']).abs()
print("\n순위 차이 큰 지역 (네이버와 매출 평가가 크게 다른 곳):")
print(compare[['네이버순위','매출순위','순위차이','네이버점수','매출복합점수']].sort_values('순위차이', ascending=False).head(8).round(3).to_string())


# ════════════════════════════════════════════════════════════
# 시각화 1. 네이버 점수 vs 매출 복합점수 산점도 (2×2)
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('네이버 트렌드 점수 vs 매출 기반 트렌드 점수 비교 (25개 지역)', fontsize=14, fontweight='bold')

# (a) 네이버점수 vs 매출복합점수
ax = axes[0]
ax.scatter(compare['네이버점수'], compare['매출복합점수'],
           color='#4e79a7', s=80, alpha=0.8, edgecolors='white', linewidths=0.8)
for grp, row in compare.iterrows():
    ax.annotate(grp, (row['네이버점수'], row['매출복합점수']),
                fontsize=8.5, xytext=(4, 3), textcoords='offset points', color='#333333')

# 회귀선
x = compare['네이버점수'].values
y = compare['매출복합점수'].values
z = np.polyfit(x, y, 1)
xr = np.linspace(x.min(), x.max(), 100)
ax.plot(xr, np.poly1d(z)(xr), color='gray', linestyle='--', linewidth=1.5,
        label=f'추세선 (ρ={rho:.3f}, p={pval:.2f})')
ax.set_xlabel('네이버 검색 트렌드 점수 (ratio 평균)', fontsize=11)
ax.set_ylabel('매출 기반 복합점수 (업종+시간패턴)', fontsize=11)
ax.set_title(f'네이버 vs 매출 복합점수\n스피어만 ρ={rho:.3f} (p={pval:.3f})', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

# (b) 네이버점수 vs 2030비중
ax = axes[1]
ax.scatter(compare['네이버점수'], compare['2030비중'] * 100,
           color='#e15759', s=80, alpha=0.8, edgecolors='white', linewidths=0.8)
for grp, row in compare.iterrows():
    ax.annotate(grp, (row['네이버점수'], row['2030비중'] * 100),
                fontsize=8.5, xytext=(4, 3), textcoords='offset points', color='#333333')

y2 = compare['2030비중'].values * 100
z2 = np.polyfit(x, y2, 1)
ax.plot(xr, np.poly1d(z2)(xr), color='gray', linestyle='--', linewidth=1.5,
        label=f'추세선 (ρ={rho2:.3f}, p={pval2:.2f})')
ax.set_xlabel('네이버 검색 트렌드 점수 (ratio 평균)', fontsize=11)
ax.set_ylabel('2030세대 매출 비중 (%)', fontsize=11)
ax.set_title(f'네이버 점수 vs 2030비중\n스피어만 ρ={rho2:.3f} (p={pval2:.3f})', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'naver_vs_sales_scatter.png'), dpi=150, bbox_inches='tight')
print("\n저장: image/naver_vs_sales_scatter.png")
plt.show()


# ════════════════════════════════════════════════════════════
# 시각화 2. 연도별 시계열 비교 — 네이버 ratio vs 2030비중
#   (두 지표가 같은 방향으로 움직이는가?)
# ════════════════════════════════════════════════════════════
# 네이버 연도별 평균
naver_annual = nv.groupby(['year', 'group'])['ratio'].mean().reset_index()
naver_annual.columns = ['year', 'group', 'naver_ratio']

# 매출 연도별 2030비중
all_mapped_dongs = [d for dongs in NAVER_TO_DONG.values() for d in dongs]
df_sub = df_sales[df_sales['행정동_코드_명'].isin(all_mapped_dongs)].copy()
df_sub['naver_group'] = df_sub['행정동_코드_명'].map(DONG_TO_NAVER)

sales_annual = (
    df_sub.groupby(['year', 'naver_group'])
    .agg(g20=('연령대_20_매출_금액','sum'),
         g30=('연령대_30_매출_금액','sum'),
         tot=('당월_매출_금액','sum'))
    .reset_index()
)
sales_annual['비중'] = (sales_annual['g20'] + sales_annual['g30']) / sales_annual['tot']
sales_annual.columns = ['year','group','g20','g30','tot','2030비중']

# 병합 (2022~2024 — 두 데이터 모두 있는 구간)
merged_ts = pd.merge(
    naver_annual[naver_annual['year'] >= 2020],
    sales_annual[['year','group','2030비중']],
    on=['year','group']
)

# 지역별 두 지표의 피어슨 상관 계산
ts_corr = (
    merged_ts.groupby('group')
    .apply(lambda g: g[['naver_ratio','2030비중']].corr().iloc[0,1])
    .sort_values(ascending=False)
)
print("\n=== 지역별 네이버 ratio vs 2030비중 연도별 상관 ===")
print(ts_corr.round(3).to_string())

# 상관 높은 상위 6 / 낮은 하위 6 선택해서 시계열 시각화
top6  = ts_corr.nlargest(6).index.tolist()
bot6  = ts_corr.nsmallest(6).index.tolist()
plot_groups = top6 + bot6

fig2, axes2 = plt.subplots(4, 3, figsize=(20, 18))
fig2.suptitle('네이버 트렌드 ratio vs 2030비중 연도별 추이 비교\n(상단: 두 지표 동행, 하단: 두 지표 불일치)', fontsize=14, fontweight='bold')
axes2 = axes2.flatten()

for idx, grp in enumerate(plot_groups):
    ax = axes2[idx]
    sub = merged_ts[merged_ts['group'] == grp].sort_values('year')
    if len(sub) < 2:
        ax.set_visible(False)
        continue

    years = sub['year'].astype(str).tolist()
    corr_val = ts_corr.get(grp, np.nan)

    ax2 = ax.twinx()
    ax.plot(years, sub['naver_ratio'], color='#e15759', marker='o', linewidth=2,
            markersize=6, label='네이버 ratio')
    ax2.plot(years, sub['2030비중'] * 100, color='#4e79a7', marker='s', linewidth=2,
             markersize=6, linestyle='--', label='2030비중 %')

    ax.set_ylabel('네이버 ratio', color='#e15759', fontsize=8)
    ax2.set_ylabel('2030비중 %', color='#4e79a7', fontsize=8)
    ax.tick_params(axis='y', colors='#e15759', labelsize=7)
    ax2.tick_params(axis='y', colors='#4e79a7', labelsize=7)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_title(f'{grp}  (r={corr_val:.2f})', fontsize=10, fontweight='bold',
                 color='#c0392b' if idx < 6 else '#2c5f8a')
    ax.grid(True, alpha=0.3)
    ax.spines[['top']].set_visible(False)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'naver_vs_sales_timeseries.png'), dpi=150, bbox_inches='tight')
print("저장: image/naver_vs_sales_timeseries.png")
plt.show()


# ════════════════════════════════════════════════════════════
# 시각화 3. 순위 비교 — 네이버 vs 매출 (bump chart 스타일)
# ════════════════════════════════════════════════════════════
cmp_sorted = compare.sort_values('네이버점수', ascending=False).reset_index()
cmp_sorted['네이버순위_idx'] = range(1, len(cmp_sorted)+1)
cmp_sorted = cmp_sorted.sort_values('매출복합점수', ascending=False).reset_index(drop=True)
cmp_sorted['매출순위_idx'] = range(1, len(cmp_sorted)+1)

fig3, ax = plt.subplots(figsize=(12, 10))
fig3.suptitle('지역별 순위 비교: 네이버 트렌드 vs 매출 복합점수\n(선이 교차할수록 두 기준이 다르게 평가)', fontsize=13, fontweight='bold')

compare_sorted2 = compare.copy()
compare_sorted2['네이버순위'] = compare['네이버점수'].rank(ascending=False).astype(int)
compare_sorted2['매출순위']   = compare['매출복합점수'].rank(ascending=False).astype(int)
compare_sorted2['순위차이']   = (compare_sorted2['네이버순위'] - compare_sorted2['매출순위'])

for grp, row in compare_sorted2.iterrows():
    diff = row['순위차이']
    color = '#e15759' if diff > 4 else '#4e79a7' if diff < -4 else '#bab0ac'
    lw    = 2.5 if abs(diff) > 4 else 1.2
    ax.plot([0, 1], [row['네이버순위'], row['매출순위']],
            color=color, linewidth=lw, alpha=0.8)
    ax.text(-0.02, row['네이버순위'], f"{int(row['네이버순위'])}. {grp}",
            ha='right', va='center', fontsize=8.5)
    ax.text(1.02, row['매출순위'], f"{int(row['매출순위'])}. {grp}",
            ha='left', va='center', fontsize=8.5)

ax.set_xlim(-0.4, 1.4)
ax.set_ylim(0.5, len(compare) + 0.5)
ax.invert_yaxis()
ax.set_xticks([0, 1])
ax.set_xticklabels(['네이버 트렌드\n순위 (ratio 기준)', '매출 복합점수\n순위 (업종+시간)'], fontsize=11)
ax.set_yticks([])
ax.spines[['top','bottom','left','right']].set_visible(False)
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0],[0], color='#e15759', lw=2.5, label='네이버↑ 매출↓ (네이버가 높게 봄)'),
    Line2D([0],[0], color='#4e79a7', lw=2.5, label='네이버↓ 매출↑ (매출이 높게 봄)'),
    Line2D([0],[0], color='#bab0ac', lw=1.2, label='순위 일치 (차이 ≤4)'),
]
ax.legend(handles=legend_elements, fontsize=9, loc='lower center', bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'naver_vs_sales_rank.png'), dpi=150, bbox_inches='tight')
print("저장: image/naver_vs_sales_rank.png")
plt.show()

print("\n완료!")
