# -*- coding: utf-8 -*-
"""
트렌드 상권 재정의 v3 — 2030 데이터 완전 배제
================================================
[트렌드 상권 정의 기준]
  1. 업종 기준 : 라이프스타일 + F&B(카페/디저트) + F&B(식사) + 주류/유흥 매출 비중
  2. 시간 기준 : 금요효과(금/목 비율) × 주말비중(토+일) × 야간비중(17~24시)
     → 두 기준을 표준화 후 합산한 복합 점수 상위 30%

[2030 비교]
  정의된 트렌드 상권에서 2030 소비 추이가 어떻게 나타나는지 사후 비교
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

data_dir = 'data'
image_dir = 'image'

df = pd.read_csv(
    os.path.join(data_dir, '서울시_상권분석서비스(추정매출-행정동)_2020_2024.csv'),
    encoding='utf-8-sig',
)
df['year'] = df['기준_년분기_코드'] // 10

# ── 업종 카테고리 매핑 ────────────────────────────────────────
CATEG_MAP = {
    'F&B (카페/디저트)': ['커피-음료', '제과점'],
    'F&B (식사)':       ['한식음식점', '일식음식점', '양식음식점', '중식음식점',
                         '분식전문점', '패스트푸드점', '치킨전문점'],
    '주류/유흥':        ['호프-간이주점', '노래방'],
    '라이프스타일':     ['일반의류', '편의점', '애완동물', '화초', '인테리어'],
}
TREND_CATEGS = list(CATEG_MAP.keys())   # 트렌드 업종 4종

df['category'] = df['서비스_업종_코드_명'].map(
    lambda x: next((c for c, items in CATEG_MAP.items() if x in items), '기타')
)

# ── 컬럼 정의 ────────────────────────────────────────────────
DAY_COLS  = ['월요일_매출_금액','화요일_매출_금액','수요일_매출_금액',
             '목요일_매출_금액','금요일_매출_금액','토요일_매출_금액','일요일_매출_금액']
TIME_COLS = ['시간대_00~06_매출_금액','시간대_06~11_매출_금액','시간대_11~14_매출_금액',
             '시간대_14~17_매출_금액','시간대_17~21_매출_금액','시간대_21~24_매출_금액']

# ════════════════════════════════════════════════════════════
# STEP 1. 업종 기준 점수
# ════════════════════════════════════════════════════════════
categ_sum = (
    df.groupby(['행정동_코드_명', 'category'])['당월_매출_금액']
    .sum().unstack(fill_value=0)
)
total_sales = categ_sum.sum(axis=1)
categ_share = categ_sum.div(total_sales, axis=0)

# 4개 트렌드 업종 합산 비중
industry_score = categ_share[TREND_CATEGS].sum(axis=1)

print("=== 업종 기준 점수 분포 ===")
print(industry_score.describe().round(3))
print(f"\n상위 10개:")
print(industry_score.nlargest(10).round(3).to_string())


# ════════════════════════════════════════════════════════════
# STEP 2. 시간 기준 점수 (2030 데이터 사용 안 함)
# ════════════════════════════════════════════════════════════
base = df.groupby('행정동_코드_명')[DAY_COLS + TIME_COLS].sum()

day_total  = base[DAY_COLS].sum(axis=1)
time_total = base[TIME_COLS].sum(axis=1)

# 금요효과: 금요일 / 목요일 매출비
금요효과 = base['금요일_매출_금액'] / base['목요일_매출_금액']

# 주말비중: (토+일) / 7일
주말비중 = (base['토요일_매출_금액'] + base['일요일_매출_금액']) / day_total

# 야간비중: (17~21 + 21~24) / 전체 시간대
야간비중 = (base['시간대_17~21_매출_금액'] + base['시간대_21~24_매출_금액']) / time_total

# 3개 지표 표준화 후 합산
def zscore(s):
    return (s - s.mean()) / s.std()

temporal_score = zscore(금요효과) + zscore(주말비중) + zscore(야간비중)

print("\n=== 시간 기준 점수 (금요효과+주말비중+야간비중 표준화 합) ===")
print(temporal_score.describe().round(3))
print(f"\n상위 10개:")
print(temporal_score.nlargest(10).round(3).to_string())


# ════════════════════════════════════════════════════════════
# STEP 3. 트렌드 상권 선정 — 두 기준 모두 상위 30%
# ════════════════════════════════════════════════════════════
ind_thr  = industry_score.quantile(0.70)
temp_thr = temporal_score.quantile(0.70)

trend_mask = (industry_score >= ind_thr) & (temporal_score >= temp_thr)
trend_dongs  = industry_score.index[trend_mask]
stable_dongs = industry_score.index[~trend_mask]

print(f"\n=== 트렌드 상권 선정 결과 ===")
print(f"업종 임계값 (상위 30%): {ind_thr:.3f}")
print(f"시간 임계값 (상위 30%): {temp_thr:.3f}")
print(f"트렌드 상권: {len(trend_dongs)}개 | 일반 상권: {len(stable_dongs)}개")

# 복합 점수 (표준화 합) — 트렌드 상권 순위용
composite = zscore(industry_score) + zscore(temporal_score)
print(f"\n트렌드 상권 Top 20 (복합점수 기준):")
top20 = composite[trend_mask].nlargest(20)
detail = pd.DataFrame({
    '복합점수': composite,
    '업종점수': industry_score,
    '시간점수': temporal_score,
    '금요효과': 금요효과,
    '주말비중': 주말비중,
    '야간비중': 야간비중,
})
print(detail.loc[top20.index].round(3).to_string())


# ════════════════════════════════════════════════════════════
# 시각화 1. 선정 기준 overview — 2×2
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle('트렌드 상권 선정 기준 (2030 데이터 배제)', fontsize=15, fontweight='bold')

COLORS = {'트렌드': '#e15759', '일반': '#aec7e8'}

# (0,0) 업종점수 vs 시간점수 산점도 — 4분면
ax = axes[0, 0]
ax.scatter(industry_score[~trend_mask], temporal_score[~trend_mask],
           alpha=0.35, color=COLORS['일반'], s=35, label='일반 상권', edgecolors='none')
ax.scatter(industry_score[trend_mask], temporal_score[trend_mask],
           alpha=0.75, color=COLORS['트렌드'], s=55, label=f'트렌드 상권 ({len(trend_dongs)}개)',
           edgecolors='white', linewidths=0.5)
for dong in top20.index[:12]:
    ax.annotate(dong, (industry_score[dong], temporal_score[dong]),
                fontsize=7, xytext=(3, 2), textcoords='offset points', color='#c0392b')
ax.axhline(temp_thr, color='gray', linestyle='--', linewidth=1)
ax.axvline(ind_thr,  color='gray', linestyle='--', linewidth=1)
ax.set_xlabel('업종 점수 (트렌드 업종 4종 매출 비중)', fontsize=10)
ax.set_ylabel('시간 점수 (금요효과+주말비중+야간비중 표준화)', fontsize=10)
ax.set_title('트렌드 상권 선정 4분면\n(점선: 상위 30% 임계값)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

# (0,1) 트렌드 상권 Top 20 복합점수 막대
ax = axes[0, 1]
top20_detail = detail.loc[top20.index].sort_values('복합점수')
bar_colors = [COLORS['트렌드']] * len(top20_detail)
bars = ax.barh(top20_detail.index, top20_detail['복합점수'], color=bar_colors, alpha=0.85, edgecolor='white')
for bar, val in zip(bars, top20_detail['복합점수']):
    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', va='center', fontsize=7.5)
ax.set_xlabel('복합 점수 (업종+시간 표준화 합)', fontsize=10)
ax.set_title('트렌드 상권 Top 20\n(업종점수 + 시간점수)', fontsize=11)
ax.grid(True, axis='x', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

# (1,0) 금요효과 분포 비교 (트렌드 vs 일반)
ax = axes[1, 0]
bins = np.linspace(금요효과.min(), min(금요효과.quantile(0.98), 2.0), 30)
ax.hist(금요효과[~trend_mask].clip(upper=2.0), bins=bins, alpha=0.6,
        color=COLORS['일반'], label='일반 상권', density=True)
ax.hist(금요효과[trend_mask].clip(upper=2.0), bins=bins, alpha=0.6,
        color=COLORS['트렌드'], label='트렌드 상권', density=True)
ax.axvline(1.0, color='black', linestyle='--', linewidth=1, label='금=목 (효과 없음)')
ax.set_xlabel('금요효과 (금요일/목요일 매출비)', fontsize=10)
ax.set_ylabel('밀도', fontsize=10)
ax.set_title('금요효과 분포 비교\n(트렌드 상권은 1.0 이상에 집중)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

# (1,1) 야간비중 vs 주말비중 (트렌드 vs 일반)
ax = axes[1, 1]
ax.scatter(야간비중[~trend_mask] * 100, 주말비중[~trend_mask] * 100,
           alpha=0.35, color=COLORS['일반'], s=35, label='일반 상권', edgecolors='none')
ax.scatter(야간비중[trend_mask] * 100, 주말비중[trend_mask] * 100,
           alpha=0.75, color=COLORS['트렌드'], s=55, label='트렌드 상권',
           edgecolors='white', linewidths=0.5)
ax.set_xlabel('야간비중 % (17~24시 매출 / 전체 시간대)', fontsize=10)
ax.set_ylabel('주말비중 % (토+일 / 7일)', fontsize=10)
ax.set_title('야간비중 vs 주말비중\n(트렌드 상권은 오른쪽 상단 집중)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'trend_v3_selection.png'), dpi=150, bbox_inches='tight')
print("\n저장: image/trend_v3_selection.png")
plt.show()


# ════════════════════════════════════════════════════════════
# STEP 4. 2030 소비 추이 비교 (사후 검증)
# ════════════════════════════════════════════════════════════
df['is_trend'] = df['행정동_코드_명'].isin(trend_dongs)

# 행정동별 전체 2030비중
gen2030 = (
    df.groupby('행정동_코드_명')
    .agg(gen20=('연령대_20_매출_금액','sum'),
         gen30=('연령대_30_매출_금액','sum'),
         total=('당월_매출_금액','sum'))
)
gen2030['비중'] = (gen2030['gen20'] + gen2030['gen30']) / gen2030['total']

trend_2030  = gen2030.loc[gen2030.index.isin(trend_dongs),  '비중']
stable_2030 = gen2030.loc[gen2030.index.isin(stable_dongs), '비중']

print(f"\n=== 2030 소비 비중 비교 (사후 검증) ===")
print(f"트렌드 상권 2030비중: mean={trend_2030.mean():.3f}, median={trend_2030.median():.3f}")
print(f"일반 상권  2030비중: mean={stable_2030.mean():.3f}, median={stable_2030.median():.3f}")

# 연도별 트렌드 vs 일반 2030비중 추이
annual_2030 = (
    df.groupby(['year', 'is_trend'])
    .agg(gen20=('연령대_20_매출_금액','sum'),
         gen30=('연령대_30_매출_금액','sum'),
         total=('당월_매출_금액','sum'))
    .reset_index()
)
annual_2030['비중'] = (annual_2030['gen20'] + annual_2030['gen30']) / annual_2030['total']

# 복합점수 vs 2030비중 상관
merged = pd.DataFrame({
    '복합점수': composite,
    '2030비중': gen2030['비중'],
    'is_trend': trend_mask,
})
corr = merged[['복합점수', '2030비중']].corr().iloc[0, 1]
print(f"\n복합점수 vs 2030비중 상관계수: r = {corr:.3f}")


# ════════════════════════════════════════════════════════════
# 시각화 2. 2030 소비 비교 — 3개 차트
# ════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(20, 7))
fig2.suptitle('트렌드 상권 vs 일반 상권: 2030세대 소비 비중 비교 (사후 검증)', fontsize=14, fontweight='bold')

# (a) 박스플롯 비교
ax = axes2[0]
bp = ax.boxplot(
    [trend_2030.values, stable_2030.values],
    labels=['트렌드 상권', '일반 상권'],
    patch_artist=True,
    medianprops=dict(color='black', linewidth=2),
    widths=0.5
)
bp['boxes'][0].set_facecolor('#e15759'); bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor('#aec7e8'); bp['boxes'][1].set_alpha(0.7)
ax.set_ylabel('2030세대 매출 비중', fontsize=10)
ax.set_title(f'2030비중 분포 비교\n트렌드 mean={trend_2030.mean():.1%} | 일반 mean={stable_2030.mean():.1%}', fontsize=11)
ax.grid(True, axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

# (b) 연도별 2030비중 추이
ax = axes2[1]
for is_t, label, color in [(True,'트렌드 상권','#e15759'), (False,'일반 상권','#4e79a7')]:
    sub = annual_2030[annual_2030['is_trend'] == is_t].sort_values('year')
    ax.plot(sub['year'].astype(str), sub['비중'] * 100,
            marker='o', linewidth=2.5, color=color, label=label, markersize=7)
    for _, row in sub.iterrows():
        ax.annotate(f"{row['비중']:.1%}", (str(int(row['year'])), row['비중'] * 100),
                    textcoords='offset points', xytext=(0, 8),
                    fontsize=8, ha='center', color=color)
ax.set_xlabel('연도', fontsize=10)
ax.set_ylabel('2030세대 매출 비중 (%)', fontsize=10)
ax.set_title('연도별 2030비중 추이\n(트렌드 vs 일반)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

# (c) 복합점수 vs 2030비중 산점도
ax = axes2[2]
ax.scatter(merged.loc[~merged['is_trend'], '복합점수'],
           merged.loc[~merged['is_trend'], '2030비중'] * 100,
           alpha=0.35, color='#aec7e8', s=35, label='일반 상권', edgecolors='none')
ax.scatter(merged.loc[merged['is_trend'], '복합점수'],
           merged.loc[merged['is_trend'], '2030비중'] * 100,
           alpha=0.75, color='#e15759', s=55, label='트렌드 상권',
           edgecolors='white', linewidths=0.5)
# 회귀선
x = merged['복합점수'].values
y = merged['2030비중'].values * 100
z = np.polyfit(x[~np.isnan(x) & ~np.isnan(y)], y[~np.isnan(x) & ~np.isnan(y)], 1)
xr = np.linspace(x.min(), x.max(), 100)
ax.plot(xr, np.poly1d(z)(xr), color='gray', linewidth=1.5, linestyle='--', label=f'추세선 (r={corr:.3f})')
for dong in top20.index[:10]:
    ax.annotate(dong, (merged.loc[dong,'복합점수'], merged.loc[dong,'2030비중']*100),
                fontsize=7, xytext=(3,2), textcoords='offset points', color='#c0392b')
ax.set_xlabel('트렌드 복합점수 (업종+시간, 2030 배제)', fontsize=10)
ax.set_ylabel('2030세대 매출 비중 (%)', fontsize=10)
ax.set_title(f'트렌드 점수 vs 2030비중\nr = {corr:.3f}', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'trend_v3_2030_comparison.png'), dpi=150, bbox_inches='tight')
print("저장: image/trend_v3_2030_comparison.png")
plt.show()


# ════════════════════════════════════════════════════════════
# 시각화 3. 행정동별 연도별 2030비중 — 트렌드 상권 Top 15
# ════════════════════════════════════════════════════════════
top15_dongs = composite[trend_mask].nlargest(15).index.tolist()

annual_dong = (
    df[df['행정동_코드_명'].isin(top15_dongs)]
    .groupby(['year','행정동_코드_명'])
    .agg(g20=('연령대_20_매출_금액','sum'),
         g30=('연령대_30_매출_금액','sum'),
         tot=('당월_매출_금액','sum'))
    .reset_index()
)
annual_dong['비중'] = (annual_dong['g20'] + annual_dong['g30']) / annual_dong['tot']

fig3, ax = plt.subplots(figsize=(14, 7))
fig3.suptitle('트렌드 상권 Top 15: 2030비중 연도별 변화 (2020→2024)', fontsize=14, fontweight='bold')

cmap = plt.colormaps.get_cmap('tab20').resampled(len(top15_dongs))
for i, dong in enumerate(top15_dongs):
    sub = annual_dong[annual_dong['행정동_코드_명'] == dong].sort_values('year')
    yrs  = sub['year'].astype(str).tolist()
    vals = (sub['비중'] * 100).tolist()
    ax.plot(yrs, vals, marker='o', linewidth=2, color=cmap(i), label=dong, markersize=6)
    if vals:
        ax.annotate(f"{dong} {vals[-1]:.1f}%",
                    xy=(yrs[-1], vals[-1]),
                    xytext=(6, 0), textcoords='offset points',
                    fontsize=7.5, color=cmap(i), va='center')

ax.set_xlabel('연도', fontsize=11)
ax.set_ylabel('2030세대 매출 비중 (%)', fontsize=11)
ax.legend(title='행정동', fontsize=7.5, loc='upper left', ncol=2)
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'trend_v3_2030_timeseries.png'), dpi=150, bbox_inches='tight')
print("저장: image/trend_v3_2030_timeseries.png")
plt.show()

print("\n완료!")
