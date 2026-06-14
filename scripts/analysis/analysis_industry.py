# -*- coding: utf-8 -*-
"""
외부 유입 트렌드 상권의 업종별 소비 변화 분석
- 외부 유입 지표: 주말 매출 비중 (주말 > 평일이면 외부 방문객 비율 높음)
- 트렌드 상권 vs 일반 상권 간 업종 구성 차이
- 2020→2024 업종별 점유율 변화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
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
df['year']    = df['기준_년분기_코드'] // 10
df['quarter'] = df['기준_년분기_코드'] % 10

# ── 업종 카테고리 분류 ────────────────────────────────────────
TREND_CATEG = {
    'F&B (카페/디저트)': ['커피-음료', '제과점'],
    'F&B (식사)':       ['한식음식점', '일식음식점', '양식음식점', '중식음식점',
                         '분식전문점', '패스트푸드점', '치킨전문점'],
    '주류/유흥':        ['호프-간이주점', '노래방'],
    '뷰티':            ['미용실', '네일숍', '피부관리실', '화장품'],
    '라이프스타일':     ['일반의류', '편의점', '애완동물', '화초', '인테리어'],
    '교육':            ['일반교습학원', '외국어학원', '예술학원'],
    '의료':            ['일반의원', '치과의원', '한의원', '의약품'],
    '기타':            [],   # 나머지 업종
}

def get_category(업종):
    for cat, items in TREND_CATEG.items():
        if 업종 in items:
            return cat
    return '기타'

df['category'] = df['서비스_업종_코드_명'].map(get_category)

# ── 트렌드 상권 정의: 주말 비중 상위 30% AND 2030 비중 상위 30% ──
# 단순 주말 비중만 쓰면 고시촌·등산로·스포츠시설이 오염됨
# 2030 비중을 동시에 요구해 실제 MZ 외부 유입 상권만 선별

weekend_ratio = (
    df.groupby('행정동_코드_명')
    .agg(주말=('주말_매출_금액', 'sum'), 전체=('당월_매출_금액', 'sum'))
    .assign(주말비중=lambda x: x['주말'] / x['전체'])
)

gen2030 = (
    df.groupby('행정동_코드_명')
    .agg(
        gen2030=('연령대_20_매출_금액', 'sum'),
        gen2030_30=('연령대_30_매출_금액', 'sum'),
        전체=('당월_매출_금액', 'sum'),
    )
    .assign(gen2030비중=lambda x: (x['gen2030'] + x['gen2030_30']) / x['전체'])
)

stats = weekend_ratio[['주말비중']].join(gen2030[['gen2030비중']])

wk_thr   = stats['주말비중'].quantile(0.70)   # 상위 30%
mz_thr   = stats['gen2030비중'].quantile(0.70)

trend_mask   = (stats['주말비중'] >= wk_thr) & (stats['gen2030비중'] >= mz_thr)
trend_dongs  = stats[trend_mask].index.tolist()
stable_dongs = stats[~trend_mask].index.tolist()

print(f"트렌드 상권 기준: 주말비중 ≥ {wk_thr:.1%} AND 2030비중 ≥ {mz_thr:.1%}")
print(f"트렌드 상권: {len(trend_dongs)}개 | 일반 상권: {len(stable_dongs)}개")
print(f"트렌드 상권 Top 15 (주말비중 기준): {stats[trend_mask].nlargest(15,'주말비중').index.tolist()}")


# ════════════════════════════════════════════════════════════
# 1. 트렌드 상권 vs 일반 상권 업종 구성 비교
# ════════════════════════════════════════════════════════════
def categ_share(dongs):
    sub = df[df['행정동_코드_명'].isin(dongs)]
    s = sub.groupby('category')['당월_매출_금액'].sum()
    return (s / s.sum() * 100).round(2)

trend_share  = categ_share(trend_dongs)
stable_share = categ_share(stable_dongs)
categ_order  = trend_share.sort_values(ascending=False).index

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('트렌드 상권 vs 일반 상권: 업종 구성 비교 (2020~2024)', fontsize=14, fontweight='bold')

CATEG_COLORS = {
    'F&B (카페/디저트)': '#e15759',
    'F&B (식사)':        '#f28e2b',
    '주류/유흥':         '#edc948',
    '뷰티':             '#76b7b2',
    '라이프스타일':      '#59a14f',
    '교육':             '#4e79a7',
    '의료':             '#b07aa1',
    '기타':             '#bab0ac',
}

for ax, share, title in [
    (axes[0], trend_share,  f'트렌드 상권 (주말비중 상위 30%, {len(trend_dongs)}개 동)'),
    (axes[1], stable_share, f'일반 상권 ({len(stable_dongs)}개 동)'),
]:
    share_ordered = share.reindex(categ_order).fillna(0)
    colors = [CATEG_COLORS.get(c, '#bab0ac') for c in share_ordered.index]
    bars = ax.bar(share_ordered.index, share_ordered.values, color=colors, alpha=0.88, edgecolor='white')
    for bar, val in zip(bars, share_ordered.values):
        if val > 1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title(title, fontsize=11)
    ax.set_ylabel('매출 비중 (%)', fontsize=10)
    ax.set_ylim(0, share_ordered.max() * 1.2)
    ax.set_xticklabels(share_ordered.index, rotation=30, ha='right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'industry_trend_vs_stable.png'), dpi=150, bbox_inches='tight')
print("저장: image/industry_trend_vs_stable.png")
plt.show()


# ════════════════════════════════════════════════════════════
# 2. 트렌드 상권 내 카테고리별 연도별 점유율 변화
# ════════════════════════════════════════════════════════════
trend_df   = df[df['행정동_코드_명'].isin(trend_dongs)]
annual_cat = trend_df.groupby(['year', 'category'])['당월_매출_금액'].sum().reset_index()
annual_total = trend_df.groupby('year')['당월_매출_금액'].sum()
annual_cat['share'] = annual_cat.apply(lambda r: r['당월_매출_금액'] / annual_total[r['year']] * 100, axis=1)

pivot_cat = annual_cat.pivot(index='year', columns='category', values='share').fillna(0)
pivot_cat = pivot_cat[categ_order]  # 트렌드 상권 기준 정렬

fig, ax = plt.subplots(figsize=(13, 7))
fig.suptitle('트렌드 상권 내 업종별 매출 점유율 연도별 변화', fontsize=14, fontweight='bold')

years_str = [str(y) for y in pivot_cat.index]
for cat in pivot_cat.columns:
    vals = pivot_cat[cat]
    color = CATEG_COLORS.get(cat, '#bab0ac')
    ax.plot(years_str, vals, marker='o', linewidth=2.2, color=color, label=cat, markersize=7)
    ax.annotate(f"{vals.iloc[-1]:.1f}%",
                xy=(years_str[-1], vals.iloc[-1]),
                xytext=(6, 0), textcoords='offset points',
                fontsize=8.5, color=color, va='center')

ax.set_xlabel('연도', fontsize=11)
ax.set_ylabel('업종 매출 비중 (%)', fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax.legend(title='업종 카테고리', fontsize=9, loc='upper left', ncol=2)
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'industry_share_trend_line.png'), dpi=150, bbox_inches='tight')
print("저장: image/industry_share_trend_line.png")
plt.show()


# ════════════════════════════════════════════════════════════
# 3. 핵심 트렌드 업종별 행정동 Top 15 (카페/디저트, F&B, 뷰티)
# ════════════════════════════════════════════════════════════
focus_cats = ['F&B (카페/디저트)', 'F&B (식사)', '뷰티', '라이프스타일']

fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle('트렌드 업종별 행정동 Top 15 매출 점유율 변화 (2020 vs 2024)', fontsize=14, fontweight='bold')

for ax, cat in zip(axes.flat, focus_cats):
    cat_df = df[df['category'] == cat]

    # 연도별 행정동 점유율
    yr_total = cat_df.groupby('year')['당월_매출_금액'].sum()
    dong_yr  = cat_df.groupby(['year', '행정동_코드_명'])['당월_매출_금액'].sum().reset_index()
    dong_yr['share'] = dong_yr.apply(lambda r: r['당월_매출_금액'] / yr_total[r['year']] * 100, axis=1)

    y2020 = dong_yr[dong_yr['year'] == 2020].set_index('행정동_코드_명')['share']
    y2024 = dong_yr[dong_yr['year'] == 2024].set_index('행정동_코드_명')['share']

    top15 = y2024.nlargest(15).index
    cmp   = pd.DataFrame({'2020': y2020.reindex(top15).fillna(0),
                           '2024': y2024.reindex(top15).fillna(0)})
    cmp   = cmp.sort_values('2024', ascending=True)

    y_pos = range(len(cmp))
    ax.barh([y - 0.18 for y in y_pos], cmp['2020'], height=0.35,
            color='#aec7e8', alpha=0.85, label='2020')
    ax.barh([y + 0.18 for y in y_pos], cmp['2024'], height=0.35,
            color=CATEG_COLORS.get(cat, '#4e79a7'), alpha=0.85, label='2024')

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(cmp.index, fontsize=8.5)
    ax.set_xlabel('매출 점유율 (%)', fontsize=9)
    ax.set_title(cat, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, axis='x', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'industry_dong_top15.png'), dpi=150, bbox_inches='tight')
print("저장: image/industry_dong_top15.png")
plt.show()


# ════════════════════════════════════════════════════════════
# 4. 행정동별 주말 비중 vs 카페/F&B 비중 (외부 유입-소비 관계)
# ════════════════════════════════════════════════════════════
dong_cat_share = (
    df.groupby(['행정동_코드_명', 'category'])['당월_매출_금액'].sum()
    .reset_index()
)
dong_total = df.groupby('행정동_코드_명')['당월_매출_금액'].sum()
dong_cat_share['share'] = dong_cat_share.apply(
    lambda r: r['당월_매출_금액'] / dong_total[r['행정동_코드_명']] * 100, axis=1
)

fnb_cafe = dong_cat_share[dong_cat_share['category'] == 'F&B (카페/디저트)'].set_index('행정동_코드_명')['share']
fnb_meal = dong_cat_share[dong_cat_share['category'] == 'F&B (식사)'].set_index('행정동_코드_명')['share']

scatter_df = pd.DataFrame({
    '주말비중': weekend_ratio['주말비중'] * 100,
    '카페비중': fnb_cafe,
    '식사비중': fnb_meal,
}).dropna()

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('외부 유입(주말 비중) vs 업종 소비 비중 관계', fontsize=14, fontweight='bold')

for ax, col, color, label in [
    (axes[0], '카페비중', '#e15759', 'F&B (카페/디저트) 비중'),
    (axes[1], '식사비중', '#f28e2b', 'F&B (식사) 비중'),
]:
    ax.scatter(scatter_df['주말비중'], scatter_df[col],
               alpha=0.55, color=color, edgecolors='white', linewidths=0.5, s=60)

    # 상관계수
    corr = scatter_df[['주말비중', col]].corr().iloc[0, 1]

    # 추세선
    z = np.polyfit(scatter_df['주말비중'], scatter_df[col], 1)
    p = np.poly1d(z)
    xs = np.linspace(scatter_df['주말비중'].min(), scatter_df['주말비중'].max(), 100)
    ax.plot(xs, p(xs), '--', color='gray', linewidth=1.5)

    # 외부 유입 상위 10 행정동 레이블
    top10 = scatter_df.nlargest(10, '주말비중')
    for dong, row in top10.iterrows():
        ax.annotate(dong, (row['주말비중'], row[col]),
                    fontsize=7, ha='left', va='bottom',
                    xytext=(3, 3), textcoords='offset points', color='#333')

    ax.set_xlabel('주말 매출 비중 (%)', fontsize=10)
    ax.set_ylabel(label + ' (%)', fontsize=10)
    ax.set_title(f'{label}\nr = {corr:.3f}', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'industry_weekend_scatter.png'), dpi=150, bbox_inches='tight')
print("저장: image/industry_weekend_scatter.png")
plt.show()


# ── 요약 출력 ─────────────────────────────────────────────────
print("\n── 트렌드 vs 일반 상권 업종 비중 비교 ──")
comp = pd.DataFrame({'트렌드': trend_share, '일반': stable_share,
                     '차이(pp)': (trend_share - stable_share).round(2)})
print(comp.to_string())

print(f"\n기준: 주말비중 ≥ {wk_thr:.1%} AND 2030비중 ≥ {mz_thr:.1%}")
print(f"트렌드 상권 상위 15개: {stats[trend_mask].nlargest(15,'주말비중').index.tolist()}")
