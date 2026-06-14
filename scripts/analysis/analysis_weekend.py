# -*- coding: utf-8 -*-
"""
평일 vs 주말 일평균 매출 비율로 트렌드 상권 재정의
- 주말 일평균 = 주말매출 / 2
- 평일 일평균 = 주중매출 / 5
- 비율 = 주말 일평균 / 평일 일평균
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

# ── 행정동별 지표 계산 ────────────────────────────────────────
stats = (
    df.groupby('행정동_코드_명')
    .agg(
        주말매출=('주말_매출_금액', 'sum'),
        주중매출=('주중_매출_금액', 'sum'),
        전체매출=('당월_매출_금액', 'sum'),
        gen20=('연령대_20_매출_금액', 'sum'),
        gen30=('연령대_30_매출_금액', 'sum'),
    )
)

stats['주말일평균'] = stats['주말매출'] / 2
stats['평일일평균'] = stats['주중매출'] / 5
stats['주말평일비율'] = stats['주말일평균'] / stats['평일일평균']   # 핵심 지표
stats['주말비중']   = stats['주말매출'] / stats['전체매출']         # 기존 지표 (비교용)
stats['2030비중']   = (stats['gen20'] + stats['gen30']) / stats['전체매출']

print("=== 주말/평일 일평균 비율 분포 ===")
print(stats['주말평일비율'].describe().round(3))

print("\n상위 20개 (외부 방문 집중 상권):")
top20 = stats.nlargest(20, '주말평일비율')[['주말평일비율','주말비중','2030비중']]
print(top20.round(3).to_string())

print("\n하위 20개 (평일 집중 / 오피스 상권):")
bot20 = stats.nsmallest(20, '주말평일비율')[['주말평일비율','주말비중','2030비중']]
print(bot20.round(3).to_string())


# ════════════════════════════════════════════════════════════
# 1. 기존 주말비중 vs 새 지표(주말/평일 비율) 산점도
#    — 두 방법의 차이를 보여줌
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('트렌드 상권 지표 비교: 주말비중 vs 주말/평일 일평균 비율', fontsize=14, fontweight='bold')

ax = axes[0]
ax.scatter(stats['주말비중'] * 100, stats['주말평일비율'],
           alpha=0.45, color='#4e79a7', edgecolors='white', linewidths=0.4, s=55)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='비율=1.0 (주말=평일)')

# 두 지표 간 순위가 크게 바뀐 상권 레이블
highlight = stats[
    (stats['주말평일비율'] > stats['주말평일비율'].quantile(0.92)) |
    (stats['주말평일비율'] < stats['주말평일비율'].quantile(0.05))
]
for dong, row in highlight.iterrows():
    ax.annotate(dong, (row['주말비중'] * 100, row['주말평일비율']),
                fontsize=6.5, xytext=(3, 3), textcoords='offset points', color='#c0392b')

corr = stats[['주말비중', '주말평일비율']].corr().iloc[0, 1]
ax.set_xlabel('주말 비중 (%) — 기존 방식', fontsize=11)
ax.set_ylabel('주말/평일 일평균 비율 — 새 방식', fontsize=11)
ax.set_title(f'두 지표 상관: r = {corr:.3f}', fontsize=12)
ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)


# ════════════════════════════════════════════════════════════
# 2. 주말/평일 비율 Top 30 / Bottom 30
# ════════════════════════════════════════════════════════════
ax = axes[1]
top30  = stats.nlargest(30, '주말평일비율')
bot10  = stats.nsmallest(10, '주말평일비율')
plot_df = pd.concat([bot10, top30]).sort_values('주말평일비율')

colors = ['#4e79a7' if v < 1.0 else '#e15759' if v > 1.3 else '#f28e2b'
          for v in plot_df['주말평일비율']]
ax.barh(plot_df.index, plot_df['주말평일비율'], color=colors, alpha=0.85, edgecolor='white')
ax.axvline(1.0, color='black', linewidth=1.2, linestyle='--', label='비율=1.0')
for i, (dong, row) in enumerate(plot_df.iterrows()):
    ax.text(row['주말평일비율'] + 0.01, i,
            f"{row['주말평일비율']:.2f}", va='center', fontsize=7.5)

ax.set_xlabel('주말/평일 일평균 매출 비율', fontsize=11)
ax.set_title('주말 집중도 상위 30 / 하위 10\n(빨강>1.3배, 주황>1.0배, 파랑<1.0배)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, axis='x', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'weekend_ratio_comparison.png'), dpi=150, bbox_inches='tight')
print("\n저장: image/weekend_ratio_comparison.png")
plt.show()


# ════════════════════════════════════════════════════════════
# 3. 새 지표로 트렌드 상권 재정의 + 업종 구성 비교
# ════════════════════════════════════════════════════════════
# 주말/평일 비율 상위 30% AND 2030비중 상위 30%
wk_thr = stats['주말평일비율'].quantile(0.70)
mz_thr = stats['2030비중'].quantile(0.70)

trend_new  = stats[(stats['주말평일비율'] >= wk_thr) & (stats['2030비중'] >= mz_thr)].index
trend_old  = stats[(stats['주말비중'] >= stats['주말비중'].quantile(0.70)) & (stats['2030비중'] >= mz_thr)].index
stable     = stats[~stats.index.isin(trend_new)].index

print(f"\n=== 트렌드 상권 재정의 비교 ===")
print(f"기존 (주말비중 + 2030비중): {len(trend_old)}개")
print(f"새 방식 (주말/평일비율 + 2030비중): {len(trend_new)}개")
print(f"\n새 방식 Top 20:")
print(stats.loc[trend_new].nlargest(20, '주말평일비율').index.tolist())

# 기존 방식에서 빠지고 새 방식에 포함된 상권 (새로 발견된 상권)
new_in  = set(trend_new) - set(trend_old)
new_out = set(trend_old) - set(trend_new)
print(f"\n새로 포함된 상권 ({len(new_in)}개): {sorted(new_in)}")
print(f"제외된 상권 ({len(new_out)}개): {sorted(new_out)}")

# 업종 구성 비교
TREND_CATEG = {
    'F&B (카페/디저트)': ['커피-음료', '제과점'],
    'F&B (식사)':       ['한식음식점', '일식음식점', '양식음식점', '중식음식점',
                         '분식전문점', '패스트푸드점', '치킨전문점'],
    '주류/유흥':        ['호프-간이주점', '노래방'],
    '뷰티':            ['미용실', '네일숍', '피부관리실', '화장품'],
    '라이프스타일':     ['일반의류', '편의점', '애완동물', '화초', '인테리어'],
    '교육':            ['일반교습학원', '외국어학원', '예술학원'],
    '의료':            ['일반의원', '치과의원', '한의원', '의약품'],
}
df['category'] = df['서비스_업종_코드_명'].map(
    lambda x: next((c for c, items in TREND_CATEG.items() if x in items), '기타')
)

def categ_share(dongs):
    s = df[df['행정동_코드_명'].isin(dongs)].groupby('category')['당월_매출_금액'].sum()
    return (s / s.sum() * 100).round(2)

share_new    = categ_share(trend_new)
share_stable = categ_share(stable)
categ_order  = share_new.sort_values(ascending=False).index

CATEG_COLORS = {
    'F&B (카페/디저트)': '#e15759', 'F&B (식사)': '#f28e2b',
    '주류/유흥': '#edc948',         '뷰티': '#76b7b2',
    '라이프스타일': '#59a14f',       '교육': '#4e79a7',
    '의료': '#b07aa1',              '기타': '#bab0ac',
}

fig2, axes2 = plt.subplots(1, 2, figsize=(18, 7))
fig2.suptitle('트렌드 상권(주말/평일비율 기준) vs 일반 상권 업종 구성', fontsize=14, fontweight='bold')

for ax, share, title in [
    (axes2[0], share_new,    f'트렌드 상권 (주말/평일≥{wk_thr:.2f} & 2030≥{mz_thr:.1%}, {len(trend_new)}개)'),
    (axes2[1], share_stable, f'일반 상권 ({len(stable)}개)'),
]:
    share_ord = share.reindex(categ_order).fillna(0)
    colors    = [CATEG_COLORS.get(c, '#bab0ac') for c in share_ord.index]
    bars = ax.bar(share_ord.index, share_ord.values, color=colors, alpha=0.88, edgecolor='white')
    for bar, val in zip(bars, share_ord.values):
        if val > 1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('매출 비중 (%)', fontsize=10)
    ax.set_ylim(0, share_ord.max() * 1.2)
    ax.set_xticks(range(len(share_ord)))
    ax.set_xticklabels(share_ord.index, rotation=30, ha='right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'weekend_ratio_industry.png'), dpi=150, bbox_inches='tight')
print("저장: image/weekend_ratio_industry.png")
plt.show()


# ════════════════════════════════════════════════════════════
# 4. 연도별 주말/평일 비율 변화 — 트렌드 상권 Top 15
# ════════════════════════════════════════════════════════════
top15_dongs = stats.loc[trend_new].nlargest(15, '주말평일비율').index.tolist()

annual = (
    df[df['행정동_코드_명'].isin(top15_dongs)]
    .groupby(['year', '행정동_코드_명'])
    .agg(주말=('주말_매출_금액','sum'), 주중=('주중_매출_금액','sum'))
    .assign(비율=lambda x: (x['주말']/2) / (x['주중']/5))
    .reset_index()
)

fig3, ax = plt.subplots(figsize=(14, 7))
fig3.suptitle('트렌드 상권 Top 15: 주말/평일 일평균 비율 연도별 변화', fontsize=14, fontweight='bold')

cmap = plt.colormaps.get_cmap('tab20').resampled(len(top15_dongs))
for i, dong in enumerate(top15_dongs):
    sub  = annual[annual['행정동_코드_명'] == dong].sort_values('year')
    yrs  = sub['year'].astype(str).tolist()
    vals = sub['비율'].tolist()
    ax.plot(yrs, vals, marker='o', linewidth=2, color=cmap(i), label=dong, markersize=6)
    ax.annotate(f"{dong} {vals[-1]:.2f}",
                xy=(yrs[-1], vals[-1]),
                xytext=(6, 0), textcoords='offset points',
                fontsize=7.5, color=cmap(i), va='center')

ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='비율=1.0 (주말=평일)')
ax.set_xlabel('연도', fontsize=11)
ax.set_ylabel('주말/평일 일평균 매출 비율', fontsize=11)
ax.legend(title='행정동', fontsize=7.5, loc='upper left', ncol=2)
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'weekend_ratio_trend_line.png'), dpi=150, bbox_inches='tight')
print("저장: image/weekend_ratio_trend_line.png")
plt.show()

print("\n완료!")
