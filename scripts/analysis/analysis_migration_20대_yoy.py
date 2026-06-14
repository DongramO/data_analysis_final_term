"""
생활이동 기반 20대 이동비율 전년도 대비 변화율 Top 15
: 연도별(2021~2024) 전년 대비 20대 이동비율 변화율이 가장 큰 행정동
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 & 이름 매핑 ───────────────────────────────────
mig = pd.read_csv('data/main_data/분기별_주말유입인구.csv', encoding='utf-8-sig')

kt = pd.read_excel('data/KT_서울생활이동데이터_행정동코드_20210907.xlsx')
kt.columns = ['시도','시군구','행정동코드','행정동명','전체명']
name_map = dict(zip(kt['행정동코드'], kt['행정동명']))
gu_map   = dict(zip(kt['행정동코드'], kt['시군구']))
label_map = {c: f"{gu_map.get(c,'')} {name_map.get(c, str(c))}".strip()
             for c in mig['도착_행정동코드'].unique()}

# ── 연간 집계 & 20대 비율 계산 ────────────────────────────────
annual = (
    mig.groupby(['도착_행정동코드', '연도', '연령대'])['합계_이동인구']
    .sum().reset_index()
)
total = annual.groupby(['도착_행정동코드','연도'])['합계_이동인구'].sum().rename('총_이동인구')
annual = annual.join(total, on=['도착_행정동코드','연도'])
annual['비율'] = annual['합계_이동인구'] / annual['총_이동인구'] * 100

age20 = (annual[annual['연령대'] == 20]
         .sort_values(['도착_행정동코드','연도'])
         .copy())
age20['전년비율'] = age20.groupby('도착_행정동코드')['비율'].shift(1)
age20['변화율'] = age20['비율'] - age20['전년비율']  # %p 변화
age20['행정동'] = age20['도착_행정동코드'].map(label_map)
yoy = age20.dropna(subset=['변화율'])

# ── 시각화: 연도별 Top 15 (변화율 큰 순) ─────────────────────
years = sorted(yoy['연도'].unique())  # 2021~2024
fig, axes = plt.subplots(2, 2, figsize=(18, 16))
fig.suptitle('연도별 전년도 대비 20대 이동비율 변화율 Top 15 행정동\n(서울 생활이동, 금·토·일 트렌드 시간대)',
             fontsize=15, fontweight='bold', y=1.01)

cmap_pos = LinearSegmentedColormap.from_list('green', ['#a8d8a8', '#1a6e2e'])
cmap_neg = LinearSegmentedColormap.from_list('red',   ['#f5b3b3', '#c0392b'])

for ax, yr in zip(axes.flatten(), years):
    df_yr = yoy[yoy['연도'] == yr].nlargest(15, '변화율').sort_values('변화율', ascending=True)
    n = len(df_yr)
    colors = [cmap_pos(i / max(n-1,1)) for i in range(n)]

    bars = ax.barh(df_yr['행정동'], df_yr['변화율'],
                   color=colors, edgecolor='white', linewidth=0.4, height=0.7)
    for bar, val in zip(bars, df_yr['변화율']):
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f'+{val:.1f}%p', va='center', ha='left',
                fontsize=9, fontweight='bold', color='#1a3a1a')

    ax.set_title(f'{yr}년 (vs {yr-1}년)', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('20대 이동비율 변화 (%p)', fontsize=10)
    ax.set_xlim(0, df_yr['변화율'].max() * 1.25)
    ax.tick_params(axis='y', labelsize=9.5)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig('image/migration_20대_yoy_top15.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: image/migration_20대_yoy_top15.png')

# ── 텍스트 요약 ───────────────────────────────────────────────
print()
for yr in years:
    print(f'=== {yr}년 (vs {yr-1}년) Top 15 ===')
    df_yr = yoy[yoy['연도'] == yr].nlargest(15, '변화율').reset_index(drop=True)
    for i, row in df_yr.iterrows():
        print(f"  {i+1:2d}위  {row['행정동']:<18s}  +{row['변화율']:.1f}%p  (비율: {row['비율']:.1f}%)")
    print()
