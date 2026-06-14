"""
생활이동 기반 20대 이동인구 비율 분석
: Top 15 행정동 × 연도별 / 연령별 / 전년도 대비 변화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 행정동 이름 매핑 구축 ──────────────────────────────────────
# migration 코드 구조: 11 + 구번호(01~25) + 동번호(3자리)
# 구번호 01=종로구, 02=중구, ... 25=강동구 (KT 자체 코드)

def build_name_map():
    kt = pd.read_excel('data/KT_서울생활이동데이터_행정동코드_20210907.xlsx')
    kt.columns = ['시도','시군구','행정동코드','행정동명','전체명']
    return dict(zip(kt['행정동코드'], kt['행정동명'])), dict(zip(kt['행정동코드'], kt['시군구']))

# ── 데이터 로드 ───────────────────────────────────────────────
mig = pd.read_csv('data/main_data/분기별_주말유입인구.csv', encoding='utf-8-sig')
mig_codes = sorted(mig['도착_행정동코드'].unique())
name_map, gu_map = build_name_map()
# 라벨: 자치구 + 행정동명
label_map = {c: f"{gu_map.get(c,'')} {name_map.get(c, str(c))}".strip() for c in mig_codes}

# ── 연간 집계 ─────────────────────────────────────────────────
annual = (
    mig.groupby(['도착_행정동코드', '연도', '연령대'])['합계_이동인구']
    .sum()
    .reset_index()
)
total = annual.groupby(['도착_행정동코드', '연도'])['합계_이동인구'].sum().rename('총_이동인구')
annual = annual.join(total, on=['도착_행정동코드', '연도'])
annual['비율'] = annual['합계_이동인구'] / annual['총_이동인구'] * 100
annual['행정동'] = annual['도착_행정동코드'].map(label_map)

age20 = annual[annual['연령대'] == 20].copy()

# ── Top 15 선정 (5년 평균 20대 비율 기준) ─────────────────────
avg20 = age20.groupby(['도착_행정동코드','행정동'])['비율'].mean().sort_values(ascending=False)
top15_codes = [c for c, _ in avg20.head(15).index]
top15_names = [n for _, n in avg20.head(15).index]

# ── 그림 1: 연도별 Top 15 20대 이동 비율 ──────────────────────
years = sorted(age20['연도'].unique())
fig1, axes = plt.subplots(1, 5, figsize=(24, 7))
fig1.suptitle('연도별 20대 이동인구 비율 Top 15 행정동\n(서울 생활이동, 금·토·일 트렌드 시간대)',
              fontsize=14, fontweight='bold', y=1.01)

cmap = LinearSegmentedColormap.from_list('blue', ['#b3d1f5', '#1a5fa8'])

for ax, yr in zip(axes, years):
    df_yr = (age20[age20['연도'] == yr]
             .nlargest(15, '비율')
             .sort_values('비율', ascending=True))
    n = len(df_yr)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
    bars = ax.barh(df_yr['행정동'], df_yr['비율'],
                   color=colors, edgecolor='white', linewidth=0.4, height=0.7)
    for bar, val in zip(bars, df_yr['비율']):
        ax.text(val + 0.15, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', ha='left', fontsize=8, color='#1a3a6a')
    ax.set_title(f'{yr}년', fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('20대 이동 비율 (%)', fontsize=9)
    ax.set_xlim(0, df_yr['비율'].max() * 1.25)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('image/migration_20대_연도별_top15.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: image/migration_20대_연도별_top15.png')

# ── 그림 2: Top 15 행정동 연령별 구성 히트맵 ─────────────────
age_pivot = (
    annual[annual['도착_행정동코드'].isin(top15_codes)]
    .groupby(['행정동', '연령대'])['비율']
    .mean()
    .unstack('연령대', fill_value=0)
)
age_pivot = age_pivot.reindex(top15_names)

age_labels = {0:'10대미만',10:'10대',20:'20대',30:'30대',40:'40대',
              50:'50대',60:'60대',70:'70대+',80:'80대+'}
age_pivot.columns = [age_labels.get(c, str(c)) for c in age_pivot.columns]

fig2, ax = plt.subplots(figsize=(13, 7))
im = ax.imshow(age_pivot.values, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(age_pivot.columns)))
ax.set_xticklabels(age_pivot.columns, fontsize=10)
ax.set_yticks(range(len(age_pivot.index)))
ax.set_yticklabels(age_pivot.index, fontsize=10)
for i in range(len(age_pivot.index)):
    for j in range(len(age_pivot.columns)):
        val = age_pivot.values[i, j]
        ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                fontsize=8.5, color='black' if val < 20 else 'white')
plt.colorbar(im, ax=ax, label='이동 비율 (%)')
ax.set_title('Top 15 행정동 연령대별 이동인구 구성 비율 (5년 평균)\n(서울 생활이동, 금·토·일 트렌드 시간대)',
             fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('image/migration_20대_연령별_구성.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: image/migration_20대_연령별_구성.png')

# ── 그림 3: 전년도 대비 20대 이동 비율 변화 ──────────────────
age20_top = age20[age20['도착_행정동코드'].isin(top15_codes)].copy()
age20_top = age20_top.sort_values(['도착_행정동코드', '연도'])
age20_top['전년비율'] = age20_top.groupby('도착_행정동코드')['비율'].shift(1)
age20_top['변화'] = age20_top['비율'] - age20_top['전년비율']
yoy = age20_top.dropna(subset=['변화'])

yoy_years = sorted(yoy['연도'].unique())
fig3, axes = plt.subplots(1, len(yoy_years), figsize=(22, 6), sharey=True)
fig3.suptitle('Top 15 행정동 전년도 대비 20대 이동 비율 변화 (%p)\n(서울 생활이동, 금·토·일 트렌드 시간대)',
              fontsize=13, fontweight='bold', y=1.01)

for ax, yr in zip(axes, yoy_years):
    df_yr = yoy[yoy['연도'] == yr].set_index('행정동')
    df_yr = df_yr.reindex(top15_names)
    vals = df_yr['변화'].values
    colors = ['#e05c5c' if (v < 0 if not np.isnan(v) else False) else '#2d7dd2' for v in vals]
    ax.barh(top15_names, vals, color=colors, edgecolor='white', linewidth=0.4, height=0.7)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    for i, val in enumerate(vals):
        if not np.isnan(val):
            ha = 'left' if val >= 0 else 'right'
            ax.text(val + (0.05 if val >= 0 else -0.05), i,
                    f'{val:+.1f}', va='center', ha=ha, fontsize=8)
    ax.set_title(f'{yr}년 (vs {yr-1}년)', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('변화량 (%p)', fontsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('image/migration_20대_전년대비_변화.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: image/migration_20대_전년대비_변화.png')

# ── 텍스트 요약 ───────────────────────────────────────────────
print('\n=== 5년 평균 20대 이동 비율 Top 15 ===')
for rank, ((code, name), val) in enumerate(zip(avg20.head(15).index, avg20.head(15).values), 1):
    print(f'  {rank:2d}위  {name}({code})  {val:.1f}%')
