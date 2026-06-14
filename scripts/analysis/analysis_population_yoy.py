import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 ──────────────────────────────────────────────
pop = pd.read_csv('data/population_quarterly.csv', encoding='utf-8-sig')
raw = pd.read_csv('data/main_data/분기별_원본피처.csv', encoding='utf-8-sig',
                  usecols=[0, 1])  # 행정동_코드, 행정동
raw.columns = ['행정동코드', '행정동']
dong_map = raw.drop_duplicates('행정동코드').set_index('행정동코드')['행정동']

# ── 연간 집계 ─────────────────────────────────────────────────
annual = (
    pop.groupby(['행정동코드', '연도'])['20대']
    .sum()
    .reset_index()
    .rename(columns={'20대': '총_유동인구'})
)
annual['행정동'] = annual['행정동코드'].map(dong_map).fillna(annual['행정동코드'].astype(str))

# ── 전년도 대비 변화율 계산 ────────────────────────────────────
annual = annual.sort_values(['행정동코드', '연도'])
annual['전년_유동인구'] = annual.groupby('행정동코드')['총_유동인구'].shift(1)
annual['변화율'] = (annual['총_유동인구'] - annual['전년_유동인구']) / annual['전년_유동인구'] * 100
annual = annual.dropna(subset=['변화율'])

# ── 연도별 Top 15 ──────────────────────────────────────────────
years = sorted(annual['연도'].unique())  # 2021, 2022, 2023, 2024
top15 = {}
for yr in years:
    mask = annual['연도'] == yr
    if yr == 2024:
        mask &= annual['행정동코드'] != 11680660  # 동일1동 이상치 제외
    top15[yr] = (
        annual[mask]
        .nlargest(15, '변화율')
        .reset_index(drop=True)
    )
    top15[yr].index = top15[yr].index + 1  # 1위부터 표시

# ── 순위 테이블 출력 ─────────────────────────────────────────
print('=== 연도별 유동인구 변화율 Top 15 ===')
for yr in years:
    print(f'\n[{yr}년] (전년 대비)')
    df_yr = top15[yr].reset_index()
    df_yr.columns = ['순위'] + list(df_yr.columns[1:])
    for _, row in df_yr.iterrows():
        print(f"  {int(row['순위']):2d}위  {row['행정동']:<12s}  +{row['변화율']:.1f}%")

# ── 시각화 ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 20))
fig.suptitle('연도별 전년 대비 20대 유동인구 변화율 Top 15 행정동\n(서울 생활인구, 분기합산 연간 기준)',
             fontsize=16, fontweight='bold', y=0.98)

# 색상 팔레트: 변화율 크기에 따른 그라데이션
cmap = LinearSegmentedColormap.from_list('rate', ['#a8d8a8', '#2d6a2d'])

for ax, yr in zip(axes.flatten(), years):
    df = top15[yr].copy()
    df = df.sort_values('변화율', ascending=True)  # 높은 순이 위로

    n = len(df)
    colors = [cmap(i / (n - 1)) for i in range(n)]

    bars = ax.barh(df['행정동'], df['변화율'], color=colors, edgecolor='white',
                   linewidth=0.5, height=0.7)

    # 수치 레이블
    for bar, val in zip(bars, df['변화율']):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f'+{val:.1f}%', va='center', ha='left',
                fontsize=9.5, fontweight='bold', color='#1a4a1a')

    ax.set_title(f'{yr}년 (전년 대비 변화율)', fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('20대 유동인구 변화율 (%)', fontsize=10)
    ax.set_xlim(0, df['변화율'].max() * 1.22)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

    # 순위 배지
    for i, (_, row) in enumerate(df.iterrows()):
        rank = 15 - i
        ax.text(-1.5, i, f'{rank}', va='center', ha='right',
                fontsize=8, color='gray', fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('image/population_yoy_20대_top15.png', dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()
print('\n저장 완료: image/population_yoy_20대_top15.png')
