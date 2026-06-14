# -*- coding: utf-8 -*-
"""
트렌드 상권 Top 15: 연도별 매출 변화 vs 네이버 검색 트렌드 비교
- 두 지표 모두 2020 = 100 기준 인덱스로 표준화해 방향성 비교
- 매출 지표: 당월_매출_금액 총합, 2030비중
- 네이버 지표: monthly ratio 연도별 평균
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as sp_stats
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

data_dir = 'data'
image_dir = 'image'

# ── 네이버 그룹 → 행정동 매핑 ────────────────────────────────
NAVER_TO_DONG = {
    '연남동':    ['연남동'],
    '이태원':    ['이태원1동', '이태원2동'],
    '홍대':      ['서교동'],
    '삼청동':    ['삼청동'],
    '화양동':    ['화양동'],
    '서촌':      ['서원동'],
    '한남동':    ['한남동'],
    '북촌':      ['가회동'],
    '합정망원':  ['합정동', '망원1동', '망원2동'],
    '대학로':    ['혜화동'],
    '잠실':      ['잠실6동'],
    '낙성대':    ['낙성대동'],
    '방배':      ['방배2동'],
    '상봉':      ['상봉1동'],
    '용산해방촌': ['후암동', '용산2가동'],
}
DONG_TO_NAVER = {d: g for g, dongs in NAVER_TO_DONG.items() for d in dongs}
ALL_DONGS = [d for dongs in NAVER_TO_DONG.values() for d in dongs]

# ════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ════════════════════════════════════════════════════════════
df_sales = pd.read_csv(
    os.path.join(data_dir, '서울시_상권분석서비스(추정매출-행정동)_2020_2024.csv'),
    encoding='utf-8-sig',
)
df_sales['year'] = df_sales['기준_년분기_코드'] // 10
df_sales['naver_group'] = df_sales['행정동_코드_명'].map(DONG_TO_NAVER)
df_mapped = df_sales[df_sales['naver_group'].notna()].copy()

nv = pd.read_csv(
    os.path.join(data_dir, '네이버트렌드_트렌드상권_20260517.csv'),
    encoding='utf-8-sig',
)
nv['period'] = pd.to_datetime(nv['period'])
nv['year'] = nv['period'].dt.year

# ════════════════════════════════════════════════════════════
# 2. 연도별 집계
# ════════════════════════════════════════════════════════════
# 매출: 그룹별 연도별 총매출 + 2030비중
sales_annual = (
    df_mapped.groupby(['naver_group', 'year'])
    .agg(매출=('당월_매출_금액', 'sum'),
         g20=('연령대_20_매출_금액', 'sum'),
         g30=('연령대_30_매출_금액', 'sum'))
    .reset_index()
)
sales_annual['2030비중'] = (sales_annual['g20'] + sales_annual['g30']) / sales_annual['매출']

# 네이버: 그룹별 연도별 ratio 평균
naver_annual = nv.groupby(['group', 'year'])['ratio'].mean().reset_index()
naver_annual.columns = ['naver_group', 'year', 'naver_ratio']

# 2020 기준 인덱스 변환
def to_index(df, group_col, year_col, val_col, base_year=2020):
    base = df[df[year_col] == base_year].set_index(group_col)[val_col]
    result = df.copy()
    result['index'] = result.apply(
        lambda r: r[val_col] / base.get(r[group_col], np.nan) * 100, axis=1
    )
    return result

sales_idx  = to_index(sales_annual, 'naver_group', 'year', '매출')
gen2030_idx = to_index(sales_annual, 'naver_group', 'year', '2030비중')
naver_idx  = to_index(naver_annual, 'naver_group', 'year', 'naver_ratio')

# 병합
merged = (
    sales_idx[['naver_group','year','index']].rename(columns={'index':'매출_idx'})
    .merge(gen2030_idx[['naver_group','year','index']].rename(columns={'index':'2030_idx'}),
           on=['naver_group','year'])
    .merge(naver_idx[['naver_group','year','index']].rename(columns={'index':'naver_idx'}),
           on=['naver_group','year'])
)

# 2020→2024 변화량 (인덱스 기준)
change = (
    merged.groupby('naver_group')
    .apply(lambda g: pd.Series({
        '매출_변화':  g.loc[g['year']==2024,'매출_idx'].values[0] - 100 if len(g[g['year']==2024])>0 else np.nan,
        '2030_변화': g.loc[g['year']==2024,'2030_idx'].values[0] - 100 if len(g[g['year']==2024])>0 else np.nan,
        '네이버_변화': g.loc[g['year']==2024,'naver_idx'].values[0] - 100 if len(g[g['year']==2024])>0 else np.nan,
    }))
    .dropna()
)

print("=== 2020→2024 변화량 (인덱스 기준, 2020=100) ===")
print(change.round(1).sort_values('네이버_변화', ascending=False).to_string())

# 상관계수
rho1, p1 = sp_stats.spearmanr(change['네이버_변화'], change['매출_변화'])
rho2, p2 = sp_stats.spearmanr(change['네이버_변화'], change['2030_변화'])
print(f"\n네이버변화 vs 매출변화  스피어만 ρ={rho1:.3f} p={p1:.3f}")
print(f"네이버변화 vs 2030변화 스피어만 ρ={rho2:.3f} p={p2:.3f}")


# ════════════════════════════════════════════════════════════
# 시각화 1. 그룹별 3개 지표 인덱스 시계열 (5×3 = 15개)
# ════════════════════════════════════════════════════════════
groups = list(NAVER_TO_DONG.keys())
fig, axes = plt.subplots(5, 3, figsize=(20, 22))
fig.suptitle('트렌드 상권 Top 15: 연도별 매출 vs 2030비중 vs 네이버 트렌드\n(2020 = 100 기준 인덱스)',
             fontsize=14, fontweight='bold')
axes = axes.flatten()

years_str = ['2020', '2021', '2022', '2023', '2024']

for idx, grp in enumerate(groups):
    ax = axes[idx]
    sub = merged[merged['naver_group'] == grp].sort_values('year')

    ax.plot(sub['year'].astype(str), sub['매출_idx'],
            color='#4e79a7', marker='o', linewidth=2, markersize=6, label='매출')
    ax.plot(sub['year'].astype(str), sub['2030_idx'],
            color='#59a14f', marker='s', linewidth=2, markersize=6, label='2030비중', linestyle='--')
    ax.plot(sub['year'].astype(str), sub['naver_idx'],
            color='#e15759', marker='^', linewidth=2, markersize=7, label='네이버 트렌드', linestyle=':')

    ax.axhline(100, color='gray', linewidth=0.8, linestyle='-', alpha=0.5)
    ax.set_title(grp, fontsize=12, fontweight='bold')
    ax.set_ylabel('인덱스 (2020=100)', fontsize=8)
    ax.set_ylim(30, None)
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', labelsize=8)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'trend_naver_timeseries.png'), dpi=150, bbox_inches='tight')
print("\n저장: image/trend_naver_timeseries.png")
plt.show()


# ════════════════════════════════════════════════════════════
# 시각화 2. 2020→2024 변화량 3개 지표 비교 막대
# ════════════════════════════════════════════════════════════
change_sorted = change.sort_values('네이버_변화', ascending=False)
x = np.arange(len(change_sorted))
w = 0.26

fig2, ax = plt.subplots(figsize=(18, 7))
fig2.suptitle('트렌드 상권 Top 15: 2020→2024 변화량 비교\n(매출 / 2030비중 / 네이버 트렌드)',
              fontsize=14, fontweight='bold')

bars1 = ax.bar(x - w, change_sorted['네이버_변화'], w, color='#e15759', alpha=0.85, label='네이버 트렌드 변화')
bars2 = ax.bar(x,     change_sorted['매출_변화'],   w, color='#4e79a7', alpha=0.85, label='매출 변화')
bars3 = ax.bar(x + w, change_sorted['2030_변화'],   w, color='#59a14f', alpha=0.85, label='2030비중 변화')

for bar in [*bars1, *bars2, *bars3]:
    h = bar.get_height()
    if abs(h) > 2:
        ax.text(bar.get_x() + bar.get_width()/2,
                h + (1.5 if h >= 0 else -4),
                f'{h:+.0f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(change_sorted.index, fontsize=10, rotation=20, ha='right')
ax.set_ylabel('변화량 (인덱스 포인트, 2020=100 기준)', fontsize=10)
ax.legend(fontsize=10)
ax.grid(True, axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'trend_naver_change_bar.png'), dpi=150, bbox_inches='tight')
print("저장: image/trend_naver_change_bar.png")
plt.show()


# ════════════════════════════════════════════════════════════
# 시각화 3. 네이버 변화 vs 매출 변화 산점도 (방향성 일치 여부)
# ════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))
fig3.suptitle('네이버 트렌드 변화 vs 매출·2030비중 변화 (2020→2024)\n방향이 일치하면 같은 분면에 위치',
              fontsize=13, fontweight='bold')

for ax, ycol, ylabel, rho, pv, color in [
    (axes3[0], '매출_변화',  '매출 변화량 (인덱스 포인트)', rho1, p1, '#4e79a7'),
    (axes3[1], '2030_변화', '2030비중 변화량 (인덱스 포인트)', rho2, p2, '#59a14f'),
]:
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    # 4분면 색상 배경
    ax.axhspan(0, change[ycol].max()*1.2+20, xmin=0.5, alpha=0.04, color='green')
    ax.axhspan(change[ycol].min()*1.2-20, 0, xmax=0.5, alpha=0.04, color='green')

    ax.scatter(change['네이버_변화'], change[ycol],
               color=color, s=90, alpha=0.8, edgecolors='white', linewidths=0.8, zorder=3)
    for grp, row in change.iterrows():
        ax.annotate(grp, (row['네이버_변화'], row[ycol]),
                    fontsize=8.5, xytext=(5, 3), textcoords='offset points')

    # 회귀선
    xv = change['네이버_변화'].values
    yv = change[ycol].values
    z  = np.polyfit(xv, yv, 1)
    xr = np.linspace(xv.min()-5, xv.max()+5, 100)
    ax.plot(xr, np.poly1d(z)(xr), color='gray', linewidth=1.5, linestyle='-',
            label=f'ρ={rho:.3f} (p={pv:.2f})')

    ax.set_xlabel('네이버 트렌드 변화량 (인덱스 포인트)', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f'네이버 vs {ylabel[:5]}\n스피어만 ρ={rho:.3f} (p={pv:.3f})', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    # 분면 레이블
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.text(xlim[1]*0.7, ylim[1]*0.9, '네이버↑\n매출↑', fontsize=8, color='#27ae60', ha='center')
    ax.text(xlim[0]*0.7, ylim[0]*0.9, '네이버↓\n매출↓', fontsize=8, color='#27ae60', ha='center')
    ax.text(xlim[1]*0.7, ylim[0]*0.9, '네이버↑\n매출↓', fontsize=8, color='#c0392b', ha='center')
    ax.text(xlim[0]*0.7, ylim[1]*0.9, '네이버↓\n매출↑', fontsize=8, color='#c0392b', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'trend_naver_scatter.png'), dpi=150, bbox_inches='tight')
print("저장: image/trend_naver_scatter.png")
plt.show()

print("\n완료!")
