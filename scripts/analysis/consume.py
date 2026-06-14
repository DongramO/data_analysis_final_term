import os
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def nfc(s):
    return unicodedata.normalize('NFC', s)

data_dir = 'data'

# ── 행정동 2020~2024 통합 데이터 로드 ────────────────────────────
dong_file = next(f for f in os.listdir(data_dir) if '2020_2024' in nfc(f) and f.endswith('.csv'))
df = pd.read_csv(os.path.join(data_dir, dong_file), encoding='utf-8-sig')
df['year']    = df['기준_년분기_코드'] // 10
df['quarter'] = df['기준_년분기_코드'] % 10

age_cols = [
    '연령대_10_매출_금액', '연령대_20_매출_금액', '연령대_30_매출_금액',
    '연령대_40_매출_금액', '연령대_50_매출_금액', '연령대_60_이상_매출_금액',
]
age_labels = ['10대', '20대', '30대', '40대', '50대', '60대 이상']
AGE_COLORS  = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948']


# ════════════════════════════════════════════════════════════
# 1. 연령대별 소비 비율 + 행정동 Top 30 (2020~2024 통합)
# ════════════════════════════════════════════════════════════
age_totals = df[age_cols].sum()
age_pct    = (age_totals / age_totals.sum() * 100).round(2)

dong_sales = (
    df.groupby('행정동_코드_명')['당월_매출_금액']
    .sum()
    .sort_values(ascending=False)
)
top30 = dong_sales.head(30)

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
fig.suptitle('서울시 소비 데이터 분석 (2020~2024년 통합 추정매출)', fontsize=15, fontweight='bold')

ax = axes[0]
bars = ax.bar(age_labels, age_pct, color=AGE_COLORS, alpha=0.88, edgecolor='white', width=0.6)
for bar, pct in zip(bars, age_pct):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f'{pct:.1f}%',
        ha='center', va='bottom', fontsize=11, fontweight='bold',
    )
ax.set_ylabel('비율 (%)', fontsize=11)
ax.set_title('연령대별 소비 비율 (2020~2024 통합)', fontsize=13)
ax.set_ylim(0, age_pct.max() * 1.18)
ax.grid(True, axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

ax = axes[1]
norm      = plt.Normalize(top30.values.min(), top30.values.max())
bar_colors = [plt.cm.RdYlGn(norm(v)) for v in top30.values[::-1]]
bars_h     = ax.barh(top30.index[::-1], top30.values[::-1] / 1e8,
                     color=bar_colors, alpha=0.88, edgecolor='white')
for bar in bars_h:
    w = bar.get_width()
    ax.text(w + top30.values.max() / 1e8 * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{w:,.0f}억', va='center', ha='left', fontsize=8)
ax.set_xlabel('매출 합계 (억원)', fontsize=11)
ax.set_title('행정동별 매출 순위 Top 30 (2020~2024 통합)', fontsize=13)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
ax.grid(True, axis='x', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
out1 = 'data/consume_분석.png'
plt.savefig(out1, dpi=150, bbox_inches='tight')
print(f"저장 완료: {out1}")

print("\n── 연령대별 소비 비율 (2020~2024 통합) ──")
for label, pct in zip(age_labels, age_pct):
    print(f"  {label}: {pct:.1f}%")

print("\n── 행정동별 매출 Top 10 (2020~2024 통합) ──")
for dong, val in dong_sales.head(10).items():
    print(f"  {dong}: {val/1e8:,.1f}억원")

plt.show()


# ════════════════════════════════════════════════════════════
# 2. 서울시 분기별 매출 비교
# ════════════════════════════════════════════════════════════
seoul_target = next(f for f in os.listdir(data_dir) if nfc(f).endswith('서울시).csv'))
ds = pd.read_csv(os.path.join(data_dir, seoul_target), encoding='utf-8-sig')

ds['year']    = ds['기준_년분기_코드'] // 10
ds['quarter'] = ds['기준_년분기_코드'] % 10
ds['label']   = ds['year'].astype(str) + '-Q' + ds['quarter'].astype(str)

qtr = (
    ds.groupby(['year', 'quarter', 'label'])
    .agg(총매출=('당월_매출_금액', 'sum'), 총건수=('당월_매출_건수', 'sum'))
    .reset_index()
    .sort_values(['year', 'quarter'])
)
qtr['yoy'] = qtr.groupby('quarter')['총매출'].pct_change() * 100
pivot      = qtr.pivot(index='year', columns='quarter', values='총매출') / 1e12

year_min, year_max = qtr['year'].min(), qtr['year'].max()

fig2, axes2 = plt.subplots(2, 2, figsize=(20, 13))
fig2.suptitle(
    f'서울시 추정매출 분기별 비교 ({year_min}Q1 ~ {year_max}Q4)',
    fontsize=15, fontweight='bold',
)
COLORS_Q = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']

ax = axes2[0][0]
ax.plot(qtr['label'], qtr['총매출'] / 1e12, marker='o', linewidth=2,
        color='steelblue', markersize=5)
ax.fill_between(qtr['label'], qtr['총매출'] / 1e12, alpha=0.12, color='steelblue')
cs = qtr[qtr['label'] == '2020-Q1'].index
ce = qtr[qtr['label'] == '2021-Q4'].index
if len(cs) and len(ce):
    ax.axvspan(cs[0], ce[0], alpha=0.1, color='red', label='코로나 기간')
ax.set_xticks(range(len(qtr)))
ax.set_xticklabels(qtr['label'], rotation=45, ha='right', fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}조'))
ax.set_ylabel('매출 합계 (조원)', fontsize=11)
ax.set_title('분기별 총 매출 추이', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

ax = axes2[0][1]
years = sorted(qtr['year'].unique())
x     = range(len(years))
w     = 0.2
for i, q in enumerate([1, 2, 3, 4]):
    vals = [
        qtr.loc[(qtr['year'] == y) & (qtr['quarter'] == q), '총매출'].values[0] / 1e12
        if len(qtr.loc[(qtr['year'] == y) & (qtr['quarter'] == q)]) else 0
        for y in years
    ]
    ax.bar([xi + i * w for xi in x], vals, w, label=f'Q{q}', color=COLORS_Q[i], alpha=0.85)
ax.set_xticks([xi + w * 1.5 for xi in x])
ax.set_xticklabels(years, fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}조'))
ax.set_ylabel('매출 합계 (조원)', fontsize=11)
ax.set_title('연도별 분기 비교', fontsize=12)
ax.legend(title='분기', fontsize=9)
ax.grid(True, axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

ax = axes2[1][0]
hm = ax.imshow(pivot.values, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(4))
ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'], fontsize=11)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=11)
valid_max = np.nanmax(pivot.values)
for i in range(len(pivot.index)):
    for j in range(4):
        v = pivot.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.1f}조', ha='center', va='center',
                    fontsize=9, color='black' if v < valid_max * 0.75 else 'white')
plt.colorbar(hm, ax=ax, label='조원')
ax.set_title('연도×분기 매출 히트맵', fontsize=12)

ax = axes2[1][1]
yoy_data       = qtr.dropna(subset=['yoy'])
bar_colors_yoy = ['#e15759' if v >= 0 else '#4e79a7' for v in yoy_data['yoy']]
ax.bar(yoy_data['label'], yoy_data['yoy'], color=bar_colors_yoy, alpha=0.85)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(range(len(yoy_data)))
ax.set_xticklabels(yoy_data['label'], rotation=45, ha='right', fontsize=8)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.set_ylabel('전년 동기 대비 성장률 (%)', fontsize=11)
ax.set_title('전년 동기 대비 매출 성장률 (YoY)', fontsize=12)
ax.grid(True, axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
out2 = 'data/consume_분기별비교.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
print(f"\n저장 완료: {out2}")
plt.show()


# ════════════════════════════════════════════════════════════
# 3. 서울시 연도별 꺾은선 + 행정동 Top 10 연도별 매출 변화
# ════════════════════════════════════════════════════════════
qtr_pivot     = qtr.pivot(index='quarter', columns='year', values='총매출') / 1e12
quarter_labels = ['Q1', 'Q2', 'Q3', 'Q4']

top10_dong = (
    df.groupby('행정동_코드_명')['당월_매출_금액']
    .sum()
    .nlargest(10)
    .index
    .tolist()
)
dong_annual = (
    df[df['행정동_코드_명'].isin(top10_dong)]
    .groupby(['year', '행정동_코드_명'])['당월_매출_금액']
    .sum()
    .reset_index()
)
dong_year_pivot = dong_annual.pivot(index='year', columns='행정동_코드_명', values='당월_매출_금액') / 1e8
years_str = [str(y) for y in dong_year_pivot.index]

fig3, axes3 = plt.subplots(1, 2, figsize=(22, 8))
fig3.suptitle('서울시 매출 추이 분석 (2020~2024)', fontsize=15, fontweight='bold')

ax = axes3[0]
cmap_yr = plt.colormaps.get_cmap('tab10').resampled(len(qtr_pivot.columns))
for i, yr in enumerate(qtr_pivot.columns):
    vals  = qtr_pivot[yr]
    style = '--' if yr >= 2025 else '-'
    lw    = 2.5 if yr in (2020, 2021) else 1.8
    ax.plot(quarter_labels, vals, marker='o', linestyle=style, linewidth=lw,
            color=cmap_yr(i), label=str(yr), markersize=7)
    last_q = vals.last_valid_index()
    if last_q is not None:
        ax.annotate(f"{vals[last_q]:.1f}조",
                    xy=(quarter_labels[last_q - 1], vals[last_q]),
                    xytext=(6, 0), textcoords='offset points',
                    fontsize=8, color=cmap_yr(i), va='center')
ax.set_xlabel('분기', fontsize=11)
ax.set_ylabel('총 매출 (조원)', fontsize=11)
ax.set_title('연도별 분기 매출 추이 (서울시 전체)', fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}조'))
ax.legend(title='연도', fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

ax = axes3[1]
cmap_dong = plt.colormaps.get_cmap('tab10').resampled(len(top10_dong))
for i, dong in enumerate(dong_year_pivot.columns):
    vals = dong_year_pivot[dong]
    ax.plot(years_str, vals, marker='o', linewidth=2,
            color=cmap_dong(i), label=dong, markersize=7)
    ax.annotate(f"{dong}\n{vals.iloc[-1]:,.0f}억",
                xy=(years_str[-1], vals.iloc[-1]),
                xytext=(6, 0), textcoords='offset points',
                fontsize=7.5, color=cmap_dong(i), va='center')
ax.set_xlabel('연도', fontsize=11)
ax.set_ylabel('매출 합계 (억원)', fontsize=11)
ax.set_title('행정동 Top 10 연도별 매출 변화 (2020~2024)', fontsize=12)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}억'))
ax.legend(title='행정동', fontsize=8, loc='upper left', ncol=2)
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
out3 = 'data/consume_연도별꺾은선.png'
plt.savefig(out3, dpi=150, bbox_inches='tight')
print(f"저장 완료: {out3}")
plt.show()


# ════════════════════════════════════════════════════════════
# 4. 연령대별 소비 변화 추이 (2020~2024)
# ════════════════════════════════════════════════════════════
age_annual     = df.groupby('year')[age_cols].sum()
age_annual_pct = age_annual.div(age_annual.sum(axis=1), axis=0) * 100
age_annual_bil = age_annual / 1e8
yr_labels      = [str(y) for y in age_annual.index]

fig4, axes4 = plt.subplots(1, 2, figsize=(20, 8))
fig4.suptitle('연령대별 소비 변화 추이 (2020~2024)', fontsize=15, fontweight='bold')

ax = axes4[0]
for col, label, color in zip(age_cols, age_labels, AGE_COLORS):
    vals = age_annual_pct[col]
    ax.plot(yr_labels, vals, marker='o', linewidth=2, color=color, label=label, markersize=7)
    ax.annotate(f"{vals.iloc[-1]:.1f}%",
                xy=(yr_labels[-1], vals.iloc[-1]),
                xytext=(6, 0), textcoords='offset points',
                fontsize=8.5, color=color, va='center')
ax.set_xlabel('연도', fontsize=11)
ax.set_ylabel('소비 비율 (%)', fontsize=11)
ax.set_title('연령대별 소비 비율 추이', fontsize=13)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax.legend(title='연령대', fontsize=9)
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

ax = axes4[1]
ax.stackplot(yr_labels,
             [age_annual_bil[col] for col in age_cols],
             labels=age_labels, colors=AGE_COLORS, alpha=0.8)
ax.set_xlabel('연도', fontsize=11)
ax.set_ylabel('매출 합계 (억원)', fontsize=11)
ax.set_title('연령대별 소비 금액 누적 추이', fontsize=13)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}억'))
ax.legend(title='연령대', fontsize=9, loc='upper left')
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
out4 = 'data/consume_연령대별추이.png'
plt.savefig(out4, dpi=150, bbox_inches='tight')
print(f"저장 완료: {out4}")
plt.show()
