# -*- coding: utf-8 -*-
"""
네이버/블로그 트렌드 vs 실제 매출 상관관계 분석 - 최종 시각화
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import unicodedata
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from scipy import stats
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def nfc(s):
    return unicodedata.normalize('NFC', s)

DATA_DIR = r'C:\Users\30032\OneDrive\바탕 화면\workspace\data_analytics_final\data'

# ── 색상 팔레트 ─────────────────────────────────────────────
C_BLUE   = '#4e79a7'
C_ORANGE = '#f28e2b'
C_RED    = '#e15759'
C_GREEN  = '#59a14f'
C_TEAL   = '#76b7b2'
C_PURPLE = '#b07aa1'
C_GRAY   = '#bab0ac'

# ══════════════════════════════════════════════════════════════
# 0. 데이터 로드 & 전처리
# ══════════════════════════════════════════════════════════════
print("데이터 로딩...")
dong_file = next(f for f in os.listdir(DATA_DIR) if '2020_2024' in nfc(f) and f.endswith('.csv'))
df = pd.read_csv(os.path.join(DATA_DIR, dong_file), encoding='utf-8-sig')
df['year']    = df['기준_년분기_코드'] // 10
df['quarter'] = df['기준_년분기_코드'] % 10

naver_adj = pd.read_csv(os.path.join(DATA_DIR, 'naver_trend_행정동보정_20260510.csv'), encoding='utf-8-sig')
naver_raw = pd.read_csv(os.path.join(DATA_DIR, 'naver_trend_행정동_20260510.csv'), encoding='utf-8-sig')
blog      = pd.read_csv(os.path.join(DATA_DIR, '블로그트렌드_상권.csv'), encoding='utf-8-sig')

# 분기별 집계
dong_qtr = (
    df.groupby(['기준_년분기_코드', 'year', 'quarter', '행정동_코드_명'])
    .agg(
        매출=('당월_매출_금액', 'sum'),
        gen20=('연령대_20_매출_금액', 'sum'),
        gen30=('연령대_30_매출_금액', 'sum'),
        gen40=('연령대_40_매출_금액', 'sum'),
    )
    .reset_index()
)
dong_qtr['gen2030'] = dong_qtr['gen20'] + dong_qtr['gen30']
dong_qtr['gen2030_ratio'] = dong_qtr['gen2030'] / dong_qtr['매출'].replace(0, np.nan)

# 네이버 분기 평균
naver = naver_adj.copy()
naver['period'] = pd.to_datetime(naver['period'])
naver['year']   = naver['period'].dt.year
naver['quarter']= naver['period'].dt.quarter
naver_qtr = (
    naver.groupby(['group', 'year', 'quarter'])
    ['adjusted_ratio'].mean()
    .reset_index(name='naver_adj')
)
naver_qtr['기준_년분기_코드'] = naver_qtr['year'] * 10 + naver_qtr['quarter']

dong_names = sorted(df['행정동_코드_명'].unique())
naver_groups = sorted(naver_qtr['group'].unique())

mapping = {}
for g in naver_groups:
    matches = [d for d in dong_names if g in d or d in g]
    if matches:
        mapping[g] = sorted(matches, key=len)[0]

# ── 상관관계 계산 ──
results = []
for naver_g, dong_name in mapping.items():
    s_ts = dong_qtr[dong_qtr['행정동_코드_명'] == dong_name].sort_values('기준_년분기_코드')
    n_ts = naver_qtr[naver_qtr['group'] == naver_g].sort_values('기준_년분기_코드')
    mg   = pd.merge(s_ts, n_ts, on='기준_년분기_코드', how='inner')
    if len(mg) < 6:
        continue

    r_sync, p_sync = stats.pearsonr(mg['naver_adj'], mg['매출'])
    r_2030, _      = stats.pearsonr(mg['naver_adj'], mg['gen2030_ratio'].fillna(mg['gen2030_ratio'].mean()))

    x = (mg['naver_adj'] - mg['naver_adj'].mean()) / (mg['naver_adj'].std() + 1e-9)
    y = (mg['매출']     - mg['매출'].mean())      / (mg['매출'].std()     + 1e-9)
    n = len(x)
    best_lag, best_r = 0, r_sync
    for lag in range(-3, 4):
        if lag == 0:
            continue
        if lag > 0:
            xi, yi = x.values[:n-lag], y.values[lag:]
        else:
            xi, yi = x.values[-lag:], y.values[:n+lag]
        if len(xi) < 4:
            continue
        rl, _ = stats.pearsonr(xi, yi)
        if abs(rl) > abs(best_r):
            best_lag, best_r = lag, rl

    results.append({
        'dong': naver_g, 'dong_full': dong_name,
        'r_sync': round(r_sync, 3), 'p_sync': round(p_sync, 3),
        'best_lag': best_lag, 'best_r': round(best_r, 3),
        'r_2030': round(r_2030, 3), 'n_obs': len(mg),
        'merged': mg,
    })

results_df_raw = pd.DataFrame(results)
results_df = results_df_raw.drop(columns=['merged']).sort_values('r_sync', ascending=False).reset_index(drop=True)
merged_dict = {row['dong']: row['merged'] for _, row in results_df_raw.iterrows()}

# ── 블로그 결합 ──
dong_annual_22 = df[df['year']==2022].groupby('행정동_코드_명')['당월_매출_금액'].sum()
dong_annual_24 = df[df['year']==2024].groupby('행정동_코드_명')['당월_매출_금액'].sum()
blog_mapping = {}
for area in blog['area'].unique():
    area_nfc = nfc(str(area))
    matches = [d for d in dong_names if area_nfc in d or d in area_nfc]
    if matches:
        blog_mapping[area_nfc] = sorted(matches, key=len)[0]

blog_list = []
for _, row in blog.iterrows():
    a = nfc(str(row['area']))
    if a not in blog_mapping:
        continue
    d = blog_mapping[a]
    s22, s24 = dong_annual_22.get(d, np.nan), dong_annual_24.get(d, np.nan)
    sg = (s24-s22)/s22*100 if pd.notna(s22) and s22 > 0 else np.nan
    bg = (row['2024']-row['2022'])/row['2022']*100
    blog_list.append({
        'area': a, 'dong': d,
        'blog_g_22_24': bg,
        'blog_g_22_25': row['growth_22_25'],
        'sales_g_22_24': round(sg, 2) if pd.notna(sg) else np.nan,
        'total_vol': row['total_vol'],
        'blog_2022': row['2022'], 'blog_2024': row['2024'],
    })

blog_df = pd.DataFrame(blog_list).dropna(subset=['sales_g_22_24'])
r_blog, p_blog = stats.pearsonr(blog_df['blog_g_22_24'], blog_df['sales_g_22_24'])

# 2030 집중도
gen2030_top = (
    dong_qtr[dong_qtr['year'].between(2022,2024)]
    .groupby('행정동_코드_명')
    .agg(총매출=('매출','sum'), gen2030합=('gen2030','sum'))
    .reset_index()
)
gen2030_top['r2030'] = gen2030_top['gen2030합'] / gen2030_top['총매출']
gen2030_top = gen2030_top[gen2030_top['총매출'] > 1e10]

# ══════════════════════════════════════════════════════════════
# FIGURE 1: 네이버 트렌드 vs 매출 상관관계 종합 대시보드
# ══════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(24, 16))
fig1.suptitle(
    '네이버 검색 트렌드(보정) vs 실제 매출 상관관계 분석\n(분석 기간: 2022Q1~2024Q4, 23개 행정동)',
    fontsize=16, fontweight='bold', y=0.98
)

gs = fig1.add_gridspec(2, 3, hspace=0.45, wspace=0.38)
ax_bar   = fig1.add_subplot(gs[0, 0])
ax_lag   = fig1.add_subplot(gs[0, 1])
ax_lag_d = fig1.add_subplot(gs[0, 2])
ax_ts    = fig1.add_subplot(gs[1, 0:2])
ax_2030  = fig1.add_subplot(gs[1, 2])

# ── 패널 1: 동시점 상관계수 막대 ──
rd = results_df.sort_values('r_sync')
bar_c = []
for r in rd['r_sync']:
    if r >= 0.5:   bar_c.append(C_GREEN)
    elif r >= 0.3: bar_c.append(C_TEAL)
    elif r >= 0:   bar_c.append(C_BLUE)
    elif r >= -0.3:bar_c.append(C_ORANGE)
    else:          bar_c.append(C_RED)

bars = ax_bar.barh(rd['dong'], rd['r_sync'], color=bar_c, alpha=0.88,
                   edgecolor='white', height=0.7)
ax_bar.axvline(0, color='#333', linewidth=1)
ax_bar.axvline(0.5, color=C_GREEN, linewidth=1.2, linestyle='--', alpha=0.6)
ax_bar.axvline(-0.5, color=C_RED, linewidth=1.2, linestyle='--', alpha=0.6)
for bar, r_v, p_v in zip(bars, rd['r_sync'], rd['p_sync']):
    sig = '**' if p_v < 0.05 else ('*' if p_v < 0.1 else '')
    offset = 0.015 if r_v >= 0 else -0.015
    ha = 'left' if r_v >= 0 else 'right'
    ax_bar.text(r_v + offset, bar.get_y() + bar.get_height()/2,
                f'{r_v:.2f}{sig}', va='center', ha=ha, fontsize=8)
ax_bar.set_xlabel('Pearson r (동시점 상관계수)', fontsize=10)
ax_bar.set_title('행정동별 네이버 트렌드-매출\n동시점 상관계수', fontsize=11, fontweight='bold')
ax_bar.set_xlim(-0.85, 0.9)
ax_bar.grid(True, axis='x', alpha=0.3)
ax_bar.spines[['top','right']].set_visible(False)
legend_patches = [
    mpatches.Patch(color=C_GREEN,  label='강양상관 (r≥0.5)'),
    mpatches.Patch(color=C_TEAL,   label='약양상관 (0.3≤r<0.5)'),
    mpatches.Patch(color=C_BLUE,   label='미약양상관 (0≤r<0.3)'),
    mpatches.Patch(color=C_ORANGE, label='약음상관 (-0.3<r<0)'),
    mpatches.Patch(color=C_RED,    label='강음상관 (r≤-0.3)'),
]
ax_bar.legend(handles=legend_patches, fontsize=7, loc='lower right')
ax_bar.text(-0.82, 0, '* p<0.1  ** p<0.05', fontsize=7.5, color='gray',
            va='bottom', style='italic')

# ── 패널 2: 선행-후행 유형 파이+막대 ──
lead = results_df[results_df['best_lag'] > 0]
sync = results_df[results_df['best_lag'] == 0]
lag_behind = results_df[results_df['best_lag'] < 0]
counts = [len(lead), len(sync), len(lag_behind)]
labels_pie = [f'선행\n(네이버→매출)\n{counts[0]}개',
              f'동행\n(동시)\n{counts[1]}개',
              f'후행\n(매출→네이버)\n{counts[2]}개']
colors_pie = [C_GREEN, C_BLUE, C_ORANGE]
wedges, texts, autotexts = ax_lag.pie(
    counts, labels=labels_pie, colors=colors_pie,
    autopct='%1.0f%%', startangle=90,
    textprops={'fontsize': 9},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2},
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight('bold')
ax_lag.set_title('네이버 트렌드의 매출 대비\n시차 유형 분포', fontsize=11, fontweight='bold')

# ── 패널 3: 최적 시차 막대 ──
lag_rd = results_df.sort_values('best_lag')
lag_bar_c = [C_GREEN if l > 0 else (C_ORANGE if l < 0 else C_BLUE) for l in lag_rd['best_lag']]
bars3 = ax_lag_d.barh(lag_rd['dong'], lag_rd['best_lag'],
                      color=lag_bar_c, alpha=0.85, edgecolor='white', height=0.7)
ax_lag_d.axvline(0, color='#333', linewidth=1)
for bar, lag_v, br in zip(bars3, lag_rd['best_lag'], lag_rd['best_r']):
    if lag_v != 0:
        offset = 0.08 if lag_v >= 0 else -0.08
        ha = 'left' if lag_v >= 0 else 'right'
        ax_lag_d.text(lag_v + offset, bar.get_y() + bar.get_height()/2,
                      f'{lag_v:+d}Q (r={br:.2f})', va='center', ha=ha, fontsize=8)
ax_lag_d.set_xlabel('최적 시차 (분기, + = 네이버 선행)', fontsize=10)
ax_lag_d.set_title('행정동별 최적 선행-후행 시차\n(양수=네이버가 매출 앞섬)', fontsize=11, fontweight='bold')
ax_lag_d.set_xlim(-4.5, 4.5)
ax_lag_d.grid(True, axis='x', alpha=0.3)
ax_lag_d.spines[['top','right']].set_visible(False)

# ── 패널 4: 상관관계 상위 4개 행정동 시계열 ──
top4 = results_df.nlargest(4, 'r_sync')
colors_ts = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE]
x_labels_set = False

for i, (_, row) in enumerate(top4.iterrows()):
    naver_g = row['dong']
    mg = merged_dict[naver_g]
    n_pts = len(mg)
    xi = list(range(n_pts))
    labels_x = [str(c) for c in mg['기준_년분기_코드']]

    # Z-score 정규화
    s_z = (mg['매출'].values - mg['매출'].mean()) / (mg['매출'].std() + 1e-9)
    n_z = (mg['naver_adj'].values - mg['naver_adj'].mean()) / (mg['naver_adj'].std() + 1e-9)

    ax_ts.plot(xi, s_z, '-o', color=colors_ts[i], linewidth=2.2, markersize=5,
               label=f"{naver_g} 매출 (r={row['r_sync']:.2f})", alpha=0.95)
    ax_ts.plot(xi, n_z, '--', color=colors_ts[i], linewidth=1.5,
               label=f"{naver_g} 네이버", alpha=0.55)
    if not x_labels_set and n_pts > 0:
        ax_ts.set_xticks(xi)
        ax_ts.set_xticklabels(labels_x, rotation=45, ha='right', fontsize=8)
        x_labels_set = True

ax_ts.set_ylabel('표준화 값 (Z-score)', fontsize=10)
ax_ts.set_title(
    '상관계수 상위 4개 행정동 시계열 비교\n실선=실제매출, 점선=네이버트렌드 (모두 Z-score 정규화)',
    fontsize=11, fontweight='bold'
)
ax_ts.legend(fontsize=8, loc='upper left', ncol=2,
             framealpha=0.85, edgecolor=C_GRAY)
ax_ts.axhline(0, color='gray', linewidth=0.7, linestyle=':')
ax_ts.grid(True, alpha=0.25)
ax_ts.spines[['top','right']].set_visible(False)

# ── 패널 5: 2030 비중 vs 네이버 상관계수 scatter ──
gc_data = results_df[['dong','r_sync','r_2030']].copy()
sc5 = ax_2030.scatter(gc_data['r_sync'], gc_data['r_2030'],
                      s=90, c=gc_data['r_sync'], cmap='RdYlGn',
                      vmin=-0.7, vmax=0.7, edgecolors='white',
                      linewidth=0.8, alpha=0.88, zorder=3)
plt.colorbar(sc5, ax=ax_2030, label='전체매출 상관계수')
for _, r in gc_data.iterrows():
    ax_2030.annotate(r['dong'], (r['r_sync'], r['r_2030']),
                     fontsize=7.5, ha='center', va='bottom',
                     xytext=(0, 4), textcoords='offset points')
ax_2030.axhline(0, color='gray', linewidth=0.7, linestyle=':')
ax_2030.axvline(0, color='gray', linewidth=0.7, linestyle=':')
r_meta, p_meta = stats.pearsonr(gc_data['r_sync'], gc_data['r_2030'])
ax_2030.set_xlabel('네이버-전체매출 상관계수', fontsize=10)
ax_2030.set_ylabel('네이버-2030비중 상관계수', fontsize=10)
ax_2030.set_title(f'전체매출 vs 2030비중 상관계수 비교\n(두 지표 간 상관: r={r_meta:.2f})',
                  fontsize=11, fontweight='bold')
ax_2030.grid(True, alpha=0.25)
ax_2030.spines[['top','right']].set_visible(False)

plt.savefig(os.path.join(DATA_DIR, 'trend_correlation_analysis.png'), dpi=150, bbox_inches='tight')
print("저장: trend_correlation_analysis.png")
plt.close()

# ══════════════════════════════════════════════════════════════
# FIGURE 2: 행정동 개별 시계열 비교 그리드 (선행/동행/후행 구분)
# ══════════════════════════════════════════════════════════════
# 상관계수 기준 상위 + 하위 포함, 최대 12개 선택
top_n = results_df_raw.sort_values('r_sync', ascending=False).head(6)
bot_n = results_df_raw.sort_values('r_sync', ascending=True).head(6)
sel_dongs = pd.concat([top_n, bot_n]).drop_duplicates('dong')

n_plots = len(sel_dongs)
ncols = 4
nrows = int(np.ceil(n_plots / ncols))

fig2, axes2 = plt.subplots(nrows, ncols, figsize=(24, nrows * 5.2))
fig2.suptitle(
    '행정동별 네이버 검색 트렌드(보정) vs 실제 매출 시계열 비교\n'
    '(상위 6개: 강 양의 상관, 하위 6개: 음의 상관)',
    fontsize=15, fontweight='bold'
)
axes2_flat = axes2.flatten()

for idx, (_, row) in enumerate(sel_dongs.iterrows()):
    ax = axes2_flat[idx]
    naver_g  = row['dong']
    dong_n   = row['dong_full']
    r_val    = row['r_sync']
    best_lag = row['best_lag']
    mg       = row['merged']

    n_pts   = len(mg)
    xi      = list(range(n_pts))
    labels_x = [str(c) for c in mg['기준_년분기_코드']]

    ax2 = ax.twinx()

    # 매출 (억원) - 파란색
    ax.fill_between(xi, mg['매출'].values / 1e8, alpha=0.18, color=C_BLUE)
    line1, = ax.plot(xi, mg['매출'].values / 1e8, '-o', color=C_BLUE, linewidth=2.2,
                     markersize=5, label='매출(억원)', zorder=3)
    ax.set_ylabel('매출 (억원)', color=C_BLUE, fontsize=8)
    ax.tick_params(axis='y', labelcolor=C_BLUE, labelsize=7)

    # 네이버 트렌드 - 주황색
    line2, = ax2.plot(xi, mg['naver_adj'].values, '--s', color=C_ORANGE, linewidth=2,
                      markersize=4, label='네이버 트렌드', alpha=0.85, zorder=2)
    ax2.set_ylabel('네이버 지수', color=C_ORANGE, fontsize=8)
    ax2.tick_params(axis='y', labelcolor=C_ORANGE, labelsize=7)

    # 시차 표시
    if best_lag > 0:
        lag_str = f'네이버 {best_lag}Q 선행'
        lag_color = C_GREEN
    elif best_lag < 0:
        lag_str = f'매출 {abs(best_lag)}Q 선행'
        lag_color = C_RED
    else:
        lag_str = '동행'
        lag_color = C_BLUE

    # r값 색상
    if abs(r_val) >= 0.5:
        r_color = C_GREEN if r_val > 0 else C_RED
    elif abs(r_val) >= 0.3:
        r_color = C_TEAL if r_val > 0 else C_ORANGE
    else:
        r_color = C_GRAY

    ax.set_title(f'{naver_g}  |  r={r_val:+.2f}  |  {lag_str}',
                 fontsize=10, color=r_color, fontweight='bold')
    ax.set_xticks(xi)
    ax.set_xticklabels(labels_x, rotation=45, ha='right', fontsize=6.5)
    ax.grid(True, alpha=0.2)
    ax.spines[['top']].set_visible(False)
    ax2.spines[['top']].set_visible(False)

    # 2030 비중 (배경 음영)
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 45))
    ax3.fill_between(xi, mg['gen2030_ratio'].values * 100, alpha=0.1, color=C_PURPLE)
    ax3.set_ylabel('2030비중(%)', color=C_PURPLE, fontsize=7)
    ax3.tick_params(axis='y', labelcolor=C_PURPLE, labelsize=6)
    ax3.spines[['top','left','bottom']].set_visible(False)

for idx in range(n_plots, len(axes2_flat)):
    axes2_flat[idx].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'trend_timeseries_grid.png'), dpi=150, bbox_inches='tight')
print("저장: trend_timeseries_grid.png")
plt.close()

# ══════════════════════════════════════════════════════════════
# FIGURE 3: 블로그 트렌드 vs 매출 성장률 분석
# ══════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 3, figsize=(24, 9))
fig3.suptitle(
    '블로그(네이버) 언급량 트렌드 vs 실제 매출 성장 분석\n(2022~2024년 기준)',
    fontsize=15, fontweight='bold'
)

# ── 패널 1: 블로그 vs 매출 성장률 산점도 ──
ax = axes3[0]
sc = ax.scatter(
    blog_df['blog_g_22_24'], blog_df['sales_g_22_24'],
    c=np.log1p(blog_df['total_vol']), cmap='YlOrRd',
    s=100, alpha=0.82, edgecolors='white', linewidth=0.7, zorder=3
)
plt.colorbar(sc, ax=ax, label='log(블로그 총량)')

xfit = blog_df['blog_g_22_24']
yfit = blog_df['sales_g_22_24']
m, b = np.polyfit(xfit, yfit, 1)
xl = np.linspace(xfit.min()-2, xfit.max()+2, 100)
ax.plot(xl, m*xl+b, '--', color=C_RED, linewidth=1.8, alpha=0.75,
        label=f'회귀선 (r={r_blog:.2f}, p={p_blog:.3f})')

# 사분면 분류
q1 = blog_df[(blog_df['blog_g_22_24']>0) & (blog_df['sales_g_22_24']>0)]
q2 = blog_df[(blog_df['blog_g_22_24']<0) & (blog_df['sales_g_22_24']>0)]
q3 = blog_df[(blog_df['blog_g_22_24']<0) & (blog_df['sales_g_22_24']<0)]
q4 = blog_df[(blog_df['blog_g_22_24']>0) & (blog_df['sales_g_22_24']<0)]
ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax.axvline(0, color='gray', linewidth=0.8, linestyle=':')

# 사분면 라벨
xmin, xmax = xfit.min()-5, xfit.max()+5
ymin, ymax = yfit.min()-5, yfit.max()+5
ax.text(xmax*0.92, ymax*0.92, f'I (선도형)\n{len(q1)}개', fontsize=8.5, color=C_GREEN,
        ha='right', va='top', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))
ax.text(xmin*0.92, ymax*0.92, f'II (독자성장)\n{len(q2)}개', fontsize=8.5, color=C_BLUE,
        ha='left', va='top', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))
ax.text(xmin*0.92, ymin*0.92, f'III (동반쇠퇴)\n{len(q3)}개', fontsize=8.5, color=C_RED,
        ha='left', va='bottom', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.3))
ax.text(xmax*0.92, ymin*0.92, f'IV (화제성↑/실적↓)\n{len(q4)}개', fontsize=8.5, color=C_ORANGE,
        ha='right', va='bottom', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.3))

for _, row in blog_df.iterrows():
    ax.annotate(row['area'], (row['blog_g_22_24'], row['sales_g_22_24']),
                fontsize=7, ha='center', va='bottom',
                xytext=(0, 3), textcoords='offset points')

ax.set_xlabel('블로그 언급량 성장률 2022→2024 (%)', fontsize=10)
ax.set_ylabel('실제 매출 성장률 2022→2024 (%)', fontsize=10)
ax.set_title('블로그 트렌드 성장 vs 실제 매출 성장\n(버블 크기 = 블로그 총 언급량)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)
ax.spines[['top','right']].set_visible(False)

# ── 패널 2: 블로그 성장률 vs 매출 성장률 비교 막대 ──
ax = axes3[1]
blog_sorted = blog_df.sort_values('blog_g_22_24', ascending=False).head(20)
x_idx = range(len(blog_sorted))
w = 0.38

bar_bg = ax.bar([xi-w/2 for xi in x_idx], blog_sorted['blog_g_22_24'],
                w, color=C_ORANGE, alpha=0.82, edgecolor='white', label='블로그 성장률(%)')
bar_sg = ax.bar([xi+w/2 for xi in x_idx], blog_sorted['sales_g_22_24'],
                w,
                color=[C_GREEN if v > 0 else C_RED for v in blog_sorted['sales_g_22_24']],
                alpha=0.82, edgecolor='white', label='매출 성장률(%)')

ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(list(x_idx))
ax.set_xticklabels(blog_sorted['area'], rotation=45, ha='right', fontsize=8)
ax.set_ylabel('성장률 (%)', fontsize=10)
ax.set_title('블로그 언급량 상위 20개 상권\n블로그 성장 vs 매출 성장 비교', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, axis='y', alpha=0.3)
ax.spines[['top','right']].set_visible(False)

# ── 패널 3: 2030세대 비중 상위 행정동 ──
ax = axes3[2]
top20_2030 = gen2030_top.nlargest(20, 'r2030').sort_values('r2030', ascending=True)
norm_c = plt.Normalize(top20_2030['r2030'].min(), top20_2030['r2030'].max())
colors_2030 = [plt.cm.RdYlGn(norm_c(v)) for v in top20_2030['r2030']]
bars_h = ax.barh(top20_2030['행정동_코드_명'], top20_2030['r2030']*100,
                 color=colors_2030, alpha=0.88, edgecolor='white', height=0.7)
for bar in bars_h:
    w2 = bar.get_width()
    ax.text(w2 + 0.3, bar.get_y() + bar.get_height()/2,
            f'{w2:.1f}%', va='center', ha='left', fontsize=8.5)
ax.set_xlabel('2030세대 매출 비중 (%)', fontsize=10)
ax.set_title('2030세대 매출 비중 상위 20 행정동\n(2022~2024 누적)', fontsize=11, fontweight='bold')
ax.set_xlim(0, top20_2030['r2030'].max()*100 * 1.18)
ax.grid(True, axis='x', alpha=0.3)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'blog_trend_sales_analysis.png'), dpi=150, bbox_inches='tight')
print("저장: blog_trend_sales_analysis.png")
plt.close()

# ══════════════════════════════════════════════════════════════
# FIGURE 4: 2030세대 집중도 심화 분석
# ══════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(2, 2, figsize=(22, 14))
fig4.suptitle('2030세대 소비 집중도 심화 분석 (2020~2024)', fontsize=15, fontweight='bold')

# ── 4-1: 2030 비중 추이 (TOP 10 행정동) ──
ax = axes4[0, 0]
top10_dongs = gen2030_top.nlargest(10, 'r2030')['행정동_코드_명'].tolist()
g2030_trend = (
    dong_qtr[dong_qtr['행정동_코드_명'].isin(top10_dongs)]
    .groupby(['year', '행정동_코드_명'])
    .agg(g2030=('gen2030','sum'), total=('매출','sum'))
    .reset_index()
)
g2030_trend['ratio_pct'] = g2030_trend['g2030'] / g2030_trend['total'] * 100
pivot_2030 = g2030_trend.pivot(index='year', columns='행정동_코드_명', values='ratio_pct')
cmap_l = plt.colormaps['tab10'].resampled(len(pivot_2030.columns))
for i, dong in enumerate(pivot_2030.columns):
    vals = pivot_2030[dong].dropna()
    ax.plot(vals.index.astype(str), vals.values, '-o', linewidth=2.2,
            color=cmap_l(i), label=dong, markersize=6)
    if len(vals) > 0:
        ax.annotate(f'{vals.iloc[-1]:.1f}%',
                    xy=(str(int(vals.index[-1])), vals.iloc[-1]),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=7.5, color=cmap_l(i), va='center')
ax.set_xlabel('연도', fontsize=10)
ax.set_ylabel('2030 매출 비중 (%)', fontsize=10)
ax.set_title('2030세대 비중 TOP10 행정동 연도별 추이', fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper left', ncol=2)
ax.grid(True, alpha=0.3)
ax.spines[['top','right']].set_visible(False)

# ── 4-2: 서울시 전체 2030 비중 변화 vs 40대 이상 ──
ax = axes4[0, 1]
age_cols_all = ['연령대_20_매출_금액','연령대_30_매출_금액','연령대_40_매출_금액',
                '연령대_50_매출_금액','연령대_60_이상_매출_금액']
seoul_age = df.groupby('year')[age_cols_all].sum().reset_index()
seoul_total = seoul_age[age_cols_all].sum(axis=1)
seoul_age['2030비중'] = (seoul_age['연령대_20_매출_금액'] + seoul_age['연령대_30_매출_금액']) / seoul_total * 100
seoul_age['4060비중'] = (seoul_age['연령대_40_매출_금액'] + seoul_age['연령대_50_매출_금액'] + seoul_age['연령대_60_이상_매출_금액']) / seoul_total * 100
yr_str = seoul_age['year'].astype(str)
ax.plot(yr_str, seoul_age['2030비중'], '-o', color=C_ORANGE, linewidth=2.5, markersize=8,
        label='2030세대 비중', zorder=3)
ax.plot(yr_str, seoul_age['4060비중'], '-s', color=C_BLUE, linewidth=2.5, markersize=8,
        label='4060세대 비중', zorder=3)
ax.fill_between(yr_str, seoul_age['2030비중'], seoul_age['4060비중'],
                alpha=0.1, color=C_ORANGE, label='비중 차이')
for y, v20, v40 in zip(yr_str, seoul_age['2030비중'], seoul_age['4060비중']):
    ax.text(y, v20+0.4, f'{v20:.1f}%', ha='center', fontsize=9, color=C_ORANGE, fontweight='bold')
    ax.text(y, v40-0.7, f'{v40:.1f}%', ha='center', fontsize=9, color=C_BLUE, fontweight='bold')
ax.set_xlabel('연도', fontsize=10)
ax.set_ylabel('매출 비중 (%)', fontsize=10)
ax.set_title('서울시 전체: 2030 vs 4060 세대 비중 추이', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(20, 80)
ax.grid(True, alpha=0.3)
ax.spines[['top','right']].set_visible(False)

# ── 4-3: 네이버 트렌드 행정동별 상관계수 히트맵 (lag × dong) ──
ax = axes4[1, 0]
# lag별 상관계수 매트릭스
lags = [-3, -2, -1, 0, 1, 2, 3]
lag_matrix = []
dong_labels_hm = []
for naver_g, dong_name in mapping.items():
    s_ts = dong_qtr[dong_qtr['행정동_코드_명'] == dong_name].sort_values('기준_년분기_코드')
    n_ts = naver_qtr[naver_qtr['group'] == naver_g].sort_values('기준_년분기_코드')
    mg   = pd.merge(s_ts, n_ts, on='기준_년분기_코드', how='inner')
    if len(mg) < 6:
        continue
    x = (mg['naver_adj'] - mg['naver_adj'].mean()) / (mg['naver_adj'].std() + 1e-9)
    y = (mg['매출']      - mg['매출'].mean())      / (mg['매출'].std()      + 1e-9)
    n = len(x)
    row_lags = []
    for lag in lags:
        if lag == 0:
            r_l, _ = stats.pearsonr(x, y)
        elif lag > 0:
            if n-lag < 4:
                row_lags.append(np.nan); continue
            r_l, _ = stats.pearsonr(x.values[:n-lag], y.values[lag:])
        else:
            if n+lag < 4:
                row_lags.append(np.nan); continue
            r_l, _ = stats.pearsonr(x.values[-lag:], y.values[:n+lag])
        row_lags.append(round(r_l, 3))
    lag_matrix.append(row_lags)
    dong_labels_hm.append(naver_g)

lag_mat_arr = np.array(lag_matrix)
im = ax.imshow(lag_mat_arr, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(lags)))
ax.set_xticklabels([f'Lag{l:+d}Q' for l in lags], fontsize=9)
ax.set_yticks(range(len(dong_labels_hm)))
ax.set_yticklabels(dong_labels_hm, fontsize=8.5)
for i in range(len(dong_labels_hm)):
    for j in range(len(lags)):
        v = lag_mat_arr[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=7, color='black' if abs(v) < 0.7 else 'white')
plt.colorbar(im, ax=ax, label='Pearson r')
ax.set_title('행정동 × Lag 상관계수 히트맵\n(양수 Lag = 네이버가 매출 선행)', fontsize=11, fontweight='bold')

# ── 4-4: 업종별 2030 집중 행정동 소비 비중 ──
ax = axes4[1, 1]
# 2030 집중 상위 10개 행정동의 업종 구성
top10_2030_for_sector = gen2030_top.nlargest(10, 'r2030')['행정동_코드_명'].tolist()
sector_2030 = (
    df[df['행정동_코드_명'].isin(top10_2030_for_sector) & df['year'].between(2022, 2024)]
    .groupby(['서비스_업종_코드_명'])
    .agg(gen2030_sales=('연령대_20_매출_금액','sum'))
    .reset_index()
)
sector_2030['gen30_sales'] = (
    df[df['행정동_코드_명'].isin(top10_2030_for_sector) & df['year'].between(2022, 2024)]
    .groupby('서비스_업종_코드_명')['연령대_30_매출_금액'].sum().values
)
sector_2030['gen2030'] = sector_2030['gen2030_sales'] + sector_2030['gen30_sales']
sector_top = sector_2030.nlargest(15, 'gen2030').sort_values('gen2030', ascending=True)

bar_colors_sector = plt.cm.tab20(np.linspace(0, 1, len(sector_top)))
ax.barh(sector_top['서비스_업종_코드_명'], sector_top['gen2030'] / 1e8,
        color=bar_colors_sector, alpha=0.88, edgecolor='white', height=0.7)
for i, (_, r) in enumerate(sector_top.iterrows()):
    w_v = r['gen2030'] / 1e8
    ax.text(w_v + sector_top['gen2030'].max()/1e8 * 0.01,
            i, f'{w_v:,.0f}억', va='center', ha='left', fontsize=8)
ax.set_xlabel('2030세대 매출 합계 (억원)', fontsize=10)
ax.set_title('2030세대 집중 행정동: 업종별 매출 Top15\n(2022~2024 누적)', fontsize=11, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)
ax.spines[['top','right']].set_visible(False)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}억'))

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'gen2030_concentration.png'), dpi=150, bbox_inches='tight')
print("저장: gen2030_concentration.png")
plt.close()

print("\n모든 시각화 완료!")
print(f"\n최종 분석 인사이트:")
print(f"  - 분석 행정동: {len(results_df)}개")
print(f"  - 통계적 유의 상관 행정동 (p<0.05): {sum(results_df['p_sync']<0.05)}개")
print(f"  - 선행 행정동 (네이버→매출): {len(results_df[results_df['best_lag']>0])}개")
print(f"  - 블로그-매출 상관계수: r={r_blog:.3f} (p={p_blog:.3f})")
print(f"  - 2030세대 최고 비중 행정동: {gen2030_top.nlargest(1,'r2030')['행정동_코드_명'].values[0]}")
