"""
비율 기반 소비 트렌드 분석 (Ratio-Based Trend Analysis)
- 절대 매출이 아닌 점유율/비중 변화로 소비 이동(shift) 포착
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── 한글 폰트 설정 ──────────────────────────────────────────────────────────────
def set_korean_font():
    candidates = [
        'Malgun Gothic', 'NanumGothic', 'NanumBarunGothic',
        'AppleGothic', 'Gulim', 'Dotum'
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            plt.rcParams['font.family'] = c
            break
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# ── 색상 팔레트 ──────────────────────────────────────────────────────────────────
RISE_COLOR  = '#E8424B'   # 상승 (따뜻한 빨강)
FALL_COLOR  = '#3B7DD8'   # 하락 (차가운 파랑)
GEN_COLOR   = '#F5A623'   # 2030세대 (노란 주황)
NEUT_COLOR  = '#7B8794'   # 중립 (회색)
BG_COLOR    = '#F8F9FA'

BASE_DIR = r'C:\Users\30032\OneDrive\바탕 화면\workspace\data_analytics_final'

# ══════════════════════════════════════════════════════════════════════════════
# 0. 데이터 로드 및 연도별 행정동 집계
# ══════════════════════════════════════════════════════════════════════════════
print("=== 데이터 로드 중 ===")
df = pd.read_csv(
    f'{BASE_DIR}/data/서울시_상권분석서비스(추정매출-행정동)_2020_2024.csv',
    encoding='utf-8-sig',
    usecols=[
        '기준_년분기_코드', '행정동_코드', '행정동_코드_명',
        '당월_매출_금액',
        '연령대_20_매출_금액', '연령대_30_매출_금액',
        '연령대_40_매출_금액', '연령대_50_매출_금액', '연령대_60_이상_매출_금액'
    ]
)
df['year']    = df['기준_년분기_코드'] // 10
df['quarter'] = df['기준_년분기_코드'] % 10
df['gen2030'] = df['연령대_20_매출_금액'] + df['연령대_30_매출_금액']
df['gen40plus'] = (df['연령대_40_매출_금액'] + df['연령대_50_매출_금액']
                   + df['연령대_60_이상_매출_금액'])

print(f"  행 수: {len(df):,}  |  연도: {sorted(df['year'].unique())}")

# 연도-행정동별 집계 (업종 합산)
annual = (
    df.groupby(['year', '행정동_코드', '행정동_코드_명'], as_index=False)
    .agg(
        total_sales=('당월_매출_금액', 'sum'),
        gen2030_sales=('gen2030', 'sum'),
        gen40plus_sales=('gen40plus', 'sum')
    )
)
annual['gen2030_share']  = annual['gen2030_sales'] / annual['total_sales']
annual['gen40plus_share'] = annual['gen40plus_sales'] / annual['total_sales']

# 분기-행정동별 집계 (네이버 트렌드 상관용)
quarterly = (
    df.groupby(['year', 'quarter', '행정동_코드', '행정동_코드_명'], as_index=False)
    .agg(
        total_sales=('당월_매출_금액', 'sum'),
        gen2030_sales=('gen2030', 'sum')
    )
)
quarterly['ym_code'] = quarterly['year'] * 10 + quarterly['quarter']

print("  집계 완료")


# ══════════════════════════════════════════════════════════════════════════════
# 1. 행정동별 매출 점유율 변화 (Shift-Share)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== [1] 행정동 매출 점유율 변화 (Shift-Share) ===")

# 연도별 서울 전체 매출
seoul_annual = annual.groupby('year')['total_sales'].sum().rename('seoul_total')
annual = annual.merge(seoul_annual, on='year')
annual['mkt_share'] = annual['total_sales'] / annual['seoul_total']

# 2020 vs 2024 피벗
share_pivot = annual.pivot_table(
    index=['행정동_코드', '행정동_코드_명'],
    columns='year',
    values='mkt_share'
).reset_index()

# 2020·2024 모두 있는 행정동만
share_pivot = share_pivot.dropna(subset=[2020, 2024])
share_pivot['share_change'] = share_pivot[2024] - share_pivot[2020]
share_pivot['share_change_pp'] = share_pivot['share_change'] * 100  # 퍼센트포인트

top20_up   = share_pivot.nlargest(20, 'share_change').reset_index(drop=True)
top20_down = share_pivot.nsmallest(20, 'share_change').reset_index(drop=True)

print(f"  상승 Top5: {top20_up['행정동_코드_명'].head().tolist()}")
print(f"  하락 Bot5: {top20_down['행정동_코드_명'].head().tolist()}")

# ── 시각화 1: Shift-Share 상승/하락 랭킹 ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 10), facecolor=BG_COLOR)
fig.suptitle('서울시 행정동별 매출 점유율 변화 (2020→2024)', fontsize=16, fontweight='bold', y=1.01)

for ax, data, title, color in [
    (axes[0], top20_up,   '상승 Top 20 (소비 유입 행정동)', RISE_COLOR),
    (axes[1], top20_down, '하락 Bottom 20 (소비 유출 행정동)', FALL_COLOR)
]:
    ax.set_facecolor(BG_COLOR)
    vals = data['share_change_pp']
    names = data['행정동_코드_명']
    y_pos = range(len(names))

    bars = ax.barh(y_pos, vals, color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('점유율 변화 (퍼센트포인트, pp)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.axvline(0, color='#333333', linewidth=0.8, linestyle='--')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)

    # 값 레이블
    for bar, v in zip(bars, vals):
        xpos = v + (0.002 if v >= 0 else -0.002)
        ha = 'left' if v >= 0 else 'right'
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f'{v:+.3f}pp', va='center', ha=ha, fontsize=7.5, color='#333333')

plt.tight_layout()
plt.savefig(f'{BASE_DIR}/data/share_shift_ranking.png', dpi=150, bbox_inches='tight',
            facecolor=BG_COLOR)
plt.close()
print("  -> share_shift_ranking.png 저장 완료")


# ══════════════════════════════════════════════════════════════════════════════
# 2. 2030세대 소비 비중 변화
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== [2] 2030세대 소비 비중 변화 (2020→2024) ===")

gen_pivot = annual.pivot_table(
    index=['행정동_코드', '행정동_코드_명'],
    columns='year',
    values='gen2030_share'
).reset_index()
gen_pivot = gen_pivot.dropna(subset=[2020, 2024])
gen_pivot['gen_change'] = gen_pivot[2024] - gen_pivot[2020]
gen_pivot['gen_change_pp'] = gen_pivot['gen_change'] * 100

gen_top20_up   = gen_pivot.nlargest(20, 'gen_change').reset_index(drop=True)
gen_top20_down = gen_pivot.nsmallest(20, 'gen_change').reset_index(drop=True)

print(f"  2030 비중 상승 Top5: {gen_top20_up['행정동_코드_명'].head().tolist()}")
print(f"  2030 비중 하락 Bot5: {gen_top20_down['행정동_코드_명'].head().tolist()}")

# ── 시각화 2A: 2030 비중 상승/하락 랭킹 ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 10), facecolor=BG_COLOR)
fig.suptitle('행정동별 2030세대 소비 비중 변화 (2020→2024)', fontsize=16, fontweight='bold', y=1.01)

for ax, data, title, color in [
    (axes[0], gen_top20_up,   '2030 비중 상승 Top 20 (2030 유입)', GEN_COLOR),
    (axes[1], gen_top20_down, '2030 비중 하락 Bottom 20 (2030 이탈)', FALL_COLOR)
]:
    ax.set_facecolor(BG_COLOR)
    vals = data['gen_change_pp']
    names = data['행정동_코드_명']
    y_pos = range(len(names))

    bars = ax.barh(y_pos, vals, color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('2030 비중 변화 (퍼센트포인트, pp)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.axvline(0, color='#333333', linewidth=0.8, linestyle='--')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)

    for bar, v in zip(bars, vals):
        xpos = v + (0.3 if v >= 0 else -0.3)
        ha = 'left' if v >= 0 else 'right'
        ax.text(xpos, bar.get_y() + bar.get_height()/2,
                f'{v:+.1f}pp', va='center', ha=ha, fontsize=7.5, color='#333333')

plt.tight_layout()
plt.savefig(f'{BASE_DIR}/data/gen2030_share_change_ranking.png', dpi=150, bbox_inches='tight',
            facecolor=BG_COLOR)
plt.close()
print("  -> gen2030_share_change_ranking.png 저장 완료")

# ── 시각화 2B: Top10 행정동 연도별 2030 비중 추이 (꺾은선) ────────────────────
top10_names = gen_top20_up['행정동_코드_명'].head(10).tolist()
# 하락 Top5도 추가해 대조
bot5_names  = gen_top20_down['행정동_코드_명'].head(5).tolist()
watch_names = top10_names + bot5_names

annual_watch = annual[annual['행정동_코드_명'].isin(watch_names)].copy()
annual_watch['gen2030_pct'] = annual_watch['gen2030_share'] * 100

fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG_COLOR)
fig.suptitle('2030세대 소비 비중 연도별 추이 (2020~2024)', fontsize=15, fontweight='bold')

years = sorted(annual['year'].unique())
cmap_up   = plt.cm.get_cmap('Oranges', len(top10_names) + 2)
cmap_down = plt.cm.get_cmap('Blues',   len(bot5_names)  + 2)

for ax, names, cmap, title in [
    (axes[0], top10_names, cmap_up,   '2030 비중 상승 Top 10'),
    (axes[1], bot5_names,  cmap_down, '2030 비중 하락 Bottom 5')
]:
    ax.set_facecolor(BG_COLOR)
    for i, name in enumerate(names):
        sub = annual_watch[annual_watch['행정동_코드_명'] == name].sort_values('year')
        color = cmap(i + 2)
        ax.plot(sub['year'], sub['gen2030_pct'], marker='o', markersize=6,
                linewidth=2, label=name, color=color)
        # 마지막 값 레이블
        last = sub.iloc[-1]
        ax.annotate(f"{name}\n{last['gen2030_pct']:.1f}%",
                    xy=(last['year'], last['gen2030_pct']),
                    xytext=(3, 0), textcoords='offset points',
                    fontsize=7, color=color, va='center')

    ax.set_xticks(years)
    ax.set_xticklabels(years)
    ax.set_xlabel('연도', fontsize=10)
    ax.set_ylabel('2030세대 소비 비중 (%)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=7, loc='upper left', ncol=2)

plt.tight_layout()
plt.savefig(f'{BASE_DIR}/data/gen2030_share_trend_line.png', dpi=150, bbox_inches='tight',
            facecolor=BG_COLOR)
plt.close()
print("  -> gen2030_share_trend_line.png 저장 완료")


# ══════════════════════════════════════════════════════════════════════════════
# 3. 네이버 트렌드 vs 매출 점유율 상관관계 (재분석)
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== [3] 네이버 트렌드 vs 매출 점유율 상관관계 (재분석) ===")

naver = pd.read_csv(
    f'{BASE_DIR}/data/naver_trend_행정동보정_20260510.csv',
    encoding='utf-8-sig'
)
print(f"  네이버 트렌드 로드: {naver.shape}  |  컬럼: {list(naver.columns)}")
naver['period'] = pd.to_datetime(naver['period'])
naver['year']    = naver['period'].dt.year
naver['quarter'] = naver['period'].dt.month.map(
    {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}
)
# 분기 평균
naver_q = (
    naver.groupby(['group', 'year', 'quarter'], as_index=False)['adjusted_ratio']
    .mean()
)
naver_q['ym_code'] = naver_q['year'] * 10 + naver_q['quarter']

# 서울 전체 분기 매출 (점유율 분모)
seoul_qtr = quarterly.groupby(['year', 'quarter'])['total_sales'].sum().reset_index()
seoul_qtr['ym_code'] = seoul_qtr['year'] * 10 + seoul_qtr['quarter']
seoul_qtr = seoul_qtr.rename(columns={'total_sales': 'seoul_qtr_total'})

quarterly = quarterly.merge(seoul_qtr[['ym_code', 'seoul_qtr_total']], on='ym_code', how='left')
quarterly['mkt_share_q'] = quarterly['total_sales'] / quarterly['seoul_qtr_total']

# 행정동명 매핑 (이전 분석과 동일 로직)
dong_names = quarterly['행정동_코드_명'].unique().tolist()
naver_groups = naver_q['group'].unique().tolist()

mapping = {}
for ng in naver_groups:
    matches = [dn for dn in dong_names if ng in dn or dn in ng]
    if matches:
        matches.sort(key=len)
        mapping[ng] = matches[0]

print(f"  매핑 성공: {len(mapping)}/{len(naver_groups)}개")

# 점유율 기반 상관관계 계산
results_share = []

for ng, dn in mapping.items():
    nav_sub  = naver_q[naver_q['group'] == ng].set_index('ym_code')['adjusted_ratio']
    sales_sub = quarterly[quarterly['행정동_코드_명'] == dn].set_index('ym_code')['mkt_share_q']

    common = nav_sub.index.intersection(sales_sub.index)
    if len(common) < 6:
        continue

    x = nav_sub.loc[common].values
    y = sales_sub.loc[common].values

    # 동시 상관
    r, p = stats.pearsonr(x, y)

    # MoM 변화율 상관 (점유율 변화율)
    # 점유율 MoM: 이번분기 - 전분기 (pp 변화)
    nav_diff   = pd.Series(x, index=common).diff().dropna()
    share_diff = pd.Series(y, index=common).diff().dropna()
    common2 = nav_diff.index.intersection(share_diff.index)
    if len(common2) >= 5:
        r_diff, p_diff = stats.pearsonr(nav_diff.loc[common2], share_diff.loc[common2])
    else:
        r_diff, p_diff = np.nan, np.nan

    # Lag 분석 (점유율 기준, lag 1~3분기)
    best_lag_r, best_lag = 0, 0
    lag_results_detail = {}
    for lag in range(-3, 4):
        shifted_nav = nav_sub.shift(-lag)  # lag>0: 네이버가 선행
        common_lag  = shifted_nav.index.intersection(sales_sub.index)
        common_lag  = common_lag.intersection(nav_sub.index).intersection(sales_sub.index)
        xlag = shifted_nav.reindex(common_lag).dropna()
        ylag = sales_sub.reindex(xlag.index).dropna()
        xlag = xlag.reindex(ylag.index)
        if len(xlag) < 5:
            continue
        rl, pl = stats.pearsonr(xlag.values, ylag.values)
        lag_results_detail[lag] = (rl, pl)
        if abs(rl) > abs(best_lag_r):
            best_lag_r, best_lag = rl, lag

    results_share.append({
        'naver_group': ng,
        'dong_name':   dn,
        'n_obs':       len(common),
        'r_simul':     r,
        'p_simul':     p,
        'r_diff':      r_diff,
        'p_diff':      p_diff,
        'best_lag':    best_lag,
        'best_lag_r':  best_lag_r,
        'sig_simul':   p < 0.05,
        'sig_diff':    p_diff < 0.05 if not np.isnan(p_diff) else False,
    })

res_df = pd.DataFrame(results_share).sort_values('r_simul', ascending=False)
print(f"\n  분석 완료: {len(res_df)}개 행정동")
print(f"\n  [점유율 기준 동시 상관] 유의미(p<0.05) 행정동:")
sig_simul = res_df[res_df['sig_simul']]
for _, row in sig_simul.iterrows():
    dir_str = '양' if row['r_simul'] > 0 else '음'
    print(f"    {row['dong_name']:12s}  r={row['r_simul']:+.3f}  p={row['p_simul']:.3f}  ({dir_str}의상관)")

print(f"\n  [점유율 변화율(QoQ) 기준 상관] 유의미 행정동:")
sig_diff = res_df[res_df['sig_diff']]
for _, row in sig_diff.iterrows():
    dir_str = '양' if row['r_diff'] > 0 else '음'
    print(f"    {row['dong_name']:12s}  r_diff={row['r_diff']:+.3f}  p={row['p_diff']:.3f}  ({dir_str}의상관)")

print(f"\n  [최적 Lag 분포] (lag>0 = 네이버가 매출 선행):")
lag_counts = res_df['best_lag'].value_counts().sort_index()
for lag, cnt in lag_counts.items():
    direction = "네이버 선행" if lag > 0 else ("매출 선행" if lag < 0 else "동기")
    print(f"    lag={lag:+d}분기: {cnt}개 행정동  ({direction})")

# ── 시각화 3: 점유율 기반 상관관계 대시보드 ──────────────────────────────────
fig = plt.figure(figsize=(20, 14), facecolor=BG_COLOR)
fig.suptitle('네이버 트렌드 vs 매출 점유율 상관관계 분석 (비율 기반 재분석)',
             fontsize=15, fontweight='bold', y=0.98)

gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

# 3-1: 동시 상관계수 (점유율 기준)
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor(BG_COLOR)
colors_bar = [RISE_COLOR if r > 0 else FALL_COLOR for r in res_df['r_simul']]
sig_markers = ['*' if s else '' for s in res_df['sig_simul']]
bars = ax1.barh(range(len(res_df)), res_df['r_simul'],
                color=colors_bar, alpha=0.8, edgecolor='white')
ax1.set_yticks(range(len(res_df)))
ax1.set_yticklabels(
    [f"{row['dong_name']}{m}" for (_, row), m in zip(res_df.iterrows(), sig_markers)],
    fontsize=8
)
ax1.invert_yaxis()
ax1.axvline(0, color='#333', linewidth=1)
ax1.set_xlabel('Pearson r (점유율 기준)', fontsize=10)
ax1.set_title('동시 상관계수 (* = p<0.05)', fontsize=11, fontweight='bold')
ax1.set_xlim(-1, 1)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.spines[['top', 'right']].set_visible(False)
for bar, v in zip(bars, res_df['r_simul']):
    ax1.text(v + (0.02 if v >= 0 else -0.02), bar.get_y() + bar.get_height()/2,
             f'{v:+.3f}', va='center', ha='left' if v >= 0 else 'right', fontsize=7)

# 3-2: 동시 vs QoQ 변화율 산점도
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor(BG_COLOR)
valid_mask = ~res_df['r_diff'].isna()
sc = ax2.scatter(res_df.loc[valid_mask, 'r_simul'],
                 res_df.loc[valid_mask, 'r_diff'],
                 c=[RISE_COLOR if r > 0 else FALL_COLOR
                    for r in res_df.loc[valid_mask, 'r_simul']],
                 alpha=0.8, s=80, edgecolors='white', linewidth=0.5)
for _, row in res_df[valid_mask].iterrows():
    ax2.annotate(row['dong_name'], (row['r_simul'], row['r_diff']),
                 fontsize=6.5, alpha=0.8, xytext=(3, 3),
                 textcoords='offset points')
ax2.axhline(0, color='#999', linewidth=0.8, linestyle='--')
ax2.axvline(0, color='#999', linewidth=0.8, linestyle='--')
ax2.set_xlabel('동시 상관 (r)', fontsize=9)
ax2.set_ylabel('QoQ 변화율 상관 (r_diff)', fontsize=9)
ax2.set_title('수준 vs 변화율 상관 비교', fontsize=10, fontweight='bold')
ax2.spines[['top', 'right']].set_visible(False)

# 3-3: Lag 분포 막대
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(BG_COLOR)
lags = sorted(res_df['best_lag'].unique())
cnt  = [res_df['best_lag'].eq(l).sum() for l in lags]
lag_colors = [RISE_COLOR if l > 0 else (FALL_COLOR if l < 0 else NEUT_COLOR) for l in lags]
ax3.bar(lags, cnt, color=lag_colors, alpha=0.85, edgecolor='white')
ax3.set_xticks(lags)
ax3.set_xticklabels([f'{l:+d}분기' for l in lags], fontsize=8)
ax3.set_ylabel('행정동 수', fontsize=9)
ax3.set_title('최적 Lag 분포\n(+: 네이버 선행, -: 매출 선행)', fontsize=10, fontweight='bold')
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.spines[['top', 'right']].set_visible(False)
for x, y in zip(lags, cnt):
    ax3.text(x, y + 0.1, str(y), ha='center', fontsize=9, fontweight='bold')

# 3-4: 점유율 기준 r vs 절대 매출 기준 차이 개념도 (시계열 예시)
# 가장 유의미한 행정동 1개 상세 시각화
if len(sig_simul) > 0:
    best_row = sig_simul.iloc[0]
    ng_sel = best_row['naver_group']
    dn_sel = best_row['dong_name']

    nav_plot   = naver_q[naver_q['group'] == ng_sel].set_index('ym_code')['adjusted_ratio']
    sales_plot = quarterly[quarterly['행정동_코드_명'] == dn_sel].set_index('ym_code')['mkt_share_q']
    abs_plot   = quarterly[quarterly['행정동_코드_명'] == dn_sel].set_index('ym_code')['total_sales']
    common_p = nav_plot.index.intersection(sales_plot.index)

    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.set_facecolor(BG_COLOR)
    ax4_twin = ax4.twinx()

    x_labels = [f"{c//10}\nQ{c%10}" for c in sorted(common_p)]
    x_idx = range(len(common_p))

    ax4.plot(x_idx, nav_plot.loc[sorted(common_p)].values,
             color='#9B59B6', linewidth=2, marker='s', markersize=5, label='네이버 검색량')
    ax4_twin.plot(x_idx, sales_plot.loc[sorted(common_p)].values * 1000,
                  color=RISE_COLOR, linewidth=2, marker='o', markersize=5,
                  label='매출 점유율(×1000)')

    ax4.set_xticks(x_idx)
    ax4.set_xticklabels(x_labels, fontsize=7)
    ax4.set_ylabel('네이버 검색량 (조정)', fontsize=9, color='#9B59B6')
    ax4_twin.set_ylabel('매출 점유율 (×1000)', fontsize=9, color=RISE_COLOR)
    ax4.set_title(f'{dn_sel}: 네이버 트렌드 vs 매출 점유율  (r={best_row["r_simul"]:+.3f})',
                  fontsize=10, fontweight='bold')
    ax4.grid(alpha=0.3, linestyle='--')
    ax4.spines[['top', 'right']].set_visible(False)
    lines1, labs1 = ax4.get_legend_handles_labels()
    lines2, labs2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labs1 + labs2, loc='upper left', fontsize=8)

plt.savefig(f'{BASE_DIR}/data/naver_share_correlation.png', dpi=150, bbox_inches='tight',
            facecolor=BG_COLOR)
plt.close()
print("  -> naver_share_correlation.png 저장 완료")


# ══════════════════════════════════════════════════════════════════════════════
# 4. 블로그 트렌드 vs 매출 점유율
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== [4] 블로그 트렌드 vs 매출 점유율 ===")

blog = pd.read_csv(f'{BASE_DIR}/data/블로그트렌드_상권.csv', encoding='utf-8-sig')
print(f"  블로그 데이터: {blog.shape}  |  컬럼: {list(blog.columns)}")

# 연도별 매출 점유율 (행정동 수준)
share_long = annual[['year', '행정동_코드_명', 'mkt_share']].copy()

# 블로그 상권 → 행정동 매핑
blog_area_col = 'area' if 'area' in blog.columns else blog.columns[0]
blog_areas = blog[blog_area_col].tolist()
dong_names_all = share_long['행정동_코드_명'].unique().tolist()

blog_mapping = {}
for area in blog_areas:
    area_clean = area.strip()
    # 1) 직접 포함 관계
    matches = [dn for dn in dong_names_all if area_clean in dn or dn in area_clean]
    if not matches:
        # 2) 유명 상권 수동 매핑
        manual = {
            '성수': '성수동2가', '홍대': '서교동', '이태원': '이태원1동',
            '강남': '역삼1동', '종로': '종로1.2.3.4가동', '명동': '명동',
            '신촌': '신촌동', '건대': '화양동', '합정': '합정동',
            '연남': '연남동', '망원': '망원1동', '을지로': '을지로동',
            '연희': '연희동', '서울숲': '성수동1가', '뚝섬': '성수동1가',
            '압구정': '압구정동', '청담': '청담동', '삼청': '삼청동',
            '익선': '익선동', '북촌': '가회동', '인사동': '인사동',
            '경리단길': '이태원2동', '해방촌': '용산2가동',
            '노량진': '노량진1동', '여의도': '여의도동',
        }
        for k, v in manual.items():
            if k in area_clean:
                matches = [v]
                break
    if matches:
        matches.sort(key=len)
        blog_mapping[area_clean] = matches[0]

print(f"  블로그 매핑: {len(blog_mapping)}/{len(blog_areas)}개 성공")

# 매핑된 상권별 분석
blog_results = []
for _, row in blog.iterrows():
    area = row[blog_area_col].strip()
    if area not in blog_mapping:
        continue
    dn = blog_mapping[area]

    # 블로그 연도별 언급량
    try:
        blog_2022 = float(row.get('2022', np.nan))
        blog_2023 = float(row.get('2023', np.nan))
        blog_2024 = float(row.get('2024', np.nan))
    except (ValueError, TypeError):
        continue

    # 매출 점유율
    share_2022 = share_long[(share_long['year'] == 2022) & (share_long['행정동_코드_명'] == dn)]['mkt_share']
    share_2023 = share_long[(share_long['year'] == 2023) & (share_long['행정동_코드_명'] == dn)]['mkt_share']
    share_2024 = share_long[(share_long['year'] == 2024) & (share_long['행정동_코드_명'] == dn)]['mkt_share']

    if share_2022.empty or share_2024.empty:
        continue

    s22 = share_2022.values[0] * 100  # %로
    s23 = share_2023.values[0] * 100 if not share_2023.empty else np.nan
    s24 = share_2024.values[0] * 100

    # 성장률 계산
    blog_growth_22_24 = ((blog_2024 - blog_2022) / blog_2022 * 100) if blog_2022 > 0 else np.nan
    share_growth_22_24 = ((s24 - s22) / s22 * 100) if s22 > 0 else np.nan
    share_change_pp_22_24 = s24 - s22

    blog_results.append({
        'area': area,
        'dong': dn,
        'blog_2022': blog_2022,
        'blog_2023': blog_2023,
        'blog_2024': blog_2024,
        'blog_growth_22_24': blog_growth_22_24,
        'share_2022': s22,
        'share_2023': s23,
        'share_2024': s24,
        'share_growth_22_24_pct': share_growth_22_24,
        'share_change_pp': share_change_pp_22_24,
    })

blog_res_df = pd.DataFrame(blog_results).dropna(subset=['blog_growth_22_24', 'share_growth_22_24_pct'])
print(f"  분석 가능 상권: {len(blog_res_df)}개")

# 상관관계 (성장률 기준)
if len(blog_res_df) >= 5:
    r_growth, p_growth = stats.pearsonr(
        blog_res_df['blog_growth_22_24'], blog_res_df['share_growth_22_24_pct']
    )
    # 점유율 변화 pp 기준
    r_pp, p_pp = stats.pearsonr(
        blog_res_df['blog_growth_22_24'], blog_res_df['share_change_pp']
    )
    print(f"\n  블로그 성장률 vs 매출 점유율 성장률: r={r_growth:+.3f}, p={p_growth:.3f}")
    print(f"  블로그 성장률 vs 매출 점유율 변화(pp): r={r_pp:+.3f}, p={p_pp:.3f}")

# 4분면 분류
blog_res_df['blog_high'] = blog_res_df['blog_growth_22_24'] > blog_res_df['blog_growth_22_24'].median()
blog_res_df['share_high'] = blog_res_df['share_growth_22_24_pct'] > blog_res_df['share_growth_22_24_pct'].median()
blog_res_df['quadrant'] = blog_res_df.apply(
    lambda r: 'Q1(언급↑매출↑)' if (r['blog_high'] and r['share_high'])
    else ('Q2(언급↑매출↓)' if (r['blog_high'] and not r['share_high'])
    else ('Q3(언급↓매출↑)' if (not r['blog_high'] and r['share_high'])
    else 'Q4(언급↓매출↓)')),
    axis=1
)
print(f"\n  4분면 분류:")
for q, grp in blog_res_df.groupby('quadrant'):
    print(f"    {q}: {grp['area'].tolist()}")

# ── 시각화 4: 블로그 vs 매출 점유율 산점도 + 4분면 ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=BG_COLOR)
fig.suptitle('블로그 언급량 성장률 vs 매출 점유율 변화 (2022→2024)',
             fontsize=15, fontweight='bold')

# 4-1: 성장률 기준 산점도
ax_s = axes[0]
ax_s.set_facecolor(BG_COLOR)
q_colors = {
    'Q1(언급↑매출↑)': RISE_COLOR,
    'Q2(언급↑매출↓)': '#FF8C42',
    'Q3(언급↓매출↑)': FALL_COLOR,
    'Q4(언급↓매출↓)': NEUT_COLOR
}
for q, grp in blog_res_df.groupby('quadrant'):
    ax_s.scatter(grp['blog_growth_22_24'], grp['share_growth_22_24_pct'],
                 color=q_colors[q], s=100, alpha=0.85, label=q,
                 edgecolors='white', linewidth=0.8, zorder=3)
    for _, row in grp.iterrows():
        ax_s.annotate(row['area'], (row['blog_growth_22_24'], row['share_growth_22_24_pct']),
                      xytext=(5, 3), textcoords='offset points', fontsize=7.5, alpha=0.9)

med_blog  = blog_res_df['blog_growth_22_24'].median()
med_share = blog_res_df['share_growth_22_24_pct'].median()
ax_s.axvline(med_blog,  color='#888', linestyle='--', linewidth=0.8, alpha=0.7)
ax_s.axhline(med_share, color='#888', linestyle='--', linewidth=0.8, alpha=0.7)
ax_s.set_xlabel('블로그 언급량 성장률 (2022→2024, %)', fontsize=10)
ax_s.set_ylabel('매출 점유율 성장률 (2022→2024, %)', fontsize=10)
if len(blog_res_df) >= 5:
    ax_s.set_title(f'블로그 성장률 vs 매출 점유율 성장률\nr={r_growth:+.3f}, p={p_growth:.3f}',
                   fontsize=11, fontweight='bold')
ax_s.legend(fontsize=8, loc='upper left')
ax_s.grid(alpha=0.3, linestyle='--')
ax_s.spines[['top', 'right']].set_visible(False)

# 4-2: 절대 점유율 변화(pp) vs 블로그 성장률
ax_b = axes[1]
ax_b.set_facecolor(BG_COLOR)
bar_data = blog_res_df.sort_values('share_change_pp', ascending=False).head(20)
bar_colors = [RISE_COLOR if v > 0 else FALL_COLOR for v in bar_data['share_change_pp']]
bars2 = ax_b.barh(range(len(bar_data)), bar_data['share_change_pp'],
                  color=bar_colors, alpha=0.85, edgecolor='white')
ax_b.set_yticks(range(len(bar_data)))
ax_b.set_yticklabels(bar_data['area'], fontsize=8)
ax_b.invert_yaxis()

# 블로그 성장률을 점으로 오버레이
ax_b_twin = ax_b.twiny()
ax_b_twin.scatter(bar_data['blog_growth_22_24'], range(len(bar_data)),
                  color='#9B59B6', s=60, zorder=5, label='블로그 성장률(%)')
ax_b_twin.set_xlabel('블로그 성장률 (%)', color='#9B59B6', fontsize=9)
ax_b_twin.tick_params(axis='x', colors='#9B59B6')

ax_b.axvline(0, color='#333', linewidth=0.8, linestyle='--')
ax_b.set_xlabel('매출 점유율 변화 (pp)', fontsize=10)
ax_b.set_title('상권별 매출 점유율 변화 vs 블로그 성장률', fontsize=11, fontweight='bold')
ax_b.grid(axis='x', alpha=0.3, linestyle='--')
ax_b.spines[['top', 'right']].set_visible(False)
ax_b_twin.legend(loc='lower right', fontsize=8)

for bar, v in zip(bars2, bar_data['share_change_pp']):
    ax_b.text(v + (0.001 if v >= 0 else -0.001), bar.get_y() + bar.get_height()/2,
              f'{v:+.3f}pp', va='center', ha='left' if v >= 0 else 'right', fontsize=7)

plt.tight_layout()
plt.savefig(f'{BASE_DIR}/data/blog_share_correlation.png', dpi=150, bbox_inches='tight',
            facecolor=BG_COLOR)
plt.close()
print("  -> blog_share_correlation.png 저장 완료")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 종합 대시보드: 2030 비중 변화 × 매출 점유율 변화 매트릭스
# ══════════════════════════════════════════════════════════════════════════════
print("\n=== [5] 종합 매트릭스: 2030 비중 변화 × 매출 점유율 변화 ===")

# 행정동별 두 지표 결합
# share_pivot, gen_pivot 모두 pivot_table + dropna + change 계산 완료 상태
# 공통 컬럼만 추출하여 merge
combined = (
    share_pivot[['행정동_코드', '행정동_코드_명', 'share_change_pp']]
    .merge(
        gen_pivot[['행정동_코드', 'gen_change_pp']],
        on='행정동_코드', how='inner'
    )
)
combined = combined.dropna(subset=['share_change_pp', 'gen_change_pp'])

# 4분면
med_sh  = combined['share_change_pp'].median()
med_gen = combined['gen_change_pp'].median()

def classify_quad(row):
    sh  = row['share_change_pp'] > med_sh
    gen = row['gen_change_pp']   > med_gen
    if sh and gen:
        return 'A형: 매출↑+2030↑\n(2030 주도 성장)'
    elif sh and not gen:
        return 'B형: 매출↑+2030↓\n(비2030 주도 성장)'
    elif not sh and gen:
        return 'C형: 매출↓+2030↑\n(2030 유입, 규모↓)'
    else:
        return 'D형: 매출↓+2030↓\n(이중 침체)'

combined['type'] = combined.apply(classify_quad, axis=1)

type_colors = {
    'A형: 매출↑+2030↑\n(2030 주도 성장)':  RISE_COLOR,
    'B형: 매출↑+2030↓\n(비2030 주도 성장)': '#3BB3A9',
    'C형: 매출↓+2030↑\n(2030 유입, 규모↓)': GEN_COLOR,
    'D형: 매출↓+2030↓\n(이중 침체)':         FALL_COLOR,
}

for t, grp in combined.groupby('type'):
    print(f"  {t.replace(chr(10),' ')}: {len(grp)}개  ex) {grp['행정동_코드_명'].head(5).tolist()}")

# ── 시각화 5: 4분면 산점도 ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 10), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

for t, grp in combined.groupby('type'):
    label = t.replace('\n', ' ')
    ax.scatter(grp['share_change_pp'], grp['gen_change_pp'],
               color=type_colors[t], s=70, alpha=0.75,
               label=f'{label} (n={len(grp)})',
               edgecolors='white', linewidth=0.8, zorder=3)

# 주요 행정동 레이블 (A형 상위 10, C형 상위 5)
highlight_a = combined[combined['type'].str.startswith('A')].nlargest(10, 'share_change_pp')
highlight_c = combined[combined['type'].str.startswith('C')].nlargest(5, 'gen_change_pp')
highlight_b = combined[combined['type'].str.startswith('B')].nlargest(5, 'share_change_pp')

for _, row in pd.concat([highlight_a, highlight_c, highlight_b]).iterrows():
    ax.annotate(row['행정동_코드_명'],
                (row['share_change_pp'], row['gen_change_pp']),
                xytext=(5, 3), textcoords='offset points',
                fontsize=7.5, fontweight='bold', alpha=0.9,
                arrowprops=dict(arrowstyle='->', color='#999', lw=0.7))

ax.axvline(med_sh,  color='#555', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(med_gen, color='#555', linestyle='--', linewidth=1, alpha=0.7)
ax.text(ax.get_xlim()[1]*0.98, med_gen + 0.1, '중앙값', fontsize=8, color='#555', ha='right')
ax.text(med_sh + 0.0001, ax.get_ylim()[1]*0.98, '중앙값', fontsize=8, color='#555')

ax.set_xlabel('매출 점유율 변화 (2020→2024, pp)', fontsize=11)
ax.set_ylabel('2030세대 소비 비중 변화 (2020→2024, pp)', fontsize=11)
ax.set_title('행정동 소비 유형 분류: 매출 점유율 × 2030 비중 변화 매트릭스\n(2020→2024)',
             fontsize=13, fontweight='bold', pad=15)
ax.legend(fontsize=9, loc='upper left',
          framealpha=0.8, edgecolor='#ddd')
ax.grid(alpha=0.25, linestyle='--')
ax.spines[['top', 'right']].set_visible(False)

# 4분면 라벨 배경
xlim = ax.get_xlim()
ylim = ax.get_ylim()
quad_labels = [
    (xlim[1]*0.7, ylim[1]*0.85, 'A형\n2030 주도 성장', RISE_COLOR),
    (xlim[0]*0.5, ylim[1]*0.85, 'B형\n非2030 성장',   '#3BB3A9'),
    (xlim[1]*0.7, ylim[0]*0.5, 'C형\n2030 유입↑',    GEN_COLOR),
    (xlim[0]*0.5, ylim[0]*0.5, 'D형\n이중 침체',      FALL_COLOR),
]
for x, y, label, c in quad_labels:
    ax.text(x, y, label, fontsize=9, color=c, fontweight='bold',
            ha='center', va='center', alpha=0.3,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=c, alpha=0.07, edgecolor='none'))

plt.tight_layout()
plt.savefig(f'{BASE_DIR}/data/share_gen2030_matrix.png', dpi=150, bbox_inches='tight',
            facecolor=BG_COLOR)
plt.close()
print("  -> share_gen2030_matrix.png 저장 완료")


# ══════════════════════════════════════════════════════════════════════════════
# 최종 요약 출력
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("분석 완료 요약")
print("="*70)

print("\n[1] 매출 점유율 Shift-Share (2020→2024)")
print("  상승 Top 5:")
for _, row in top20_up.head(5).iterrows():
    print(f"    {row['행정동_코드_명']:15s}  {row['share_change_pp']:+.4f}pp")
print("  하락 Bottom 5:")
for _, row in top20_down.head(5).iterrows():
    print(f"    {row['행정동_코드_명']:15s}  {row['share_change_pp']:+.4f}pp")

print("\n[2] 2030세대 비중 변화 (2020→2024)")
print("  유입 Top 5:")
for _, row in gen_top20_up.head(5).iterrows():
    print(f"    {row['행정동_코드_명']:15s}  2020: {row[2020]*100:.1f}%  →  2024: {row[2024]*100:.1f}%  ({row['gen_change_pp']:+.1f}pp)")
print("  이탈 Bottom 5:")
for _, row in gen_top20_down.head(5).iterrows():
    print(f"    {row['행정동_코드_명']:15s}  2020: {row[2020]*100:.1f}%  →  2024: {row[2024]*100:.1f}%  ({row['gen_change_pp']:+.1f}pp)")

print("\n[3] 네이버 트렌드 vs 매출 점유율 상관 (점유율 기준)")
print(f"  동시 상관 유의미 행정동: {len(sig_simul)}개")
print(f"  변화율 상관 유의미 행정동: {len(sig_diff)}개")
lag_pos = (res_df['best_lag'] > 0).sum()
lag_neg = (res_df['best_lag'] < 0).sum()
lag_zero = (res_df['best_lag'] == 0).sum()
print(f"  Lag 분포: 네이버 선행={lag_pos}개, 매출 선행={lag_neg}개, 동기={lag_zero}개")

print("\n[4] 블로그 vs 매출 점유율")
if len(blog_res_df) >= 5:
    print(f"  블로그 성장률 vs 매출 점유율 성장률: r={r_growth:+.3f}, p={p_growth:.3f}")
    print(f"  블로그 성장률 vs 점유율 변화(pp):    r={r_pp:+.3f}, p={p_pp:.3f}")

print("\n생성된 파일:")
files = [
    'share_shift_ranking.png',
    'gen2030_share_change_ranking.png',
    'gen2030_share_trend_line.png',
    'naver_share_correlation.png',
    'blog_share_correlation.png',
    'share_gen2030_matrix.png',
]
for f in files:
    print(f"  data/{f}")
