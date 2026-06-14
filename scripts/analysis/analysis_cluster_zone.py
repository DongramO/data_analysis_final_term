# -*- coding: utf-8 -*-
"""
상권 클러스터 단위 분석
========================
문제 제기:
  행정동 단위 분석은 실제 상권 경계와 불일치 →
  하나의 상권이 여러 행정동으로 쪼개지면 신호가 분산됨

3가지 확인:
  1. 분류 사각지대 진단  — 분위는 높은데 ⑦일반인 행정동
  2. 클러스터 단위 재계산 — 상권 묶음 기준 가중 복합점수
  3. 개별 vs 클러스터 티어 비교 시각화
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

data_dir  = 'data'
image_dir = 'image'

master = pd.read_csv(os.path.join(data_dir, '트렌드매출_마스터.csv'), encoding='utf-8-sig')
YEARS  = [2020, 2021, 2022, 2023, 2024]

# ── 상권 클러스터 정의 ────────────────────────────────────────────────
# 실제 소비자가 인식하는 상권 단위로 묶음 (인접 + 동일 생활권)
ZONES = {
    '성수동':      ['성수1가1동', '성수1가2동', '성수2가1동', '성수2가3동'],
    '이태원·한남': ['이태원1동', '이태원2동', '한남동', '보광동'],
    '홍대·합정':   ['서교동', '합정동', '동교동'],
    '연남·망원':   ['연남동', '망원1동', '망원2동'],
    '북촌·서촌':   ['가회동', '삼청동', '청운효자동', '서원동'],
    '혜화·대학로': ['혜화동', '명륜3가동'],
    '용산':        ['용산2가동', '후암동', '남영동'],
    '낙성대·서울대': ['낙성대동', '서원동'],
    '압구정·청담': ['압구정동', '청담동', '신사동', '논현2동'],
    '방배':        ['방배2동', '방배3동', '방배4동'],
    '왕십리':      ['왕십리2동', '행당1동'],
    '잠실·송파':   ['잠실6동', '잠실7동', '잠실본동'],
}

TIER_COLORS = {
    '① 레전드':    '#8b0000',
    '② 라이징스타': '#e15759',
    '③ 급성장 신흥': '#59a14f',
    '④ 잠재 후보':  '#4e79a7',
    '⑤ 성숙·정체': '#f28e2b',
    '⑥ 쇠퇴 주의': '#9467bd',
    '⑦ 일반':      '#aaa',
    '클러스터':    '#1f77b4',
}


# ════════════════════════════════════════════════════════════
# 1. 분류 사각지대 진단
#    "분위 ≥ 0.80 인데 ⑦ 일반으로 분류된 행정동은?"
# ════════════════════════════════════════════════════════════
df24 = master[master['연도'] == 2024].copy()
blind_spots = (df24[(df24['복합점수_분위'] >= 0.80) & (df24['상권티어'] == '⑦ 일반')]
               .sort_values('복합점수_분위', ascending=False)
               [['행정동', '복합점수_분위', '트렌드매출_비중', '업종점수', '2030비중']])

print("=== 사각지대: 분위 ≥ 0.80 이면서 ⑦ 일반 ===")
print(blind_spots.to_string(index=False))
print(f"\n→ 총 {len(blind_spots)}개 행정동이 높은 분위임에도 어떤 티어에도 해당되지 않음")

# 해당 행정동이 어느 클러스터에 속하는지
print("\n→ 클러스터 귀속:")
for _, row in blind_spots.iterrows():
    dong = row['행정동']
    found = [z for z, dongs in ZONES.items() if dong in dongs]
    label = ', '.join(found) if found else '미정의 클러스터'
    print(f"   {dong} ({row['복합점수_분위']:.3f}) → {label}")


# ════════════════════════════════════════════════════════════
# 2. 클러스터 단위 재계산
#    매출 가중 평균으로 복합점수·트렌드매출_비중 집계
# ════════════════════════════════════════════════════════════
cluster_records = []

for zone, dongs in ZONES.items():
    for yr in YEARS:
        yr_data = master[(master['연도'] == yr) & (master['행정동'].isin(dongs))]
        if yr_data.empty:
            continue

        total_sales = yr_data['총_매출'].sum()
        if total_sales == 0:
            continue

        w = yr_data['총_매출'] / total_sales  # 매출 가중치

        cluster_records.append({
            '클러스터':         zone,
            '연도':             yr,
            '포함_행정동수':    len(yr_data),
            '총_매출_합':       total_sales,
            '복합점수_분위_가중': (yr_data['복합점수_분위'] * w).sum(),
            '트렌드매출_비중_가중': (yr_data['트렌드매출_비중'] * w).sum(),
            '업종점수_가중':    (yr_data['업종점수'] * w).sum(),
            '2030비중_가중':    (yr_data['2030비중'] * w).sum(),
        })

cluster_df = pd.DataFrame(cluster_records)

# 클러스터 momentum (2020→2024 분위 변화)
cluster_mom = {}
for zone in ZONES:
    sub = cluster_df[cluster_df['클러스터'] == zone].set_index('연도')
    if 2020 in sub.index and 2024 in sub.index:
        cluster_mom[zone] = (sub.loc[2024, '복합점수_분위_가중'] -
                             sub.loc[2020, '복합점수_분위_가중'])

print("\n\n=== 클러스터 단위 2024년 분위 vs 행정동 개별 최고/최저 ===")
df24_full = master[master['연도'] == 2024].set_index('행정동')
print(f"{'클러스터':<15} {'클러스터분위':>10} {'개별최고':>8} {'개별최저':>8} "
      f"{'모멘텀':>8} {'행정동들(티어)'}")
print("-" * 90)
for zone, dongs in ZONES.items():
    c_sub = cluster_df[(cluster_df['클러스터'] == zone) & (cluster_df['연도'] == 2024)]
    if c_sub.empty:
        continue
    c_q = c_sub['복합점수_분위_가중'].iloc[0]
    exist = [d for d in dongs if d in df24_full.index]
    if not exist:
        continue
    indiv_max = df24_full.loc[exist, '복합점수_분위'].max()
    indiv_min = df24_full.loc[exist, '복합점수_분위'].min()
    mom = cluster_mom.get(zone, np.nan)

    tier_info = ', '.join(
        f"{d}({df24_full.loc[d,'상권티어']})" for d in exist
    )
    print(f"{zone:<15} {c_q:>10.3f} {indiv_max:>8.3f} {indiv_min:>8.3f} "
          f"{mom:>+8.3f}  {tier_info}")


# ════════════════════════════════════════════════════════════
# 3. 시각화
# ════════════════════════════════════════════════════════════
FOCUS_ZONES = ['성수동', '이태원·한남', '홍대·합정', '연남·망원',
               '북촌·서촌', '압구정·청담']

fig = plt.figure(figsize=(22, 18))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35,
                        height_ratios=[1, 1, 0.9])

# ── 상단 2행: 클러스터별 개별 행정동 궤적 + 클러스터 평균 ──────────
for idx, zone in enumerate(FOCUS_ZONES):
    row_i, col_i = divmod(idx, 3)
    ax = fig.add_subplot(gs[row_i, col_i])
    dongs = ZONES[zone]
    exist = [d for d in dongs if d in master['행정동'].values]

    # 개별 행정동 궤적
    for dong in exist:
        sub = master[master['행정동'] == dong].sort_values('연도')
        if len(sub) < 2:
            continue
        tier = sub[sub['연도'] == 2024]['상권티어'].values
        tier_str = tier[0] if len(tier) else '⑦ 일반'
        tc = TIER_COLORS.get(tier_str, '#aaa')
        ax.plot(sub['연도'], sub['복합점수_분위'],
                color=tc, alpha=0.55, linewidth=1.6, marker='o', markersize=4)
        ax.annotate(f"{dong}\n({tier_str})",
                    xy=(sub['연도'].iloc[-1], sub['복합점수_분위'].iloc[-1]),
                    xytext=(3, 0), textcoords='offset points',
                    fontsize=6.2, color=tc, va='center')

    # 클러스터 가중 평균 궤적 (파란 굵은 선)
    c_sub = cluster_df[cluster_df['클러스터'] == zone].sort_values('연도')
    if not c_sub.empty:
        ax.plot(c_sub['연도'], c_sub['복합점수_분위_가중'],
                color=TIER_COLORS['클러스터'], linewidth=3,
                linestyle='--', marker='D', markersize=6,
                label='클러스터 가중평균', zorder=6)
        last = c_sub.iloc[-1]
        ax.annotate(f"클러스터\n{last['복합점수_분위_가중']:.3f}",
                    xy=(last['연도'], last['복합점수_분위_가중']),
                    xytext=(-55, 8), textcoords='offset points',
                    fontsize=7.5, color=TIER_COLORS['클러스터'], fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=TIER_COLORS['클러스터'], lw=1.2))

    # 레전드 기준선
    ax.axhline(0.90, color='#8b0000', linewidth=0.7, linestyle=':', alpha=0.7)
    ax.axhline(0.70, color='#e15759', linewidth=0.7, linestyle=':', alpha=0.7)
    ax.text(YEARS[0], 0.905, '레전드선 0.90', fontsize=6.5, color='#8b0000', alpha=0.8)

    ax.set_title(f'{zone}\n({len(exist)}개 행정동)', fontsize=11, fontweight='bold')
    ax.set_xlabel('연도', fontsize=9)
    ax.set_ylabel('복합점수 분위', fontsize=9)
    ax.set_xticks(YEARS)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=7.5, loc='lower right')
    ax.grid(True, alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)


# ── 하단 왼쪽: 사각지대 행정동 vs 클러스터 ─────────────────────────
ax_gap = fig.add_subplot(gs[2, 0])
gap_zones = ['이태원·한남', '홍대·합정', '성수동', '북촌·서촌']
for zone in gap_zones:
    c_sub = cluster_df[cluster_df['클러스터'] == zone].sort_values('연도')
    ax_gap.plot(c_sub['연도'], c_sub['복합점수_분위_가중'],
                linewidth=2.2, marker='o', markersize=6, label=f'{zone}(클러스터)')

ax_gap.axhline(0.90, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
ax_gap.text(2024.05, 0.91, '레전드 기준선', fontsize=7, color='gray')
ax_gap.set_title('주요 클러스터 분위 궤적\n(클러스터 가중 평균)', fontsize=11, fontweight='bold')
ax_gap.set_xlabel('연도')
ax_gap.set_ylabel('복합점수 분위 (가중)')
ax_gap.set_xticks(YEARS)
ax_gap.set_ylim(0.7, 1.05)
ax_gap.legend(fontsize=8)
ax_gap.grid(True, alpha=0.25)
ax_gap.spines[['top', 'right']].set_visible(False)


# ── 하단 가운데: 클러스터 모멘텀 vs 가중 분위 버블차트 ──────────────
ax_bub = fig.add_subplot(gs[2, 1])
for zone in ZONES:
    c24 = cluster_df[(cluster_df['클러스터'] == zone) & (cluster_df['연도'] == 2024)]
    if c24.empty:
        continue
    q24  = c24['복합점수_분위_가중'].iloc[0]
    mom  = cluster_mom.get(zone, np.nan)
    size = c24['총_매출_합'].iloc[0] / 1e10 * 60  # 버블 크기 = 매출 규모

    ax_bub.scatter(mom, q24, s=max(size, 30), alpha=0.75,
                   color='#1f77b4', edgecolors='white', linewidths=1.2)
    ax_bub.annotate(zone, (mom, q24),
                    textcoords='offset points', xytext=(5, 3), fontsize=8)

ax_bub.axhline(0.90, color='#8b0000', linewidth=0.8, linestyle='--', alpha=0.6)
ax_bub.axhline(0.70, color='#e15759', linewidth=0.8, linestyle='--', alpha=0.6)
ax_bub.axvline(0,    color='gray',    linewidth=0.8, linestyle='--', alpha=0.5)
ax_bub.text(ax_bub.get_xlim()[0] if ax_bub.get_xlim()[0] > -1 else -0.35,
            0.905, '레전드선', fontsize=7, color='#8b0000')
ax_bub.set_xlabel('클러스터 모멘텀 (2020→2024 분위 변화)', fontsize=9)
ax_bub.set_ylabel('2024 클러스터 가중 분위', fontsize=9)
ax_bub.set_title('클러스터 수준 포지셔닝\n(버블 크기 = 총 매출 규모)', fontsize=11, fontweight='bold')
ax_bub.grid(True, alpha=0.25)
ax_bub.spines[['top', 'right']].set_visible(False)


# ── 하단 오른쪽: 성수동 행정동 분리 vs 통합 비교 ──────────────────────
ax_ss = fig.add_subplot(gs[2, 2])

seongsu_dongs = ZONES['성수동']
exist_ss = [d for d in seongsu_dongs if d in master['행정동'].values]

for dong in exist_ss:
    sub = master[master['행정동'] == dong].sort_values('연도')
    tier_val = sub[sub['연도'] == 2024]['상권티어'].values
    ts = tier_val[0] if len(tier_val) else '⑦ 일반'
    tc = TIER_COLORS.get(ts, '#aaa')
    ax_ss.plot(sub['연도'], sub['복합점수_분위'],
               color=tc, alpha=0.55, linewidth=1.8, marker='o', markersize=5,
               label=f'{dong}({ts})')

c_ss = cluster_df[cluster_df['클러스터'] == '성수동'].sort_values('연도')
ax_ss.plot(c_ss['연도'], c_ss['복합점수_분위_가중'],
           color='black', linewidth=3, linestyle='--', marker='D', markersize=7,
           label='성수동 클러스터 통합', zorder=6)

ax_ss.axhline(0.90, color='#8b0000', linewidth=0.9, linestyle=':', alpha=0.8)
ax_ss.text(YEARS[0], 0.905, '레전드 기준선', fontsize=7, color='#8b0000')

ax_ss.set_title('성수동: 행정동 분리 분석 vs 클러스터 통합\n"개별로 보면 ②③, 합치면?"',
                fontsize=11, fontweight='bold')
ax_ss.set_xlabel('연도')
ax_ss.set_ylabel('복합점수 분위')
ax_ss.set_xticks(YEARS)
ax_ss.set_ylim(0, 1.08)
ax_ss.legend(fontsize=8, loc='lower right')
ax_ss.grid(True, alpha=0.25)
ax_ss.spines[['top', 'right']].set_visible(False)


fig.suptitle('상권 클러스터 단위 분석\n"행정동 경계가 상권 신호를 얼마나 쪼개는가"',
             fontsize=14, fontweight='bold')

plt.savefig(os.path.join(image_dir, 'cluster_zone_analysis.png'), dpi=150, bbox_inches='tight')
print("\n저장: image/cluster_zone_analysis.png")
plt.show()


# ── 최종 요약 ──────────────────────────────────────────────────────────
print("\n=== 클러스터 통합 시 분위 변화 요약 (2024) ===")
print(f"{'클러스터':<15} {'통합분위':>8}  {'개별범위':<20} {'모멘텀':>8}  {'행정동 티어 현황'}")
print("-"*85)
for zone in ZONES:
    c24 = cluster_df[(cluster_df['클러스터'] == zone) & (cluster_df['연도'] == 2024)]
    if c24.empty:
        continue
    q = c24['복합점수_분위_가중'].iloc[0]
    exist = [d for d in ZONES[zone] if d in df24_full.index]
    if not exist:
        continue
    qs = df24_full.loc[exist, '복합점수_분위']
    mom = cluster_mom.get(zone, np.nan)
    tiers = sorted(set(df24_full.loc[exist, '상권티어'].values))
    print(f"{zone:<15} {q:>8.3f}  "
          f"{qs.min():.3f}~{qs.max():.3f}          "
          f"{mom:>+8.3f}  {', '.join(tiers)}")

print("\n완료!")
