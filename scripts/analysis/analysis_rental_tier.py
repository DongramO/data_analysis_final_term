# -*- coding: utf-8 -*-
"""
티어별 임대료·공실률 변화 분석 (2020~2024)
==========================================
① 레전드 / ② 라이징스타 / ③ 급성장 신흥의
임대료(천원/㎡)·공실률(%) 변화 양상 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

data_dir  = 'data'
image_dir = 'image'

# ── 데이터 로드 ──────────────────────────────────────────────────────
rental  = pd.read_csv(os.path.join(data_dir, '임대료_공실률.csv'),   encoding='utf-8-sig')
dongmap = pd.read_csv(os.path.join(data_dir, 'rental_vacancy_dongmap.csv'), encoding='utf-8-sig')
master  = pd.read_csv(os.path.join(data_dir, '트렌드매출_마스터.csv'),     encoding='utf-8-sig')

tier_2024 = (master[master['연도'] == 2024]
             .drop_duplicates('행정동')
             .set_index('행정동')['상권티어'])

dongmap['상권티어'] = dongmap['대표_행정동'].map(tier_2024)

TARGET_TIERS = ['① 레전드', '② 라이징스타', '③ 급성장 신흥']
COLORS = {'① 레전드': '#8b0000', '② 라이징스타': '#e15759', '③ 급성장 신흥': '#59a14f'}
YEARS  = [2020, 2021, 2022, 2023, 2024]

# 2020~2024, 3개 티어 필터
df = (rental[rental['연도'] <= 2024]
      .merge(dongmap[['상권명', '상권티어']], on='상권명', how='left'))
df = df[df['상권티어'].isin(TARGET_TIERS)].copy()


# ════════════════════════════════════════════════════════════
# 시각화
# Layout:
#   Row 0-1: 티어별 임대료(row0) / 공실률(row1) 개별 궤적
#   Row 2  : 3 티어 평균 오버레이(임대료) | 오버레이(공실률) | 임대료 변화율 비교
# ════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 16))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.35,
                        height_ratios=[1, 1, 0.9])

tier_avgs_rent = {}
tier_avgs_vac  = {}

for col_i, tier in enumerate(TARGET_TIERS):
    sub   = df[df['상권티어'] == tier]
    color = COLORS[tier]

    r_avg = sub.groupby('연도')['임대료'].mean()
    v_avg = sub.groupby('연도')['공실률'].mean()
    tier_avgs_rent[tier] = r_avg
    tier_avgs_vac[tier]  = v_avg

    for row_i, (metric, ylabel) in enumerate([('임대료', '임대료 (천원/㎡)'),
                                               ('공실률', '공실률 (%)')]):
        ax    = fig.add_subplot(gs[row_i, col_i])
        pivot = sub.pivot_table(index='연도', columns='상권명', values=metric)
        avg   = pivot.mean(axis=1)

        for name in pivot.columns:
            vals = pivot[name].dropna()
            if len(vals) < 2:
                continue
            ax.plot(vals.index, vals.values,
                    color=color, alpha=0.28, linewidth=1.4)
            ax.annotate(name,
                        xy=(vals.index[-1], vals.iloc[-1]),
                        xytext=(3, 0), textcoords='offset points',
                        fontsize=6.5, color='#555555', va='center')

        # 티어 평균 (굵은 선)
        ax.plot(avg.dropna().index, avg.dropna().values,
                color=color, linewidth=3, zorder=5, label='티어 평균')

        # 평균 끝점 값 주석
        last = avg.dropna()
        if not last.empty:
            ax.annotate(f'평균 {last.iloc[-1]:.1f}',
                        xy=(last.index[-1], last.iloc[-1]),
                        xytext=(-52, 10), textcoords='offset points',
                        fontsize=8, color=color, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

        if row_i == 0:
            n_sang = sub['상권명'].nunique()
            ax.set_title(f'{tier}  ({n_sang}개 상권)',
                         fontsize=11, fontweight='bold', color=color)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel('연도', fontsize=9)
        ax.set_xticks(YEARS)
        ax.set_xticklabels([str(y) for y in YEARS], fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)


# ── 하단 왼쪽: 임대료 3 티어 평균 오버레이 ──────────────────────────
ax_r = fig.add_subplot(gs[2, 0])
for tier in TARGET_TIERS:
    vals = tier_avgs_rent[tier].dropna()
    ax_r.plot(vals.index, vals.values,
              color=COLORS[tier], linewidth=2.5, marker='o', markersize=7,
              label=tier)
    for yr, v in zip(vals.index, vals.values):
        ax_r.annotate(f'{v:.0f}', (yr, v),
                      textcoords='offset points', xytext=(0, 8),
                      ha='center', fontsize=7.5, color=COLORS[tier])

ax_r.set_title('3 티어 평균 임대료 비교', fontsize=11, fontweight='bold')
ax_r.set_ylabel('임대료 (천원/㎡)', fontsize=9)
ax_r.set_xlabel('연도', fontsize=9)
ax_r.set_xticks(YEARS)
ax_r.set_xticklabels([str(y) for y in YEARS], fontsize=8)
ax_r.legend(fontsize=8, loc='upper left')
ax_r.grid(True, alpha=0.25)
ax_r.spines[['top', 'right']].set_visible(False)


# ── 하단 가운데: 공실률 3 티어 평균 오버레이 ────────────────────────
ax_v = fig.add_subplot(gs[2, 1])
for tier in TARGET_TIERS:
    vals = tier_avgs_vac[tier].dropna()
    ax_v.plot(vals.index, vals.values,
              color=COLORS[tier], linewidth=2.5, marker='o', markersize=7,
              label=tier)
    for yr, v in zip(vals.index, vals.values):
        ax_v.annotate(f'{v:.1f}%', (yr, v),
                      textcoords='offset points', xytext=(0, 8),
                      ha='center', fontsize=7.5, color=COLORS[tier])

ax_v.set_title('3 티어 평균 공실률 비교', fontsize=11, fontweight='bold')
ax_v.set_ylabel('공실률 (%)', fontsize=9)
ax_v.set_xlabel('연도', fontsize=9)
ax_v.set_xticks(YEARS)
ax_v.set_xticklabels([str(y) for y in YEARS], fontsize=8)
ax_v.legend(fontsize=8, loc='upper right')
ax_v.grid(True, alpha=0.25)
ax_v.spines[['top', 'right']].set_visible(False)


# ── 하단 오른쪽: 2020 대비 임대료 변화율 ──────────────────────────────
ax_idx = fig.add_subplot(gs[2, 2])
for tier in TARGET_TIERS:
    vals  = tier_avgs_rent[tier].dropna()
    base  = vals.get(2020, np.nan)
    if pd.isna(base) or base == 0:
        continue
    idx = (vals / base - 1) * 100
    ax_idx.plot(idx.index, idx.values,
                color=COLORS[tier], linewidth=2.5, marker='o', markersize=7,
                label=tier)
    for yr, v in zip(idx.index, idx.values):
        ax_idx.annotate(f'{v:+.1f}%', (yr, v),
                        textcoords='offset points', xytext=(0, 9),
                        ha='center', fontsize=7.5, color=COLORS[tier])

ax_idx.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
ax_idx.set_title('2020 대비 임대료 변화율\n(2020 = 0%)', fontsize=11, fontweight='bold')
ax_idx.set_ylabel('변화율 (%p)', fontsize=9)
ax_idx.set_xlabel('연도', fontsize=9)
ax_idx.set_xticks(YEARS)
ax_idx.set_xticklabels([str(y) for y in YEARS], fontsize=8)
ax_idx.legend(fontsize=8)
ax_idx.grid(True, alpha=0.25)
ax_idx.spines[['top', 'right']].set_visible(False)


fig.suptitle('상권 티어별 임대료·공실률 변화 양상 (2020~2024)\n[매장용 빌딩 기준 · 상단: 개별 상권, 하단: 티어 평균 비교]',
             fontsize=14, fontweight='bold')

plt.savefig(os.path.join(image_dir, 'rental_vacancy_tier.png'), dpi=150, bbox_inches='tight')
print("저장: image/rental_vacancy_tier.png")
plt.show()


# ── 수치 요약 ──────────────────────────────────────────────────────────
print("\n=== 티어별 임대료·공실률 요약 ===")
for tier in TARGET_TIERS:
    r2020 = tier_avgs_rent[tier].get(2020, np.nan)
    r2024 = tier_avgs_rent[tier].get(2024, np.nan)
    v2020 = tier_avgs_vac[tier].get(2020,  np.nan)
    v2024 = tier_avgs_vac[tier].get(2024,  np.nan)
    r_chg = (r2024/r2020 - 1)*100 if r2020 else np.nan
    v_chg = v2024 - v2020
    print(f"\n{tier}")
    print(f"  임대료: {r2020:.1f} → {r2024:.1f} 천원/㎡  ({r_chg:+.1f}%)")
    print(f"  공실률: {v2020:.1f} → {v2024:.1f} %  ({v_chg:+.1f}pp)")

# 개별 상권별 임대료 변화 상위/하위
print("\n=== 임대료 상승률 Top 상권 (2020→2024, 2020 데이터 있는 상권) ===")
rent_chg = []
for name in df['상권명'].unique():
    sub = df[df['상권명'] == name].set_index('연도')['임대료']
    if 2020 in sub.index and 2024 in sub.index:
        r20, r24 = sub[2020], sub[2024]
        if pd.notna(r20) and pd.notna(r24) and r20 > 0:
            tier = df[df['상권명'] == name]['상권티어'].iloc[0]
            rent_chg.append({'상권명': name, '상권티어': tier,
                              '2020': r20, '2024': r24,
                              '변화율(%)': round((r24/r20-1)*100, 1)})

rent_chg_df = (pd.DataFrame(rent_chg)
               .sort_values('변화율(%)', ascending=False))
print(rent_chg_df.to_string(index=False))

print("\n완료!")
