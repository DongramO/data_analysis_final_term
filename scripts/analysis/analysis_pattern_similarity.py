# -*- coding: utf-8 -*-
"""
티어 간 성장 패턴 유사성 분석
================================
가설: 레전드 / 라이징스타 / 급성장 신흥은 같은 성장 패턴을
      서로 다른 시점에서 밟고 있는 것인가?

방법:
  1. 개별 궤적 정규화 — 각 상권의 2020년 값을 0으로 맞춰 형태만 비교
  2. 티어 평균 궤적 오버레이 — 세 티어의 평균 궤적이 유사한 형태인지 시각 확인
  3. Lag 상관 분석 — 급성장 신흥을 n년 앞당겼을 때 라이징스타와 얼마나 닮는가?
  4. 미래 경로 추정 — 현재 궤적이 유지되면 언제 다음 티어에 진입하는가?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as sp_stats
from scipy.spatial.distance import euclidean
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

data_dir  = 'data'
image_dir = 'image'

master = pd.read_csv(os.path.join(data_dir, '트렌드매출_마스터.csv'), encoding='utf-8-sig')

TIERS  = ['① 레전드', '② 라이징스타', '③ 급성장 신흥']
COLORS = {'① 레전드': '#8b0000', '② 라이징스타': '#e15759', '③ 급성장 신흥': '#59a14f'}
YEARS  = [2020, 2021, 2022, 2023, 2024]

# ── 피벗: 행정동 × 연도 → 복합점수_분위 매트릭스 ────────────────
pivot = (
    master[master['상권티어'].isin(TIERS)]
    .pivot_table(index='행정동', columns='연도', values='복합점수_분위')
    .dropna()
)
pivot['티어'] = master[master['연도'] == 2024].set_index('행정동')['상권티어']
pivot = pivot.dropna(subset=['티어'])

# ── 정규화: 각 상권의 2020년 분위를 0 기준으로 변환 ────────────
norm = pivot[YEARS].copy()
norm = norm.sub(norm[2020], axis=0)   # 2020값을 빼서 변화량만


# ════════════════════════════════════════════════════════════
# 시각화 1. 개별 궤적 + 티어 평균 — 정규화된 형태 비교
#   "각 상권이 2020 이후 어떤 방향·속도로 움직였는가"
# ════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 14))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.3)

# 상단: 티어별 개별 궤적 (얇은 선) + 평균 (굵은 선)
tier_means = {}
axes_top = []
for i, tier in enumerate(TIERS):
    ax = fig.add_subplot(gs[0, i])
    axes_top.append(ax)
    sub = norm[pivot['티어'] == tier]
    tier_means[tier] = sub.mean()

    for _, row in sub.iterrows():
        ax.plot(YEARS, row[YEARS].values,
                color=COLORS[tier], alpha=0.2, linewidth=1.2)

    mean_vals = tier_means[tier][YEARS].values
    ax.plot(YEARS, mean_vals,
            color=COLORS[tier], linewidth=3, label='평균 궤적', zorder=5)
    ax.fill_between(YEARS,
                    sub.min()[YEARS].values,
                    sub.max()[YEARS].values,
                    color=COLORS[tier], alpha=0.08)

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_title(f'{tier}\n({len(sub)}개 상권)', fontsize=11, fontweight='bold',
                 color=COLORS[tier])
    ax.set_xlabel('연도', fontsize=9)
    ax.set_ylabel('분위 변화량 (2020 기준 = 0)', fontsize=9)
    ax.set_xticks(YEARS)
    ax.set_xticklabels([str(y) for y in YEARS], fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)

fig.suptitle('티어 간 성장 패턴 유사성 분석\n(2020 기준 정규화 — 형태만 비교)',
             fontsize=14, fontweight='bold')


# ════════════════════════════════════════════════════════════
# 하단 왼쪽: 세 티어 평균 궤적 오버레이
#   "형태가 유사한가?"
# ════════════════════════════════════════════════════════════
ax_overlay = fig.add_subplot(gs[1, 0])

for tier in TIERS:
    vals = tier_means[tier][YEARS].values
    ax_overlay.plot(YEARS, vals,
                    color=COLORS[tier], linewidth=2.5, marker='o', markersize=7,
                    label=tier)
    for yr, v in zip(YEARS, vals):
        ax_overlay.annotate(f'{v:+.3f}', (yr, v),
                            textcoords='offset points', xytext=(0, 8),
                            ha='center', fontsize=7, color=COLORS[tier])

ax_overlay.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax_overlay.set_title('3개 티어 평균 궤적 오버레이\n"형태가 유사한가?"', fontsize=11, fontweight='bold')
ax_overlay.set_xlabel('연도', fontsize=9)
ax_overlay.set_ylabel('분위 변화량 (평균)', fontsize=9)
ax_overlay.legend(fontsize=8)
ax_overlay.set_xticks(YEARS)
ax_overlay.grid(True, alpha=0.25)
ax_overlay.spines[['top', 'right']].set_visible(False)


# ════════════════════════════════════════════════════════════
# 하단 가운데: Lag 상관 분석
#   급성장 신흥을 1~3년 앞당기면 라이징스타 / 레전드와 얼마나 닮는가?
# ════════════════════════════════════════════════════════════
ax_lag = fig.add_subplot(gs[1, 1])

# 각 티어의 연도별 평균 변화량 벡터
vec = {t: tier_means[t][YEARS].values for t in TIERS}

# Lag 상관: 급성장 신흥을 lag만큼 앞당겨 나머지 두 티어와 비교
lag_results = []
for lag in range(0, 4):
    for target_tier in ['① 레전드', '② 라이징스타']:
        # 급성장 신흥을 lag 앞당기면: 급성장의 [lag:] vs target의 [:len-lag]
        src = vec['③ 급성장 신흥']
        tgt = vec[target_tier]
        n   = len(src) - lag
        if n < 3:
            continue
        r, p = sp_stats.pearsonr(src[lag:], tgt[:n])
        lag_results.append({'lag': lag, '비교대상': target_tier, 'r': r, 'p': p})

lag_df = pd.DataFrame(lag_results)
print("=== Lag 상관 분석 (급성장 신흥 → 각 티어) ===")
print(lag_df.to_string(index=False))

for target_tier in ['① 레전드', '② 라이징스타']:
    sub = lag_df[lag_df['비교대상'] == target_tier]
    ax_lag.plot(sub['lag'], sub['r'],
                color=COLORS[target_tier], marker='o', linewidth=2,
                markersize=7, label=f'급성장 → {target_tier}')
    for _, row in sub.iterrows():
        sig = '*' if row['p'] < 0.05 else ''
        ax_lag.annotate(f'{row["r"]:.2f}{sig}',
                        (row['lag'], row['r']),
                        textcoords='offset points', xytext=(0, 8),
                        ha='center', fontsize=8, color=COLORS[target_tier])

ax_lag.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax_lag.set_xlabel('급성장 신흥을 앞당긴 연수 (Lag)', fontsize=9)
ax_lag.set_ylabel('피어슨 상관계수 r', fontsize=9)
ax_lag.set_title('Lag 상관\n"급성장을 N년 앞당기면 얼마나 닮는가?"', fontsize=11, fontweight='bold')
ax_lag.set_xticks([0, 1, 2, 3])
ax_lag.set_xticklabels(['0년\n(현재)', '1년 앞당김', '2년 앞당김', '3년 앞당김'], fontsize=8)
ax_lag.legend(fontsize=8)
ax_lag.set_ylim(-1.1, 1.1)
ax_lag.grid(True, alpha=0.25)
ax_lag.spines[['top', 'right']].set_visible(False)


# ════════════════════════════════════════════════════════════
# 하단 오른쪽: 현재 궤적 유지 시 미래 경로 추정
#   "지금 속도를 유지하면 언제 다음 단계에 도달하는가?"
# ════════════════════════════════════════════════════════════
ax_proj = fig.add_subplot(gs[1, 2])

# 각 티어의 실제 분위 평균 (정규화 아닌 원래 값)
pivot_abs = (
    master[master['상권티어'].isin(TIERS)]
    .pivot_table(index='행정동', columns='연도', values='복합점수_분위')
    .dropna()
)
pivot_abs['티어'] = master[master['연도'] == 2024].set_index('행정동')['상권티어']
pivot_abs = pivot_abs.dropna(subset=['티어'])

tier_abs_means = {
    t: pivot_abs[pivot_abs['티어'] == t][YEARS].mean()
    for t in TIERS
}

future_years = [2025, 2026, 2027]
proj_years   = YEARS + future_years
all_years_str = [str(y) for y in proj_years]

for tier in TIERS:
    vals  = tier_abs_means[tier][YEARS].values
    slope, intercept, r, p, se = sp_stats.linregress(YEARS, vals)
    proj  = [intercept + slope * y for y in proj_years]
    proj_clipped = [min(max(v, 0), 1) for v in proj]

    # 실제값
    ax_proj.plot(YEARS, vals,
                 color=COLORS[tier], linewidth=2.5, marker='o', markersize=7,
                 label=tier, zorder=4)
    # 추정 구간
    ax_proj.plot(future_years, proj_clipped[len(YEARS):],
                 color=COLORS[tier], linewidth=1.8, linestyle=':', marker='s',
                 markersize=5, alpha=0.75, zorder=3)

# 티어 경계선
for threshold, label in [(0.90, '레전드 진입선 (0.90)'),
                          (0.70, '상위 30% 선 (0.70)')]:
    ax_proj.axhline(threshold, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
    ax_proj.text(proj_years[-1] + 0.1, threshold + 0.01, label, fontsize=7.5, color='gray')

# 2024/2025 구분선
ax_proj.axvline(2024.5, color='black', linewidth=0.8, linestyle=':', alpha=0.5)
ax_proj.text(2024.6, 0.15, '← 실제  추정 →', fontsize=8, color='#555555')

ax_proj.set_xlim(2019.5, 2027.5)
ax_proj.set_ylim(0, 1.05)
ax_proj.set_xticks(proj_years)
ax_proj.set_xticklabels(all_years_str, fontsize=7.5, rotation=30)
ax_proj.set_xlabel('연도', fontsize=9)
ax_proj.set_ylabel('복합점수 분위 (티어 평균)', fontsize=9)
ax_proj.set_title('현재 기울기 유지 시 미래 경로\n(점선 = 선형 외삽)', fontsize=11, fontweight='bold')
ax_proj.legend(fontsize=8)
ax_proj.grid(True, alpha=0.25)
ax_proj.spines[['top', 'right']].set_visible(False)

plt.savefig(os.path.join(image_dir, 'pattern_similarity.png'), dpi=150, bbox_inches='tight')
print("\n저장: image/pattern_similarity.png")
plt.show()

# ════════════════════════════════════════════════════════════
# 수치 요약 출력
# ════════════════════════════════════════════════════════════
print("\n=== 티어 평균 궤적 (원래 분위 값) ===")
for tier in TIERS:
    vals = tier_abs_means[tier][YEARS].values
    slope, intercept, r, p, se = sp_stats.linregress(YEARS, vals)
    proj_2025 = intercept + slope * 2025
    proj_2026 = intercept + slope * 2026
    print(f"\n{tier}")
    print(f"  2020: {vals[0]:.3f} → 2024: {vals[4]:.3f}  (변화: {vals[4]-vals[0]:+.3f})")
    print(f"  연간 기울기: {slope:+.4f}  R²={r**2:.3f}")
    print(f"  추정 2025: {min(proj_2025,1):.3f} / 2026: {min(proj_2026,1):.3f}")

print("\n완료!")
