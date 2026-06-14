# -*- coding: utf-8 -*-
"""
티어별 상권 프로파일 비교 시각화
================================
"각 티어가 어떻게 다른가" — 집계 합계가 아닌 상권당 평균으로 비교
스토리라인:
  1. 티어별 핵심 지표 평균 비교 (2024 기준) → 지금 어떤 상태인가
  2. 2020→2024 변화량 비교 → 어떻게 달라졌는가
  3. 대표 상권 경로 → 각 티어의 실제 사례
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

data_dir  = 'data'
image_dir = 'image'

master = pd.read_csv(os.path.join(data_dir, '트렌드매출_마스터.csv'), encoding='utf-8-sig')

KEY_TIERS  = ['① 레전드', '② 라이징스타', '③ 급성장 신흥', '⑤ 성숙·정체']
TIER_COLORS = {
    '① 레전드':    '#8b0000',
    '② 라이징스타': '#e15759',
    '③ 급성장 신흥': '#59a14f',
    '⑤ 성숙·정체': '#f28e2b',
}
TIER_LABELS = {
    '① 레전드':    '① 레전드\n(21개)',
    '② 라이징스타': '② 라이징스타\n(25개)',
    '③ 급성장 신흥': '③ 급성장 신흥\n(18개)',
    '⑤ 성숙·정체': '⑤ 성숙·정체\n(53개)',
}

# 각 티어 대표 상권 (출력에서 확인된 실제 상권)
TIER_REPS = {
    '① 레전드':    ['이태원1동', '혜화동', '서교동', '방배2동'],
    '② 라이징스타': ['성수1가2동', '청운효자동', '금호1가동', '명동'],
    '③ 급성장 신흥': ['성수1가1동', '서빙고동', '여의동', '연희동'],
    '⑤ 성숙·정체': ['연남동', '삼청동', '화양동', '가회동'],
}

data_2020 = master[master['연도'] == 2020].set_index('행정동')
data_2024 = master[master['연도'] == 2024].set_index('행정동')

METRICS = ['트렌드매출_비중', '업종점수', '금요효과', '주말비중', '야간비중', '2030비중']
METRIC_LABELS = ['트렌드매출\n비중', '업종점수\n(F&B·라이프)', '금요일\n효과', '주말\n비중', '야간\n비중', '2030\n비중']


# ════════════════════════════════════════════════════════════
# 시각화 1. 티어별 지표 평균 비교 (2024 기준 막대)
#   → "지금 이 티어 상권들은 어떤 특성을 갖고 있는가"
# ════════════════════════════════════════════════════════════
tier_avg_2024 = (
    master[master['연도'] == 2024]
    .groupby('상권티어')[METRICS].mean()
)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('티어별 상권 프로파일 (2024년 기준, 상권당 평균)\n"각 티어는 어떤 특성을 갖는가"',
             fontsize=14, fontweight='bold')
axes = axes.flatten()

for i, (metric, label) in enumerate(zip(METRICS, METRIC_LABELS)):
    ax = axes[i]
    vals  = [tier_avg_2024.loc[t, metric] if t in tier_avg_2024.index else 0 for t in KEY_TIERS]
    colors = [TIER_COLORS[t] for t in KEY_TIERS]
    xlabels = [TIER_LABELS[t] for t in KEY_TIERS]

    bars = ax.bar(xlabels, vals, color=colors, alpha=0.85, edgecolor='white', width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                f'{v:.2f}' if metric not in ['트렌드매출_비중','주말비중','야간비중','2030비중','업종점수']
                else f'{v:.1%}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(vals) * 1.25)
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'tier_profile_2024.png'), dpi=150, bbox_inches='tight')
print("저장: image/tier_profile_2024.png")
plt.show()


# ════════════════════════════════════════════════════════════
# 시각화 2. 2020→2024 변화량 비교
#   → "각 티어는 어떻게 달라졌는가"
# ════════════════════════════════════════════════════════════
tier_avg_2020 = (
    master[master['연도'] == 2020]
    .groupby('상권티어')[METRICS].mean()
)

CHANGE_METRICS = ['트렌드매출_비중', '업종점수', '금요효과', '주말비중', '2030비중']
CHANGE_LABELS  = ['트렌드매출 비중', '업종점수', '금요효과', '주말비중', '2030비중']

fig2, ax = plt.subplots(figsize=(16, 7))
fig2.suptitle('티어별 2020→2024 지표 변화량 비교 (상권당 평균)\n"어떻게 달라졌는가"',
              fontsize=14, fontweight='bold')

n = len(CHANGE_METRICS)
x = np.arange(n)
w = 0.18

for idx, t in enumerate(KEY_TIERS):
    if t not in tier_avg_2020.index or t not in tier_avg_2024.index:
        continue
    changes = [tier_avg_2024.loc[t, m] - tier_avg_2020.loc[t, m] for m in CHANGE_METRICS]
    offset  = (idx - 1.5) * w
    bars = ax.bar(x + offset, changes, w,
                  color=TIER_COLORS[t], alpha=0.85, label=TIER_LABELS[t].replace('\n',' '),
                  edgecolor='white')
    for bar, v in zip(bars, changes):
        if abs(v) > 0.005:
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + (0.003 if v >= 0 else -0.003),
                    f'{v:+.3f}', ha='center',
                    va='bottom' if v >= 0 else 'top',
                    fontsize=7, color=TIER_COLORS[t], fontweight='bold')

ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(CHANGE_LABELS, fontsize=11)
ax.set_ylabel('변화량 (2024 - 2020, 평균)', fontsize=10)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, axis='y', alpha=0.3)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'tier_profile_change.png'), dpi=150, bbox_inches='tight')
print("저장: image/tier_profile_change.png")
plt.show()


# ════════════════════════════════════════════════════════════
# 시각화 3. 대표 상권 4개씩 — 복합점수_분위 5년 궤적
#   → "실제로 이 상권들이 어떻게 움직였는가"
# ════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 2, figsize=(18, 12))
fig3.suptitle('티어별 대표 상권 5년 궤적 (복합점수 분위)\n"실제 상권들이 어떻게 움직였는가"',
              fontsize=14, fontweight='bold')
axes3 = axes3.flatten()

for idx, (tier, reps) in enumerate(TIER_REPS.items()):
    ax = axes3[idx]
    color = TIER_COLORS[tier]

    for dong in reps:
        sub = master[master['행정동'] == dong].sort_values('연도')
        if len(sub) == 0:
            continue
        yrs  = sub['연도'].astype(str).tolist()
        vals = sub['복합점수_분위'].tolist()
        ax.plot(yrs, vals, marker='o', linewidth=2.2, markersize=7,
                alpha=0.85, label=dong)
        ax.annotate(f'{dong}\n({vals[-1]:.2f})',
                    xy=(yrs[-1], vals[-1]),
                    xytext=(6, 0), textcoords='offset points',
                    fontsize=8, va='center', color='#333333')

    ax.axhspan(0.70, 1.0, alpha=0.06, color=color, label='상위 30%')
    ax.axhline(0.70, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{tier}\n대표 상권 궤적', fontsize=11, fontweight='bold', color=color)
    ax.set_ylabel('복합점수 분위 (0=하위, 1=상위)', fontsize=9)
    ax.legend(fontsize=8.5, loc='lower right' if idx in [0,1] else 'upper right')
    ax.grid(True, alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'tier_profile_trajectory.png'), dpi=150, bbox_inches='tight')
print("저장: image/tier_profile_trajectory.png")
plt.show()

print("\n완료!")
