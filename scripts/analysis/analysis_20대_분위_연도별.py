# -*- coding: utf-8 -*-
"""
20대 매출 비중 분위별 분기별 분석
- QoQ 성장률 (분위별 중앙값)
- 절대 매출 규모 (분기별 중앙값)
- 20대 비중 추이 (분위별 분기별 중앙값)

데이터: 분기별_원본피처.csv
출력: image/20대_분위_연도별_분석.png
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parent
SRC  = ROOT / 'data/main_data/분기별_원본피처.csv'

Q_LABELS  = ['하위 25%','25~50%','50~75%','상위 25%']
Q_COLORS  = ['#AAAAAA','#F5A623','#E05C5C','#8B1A1A']
SEOUL_COLOR = '#3A5CA9'

# ══ 1. 데이터 준비 ════════════════════════════════════════════
raw = pd.read_csv(SRC, encoding='utf-8-sig')
raw.columns = [c.strip() for c in raw.columns]

agg = raw.groupby(['행정동','연도분기','연도']).agg(
    총매출=('총_매출금액','sum'),
    매출20=('연령대_20_매출_금액','sum'),
).reset_index()
agg['비중20'] = agg['매출20'] / agg['총매출'] * 100

# 분기 순서 확정
QUARTERS = sorted(agg['연도분기'].unique())
NQ = len(QUARTERS)   # 20

# ── 행정동 × 분기 wide ────────────────────────────────────────
sales_wide = agg.pivot_table(index='행정동', columns='연도분기', values='총매출')
비중_wide  = agg.pivot_table(index='행정동', columns='연도분기', values='비중20')
sales_wide.columns = list(sales_wide.columns)
비중_wide.columns  = list(비중_wide.columns)

# 마지막 분기 기준 20대 비중으로 분위 구분
last_q = QUARTERS[-1]
비중_last = 비중_wide[last_q].dropna()

dong_df = sales_wide.copy()
dong_df['비중20_last'] = 비중_last
dong_df = dong_df.dropna(subset=QUARTERS + ['비중20_last'])
dong_df = dong_df[(dong_df[QUARTERS[0]] > 0) & (dong_df[last_q] > 0)]
dong_df['분위'] = pd.qcut(dong_df['비중20_last'], 4, labels=Q_LABELS)

비중_df = 비중_wide.copy()
비중_df['분위'] = dong_df['분위']
비중_df = 비중_df.dropna(subset=['분위'])

# ── QoQ 성장률 ────────────────────────────────────────────────
QOQ_PAIRS  = list(zip(QUARTERS[:-1], QUARTERS[1:]))
for q1, q2 in QOQ_PAIRS:
    dong_df[f'qoq_{q1}'] = (dong_df[q2] - dong_df[q1]) / dong_df[q1] * 100

qoq_x       = list(range(len(QOQ_PAIRS)))
qoq_labels  = [f"'{str(q2)[2:4]}Q{str(q2)[-1]}" for _, q2 in QOQ_PAIRS]
qoq_med     = {q: [dong_df[dong_df['분위']==q][f'qoq_{q1}'].median() for q1,_ in QOQ_PAIRS]
               for q in Q_LABELS}
seoul_qoq   = [dong_df[f'qoq_{q1}'].median() for q1,_ in QOQ_PAIRS]

# ── 절대 매출 (분기별 중앙값) ─────────────────────────────────
sales_med      = {q: [dong_df[dong_df['분위']==q][qq].median()/1e8 for qq in QUARTERS]
                  for q in Q_LABELS}
seoul_sales_med = [dong_df[qq].median()/1e8 for qq in QUARTERS]
ratios_last    = {q: dong_df[dong_df['분위']==q][last_q].median() / dong_df[last_q].median()
                  for q in Q_LABELS}

# ── 20대 비중 추이 ────────────────────────────────────────────
비중_med      = {q: [비중_df[비중_df['분위']==q][qq].median() for qq in QUARTERS]
                 for q in Q_LABELS}
seoul_비중_med = [비중_df[qq].median() for qq in QUARTERS]

# ── X축 레이블 (연도 첫 분기만 연도 표기) ────────────────────
def q_labels(quarters):
    labels, prev_yr = [], None
    for q in quarters:
        yr = str(q)[:4]
        label = f"'{yr[2:]}Q{str(q)[-1]}" if yr != prev_yr else f"Q{str(q)[-1]}"
        labels.append(label)
        prev_yr = yr
    return labels

Q_XTICK  = q_labels(QUARTERS)
X_QUARTERS = np.arange(NQ)
X_QOQ      = np.arange(len(QOQ_PAIRS))

# 연도 경계 위치 (Q1 시작 인덱스)
def yr_boundaries(quarters, offset=0):
    bounds = []
    prev_yr = None
    for i, q in enumerate(quarters):
        yr = str(q)[:4]
        if yr != prev_yr and i > 0:
            bounds.append(i - 0.5 + offset)
        prev_yr = yr
    return bounds

# ══ 2. 시각화 ═════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(28, 7))
fig.suptitle(
    '20대 매출 비중 분위별 분기별 분석  │  QoQ 성장률 / 절대 매출 규모 / 20대 비중 추이\n'
    f'(분위 기준: {last_q[:4]}년 {last_q[-1]}분기 20대 매출 비중 4분위 / 서울 기준선은 중앙값)',
    fontsize=12, fontweight='bold', y=1.02)

# ── Panel 1: QoQ 성장률 ──────────────────────────────────────
ax1 = axes[0]
for i, (q, color) in enumerate(zip(Q_LABELS, Q_COLORS)):
    vals = qoq_med[q]
    ax1.plot(X_QOQ, vals, color=color, lw=2, marker='o', markersize=5, label=q, zorder=3)

ax1.plot(X_QOQ, seoul_qoq, color=SEOUL_COLOR, lw=1.8, ls='--', marker='D',
         markersize=5, label='서울 전체 (중앙값)', zorder=4)
ax1.axhline(0, color='gray', lw=0.8, ls=':')
for xb in yr_boundaries(QUARTERS[1:]):   # QOQ_PAIRS의 q2 기준
    ax1.axvline(xb, color='#DDDDDD', lw=0.8, ls='--', zorder=0)
ax1.set_xticks(X_QOQ)
ax1.set_xticklabels(qoq_labels, fontsize=8, rotation=45, ha='right')
ax1.set_ylabel('총매출 QoQ 성장률 중앙값 (%)', fontsize=11)
ax1.set_title('① 분기별 QoQ 총매출 성장률\n(분위 내 행정동들의 중앙값)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(alpha=0.2, axis='y')

# ── Panel 2: 절대 매출 규모 ──────────────────────────────────
ax2 = axes[1]
for q, color in zip(Q_LABELS, Q_COLORS):
    vals = sales_med[q]
    ax2.plot(X_QUARTERS, vals, color=color, lw=2, marker='o', markersize=5, label=q, zorder=3)
    r = ratios_last[q]
    ax2.annotate(f'{vals[-1]:.0f}억\n({r:.1f}배)',
                 (NQ-1, vals[-1]), fontsize=8, color=color,
                 xytext=(8, 0), textcoords='offset points', va='center')

ax2.plot(X_QUARTERS, seoul_sales_med, color=SEOUL_COLOR, lw=1.8, ls='--', marker='D',
         markersize=5, label='서울 전체 중앙값', zorder=4)
ax2.annotate(f'{seoul_sales_med[-1]:.0f}억',
             (NQ-1, seoul_sales_med[-1]), fontsize=8, color=SEOUL_COLOR,
             xytext=(8, 0), textcoords='offset points', va='center')
for xb in yr_boundaries(QUARTERS):
    ax2.axvline(xb, color='#DDDDDD', lw=0.8, ls='--', zorder=0)
ax2.set_xticks(X_QUARTERS)
ax2.set_xticklabels(Q_XTICK, fontsize=8, rotation=45, ha='right')
ax2.set_ylabel('행정동 총매출 (억 원)', fontsize=11)
ax2.set_title('② 분기별 절대 매출 규모\n(분위별 중앙값 / 서울 전체 중앙값 기준)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(alpha=0.2)

# ── Panel 3: 20대 비중 추이 ──────────────────────────────────
ax3 = axes[2]
for q, color in zip(Q_LABELS, Q_COLORS):
    vals = 비중_med[q]
    ax3.plot(X_QUARTERS, vals, color=color, lw=2, marker='o', markersize=5, label=q, zorder=3)

ax3.plot(X_QUARTERS, seoul_비중_med, color=SEOUL_COLOR, lw=1.8, ls='--', marker='D',
         markersize=5, label='서울 전체 중앙값', zorder=4)
for xb in yr_boundaries(QUARTERS):
    ax3.axvline(xb, color='#DDDDDD', lw=0.8, ls='--', zorder=0)
ax3.set_xticks(X_QUARTERS)
ax3.set_xticklabels(Q_XTICK, fontsize=8, rotation=45, ha='right')
ax3.set_ylabel('20대 매출 비중 중앙값 (%)', fontsize=11)
ax3.set_title('③ 분기별 20대 매출 비중 추이\n(비중 자체의 분기별 변화)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9, loc='upper right')
ax3.grid(alpha=0.2)

# 하단 주석
q25, q50, q75 = dong_df['비중20_last'].quantile([0.25, 0.5, 0.75])
fig.text(0.5, -0.04,
         f'※ 분위 기준 ({last_q[:4]}Q{last_q[-1]} 20대 매출 비중):  '
         f'하위25% ≤ {q25:.1f}%  |  {q25:.1f}~{q50:.1f}%  |  '
         f'{q50:.1f}~{q75:.1f}%  |  상위25% ≥ {q75:.1f}%   '
         f'(서울 전체 중앙값: {q50:.1f}%)',
         ha='center', fontsize=9, color='#555555')

plt.tight_layout()
out = ROOT / 'image/20대_분위_연도별_분석.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'[저장] {out.name}')

# 수치 요약
print('\n=== 20대 비중 중앙값 추이 (%) - 분기별 ===')
print(f"{'분위':<12}", "  ".join(f"{q:>8}" for q in QUARTERS))
for q in Q_LABELS:
    print(f"{q:<12}", "  ".join(f"{v:>7.1f}%" for v in 비중_med[q]))
print(f"{'서울 전체':<12}", "  ".join(f"{v:>7.1f}%" for v in seoul_비중_med))
