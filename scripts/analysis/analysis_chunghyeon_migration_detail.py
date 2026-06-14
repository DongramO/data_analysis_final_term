# -*- coding: utf-8 -*-
"""
충현동 20대 주말 유입 이동성장세 상세 분석
출력: image/chunghyeon_migration_detail.png
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy.stats import linregress
from pathlib import Path

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT     = Path('.')
MIG_PATH = ROOT / 'data/main_data/분기별_주말유입인구.csv'
MAP_PATH = ROOT / 'data/main_data/코드매핑_유입인구_매출.csv'
RF_PATH  = ROOT / 'data/main_data/분기별_원본피처.csv'

_map = pd.read_csv(MAP_PATH, encoding='utf-8-sig')
_map.columns = ['mig_code', 'raw_code', 'gu', 'dong_map']

code2dong = (
    pd.read_csv(RF_PATH, encoding='utf-8-sig', usecols=['행정동_코드', '행정동'])
    .drop_duplicates()
    .rename(columns={'행정동_코드': 'raw_code', '행정동': 'dong'})
    .set_index('raw_code')['dong'].to_dict()
)

mig = pd.read_csv(MIG_PATH, encoding='utf-8-sig')
mig.columns = ['mig_code', 'yr', 'q', 'yrq', 'age', 'HE', 'WE', 'EE', 'mig_total']
mig['raw_code'] = mig['mig_code'].map(_map.set_index('mig_code')['raw_code'].to_dict())
mig = mig.dropna(subset=['raw_code'])
mig['raw_code'] = mig['raw_code'].astype(int)
mig['dong'] = mig['raw_code'].map(code2dong)
mig = mig.dropna(subset=['dong'])

ch = mig[mig['dong'] == '충현동'].copy()

# ── 분기 설정 ─────────────────────────────────────────────────
Q2023 = ['2023Q1', '2023Q2', '2023Q3', '2023Q4']
Q2024 = ['2024Q1', '2024Q2', '2024Q3', '2024Q4']
ANA_Q = Q2023 + Q2024

# 20대 분기별 피벗
q20  = ch[ch['age'] == 20].set_index('yrq')['mig_total'].sort_index()
all_q = sorted(q20.index.tolist())

# QoQ 성장률 (2022Q4 베이스 포함)
all_q_with_base = all_q  # 2022Q4가 포함돼 있음
qoq = {}
for q in ANA_Q:
    idx = all_q.index(q)
    prev = all_q[idx - 1]
    qoq[q] = (q20[q] - q20[prev]) / abs(q20[prev]) * 100

# 연간 합계 및 YoY
sum23 = q20[Q2023].sum()
sum24 = q20[Q2024].sum()
yoy   = (sum24 - sum23) / abs(sum23) * 100

# 선형 기울기 (2023~2024 8분기)
X = np.arange(8, dtype=float)
y = q20[ANA_Q].values.astype(float)
slope, intercept, r_val, _, _ = linregress(X, y)

# 연령대별 합계 비교
age_2023 = ch[ch['yrq'].isin(Q2023)].groupby('age')['mig_total'].sum()
age_2024 = ch[ch['yrq'].isin(Q2024)].groupby('age')['mig_total'].sum()

# ── 시각화 ───────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 13))
fig.suptitle('충현동  20대 주말 유입 이동성장세 상세 분석  (2020~2024)',
             fontsize=15, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig,
                       hspace=0.45, wspace=0.32,
                       left=0.08, right=0.95, top=0.92, bottom=0.06)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 0])
ax5 = fig.add_subplot(gs[2, 1])

# ── 1. 전체 분기 추이 ─────────────────────────────────────────
xvals  = np.arange(len(all_q))
y_all  = [q20[q] / 1e4 for q in all_q]
idx_23 = all_q.index('2023Q1')

ax1.fill_between(xvals, y_all, alpha=0.12, color='#2E6FAD')
ax1.plot(xvals, y_all, 'o-', color='#2E6FAD', linewidth=2.0,
         markersize=5, label='20대 유입 (만명)')

# 2023~2024 구간 추세선
x_trend = xvals[idx_23:]
y_trend = (intercept + slope * X) / 1e4
ax1.plot(x_trend, y_trend, '--', color='#E53935', linewidth=1.5,
         label=f'선형 추세 기울기 {slope/1e4:+.2f} 만명/분기  (R²={r_val**2:.3f})')

ax1.axvspan(idx_23 - 0.5, len(all_q) - 0.5, alpha=0.07, color='#FFC107')
ax1.set_xticks(xvals)
ax1.set_xticklabels([q[2:] for q in all_q], fontsize=8, rotation=45)
ax1.set_ylabel('유입 인구 (만명)', fontsize=10)
ax1.set_title('20대 주말 유입 분기별 추이 (전 구간)', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9, loc='upper left')
ax1.spines[['top', 'right']].set_visible(False)
ax1.grid(axis='y', alpha=0.2)
ax1.text(idx_23 - 0.2, max(y_all) * 0.97,
         '← 분석 구간 (2023~2024)', fontsize=8, color='#999', va='top')

# 주요 포인트 강조
for q, label_offset in [('2023Q1', 0.3), ('2023Q4', 0.3), ('2024Q4', 0.3)]:
    xi = all_q.index(q)
    yi = q20[q] / 1e4
    ax1.annotate(f'{yi:.1f}만', xy=(xi, yi), xytext=(xi, yi + label_offset),
                 ha='center', fontsize=8, color='#333',
                 arrowprops=dict(arrowstyle='->', color='#888', lw=0.8))

# ── 2. QoQ 성장률 ─────────────────────────────────────────────
q_labels = [q[2:] for q in ANA_Q]
qoq_vals = [qoq[q] for q in ANA_Q]
bar_colors = ['#43A047' if v >= 0 else '#E53935' for v in qoq_vals]

bars = ax2.bar(q_labels, qoq_vals, color=bar_colors, edgecolor='white', width=0.6)
ax2.axhline(0, color='#888', linewidth=0.8)
for bar, val in zip(bars, qoq_vals):
    va  = 'bottom' if val >= 0 else 'top'
    off = 1.2 if val >= 0 else -1.2
    ax2.text(bar.get_x() + bar.get_width() / 2, val + off,
             f'{val:.1f}%', ha='center', va=va, fontsize=8.5, fontweight='bold')

ax2.set_title('20대 유입 QoQ 성장률 (전분기 대비)', fontsize=11, fontweight='bold')
ax2.set_ylabel('QoQ 성장률 (%)', fontsize=10)
ax2.spines[['top', 'right']].set_visible(False)
ax2.grid(axis='y', alpha=0.2)
ax2.legend(handles=[
    mpatches.Patch(color='#43A047', label='증가'),
    mpatches.Patch(color='#E53935', label='감소'),
], fontsize=8.5, loc='lower right')

# ── 3. 연령대별 유입 비교 ─────────────────────────────────────
ages = sorted(age_2023.index.tolist())
age_labels = {0: '10세미만', 10: '10대', 20: '20대', 30: '30대',
              40: '40대', 50: '50대', 60: '60대', 70: '70대', 80: '80대'}
x = np.arange(len(ages))
w = 0.35
ax3.bar(x - w/2, [age_2023[a]/1e4 for a in ages], w,
        color='#90CAF9', edgecolor='white', label='2023 합계')
ax3.bar(x + w/2, [age_2024[a]/1e4 for a in ages], w,
        color='#1565C0', edgecolor='white', label='2024 합계')

ax3.set_xticks(x)
ax3.set_xticklabels([age_labels.get(a, str(a)) for a in ages], fontsize=8.5)
ax3.set_ylabel('유입 합계 (만명)', fontsize=10)
ax3.set_title('연령대별 주말 유입 합계 비교 (2023 vs 2024)', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.spines[['top', 'right']].set_visible(False)
ax3.grid(axis='y', alpha=0.2)

idx_20 = ages.index(20)
ax3.axvspan(idx_20 - 0.5, idx_20 + 0.5, alpha=0.12, color='#FF6F00')
ax3.text(idx_20, max(age_2024.values)/1e4 * 1.03, '20대',
         ha='center', fontsize=8.5, color='#FF6F00', fontweight='bold')

# 20대 변화율 표시
v23 = age_2023[20] / 1e4
v24 = age_2024[20] / 1e4
ch_pct = (v24 - v23) / v23 * 100
ax3.annotate(f'{ch_pct:+.1f}%', xy=(idx_20 + w/2, v24),
             xytext=(idx_20 + w/2 + 0.5, v24 + 5),
             fontsize=8.5, color='#E53935' if ch_pct < 0 else '#1B5E20',
             fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#888', lw=0.8))

# ── 4. 연도별 20대 유입 합계 ─────────────────────────────────
yrs = ['2020', '2021', '2022', '2023', '2024']
yr_sums = []
for yr in yrs:
    qs = [q for q in all_q if q.startswith(yr)]
    yr_sums.append(q20[qs].sum() / 1e4)

bar_c4 = ['#BBDEFB', '#90CAF9', '#64B5F6', '#1565C0', '#0D47A1']
ax4.bar(yrs, yr_sums, color=bar_c4, edgecolor='white', width=0.6)
for i, (yr, v) in enumerate(zip(yrs, yr_sums)):
    ax4.text(i, v + 0.5, f'{v:.1f}만', ha='center', fontsize=9.5, fontweight='bold')

ax4.set_title('연도별 20대 주말 유입 합계', fontsize=11, fontweight='bold')
ax4.set_ylabel('연간 유입 합계 (만명)', fontsize=10)
ax4.spines[['top', 'right']].set_visible(False)
ax4.grid(axis='y', alpha=0.2)

# 23→24 화살표
mid_y = (yr_sums[3] + yr_sums[4]) / 2
ax4.annotate('', xy=(4, yr_sums[4] - 0.5), xytext=(3, yr_sums[3] - 0.5),
             arrowprops=dict(arrowstyle='->', color='#E53935', lw=1.8))
ax4.text(3.5, mid_y + 2.5, f'YoY {yoy:+.1f}%',
         ha='center', fontsize=9.5, color='#E53935', fontweight='bold')

# ── 5. 지표 요약 ──────────────────────────────────────────────
ax5.axis('off')
ax5.set_title('핵심 지표 요약', fontsize=11, fontweight='bold', pad=8)

rows = [
    ('전체 순위',          '421위  /  422개 행정동',    '#C62828'),
    ('이동성장세점수',     '-2.277  (하위 0.2%)',       '#C62828'),
    ('z_증가율 (연간YoY)', f'{yoy:+.1f}%  →  z=-2.72', '#E53935'),
    ('z_기울기 (선형추세)', f'{slope/1e4:+.3f}만/분기  →  z=-3.57', '#E53935'),
    ('z_모멘텀 (QoQ평균)', 'QoQ 분기 평균  →  z=-0.54', '#E53935'),
    ('2023 합계',          f'{sum23/1e4:.1f} 만명',    '#555555'),
    ('2024 합계',          f'{sum24/1e4:.1f} 만명',    '#555555'),
    ('23→24 YoY',         f'{yoy:+.1f}%',             '#C62828'),
]

y_pos = 0.95
for label, val, color in rows:
    ax5.text(0.02, y_pos, label, transform=ax5.transAxes,
             fontsize=9.5, color='#666666', va='top')
    ax5.text(0.48, y_pos, val, transform=ax5.transAxes,
             fontsize=9.5, color=color, va='top', fontweight='bold')
    y_pos -= 0.10
    ax5.plot([0.0, 1.0], [y_pos + 0.055, y_pos + 0.055],
             color='#EEEEEE', linewidth=0.8, transform=ax5.transAxes)

ax5.text(0.5, 0.02,
    '20대 유입 -38% YoY, 기울기·증가율 모두 최하위권\n'
    '2023Q4 이후 급감 — 트렌드 상권으로 보기 어려운 수준',
    transform=ax5.transAxes, ha='center', va='bottom',
    fontsize=8.5, color='#555555',
    bbox=dict(boxstyle='round,pad=0.45', facecolor='#FFF9C4',
              edgecolor='#FBC02D', alpha=0.9))

out = ROOT / 'image/chunghyeon_migration_detail.png'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'[저장] {out}')
