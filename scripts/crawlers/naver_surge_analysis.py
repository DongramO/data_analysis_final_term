# -*- coding: utf-8 -*-
"""
후보군 네이버 트렌드 월별 급등 시점 분석
- 레퍼런스 vs 후보 군 비교
- 3개월 롤링 평균으로 급등 시점 감지
- 출력: image/naver_surge_01_reference.png, naver_surge_02_candidates.png, naver_surge_03_ranking.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 ──────────────────────────────────────────────────────────
df = pd.read_csv('data/main_data/naver_trend_candidates.csv', encoding='utf-8-sig')
df['period'] = pd.to_datetime(df['period'])
df = df.sort_values(['group','period']).reset_index(drop=True)

REFS  = ['성수','문래','신대방','여의','동화']
CANDS = ['역삼2동','문래동','서초동','장안동','아현동',
         '효창동','교남동','도곡동','양재동','풍납동']

# ── 급등 지표 계산 ─────────────────────────────────────────────────────────
def compute_surge_metrics(sub):
    sub = sub.copy().sort_values('period')
    sub['roll3'] = sub['ratio'].rolling(3, min_periods=1).mean()
    base = sub[sub['period'].dt.year.isin([2020,2021])]['ratio'].mean()
    base = base if base > 0 else 1.0
    sub['norm'] = sub['ratio'] / base          # 기준선 = 1.0
    sub['norm_roll'] = sub['roll3'] / base
    sub['yoy_chg'] = sub['ratio'].pct_change(12) * 100  # 전년 동월 대비 %
    return sub

processed = {}
for grp, sub in df.groupby('group'):
    processed[grp] = compute_surge_metrics(sub)

def surge_month(sub, threshold=1.8):
    """norm_roll이 처음으로 threshold 초과한 월 반환"""
    hit = sub[sub['norm_roll'] >= threshold]
    return hit['period'].min() if len(hit) > 0 else pd.NaT

def peak_norm(sub):
    return sub['norm_roll'].max()

def recent_yoy(sub, n_months=6):
    """최근 n개월 평균 YoY 변화율"""
    return sub.tail(n_months)['yoy_chg'].mean()


# ═══════════════════════════════════════════════════════════════
# Chart 1: 레퍼런스 상권 월별 트렌드 (성수·문래 등)
# ═══════════════════════════════════════════════════════════════
colors_ref = {'성수':'#E63946','문래':'#457B9D','신대방':'#2A9D8F','여의':'#E9C46A','동화':'#F4A261'}

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('레퍼런스 상권 네이버 트렌드 — 급부상 패턴 확인\n(2020~2024 월별, 3개월 롤링 평균, 2020-21 평균=1.0 정규화)',
             fontsize=14, fontweight='bold', y=1.01)
axes = axes.flatten()

for i, grp in enumerate(REFS):
    ax = axes[i]
    sub = processed[grp]
    sm = surge_month(sub)
    pk = peak_norm(sub)

    ax.fill_between(sub['period'], sub['norm_roll'], alpha=0.15, color=colors_ref[grp])
    ax.plot(sub['period'], sub['norm_roll'], color=colors_ref[grp], lw=2.5, label='3M 롤링')
    ax.plot(sub['period'], sub['norm'], color=colors_ref[grp], lw=0.8, alpha=0.4)
    ax.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(1.8, color='#E63946', lw=0.8, ls=':', alpha=0.6, label='급등 임계값(1.8x)')

    if pd.notna(sm):
        ax.axvline(sm, color='#E63946', lw=1.5, ls='--', alpha=0.7)
        ax.text(sm, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else pk*1.05,
                sm.strftime('%y.%m'), fontsize=8, color='#E63946',
                ha='left', va='bottom', rotation=90)

    ax.set_title(f'{grp}  (최고 {pk:.1f}x)', fontsize=11, fontweight='bold', color=colors_ref[grp])
    ax.set_xlabel('')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y.%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=8, loc='upper left')

axes[-1].axis('off')
plt.tight_layout()
plt.savefig('image/naver_surge_01_reference.png', dpi=150, bbox_inches='tight')
plt.close()
print('저장: image/naver_surge_01_reference.png')


# ═══════════════════════════════════════════════════════════════
# Chart 2: 후보군 월별 트렌드 (10개)
# ═══════════════════════════════════════════════════════════════
cmap = plt.cm.tab10
cand_colors = {c: cmap(i/10) for i, c in enumerate(CANDS)}

fig, axes = plt.subplots(2, 5, figsize=(22, 9))
fig.suptitle('차세대 후보군 네이버 트렌드 — 월별 급등 시점 분석\n(2020~2024, 3개월 롤링 평균, 2020-21 평균=1.0 정규화)',
             fontsize=14, fontweight='bold')
axes = axes.flatten()

# 후보군별 패턴 유사도 (validate_scorecard 기준)
pattern_sim = {
    '역삼2동': 0.870, '문래동': 0.830, '서초동': 0.809, '장안동': 0.775,
    '아현동': 0.728, '효창동': 0.724, '교남동': 0.771, '도곡동': 0.826,
    '양재동': 0.879, '풍납동': 0.776
}

for i, grp in enumerate(CANDS):
    ax = axes[i]
    sub = processed[grp]
    sm = surge_month(sub)
    pk = peak_norm(sub)
    ryoy = recent_yoy(sub)
    col = cand_colors[grp]

    ax.fill_between(sub['period'], sub['norm_roll'], alpha=0.15, color=col)
    ax.plot(sub['period'], sub['norm_roll'], color=col, lw=2.5)
    ax.plot(sub['period'], sub['norm'], color=col, lw=0.8, alpha=0.35)
    ax.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(1.8, color='#E63946', lw=0.7, ls=':', alpha=0.5)

    if pd.notna(sm):
        ax.axvline(sm, color='#E63946', lw=1.5, ls='--', alpha=0.8)
        ymax = sub['norm_roll'].max() * 1.05
        ax.text(sm, ymax, sm.strftime('%y.%m'), fontsize=8, color='#E63946',
                ha='left', va='bottom', rotation=90)

    sim = pattern_sim.get(grp, None)
    title_extra = f' [유사도 {sim:.3f}]' if sim else ''
    ax.set_title(f'{grp}{title_extra}', fontsize=9.5, fontweight='bold')

    # 최근 YoY 레이블
    yoy_str = f'최근YoY: {ryoy:+.1f}%' if not np.isnan(ryoy) else ''
    ax.text(0.03, 0.97, yoy_str, transform=ax.transAxes,
            fontsize=8, va='top', color='#333')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y.%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=7)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('image/naver_surge_02_candidates.png', dpi=150, bbox_inches='tight')
plt.close()
print('저장: image/naver_surge_02_candidates.png')


# ═══════════════════════════════════════════════════════════════
# Chart 3: 종합 랭킹 — 급등 시점 × 최고 배율 × 최근 YoY
# ═══════════════════════════════════════════════════════════════
summary_rows = []
for grp in REFS + CANDS:
    sub = processed[grp]
    sm = surge_month(sub)
    pk = peak_norm(sub)
    ryoy = recent_yoy(sub)
    summary_rows.append({
        'group': grp,
        'type': 'reference' if grp in REFS else 'candidate',
        'surge_month': sm,
        'peak_norm': round(pk, 2),
        'recent_yoy': round(ryoy, 1) if not np.isnan(ryoy) else None,
        'surge_lag_months': (sm.year-2020)*12 + sm.month - 1 if pd.notna(sm) else None,
    })

sumdf = pd.DataFrame(summary_rows).sort_values(['type','surge_lag_months'], na_position='last')

# 시각화
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

# 3-1: 급등 시점 타임라인
ax1 = fig.add_subplot(gs[0, :])
refs_s = sumdf[sumdf['type']=='reference'].dropna(subset=['surge_lag_months'])
cands_s = sumdf[sumdf['type']=='candidate'].dropna(subset=['surge_lag_months'])

for _, row in refs_s.iterrows():
    ax1.scatter(row['surge_lag_months'], 1.5, s=200, color='#E63946',
                zorder=5, marker='*')
    ax1.text(row['surge_lag_months'], 1.65, row['group'],
             fontsize=9, ha='center', color='#E63946', fontweight='bold')

for _, row in cands_s.iterrows():
    ax1.scatter(row['surge_lag_months'], 0.5, s=120, color='#457B9D',
                zorder=5, marker='D')
    ax1.text(row['surge_lag_months'], 0.62, row['group'],
             fontsize=9, ha='center', color='#457B9D')

ax1.set_xlim(-2, 60)
ax1.set_yticks([0.5, 1.5])
ax1.set_yticklabels(['후보군', '레퍼런스'], fontsize=11)
xticks = list(range(0, 60, 6))
xlabels = [f'{2020 + m//12}.{(m%12)+1:02d}' for m in xticks]
ax1.set_xticks(xticks)
ax1.set_xticklabels(xlabels, fontsize=8)
ax1.set_title('급등 시점 타임라인 (1.8× 최초 돌파 월)', fontsize=12, fontweight='bold')
ax1.axvline(24, color='gray', lw=0.8, ls='--', alpha=0.4, label='2022년 초')
ax1.grid(axis='x', alpha=0.2)
ax1.set_ylim(0, 2.2)

# 3-2: 최고 배율 비교
ax2 = fig.add_subplot(gs[1, 0])
all_g = list(processed.keys())
peaks = [processed[g]['norm_roll'].max() for g in all_g]
colors_bar = ['#E63946' if g in REFS else '#457B9D' for g in all_g]
bars = ax2.barh(all_g, peaks, color=colors_bar, alpha=0.8)
ax2.axvline(1.8, color='#E63946', lw=1, ls=':', alpha=0.7, label='급등 임계값 1.8×')
ax2.set_xlabel('2020-21 대비 최고 배율')
ax2.set_title('최고 배율 (peak norm)', fontsize=11, fontweight='bold')
for bar, val in zip(bars, peaks):
    ax2.text(val + 0.03, bar.get_y() + bar.get_height()/2,
             f'{val:.2f}×', va='center', fontsize=8)
ax2.legend(fontsize=8)
ax2.grid(axis='x', alpha=0.3)

# 3-3: 최근 6개월 YoY
ax3 = fig.add_subplot(gs[1, 1])
ryoys = [processed[g]['yoy_chg'].tail(6).mean() for g in all_g]
colors_yoy = ['#E63946' if g in REFS else '#457B9D' for g in all_g]
ryoys_arr = np.array(ryoys)
bars2 = ax3.barh(all_g, ryoys_arr,
                 color=['#2ecc71' if v > 0 else '#e74c3c' for v in ryoys_arr], alpha=0.8)
ax3.axvline(0, color='black', lw=0.8)
ax3.set_xlabel('전년 동월 대비 변화율 (%)')
ax3.set_title('최근 6개월 평균 YoY 변화율', fontsize=11, fontweight='bold')
for bar, val in zip(bars2, ryoys_arr):
    if not np.isnan(val):
        ax3.text(val + (1 if val >= 0 else -1), bar.get_y() + bar.get_height()/2,
                 f'{val:+.1f}%', va='center', fontsize=8)
ax3.grid(axis='x', alpha=0.3)

# 범례
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#E63946', label='레퍼런스 (이미 급부상)'),
                   Patch(facecolor='#457B9D', label='후보군 (예측 대상)')]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, -0.02))

plt.suptitle('후보군 네이버 트렌드 종합 분석', fontsize=14, fontweight='bold', y=1.01)
plt.savefig('image/naver_surge_03_ranking.png', dpi=150, bbox_inches='tight')
plt.close()
print('저장: image/naver_surge_03_ranking.png')


# ─ 수치 요약 출력 ──────────────────────────────────────────────────────────
print('\n=== 그룹별 요약 ===')
sumdf['surge_month_str'] = sumdf['surge_month'].dt.strftime('%Y.%m').fillna('미돌파')
print(sumdf[['group','type','surge_month_str','peak_norm','recent_yoy']].to_string(index=False))
