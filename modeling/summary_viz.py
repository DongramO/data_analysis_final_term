"""
전체 분석 흐름 요약 시각화

Chart 1  summary_01_reference.png    — 레퍼런스 선정 기준 + 공통 변화 패턴
Chart 2  summary_02_signals.png      — 선행/후행 지표 분석
Chart 3  summary_03_validation.png   — 후보 선정 + 검증 + 추가 데이터 추천
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

IMG_DIR    = Path('modeling/image')
MASTER     = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig')
TRAIN      = pd.read_csv('data/main_data/ml_dataset_quarterly_train.csv', encoding='utf-8-sig')
PRED       = pd.read_csv('data/main_data/ml_dataset_quarterly_predict.csv', encoding='utf-8-sig')
BLOG       = pd.read_csv('data/main_data/블로그트렌드_상권.csv', encoding='utf-8-sig')
RENT       = pd.read_csv('data/main_data/임대료_공실률.csv', encoding='utf-8-sig')

REFS   = ['성수2가1동', '신대방1동', '동화동', '여의동']
REF_COLORS = {'성수2가1동': '#d62728', '신대방1동': '#1f77b4',
               '동화동': '#2ca02c',     '여의동': '#ff7f0e'}
YEARS = [2020, 2021, 2022, 2023, 2024]

# 연간 feature 값 (분기 평균)
ALL_DATA = pd.concat([TRAIN, PRED], ignore_index=True)
annual_feat = ALL_DATA.groupby(['행정동','연도'])[[
    '트렌드시간대_비중','방문형_강도','주말_비중','야간_비중','금요효과',
    '평일_비중','FB카페_비중','주류유흥_비중','라이프스타일_비중',
    '2030소비_비중','2030_이동비율','2030_이동비율_변화',
]].mean().reset_index()

# 서울 전체 연간 평균
seoul_avg = annual_feat.groupby('연도').mean(numeric_only=True).reset_index()

# ══════════════════════════════════════════════════════════════════════════
# Chart 1: 레퍼런스 선정 기준 + 연도별 공통 패턴
# ══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 16))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.38)

# ── Panel A: 후보풀 산점도 (선정 기준 시각화) ──────────────────────────
ax_scatter = fig.add_subplot(gs[0, :2])

pivot = MASTER.pivot(index='행정동', columns='연도', values='복합점수_분위').dropna()
pivot['변화'] = pivot[2024] - pivot[2020]
pool = pivot[(pivot[2020] < 0.55) & (pivot[2024] > 0.60)].copy()

# 전체 가려진 동 (회색)
ax_scatter.scatter(pivot[2020], pivot[2024] - pivot[2020],
                   color='#cccccc', s=30, alpha=0.4, zorder=1, label='일반 행정동')
# 급부상 후보풀 (하늘색)
ax_scatter.scatter(pool[2020], pool['변화'],
                   color='#6baed6', s=60, alpha=0.7, zorder=2, label='급부상 후보풀\n(2020<0.55, 변화>0.40, 2024>0.60)')
# 레퍼런스 4개 (색상 강조)
for d in REFS:
    if d in pool.index:
        ax_scatter.scatter(pool.loc[d,2020], pool.loc[d,'변화'],
                           color=REF_COLORS[d], s=180, zorder=4,
                           edgecolors='black', linewidth=1.2)
        ax_scatter.annotate(d, (pool.loc[d,2020], pool.loc[d,'변화']),
                            fontsize=7.5, xytext=(5, 3), textcoords='offset points',
                            color=REF_COLORS[d], fontweight='bold')

# 선정 기준선
ax_scatter.axvline(0.55, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax_scatter.axhline(0.40, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax_scatter.text(0.57, 0.02, '2020 분위 < 0.55\n(아직 상권화 전)', fontsize=7, color='gray')
ax_scatter.text(0.01, 0.41, '분위 변화 > 0.40', fontsize=7, color='gray')

ax_scatter.set_xlabel('2020년 복합점수 분위 (출발점)', fontsize=9)
ax_scatter.set_ylabel('2020→2024 분위 변화량', fontsize=9)
ax_scatter.set_title('레퍼런스 선정 기준\n낮은 출발점에서 급격히 상승한 행정동', fontsize=10, fontweight='bold')
ax_scatter.legend(fontsize=7.5, loc='upper left')
ax_scatter.spines[['top','right']].set_visible(False)

# ── Panel B: 레퍼런스 4개 동 분위 궤적 ────────────────────────────────
ax_traj = fig.add_subplot(gs[0, 2:])

# 서울 전체 분위 분포 (상위25%, 중앙, 하위25%)
dist_data = {y: MASTER[MASTER['연도']==y]['복합점수_분위'] for y in YEARS}
p25 = [dist_data[y].quantile(0.25) for y in YEARS]
p75 = [dist_data[y].quantile(0.75) for y in YEARS]
ax_traj.fill_between(YEARS, p25, p75, alpha=0.1, color='gray', label='서울 중간 50%')
ax_traj.plot(YEARS, [dist_data[y].median() for y in YEARS],
             color='gray', linewidth=1, linestyle=':', label='서울 중앙값')

for d in REFS:
    sub = MASTER[MASTER['행정동']==d].set_index('연도')['복합점수_분위']
    traj = [sub.get(y, np.nan) for y in YEARS]
    ax_traj.plot(YEARS, traj, color=REF_COLORS[d], linewidth=2.5, marker='o',
                 markersize=7, label=d, zorder=3)
    # 2020→2024 변화 표시
    if not np.isnan(traj[0]) and not np.isnan(traj[-1]):
        ax_traj.annotate(f'+{traj[-1]-traj[0]:.2f}',
                         (2024, traj[-1]), xytext=(5,0), textcoords='offset points',
                         fontsize=7.5, color=REF_COLORS[d], fontweight='bold')

ax_traj.set_ylim(0, 1)
ax_traj.set_xticks(YEARS)
ax_traj.set_ylabel('복합점수 분위 (0=최하위, 1=최상위)', fontsize=9)
ax_traj.set_title('레퍼런스 4개 동 복합점수 분위 궤적\n회색: 서울 전체 분포', fontsize=10, fontweight='bold')
ax_traj.legend(fontsize=7.5, loc='upper left')
ax_traj.spines[['top','right']].set_visible(False)
ax_traj.set_xlim(2019.7, 2024.8)

# ── Panel C-F: 레퍼런스 동별 핵심 지표 연도별 변화 ────────────────────
key_metrics = [
    ('트렌드시간대_비중', '트렌드시간대 비중\n(금요저녁 + 토전체 + 일전체)'),
    ('방문형_강도',      '방문형 강도\n(외부 방문객 중심도)'),
    ('주말_비중',        '주말 매출 비중'),
    ('2030_이동비율_변화','2030 이동비율 변화\n(전년 대비, 선행지표)'),
]
for idx, (feat, feat_label) in enumerate(key_metrics):
    row, col = 1 + idx // 2, (idx % 2) * 2
    ax_m = fig.add_subplot(gs[row, col:col+2])

    if feat not in annual_feat.columns:
        continue

    seoul_vals = [seoul_avg[seoul_avg['연도']==y][feat].values[0]
                  if y in seoul_avg['연도'].values else np.nan for y in YEARS]
    ax_m.plot(YEARS, seoul_vals, color='gray', linewidth=1, linestyle='--',
              label='서울 평균', alpha=0.8)

    for d in REFS:
        sub_d = annual_feat[annual_feat['행정동']==d].set_index('연도')
        vals = [sub_d.loc[y, feat] if y in sub_d.index else np.nan for y in YEARS]
        ax_m.plot(YEARS, vals, color=REF_COLORS[d], linewidth=2.2,
                  marker='o', markersize=5, label=d, zorder=3)

    ax_m.set_title(feat_label, fontsize=9, fontweight='bold')
    ax_m.set_xticks(YEARS)
    ax_m.set_xticklabels([str(y)[2:] for y in YEARS], fontsize=8)
    ax_m.tick_params(axis='y', labelsize=7.5)
    ax_m.spines[['top','right']].set_visible(False)
    ax_m.legend(fontsize=6.5, ncol=3)

    # 선행/후행 레이블
    if feat == '2030_이동비율_변화':
        ax_m.text(0.98, 0.95, '선행 신호', transform=ax_m.transAxes,
                  ha='right', va='top', fontsize=8, color='#2ca02c',
                  fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#e6f4ea'))
    else:
        ax_m.text(0.98, 0.95, '후행 지표', transform=ax_m.transAxes,
                  ha='right', va='top', fontsize=8, color='#d62728',
                  fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='#fce8e8'))

fig.suptitle('① 레퍼런스 선정 기준과 연도별 공통 변화 패턴\n'
             '상권 정의: 복합점수 = 업종구성(F&B·라이프스타일 비중) + 시간패턴(금요효과·주말·야간비중)\n'
             '2030비중은 복합점수에서 의도적으로 제외 — 독립 검증 변수로 사용',
             fontsize=11, fontweight='bold', y=0.99)

plt.savefig(IMG_DIR / 'summary_01_reference.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print('저장: summary_01_reference.png')

# ══════════════════════════════════════════════════════════════════════════
# Chart 2: 선행 vs 후행 지표 분석
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(22, 9))
fig.suptitle('② 선행 지표 vs 후행 지표 — 레퍼런스 동들이 뜨기 전/후에 무엇이 먼저 움직였나',
             fontsize=12, fontweight='bold', y=1.01)

# ── Panel A: 레퍼런스 시그니처 (뜨기 전 z-score) ─────────────────────
ax_sig = axes[0]
sig_data = [
    ('2030 이동비율 변화', +0.535, '선행'),
    ('라이프스타일 비중',  +0.299, '선행'),
    ('HE 이동 비율',      +0.235, '선행'),
    ('2030 이동비율',     +0.192, '선행'),
    ('20대 소비 비중',    +0.106, '선행'),
    ('2030 소비비중',     +0.044, '선행'),
    ('평일 비중',         +0.752, '선행'),
    ('30대 소비 비중',    -0.069, '후행'),
    ('방문형 변화',       -0.104, '후행'),
    ('트렌드 변화',       -0.116, '후행'),
    ('FB 카페 비중',      -0.156, '후행'),
    ('금요효과',          -0.250, '후행'),
    ('주류유흥 비중',     -0.300, '후행'),
    ('야간 비중',         -0.393, '후행'),
    ('주말 비중',         -0.671, '후행'),
    ('방문형 강도',       -0.690, '후행'),
    ('트렌드시간대 비중', -0.698, '후행'),
]
sig_data_sorted = sorted(sig_data, key=lambda x: x[1])
labels = [x[0] for x in sig_data_sorted]
vals   = [x[1] for x in sig_data_sorted]
types  = [x[2] for x in sig_data_sorted]
colors = ['#2166ac' if t == '선행' else '#d6604d' for t in types]

bars = ax_sig.barh(labels, vals, color=colors, edgecolor='white', height=0.65)
ax_sig.axvline(0, color='black', linewidth=0.9)
for bar, v in zip(bars, vals):
    ax_sig.text(v + (0.02 if v >= 0 else -0.02),
                bar.get_y() + bar.get_height()/2,
                f'{v:+.2f}', va='center',
                ha='left' if v >= 0 else 'right', fontsize=7.5)

ax_sig.set_xlabel('서울 평균 대비 z-score (뜨기 전)', fontsize=9)
ax_sig.set_title('뜨기 직전 레퍼런스 동의 특성\n(4개 동 × 뜨기 전 분기 평균)', fontsize=9.5, fontweight='bold')
ax_sig.spines[['top','right']].set_visible(False)

lead_patch = mpatches.Patch(color='#2166ac', label='서울 평균보다 이미 높음 (선행)')
lag_patch  = mpatches.Patch(color='#d6604d', label='서울 평균보다 아직 낮음 (후행)')
ax_sig.legend(handles=[lead_patch, lag_patch], fontsize=7.5, loc='lower right')

# ── Panel B: 선행 지표 시계열 (레퍼런스 동 공통 궤적) ─────────────────
ax_lead = axes[1]
ax_lead.set_title('선행 지표: 뜨기 전 이미 움직이는 신호들\n(2030 이동 변화율 — 상권 지수보다 먼저 상승)',
                   fontsize=9.5, fontweight='bold')

# 2030 이동비율_변화 (선행)
for d in REFS:
    sub_d = annual_feat[annual_feat['행정동']==d].set_index('연도')
    vals_mv = [sub_d.loc[y,'2030_이동비율_변화'] if y in sub_d.index else np.nan for y in YEARS]
    ax_lead.plot(YEARS, vals_mv, color=REF_COLORS[d], linewidth=2.2,
                 marker='o', markersize=6, label=f'{d} (이동변화)', zorder=3)

# 서울 평균
seoul_mv = [seoul_avg[seoul_avg['연도']==y]['2030_이동비율_변화'].values[0]
            if y in seoul_avg['연도'].values else np.nan for y in YEARS]
ax_lead.plot(YEARS, seoul_mv, color='gray', linewidth=1.2,
             linestyle='--', label='서울 평균', alpha=0.8)
ax_lead.axhline(0, color='black', linewidth=0.6, linestyle=':')

ax_lead.set_xticks(YEARS)
ax_lead.set_ylabel('2030 이동비율 전년대비 변화 (%p)', fontsize=9)
ax_lead.legend(fontsize=7, loc='upper right')
ax_lead.spines[['top','right']].set_visible(False)

# 배경 구분 (뜨기 전 / 뜨는 중)
ax_lead.axvspan(2019.8, 2021.5, alpha=0.05, color='blue', label='_선행 구간')
ax_lead.text(2020.5, ax_lead.get_ylim()[1]*0.92, '선행 구간\n(이동 증가)',
             ha='center', fontsize=7.5, color='#2166ac',
             bbox=dict(boxstyle='round', facecolor='#eef4fb', alpha=0.8))

# ── Panel C: 후행 지표 시계열 ──────────────────────────────────────────
ax_lag = axes[2]
ax_lag.set_title('후행 지표: 상권이 뜬 이후 따라오는 신호들\n(트렌드시간대 비중 — 방문객이 몰리기 시작하면 급등)',
                  fontsize=9.5, fontweight='bold')

for d in REFS:
    sub_d = annual_feat[annual_feat['행정동']==d].set_index('연도')
    vals_tr = [sub_d.loc[y,'트렌드시간대_비중'] if y in sub_d.index else np.nan for y in YEARS]
    ax_lag.plot(YEARS, vals_tr, color=REF_COLORS[d], linewidth=2.2,
                marker='o', markersize=6, label=d, zorder=3)

seoul_tr = [seoul_avg[seoul_avg['연도']==y]['트렌드시간대_비중'].values[0]
            if y in seoul_avg['연도'].values else np.nan for y in YEARS]
ax_lag.plot(YEARS, seoul_tr, color='gray', linewidth=1.2,
            linestyle='--', label='서울 평균', alpha=0.8)

ax_lag.set_xticks(YEARS)
ax_lag.set_ylabel('트렌드시간대 매출 비중\n(금요저녁+토+일)', fontsize=9)
ax_lag.legend(fontsize=7.5, loc='upper left')
ax_lag.spines[['top','right']].set_visible(False)

ax_lag.axvspan(2021.5, 2024.2, alpha=0.05, color='red')
ax_lag.text(2022.8, ax_lag.get_ylim()[1]*0.92, '후행 급등 구간\n(방문객 유입 반영)',
            ha='center', fontsize=7.5, color='#d62728',
            bbox=dict(boxstyle='round', facecolor='#fef0f0', alpha=0.8))

# 하단 요약 텍스트
fig.text(0.5, -0.04,
         '인과 흐름:  ① 2030 이동비율 증가 (선행, 1~2년 先)  →  ② 라이프스타일·카페 업종 입점  →  ③ 트렌드시간대·방문형·주말 매출 급등 (후행)',
         ha='center', fontsize=10, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#fffde7', edgecolor='#f9a825', alpha=0.9))

plt.tight_layout()
plt.savefig(IMG_DIR / 'summary_02_signals.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print('저장: summary_02_signals.png')

# ══════════════════════════════════════════════════════════════════════════
# Chart 3: 후보 선정 + 검증 + 추가 데이터 추천
# ══════════════════════════════════════════════════════════════════════════
CANDIDATES = [
    ('양재2동',  0.8794, 0.158, None,   '+17%*', '3.1%'),
    ('역삼2동',  0.8698, 0.161, '+12.8%','+17.0%','12.1%'),
    ('문래동',   0.8295, 0.040, '+68.2%', None,    None),
    ('도곡1동',  0.8260, 0.038, None,    None,    None),
    ('서초1동',  0.8085, 0.265, '+21.3%', None,    None),
    ('구로2동',  0.7955, 0.005, None,    None,    None),
    ('풍납2동',  0.7760, 0.076, None,    None,    None),
    ('장안1동',  0.7750, 0.043, '+16.7%','+14.1%','8.1%'),
    ('문정2동',  0.7731, 0.163, None,    None,    None),
    ('교남동',   0.7713, 0.066, None,    None,    None),
    ('원효로1동',0.7598, 0.002, None,    None,    None),
    ('무악동',   0.7538, 0.061, None,    None,    None),
    ('원효로2동',0.7519, 0.017, None,    None,    None),
    ('아현동',   0.7284, 0.095, None,   '+8.5%', '8.7%'),
    ('효창동',   0.7239, 0.109, None,   '-5.4%', '1.1%'),
]

fig = plt.figure(figsize=(22, 15))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

# ── Panel A: 후보 랭킹 + 검증 신호 통합 ───────────────────────────────
ax_rank = fig.add_subplot(gs[:, 0])

y_pos = np.arange(len(CANDIDATES))[::-1]
bar_colors = ['#08306b' if i < 3 else '#2171b5' if i < 6 else '#6baed6' if i < 10 else '#c6dbef'
              for i in range(len(CANDIDATES))]

bars = ax_rank.barh(y_pos, [c[1] for c in CANDIDATES],
                    color=bar_colors, edgecolor='white', height=0.7)

for i, (dong, sim, cur, blog, rent_chg, vac) in enumerate(CANDIDATES):
    yp = y_pos[i]
    # 행정동명
    ax_rank.text(-0.01, yp, f'{i+1:2d}. {dong}', va='center',
                 ha='right', fontsize=8.5, fontweight='bold' if i < 3 else 'normal')
    # 현재분위
    ax_rank.text(sim + 0.005, yp + 0.25,
                 f'현재분위 {cur:.2f}', va='center', fontsize=6.5, color='#555555')
    # 검증 신호 아이콘
    tags = []
    if blog:   tags.append(f'블로그{blog}')
    if rent_chg: tags.append(f'임대료{rent_chg}')
    if vac:    tags.append(f'공실{vac}')
    if tags:
        ax_rank.text(sim + 0.005, yp - 0.28,
                     '  '.join(tags), va='center', fontsize=6,
                     color='#2ca02c')

ax_rank.set_xlim(0.6, 1.05)
ax_rank.set_yticks([])
ax_rank.set_xlabel('레퍼런스 패턴과의 코사인 유사도', fontsize=9)
ax_rank.set_title('차세대 급부상 후보 Top 15\n(현재분위 0.65 미만 필터)', fontsize=10, fontweight='bold')
ax_rank.spines[['top','right','left']].set_visible(False)
ax_rank.axvline(0.85, color='#d62728', linestyle='--', linewidth=0.8, alpha=0.6)
ax_rank.text(0.85, len(CANDIDATES)-0.3, '강한 유사도\n기준선', fontsize=7,
             color='#d62728', ha='center')

# ── Panel B: 데이터 타임라인 ──────────────────────────────────────────
ax_time = fig.add_subplot(gs[0, 1:])
ax_time.set_xlim(2019, 2028)
ax_time.set_ylim(-0.5, 5.5)
ax_time.axis('off')
ax_time.set_title('데이터 보유 구간 vs 예측 구간', fontsize=10, fontweight='bold')

timeline_items = [
    (2020, 2024, 0, '#2166ac', '매출 원본 데이터\n(423개 행정동 × 분기별 요일/시간대/연령대)', '보유'),
    (2020, 2024, 1, '#31a354', '생활이동 데이터\n(2030 이동비율, HE/WE/EE 유형별)', '보유'),
    (2020, 2025, 2, '#fd8d3c', '블로그 언급량\n(46개 상권, 2022~2025)', '보유'),
    (2020, 2025, 3, '#756bb1', '임대료·공실률\n(68개 상권, 2020~2025)', '보유'),
    (2025, 2027, 4, '#d62728', '예측 구간\n(패턴매칭 결과 적용)', '예측'),
]

for x1, x2, y, color, label, status in timeline_items:
    alpha = 1.0 if status == '보유' else 0.4
    hatch = '' if status == '보유' else '//'
    ax_time.barh(y, x2-x1, left=x1, height=0.55, color=color,
                 alpha=alpha, hatch=hatch, edgecolor='white')
    ax_time.text(x1 + (x2-x1)/2, y, label, ha='center', va='center',
                 fontsize=7.5, color='white', fontweight='bold')

for yr in [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027]:
    ax_time.axvline(yr, color='white', linewidth=0.5, alpha=0.5)
    ax_time.text(yr, -0.4, str(yr), ha='center', fontsize=8, color='#333333')

ax_time.axvline(2024.5, color='black', linewidth=2, linestyle='--')
ax_time.text(2024.5, 5.1, '현재 데이터 한계\n(2024년 말)', ha='center',
             fontsize=8.5, color='black', fontweight='bold')

ax_time.text(2026, 5.1, '예측 구간\n2025~2026년', ha='center',
             fontsize=8.5, color='#d62728', fontweight='bold')

# 범례
boyu_patch = mpatches.Patch(color='#555555', label='보유 데이터')
pred_patch  = mpatches.Patch(color='#555555', alpha=0.4, hatch='//', label='예측 구간')
ax_time.legend(handles=[boyu_patch, pred_patch], fontsize=8,
               loc='lower right', framealpha=0.8)

# ── Panel C: 추가 검증 데이터 추천 ────────────────────────────────────
ax_add = fig.add_subplot(gs[1, 1:])
ax_add.axis('off')
ax_add.set_title('추가 검증에 활용 가능한 데이터 추천', fontsize=10, fontweight='bold')

add_data = [
    ('즉시 수집 가능', '#2ca02c', [
        ('인스타그램 해시태그 수',
         '지역명 해시태그 게시물 수 — 블로그보다 빠른 바이럴 신호. 성수·연남·홍대는 급등 2~3년 전부터 해시태그 증가'),
        ('카카오맵 신규 등록 업체 수',
         '카카오/네이버 지도 신규 업체 등록 건수 (특히 카페·음식점) — 업종 전환 신호의 선행지표'),
        ('네이버 부동산 시세 변화',
         '상업용 건물 매물 시세 (네이버 부동산 API) — 임대료 공식 통계보다 6~12개월 선행'),
    ]),
    ('공공 데이터 (수집 난이도 중)', '#1f77b4', [
        ('사업자 등록 건수 (국세청)',
         '행정동별 신규 사업자 등록 수 — 카페/요식업 증가율이 상권 형성의 직접 신호'),
        ('지하철 승하차 인원 (서울 열린데이터)',
         '역별 시간대별 승하차 — 금·토 저녁 시간대 승하차 증가가 주말형 상권화의 선행 신호'),
        ('유동인구 히트맵 (SKT/KT ADOT)',
         '통신 기반 읍면동별 유동인구 밀도 — 현재 생활이동 데이터보다 단위가 세밀'),
    ]),
    ('대안 검증 방법 (현재 데이터로)', '#9467bd', [
        ('네이버 검색 트렌드 (DataLab)',
         '후보 행정동명 + "맛집/카페" 키워드 — 이미 일부 지역 수집됨, 나머지 후보 동 수집 확장 가능'),
        ('리뷰 플랫폼 리뷰 수 (카카오/네이버)',
         '식당·카페 리뷰 수 증가율 — 방문 경험의 축적 지표. API 또는 크롤링으로 수집 가능'),
    ]),
]

y_start = 0.97
for category, color, items in add_data:
    ax_add.text(0.01, y_start, category, fontsize=9, fontweight='bold',
                color=color, transform=ax_add.transAxes)
    y_start -= 0.07
    for title, desc in items:
        ax_add.text(0.03, y_start, f'- {title}:', fontsize=8.5, fontweight='bold',
                    color='#333333', transform=ax_add.transAxes)
        ax_add.text(0.32, y_start, desc, fontsize=8, color='#555555',
                    transform=ax_add.transAxes, wrap=True)
        y_start -= 0.07
    y_start -= 0.03

fig.suptitle('③ 후보 선정 결과 + 검증 방식 + 추가 데이터 추천',
             fontsize=12, fontweight='bold', y=1.01)

plt.savefig(IMG_DIR / 'summary_03_validation.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print('저장: summary_03_validation.png')

print('\n완료: summary_01~03 저장됨 (modeling/image/)')
