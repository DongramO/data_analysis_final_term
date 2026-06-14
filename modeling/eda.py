"""
EDA — 행정동 패널 피처 분석 (동편차, 모멘텀, lag1)
핵심 질문: 행정동의 '자기 자신 대비' 변화가 향후 상권 활성화를 예측하는가?

출력 이미지 (modeling/image/):
  eda_01_deviation_target.png   — 동편차 × 타겟 분포
  eda_02_momentum_effect.png    — 모멘텀 × 상승 비율
  eda_03_quadrant.png           — 동편차 × 모멘텀 사분면
  eda_04_lag1_trajectory.png    — lag1 vs 현재값 궤적 유형
  eda_05_dong_profiles.png      — 대표 행정동 시계열 궤적
  eda_06_rising_candidates.png  — 2024 상승 후보 행정동 랭킹
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import zscore as _zscore

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

IMG_DIR = Path('modeling/image')
IMG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = 'data/main_data/ml_dataset_quarterly_train.csv'
PRED_PATH  = 'data/main_data/ml_dataset_quarterly_predict.csv'

LABEL_ORDER  = ['상승', '유지', '하락']
LABEL_COLORS = {'상승': '#2166ac', '유지': '#999999', '하락': '#d6604d'}

train = pd.read_csv(TRAIN_PATH, encoding='utf-8-sig')
pred  = pd.read_csv(PRED_PATH,  encoding='utf-8-sig')

df = train.dropna(subset=['타겟_레이블']).copy()

# ══════════════════════════════════════════════════════════════
# Chart 1: 동편차 × 타겟 분포
# 자기 평균보다 높은 동이 '상승' 클래스에 더 많이 속하는가?
# ══════════════════════════════════════════════════════════════
DEV_FEATS = [
    ('트렌드시간대_비중_동편차', '트렌드\n시간대'),
    ('방문형_강도_동편차',       '방문형\n강도'),
    ('2030소비_비중_동편차',     '2030\n소비'),
    ('주말_비중_동편차',         '주말\n비중'),
    ('금요효과_동편차',           '금요\n효과'),
]

fig, axes = plt.subplots(1, len(DEV_FEATS), figsize=(18, 5))
fig.suptitle('행정동 동편차 분포 by 타겟 클래스\n(동편차 = 현재값 - 해당 행정동 역대 평균)',
             fontsize=13, fontweight='bold')

for ax, (col, label) in zip(axes, DEV_FEATS):
    if col not in df.columns:
        ax.set_visible(False)
        continue
    data = [df[df['타겟_레이블'] == lbl][col].dropna() for lbl in LABEL_ORDER]
    parts = ax.violinplot(data, positions=[0, 1, 2], showmedians=True,
                          showextrema=False, widths=0.6)
    for pc, lbl in zip(parts['bodies'], LABEL_ORDER):
        pc.set_facecolor(LABEL_COLORS[lbl])
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)

    for i, lbl in enumerate(LABEL_ORDER):
        med = df[df['타겟_레이블'] == lbl][col].median()
        ax.text(i, med + 0.002, f'{med:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(LABEL_ORDER, fontsize=10)
    ax.set_title(label, fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)

axes[0].set_ylabel('동편차 (현재 - 자기 평균)')
plt.tight_layout()
plt.savefig(IMG_DIR / 'eda_01_deviation_target.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: eda_01_deviation_target.png')

# ══════════════════════════════════════════════════════════════
# Chart 2: 모멘텀 × 다음 분기 타겟 상승 비율
# 연속 상승 중인 동이 다음에도 상승하는가?
# ══════════════════════════════════════════════════════════════
MOM_FEATS = [
    ('트렌드시간대_비중_모멘텀', '트렌드시간대'),
    ('방문형_강도_모멘텀',       '방문형 강도'),
    ('2030소비_비중_모멘텀',     '2030소비'),
    ('주말_비중_모멘텀',         '주말비중'),
    ('금요효과_모멘텀',           '금요효과'),
]

fig, axes = plt.subplots(1, len(MOM_FEATS), figsize=(20, 5), sharey=True)
fig.suptitle('모멘텀 값별 다음 분기 타겟 분포\n(모멘텀: 양수=연속상승, 음수=연속하락)',
             fontsize=13, fontweight='bold')

for ax, (col, label) in zip(axes, MOM_FEATS):
    if col not in df.columns:
        ax.set_visible(False)
        continue

    mom_vals = sorted(df[col].dropna().unique())
    mom_vals = [v for v in mom_vals if df[df[col] == v].shape[0] >= 20]

    rise_rates, hold_rates, fall_rates = [], [], []
    for mv in mom_vals:
        sub = df[df[col] == mv]['타겟_레이블']
        rise_rates.append((sub == '상승').mean() * 100)
        hold_rates.append((sub == '유지').mean() * 100)
        fall_rates.append((sub == '하락').mean() * 100)

    x = np.arange(len(mom_vals))
    ax.bar(x, rise_rates, color='#2166ac', alpha=0.8)
    ax.bar(x, hold_rates, bottom=rise_rates, color='#999999', alpha=0.8)
    ax.bar(x, fall_rates,
           bottom=[r+h for r, h in zip(rise_rates, hold_rates)],
           color='#d6604d', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in mom_vals], fontsize=9)
    ax.set_xlabel('모멘텀 값')
    ax.set_title(label, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

    for xi, r in zip(x, rise_rates):
        if r > 5:
            ax.text(xi, r/2, f'{r:.0f}%', ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold')

axes[0].set_ylabel('비율 (%)')
axes[0].set_ylim(0, 100)
handles = [mpatches.Patch(color=c, label=l)
           for l, c in zip(['상승', '유지', '하락'], ['#2166ac', '#999999', '#d6604d'])]
axes[-1].legend(handles=handles, loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig(IMG_DIR / 'eda_02_momentum_effect.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: eda_02_momentum_effect.png')

# ══════════════════════════════════════════════════════════════
# Chart 3: 동편차 × 모멘텀 사분면
# 양의 편차 + 양의 모멘텀 조합이 가장 높은 상승률을 보이는가?
# ══════════════════════════════════════════════════════════════
col_dev = '트렌드시간대_비중_동편차'
col_mom = '트렌드시간대_비중_모멘텀'

sub = df[[col_dev, col_mom, '타겟_레이블', '행정동', '연도', '분기']].dropna()
sub = sub.copy()
sub['편차부호']   = sub[col_dev].apply(lambda x: '자기평균 이상' if x >= 0 else '자기평균 미만')
sub['모멘텀부호'] = sub[col_mom].apply(lambda x: '상승 모멘텀' if x > 0 else ('하락 모멘텀' if x < 0 else '모멘텀 없음'))

quadrant_order  = [('자기평균 이상', '상승 모멘텀'), ('자기평균 이상', '하락 모멘텀'),
                   ('자기평균 미만', '상승 모멘텀'), ('자기평균 미만', '하락 모멘텀')]
quadrant_labels = ['위+상승\n(최고 기대)', '위+하락\n(주의)', '아래+상승\n(반등 기대)', '아래+하락\n(최저 기대)']
quad_colors     = ['#08519c', '#74a9cf', '#f4a582', '#ca0020']

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('동편차 × 모멘텀 사분면 분석 (트렌드시간대_비중 기준)',
             fontsize=13, fontweight='bold')

ax = axes[0]
rise_by_q, n_by_q = [], []
for dev_s, mom_s in quadrant_order:
    mask = (sub['편차부호'] == dev_s) & (sub['모멘텀부호'] == mom_s)
    s = sub[mask]['타겟_레이블']
    rise_by_q.append((s == '상승').mean() * 100 if len(s) > 0 else 0)
    n_by_q.append(len(s))

bars = ax.bar(range(4), rise_by_q, color=quad_colors, edgecolor='white', width=0.6)
for bar, r, n in zip(bars, rise_by_q, n_by_q):
    ax.text(bar.get_x() + bar.get_width()/2, r + 0.5,
            f'{r:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_xticks(range(4))
ax.set_xticklabels(quadrant_labels, fontsize=10)
ax.set_ylabel('상승 비율 (%)')
ax.set_title('사분면별 상승 클래스 비율', fontsize=11)
ax.spines[['top', 'right']].set_visible(False)

ax2 = axes[1]
for lbl in LABEL_ORDER:
    s = sub[sub['타겟_레이블'] == lbl]
    ax2.scatter(s[col_dev], s[col_mom], c=LABEL_COLORS[lbl], alpha=0.3, s=15, label=lbl)
ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax2.axvline(0, color='black', linestyle='--', linewidth=0.8)
ax2.set_xlabel('트렌드시간대 동편차')
ax2.set_ylabel('트렌드시간대 모멘텀')
ax2.set_title('동편차 vs 모멘텀 분포', fontsize=11)
ax2.legend(fontsize=9)
ax2.text(0.02, 0.98, '위+상승(최고)', transform=ax2.transAxes,
         ha='left', va='top', fontsize=8, color='#08519c', fontweight='bold')
ax2.text(0.98, 0.02, '아래+하락(최저)', transform=ax2.transAxes,
         ha='right', va='bottom', fontsize=8, color='#ca0020', fontweight='bold')
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(IMG_DIR / 'eda_03_quadrant.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: eda_03_quadrant.png')

# ══════════════════════════════════════════════════════════════
# Chart 4: lag1 vs 현재값 궤적 유형
# ══════════════════════════════════════════════════════════════
lag_col = '트렌드시간대_비중_lag1'
cur_col = '트렌드시간대_비중'

sub2 = df[[lag_col, cur_col, '타겟_레이블']].dropna()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('lag1(1년 전) vs 현재 트렌드시간대_비중 — 타겟 클래스별 궤적',
             fontsize=13, fontweight='bold')

ax = axes[0]
for lbl in LABEL_ORDER:
    s = sub2[sub2['타겟_레이블'] == lbl]
    ax.scatter(s[lag_col], s[cur_col], c=LABEL_COLORS[lbl], alpha=0.3, s=15, label=lbl)
lim_min = min(sub2[lag_col].min(), sub2[cur_col].min())
lim_max = max(sub2[lag_col].max(), sub2[cur_col].max())
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=0.8, alpha=0.5, label='변화 없음')
ax.set_xlabel('1년 전 같은 분기 (lag1)')
ax.set_ylabel('현재값')
ax.set_title('scatter: lag1 vs 현재값', fontsize=11)
ax.legend(fontsize=9)
ax.spines[['top', 'right']].set_visible(False)

ax2 = axes[1]
sub2 = sub2.copy()
sub2['궤적'] = np.where(sub2[cur_col] > sub2[lag_col] + 0.01, '상승궤적',
               np.where(sub2[cur_col] < sub2[lag_col] - 0.01, '하락궤적', '유지궤적'))

traj_order = ['상승궤적', '유지궤적', '하락궤적']
x = np.arange(3)
width = 0.25
for i, lbl in enumerate(LABEL_ORDER):
    s = sub2[sub2['타겟_레이블'] == lbl]
    rates = [(s['궤적'] == t).mean() * 100 for t in traj_order]
    bars = ax2.bar(x + i*width, rates, width, label=lbl,
                   color=LABEL_COLORS[lbl], edgecolor='white', alpha=0.85)
    for bar, r in zip(bars, rates):
        if r > 5:
            ax2.text(bar.get_x() + bar.get_width()/2, r + 0.5,
                     f'{r:.0f}%', ha='center', va='bottom', fontsize=7)

ax2.set_xticks(x + width)
ax2.set_xticklabels(traj_order, fontsize=10)
ax2.set_ylabel('비율 (%)')
ax2.set_title('타겟 클래스별 궤적 유형 분포', fontsize=11)
ax2.legend(fontsize=9)
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(IMG_DIR / 'eda_04_lag1_trajectory.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: eda_04_lag1_trajectory.png')

# ══════════════════════════════════════════════════════════════
# Chart 5: 대표 행정동 시계열 궤적
# ══════════════════════════════════════════════════════════════
dong_summary = (df[df['연도'] == 2022]
                .groupby('행정동')
                .agg(
                    dev_mean=('트렌드시간대_비중_동편차', 'mean'),
                    mom_mean=('트렌드시간대_비중_모멘텀', 'mean'),
                    cur_mean=('트렌드시간대_비중', 'mean'),
                    lag1_mean=('트렌드시간대_비중_lag1', 'mean'),
                ).dropna())

rising  = dong_summary[(dong_summary['dev_mean'] > 0) & (dong_summary['mom_mean'] > 0)].nlargest(5, 'dev_mean').index.tolist()
falling = dong_summary[(dong_summary['dev_mean'] < 0) & (dong_summary['mom_mean'] < 0)].nsmallest(5, 'dev_mean').index.tolist()
rebound = dong_summary[(dong_summary['cur_mean'] > dong_summary['lag1_mean'] + 0.02)].nlargest(5, 'dev_mean').index.tolist()

ts = df.groupby(['행정동', '연도'])['트렌드시간대_비중'].mean().reset_index()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('대표 행정동 시계열 궤적 (트렌드시간대_비중 연도별 평균)', fontsize=13, fontweight='bold')

type_info = [
    (rising,  '연속 상승형\n(편차↑ + 모멘텀↑)', '#2166ac'),
    (falling, '연속 하락형\n(편차↓ + 모멘텀↓)', '#d6604d'),
    (rebound, '반등형\n(lag1 < 현재)',          '#4dac26'),
]

for ax, (dongs, title, color) in zip(axes, type_info):
    for dong in dongs[:5]:
        sub_ts = ts[ts['행정동'] == dong].sort_values('연도')
        if len(sub_ts) < 2:
            continue
        ax.plot(sub_ts['연도'], sub_ts['트렌드시간대_비중'],
                marker='o', markersize=4, alpha=0.8, color=color, linewidth=1.5)
        last = sub_ts.iloc[-1]
        ax.text(last['연도'] + 0.05, last['트렌드시간대_비중'],
                dong[:4], fontsize=7, va='center', color=color)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('연도')
    ax.set_ylabel('트렌드시간대_비중')
    ax.set_xticks([2020, 2021, 2022, 2023])
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(IMG_DIR / 'eda_05_dong_profiles.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: eda_05_dong_profiles.png')

# ══════════════════════════════════════════════════════════════
# Chart 6: 2024 상승 후보 행정동 랭킹
# ══════════════════════════════════════════════════════════════
pred_dong = (pred.groupby('행정동')
             .agg(
                 트렌드편차=('트렌드시간대_비중_동편차', 'mean'),
                 방문형편차=('방문형_강도_동편차', 'mean'),
                 소비편차=('2030소비_비중_동편차', 'mean'),
                 트렌드모멘텀=('트렌드시간대_비중_모멘텀', 'mean'),
                 방문형모멘텀=('방문형_강도_모멘텀', 'mean'),
                 소비모멘텀=('2030소비_비중_모멘텀', 'mean'),
                 트렌드현재=('트렌드시간대_비중', 'mean'),
                 트렌드lag1=('트렌드시간대_비중_lag1', 'mean'),
             ).dropna(subset=['트렌드편차', '트렌드모멘텀']))

pred_dong['편차점수']   = _zscore(pred_dong['트렌드편차'].fillna(0))
pred_dong['모멘텀점수'] = _zscore(pred_dong['트렌드모멘텀'].fillna(0))
pred_dong['소비점수']   = _zscore(pred_dong['소비편차'].fillna(0))
pred_dong['종합점수']   = pred_dong['편차점수'] + pred_dong['모멘텀점수'] + pred_dong['소비점수']

cands = pred_dong[
    (pred_dong['트렌드편차'] > 0) & (pred_dong['트렌드모멘텀'] > 0)
].nlargest(25, '종합점수').reset_index()

fig, axes = plt.subplots(1, 2, figsize=(18, 10))
fig.suptitle('2024 기준 상업 활성화 여지 상위 행정동 (→ 2025 상승 후보)\n'
             '조건: 트렌드시간대 동편차 양수 + 모멘텀 양수',
             fontsize=13, fontweight='bold')

ax = axes[0]
colors_bar = ['#08519c' if i < 5 else '#4292c6' if i < 10 else '#9ecae1'
              for i in range(len(cands))]
ax.barh(cands['행정동'][::-1], cands['종합점수'][::-1],
        color=colors_bar[::-1], edgecolor='white', height=0.7)
ax.set_xlabel('종합 활성화 점수')
ax.set_title('상위 25 행정동 종합 점수', fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
ax.tick_params(axis='y', labelsize=8)

ax2 = axes[1]
ax2.scatter(pred_dong['트렌드편차'], pred_dong['트렌드모멘텀'],
            c='#cccccc', alpha=0.4, s=20, zorder=1)
sc = ax2.scatter(cands['트렌드편차'], cands['트렌드모멘텀'],
                 c=cands['종합점수'], cmap='Blues', s=80, zorder=2,
                 edgecolors='#333333', linewidths=0.5)
for _, row in cands.head(10).iterrows():
    ax2.annotate(row['행정동'][:4],
                 (row['트렌드편차'], row['트렌드모멘텀']),
                 fontsize=7, ha='left', va='bottom',
                 xytext=(3, 3), textcoords='offset points')
ax2.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax2.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax2.set_xlabel('트렌드시간대 동편차 (자기 평균 대비)')
ax2.set_ylabel('트렌드시간대 모멘텀 (연속 방향)')
ax2.set_title('2024 행정동 분포\n(파란 점 = 상위 25 후보)', fontsize=11)
ax2.spines[['top', 'right']].set_visible(False)
plt.colorbar(sc, ax=ax2, label='종합 점수')

plt.tight_layout()
plt.savefig(IMG_DIR / 'eda_06_rising_candidates.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: eda_06_rising_candidates.png')

# ── 텍스트 출력: 상위 후보 목록 ──────────────────────────────
print('\n' + '='*60)
print('2025 상업 활성화 여지 상위 행정동 (2024 기준)')
print('='*60)
print(f'{"순위":>4}  {"행정동":<12}  {"트렌드편차":>8}  {"모멘텀":>6}  {"2030소비편차":>10}  {"종합점수":>8}')
print('-'*60)
for rank, (_, row) in enumerate(cands.iterrows(), 1):
    print(f'{rank:>4}  {row["행정동"]:<12}  '
          f'{row["트렌드편차"]:>8.4f}  {row["트렌드모멘텀"]:>6.2f}  '
          f'{row["소비편차"]:>10.4f}  {row["종합점수"]:>8.3f}')
