"""
차세대 후보군 외부 검증
- 블로그 언급량 성장률
- 임대료·공실률 추이
- 복합점수 분위 궤적
- 종합 검증 스코어카드

출력:
  validate_01_trajectories.png   — 후보 Top 12 분위 궤적
  validate_02_blog.png           — 블로그 언급량 비교
  validate_03_rental.png         — 임대료·공실률 추이
  validate_04_scorecard.png      — 종합 검증 히트맵
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from scipy.spatial.distance import cosine

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

IMG_DIR    = Path('modeling/image')
MASTER_PATH = 'data/main_data/트렌드매출_마스터.csv'
BLOG_PATH   = 'data/main_data/블로그트렌드_상권.csv'
RENT_PATH   = 'data/main_data/임대료_공실률.csv'
TRAIN_PATH  = 'data/main_data/ml_dataset_quarterly_train.csv'
PRED_PATH   = 'data/main_data/ml_dataset_quarterly_predict.csv'

# ── 후보 Top 25 (pattern_match.py 결과) ────────────────────────
CANDIDATES_RANKED = [
    '양재2동','역삼2동','문래동','도곡1동','서초1동','구로2동',
    '풍납2동','장안1동','문정2동','교남동','원효로1동','무악동',
    '원효로2동','아현동','효창동','염창동','서초3동','등촌3동',
    '용답동','방학2동','가양3동','마천1동','대림3동','송정동','신수동',
]
SIMILARITY_SCORES = {
    '양재2동':0.8794,'역삼2동':0.8698,'문래동':0.8295,'도곡1동':0.8260,
    '서초1동':0.8085,'구로2동':0.7955,'풍납2동':0.7760,'장안1동':0.7750,
    '문정2동':0.7731,'교남동':0.7713,'원효로1동':0.7598,'무악동':0.7538,
    '원효로2동':0.7519,'아현동':0.7284,'효창동':0.7239,'염창동':0.7235,
    '서초3동':0.7211,'등촌3동':0.7165,'용답동':0.7073,'방학2동':0.7043,
    '가양3동':0.6941,'마천1동':0.6940,'대림3동':0.6862,'송정동':0.6858,
    '신수동':0.6713,
}

# ── 레퍼런스 동 (비교 기준) ────────────────────────────────────
REFERENCE_DONGS = ['성수2가1동','신대방1동','동화동','여의동']

# ── 블로그: 후보 행정동 → 상권명 매핑 ────────────────────────
BLOG_MAP = {
    '역삼2동':  '역삼',
    '문래동':   '문래',
    '서초1동':  '서초',
    '서초3동':  '서초',
    '장안1동':  '장안',
}
# 블로그 비교군 (레퍼런스 상권 + 알려진 신흥)
BLOG_REF = ['성수','뚝섬','노량진','화양','홍대','왕십리']

# ── 임대료: 후보 행정동 → 상권명 매핑 ────────────────────────
RENT_MAP = {
    '양재2동':  '양재말죽거리',
    '역삼2동':  '강남대로',
    '장안1동':  '장안동',
    '아현동':   '공덕역',
    '효창동':   '숙명여대',
    '서초3동':  '교대역',
    '신수동':   '신촌/이대',
}

# ──────────────────────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────────────────────
master = pd.read_csv(MASTER_PATH, encoding='utf-8-sig')
blog   = pd.read_csv(BLOG_PATH,   encoding='utf-8-sig')
rent   = pd.read_csv(RENT_PATH,   encoding='utf-8-sig')

top12 = CANDIDATES_RANKED[:12]

# ══════════════════════════════════════════════════════════════
# Chart 1: 분위 궤적 (Top 12 + 레퍼런스)
# ══════════════════════════════════════════════════════════════
years = [2020, 2021, 2022, 2023, 2024]

ref_trajs = {}
for d in REFERENCE_DONGS:
    sub = master[master['행정동'] == d][['연도','복합점수_분위']].set_index('연도')
    ref_trajs[d] = [sub.loc[y,'복합점수_분위'] if y in sub.index else np.nan for y in years]

cand_trajs = {}
for d in top12:
    sub = master[master['행정동'] == d][['연도','복합점수_분위']].set_index('연도')
    cand_trajs[d] = [sub.loc[y,'복합점수_분위'] if y in sub.index else np.nan for y in years]

fig, axes = plt.subplots(3, 4, figsize=(20, 13))
fig.suptitle('차세대 후보 Top 12 — 복합점수 분위 궤적 (2020-2024)\n회색 점선: 레퍼런스 동(성수2가1동·신대방1동·동화동·여의동)',
             fontsize=13, fontweight='bold')

ref_colors = ['#aec7e8','#c5b0d5','#c49c94','#dbdb8d']

for ax, dong in zip(axes.flatten(), top12):
    sim = SIMILARITY_SCORES.get(dong, 0)
    rank = CANDIDATES_RANKED.index(dong) + 1

    # 레퍼런스 궤적 (회색 배경)
    for i, (ref_d, ref_traj) in enumerate(ref_trajs.items()):
        ax.plot(years, ref_traj, color=ref_colors[i], linewidth=1.2,
                linestyle='--', alpha=0.7, zorder=1)

    # 후보 궤적 (강조)
    traj = cand_trajs[dong]
    ax.plot(years, traj, color='#d62728', linewidth=2.5, marker='o',
            markersize=5, zorder=3)

    # 현재분위 표시
    cur_rank = traj[-1] if not np.isnan(traj[-1]) else 0
    ax.set_title(f'{rank}위 {dong}\n유사도 {sim:.3f}  현재분위 {cur_rank:.2f}',
                 fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.axhline(0.65, color='gray', linewidth=0.6, linestyle=':', alpha=0.5)
    ax.set_xticks(years)
    ax.set_xticklabels([str(y)[2:] for y in years], fontsize=7)
    ax.tick_params(axis='y', labelsize=7)
    ax.spines[['top','right']].set_visible(False)

    # 블로그/임대료 데이터 가용 여부 표시
    tags = []
    if dong in BLOG_MAP: tags.append('블로그O')
    if dong in RENT_MAP: tags.append('임대료O')
    if tags:
        ax.text(0.02, 0.05, ' '.join(tags), transform=ax.transAxes,
                fontsize=6.5, color='#2ca02c', va='bottom')

handles = [mpatches.Patch(color=c, label=d, alpha=0.7)
           for c, d in zip(ref_colors, REFERENCE_DONGS)]
handles.append(plt.Line2D([0],[0], color='#d62728', linewidth=2, marker='o', label='후보 동'))
fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8,
           bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(IMG_DIR / 'validate_01_trajectories.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: validate_01_trajectories.png')

# ══════════════════════════════════════════════════════════════
# Chart 2: 블로그 언급량 비교 (2020~2025 전체)
# ══════════════════════════════════════════════════════════════
blog_yrs = ['2020','2021','2022','2023','2024','2025']

# 후보 중 블로그 매핑 있는 것 (중복 상권명 제거)
cand_blog_areas = {}
for dong, area in BLOG_MAP.items():
    if dong in CANDIDATES_RANKED:
        rank = CANDIDATES_RANKED.index(dong) + 1
        cand_blog_areas[f'{rank}위 {dong}\n→ {area}'] = area

# 데이터 준비
def get_blog(area_name):
    row = blog[blog['area'] == area_name]
    if row.empty:
        return None
    return row.iloc[0]

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle('블로그 언급량 검증 (2020~2025)\n후보 상권(빨강) vs 레퍼런스·비교군(파랑)',
             fontsize=13, fontweight='bold')

# 패널 1: 연도별 언급량 추이 (2020~2025)
ax = axes[0]
for label, area in cand_blog_areas.items():
    row = get_blog(area)
    if row is None:
        continue
    vals = [row[y] for y in blog_yrs]
    ax.plot(blog_yrs, vals, marker='o', linewidth=2.2, markersize=6,
            color='#d62728', zorder=3)
    ax.text(blog_yrs[-1], vals[-1] + 80, label.split('\n')[0],
            fontsize=7.5, color='#d62728', ha='left', fontweight='bold')

for area in BLOG_REF:
    row = get_blog(area)
    if row is None:
        continue
    vals = [row[y] for y in blog_yrs]
    alpha = 0.95 if area in ['성수', '뚝섬'] else 0.55
    lw    = 2.0  if area in ['성수', '뚝섬'] else 1.3
    ax.plot(blog_yrs, vals, marker='s', linewidth=lw, markersize=4,
            color='#1f77b4', alpha=alpha, zorder=2)
    ax.text(blog_yrs[-1], vals[-1] + 80, area,
            fontsize=7.5, color='#1f77b4', alpha=alpha, ha='left')

ax.axvspan('2020', '2021', alpha=0.06, color='green')
ax.text('2020', ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 100,
        '매출\n데이터\n시작', ha='left', fontsize=7, color='green', alpha=0.7)

ax.set_xlabel('연도', fontsize=9)
ax.set_ylabel('블로그 언급 건수', fontsize=9)
ax.set_title('연도별 언급량 추이 (2020~2025)\n녹색=매출 데이터 시작 구간', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=0.3)

# 패널 2: 2020→2025 성장률 비교 막대
all_areas_plot = {}
for label, area in cand_blog_areas.items():
    row = get_blog(area)
    if row is not None:
        g = row['growth_20_25'] if 'growth_20_25' in row.index else row['growth_22_25']
        all_areas_plot[label] = (g, '#d62728', row['total_vol'])

for area in BLOG_REF:
    row = get_blog(area)
    if row is not None:
        g = row['growth_20_25'] if 'growth_20_25' in row.index else row['growth_22_25']
        all_areas_plot[area] = (g, '#1f77b4', row['total_vol'])

ax2 = axes[1]
sorted_areas = sorted(all_areas_plot.items(), key=lambda x: x[1][0], reverse=True)
labels_b = [x[0] for x in sorted_areas]
vals_b   = [x[1][0] for x in sorted_areas]
colors_b = [x[1][1] for x in sorted_areas]

bars = ax2.barh(labels_b[::-1], vals_b[::-1], color=colors_b[::-1],
                edgecolor='white', height=0.65)
for bar, v in zip(bars[::-1], vals_b):
    ax2.text(v + 1, bar.get_y() + bar.get_height() / 2,
             f'{v:+.1f}%', va='center', fontsize=8.5)

ax2.axvline(0, color='black', linewidth=0.8)
ax2.set_xlabel('2020→2025 블로그 언급량 성장률 (%)', fontsize=9)
ax2.set_title('블로그 성장률 비교 (2020→2025 기준)\n빨강=후보, 파랑=비교군', fontsize=10)
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(IMG_DIR / 'validate_02_blog.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: validate_02_blog.png')

# ══════════════════════════════════════════════════════════════
# Chart 3: 임대료·공실률 추이
# ══════════════════════════════════════════════════════════════
rent_yrs = [2020, 2021, 2022, 2023, 2024]

fig, axes = plt.subplots(2, len(RENT_MAP), figsize=(22, 10))
fig.suptitle('후보 상권 임대료·공실률 추이 (2020-2024)\n매핑 가능한 7개 후보 표시',
             fontsize=13, fontweight='bold')

rent_dongs_sorted = sorted(RENT_MAP.keys(), key=lambda x: CANDIDATES_RANKED.index(x))

for col, dong in enumerate(rent_dongs_sorted):
    area = RENT_MAP[dong]
    sub = rent[rent['상권명'] == area].sort_values('연도')
    sub_2024 = sub[sub['연도'] <= 2024]
    rank = CANDIDATES_RANKED.index(dong) + 1
    sim = SIMILARITY_SCORES.get(dong, 0)

    # 임대료
    ax_r = axes[0, col]
    ax_r.plot(sub_2024['연도'], sub_2024['임대료'], marker='o', linewidth=2,
              markersize=5, color='#2166ac')
    ax_r.fill_between(sub_2024['연도'], sub_2024['임대료'],
                       alpha=0.15, color='#2166ac')
    ax_r.set_title(f'{rank}위 {dong}\n[{area}]', fontsize=8.5, fontweight='bold')
    ax_r.set_ylabel('임대료(천원/㎡)', fontsize=8)
    ax_r.set_xticks(rent_yrs)
    ax_r.set_xticklabels([str(y)[2:] for y in rent_yrs], fontsize=7)
    ax_r.tick_params(axis='y', labelsize=7)
    ax_r.spines[['top','right']].set_visible(False)

    # 임대료 변화율 표시
    if len(sub_2024) >= 2:
        r_start = sub_2024[sub_2024['연도'] == 2020]['임대료'].values
        r_end   = sub_2024[sub_2024['연도'] == 2024]['임대료'].values
        if len(r_start) > 0 and len(r_end) > 0:
            chg = (r_end[0] - r_start[0]) / r_start[0] * 100
            color = '#d62728' if chg > 10 else '#2ca02c' if chg > 0 else '#888888'
            ax_r.text(0.97, 0.05, f'20→24\n{chg:+.1f}%',
                      transform=ax_r.transAxes, ha='right', va='bottom',
                      fontsize=7.5, color=color, fontweight='bold')

    # 공실률
    ax_v = axes[1, col]
    ax_v.bar(sub_2024['연도'], sub_2024['공실률'], color='#fdae6b',
             edgecolor='white', width=0.6)
    ax_v.set_ylabel('공실률(%)', fontsize=8)
    ax_v.set_xticks(rent_yrs)
    ax_v.set_xticklabels([str(y)[2:] for y in rent_yrs], fontsize=7)
    ax_v.tick_params(axis='y', labelsize=7)
    ax_v.spines[['top','right']].set_visible(False)

    # 2024 공실률 표시
    v24 = sub_2024[sub_2024['연도'] == 2024]['공실률'].values
    if len(v24) > 0:
        ax_v.text(0.97, 0.95, f'2024\n{v24[0]:.1f}%',
                  transform=ax_v.transAxes, ha='right', va='top',
                  fontsize=7.5, color='#636363', fontweight='bold')

axes[0, 0].set_ylabel('임대료(천원/㎡)', fontsize=8)
axes[1, 0].set_ylabel('공실률(%)', fontsize=8)

plt.tight_layout()
plt.savefig(IMG_DIR / 'validate_03_rental.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: validate_03_rental.png')

# ══════════════════════════════════════════════════════════════
# Chart 4: 종합 검증 스코어카드 (히트맵)
# ══════════════════════════════════════════════════════════════
top15 = CANDIDATES_RANKED[:15]

scorecard_data = {}
for dong in top15:
    row = {}
    rank = CANDIDATES_RANKED.index(dong) + 1

    # 1) 패턴 유사도
    row['패턴\n유사도'] = SIMILARITY_SCORES.get(dong, np.nan)

    # 2) 현재분위 (낮을수록 상승 여지 큼 → 역수)
    sub = master[(master['행정동'] == dong) & (master['연도'] == 2024)]
    cur = sub['복합점수_분위'].values[0] if len(sub) > 0 else np.nan
    row['상승여지\n(1-분위)'] = 1 - cur if not np.isnan(cur) else np.nan

    # 3) 분위 모멘텀 (2022→2024)
    sub22 = master[(master['행정동'] == dong) & (master['연도'] == 2022)]['복합점수_분위'].values
    sub24 = master[(master['행정동'] == dong) & (master['연도'] == 2024)]['복합점수_분위'].values
    row['분위\n모멘텀'] = (sub24[0] - sub22[0]) if (len(sub22) > 0 and len(sub24) > 0) else np.nan

    # 4) 블로그 성장률 (22→25, 없으면 NaN)
    area = BLOG_MAP.get(dong)
    if area:
        b_row = blog[blog['area'] == area]
        if not b_row.empty:
            col_g = 'growth_20_25' if 'growth_20_25' in b_row.columns else 'growth_22_25'
            row['블로그\n성장률'] = b_row.iloc[0][col_g] / 100
        else:
            row['블로그\n성장률'] = np.nan
    else:
        row['블로그\n성장률'] = np.nan

    # 5) 임대료 변화 (20→24, 없으면 NaN)
    area_r = RENT_MAP.get(dong)
    if area_r:
        r20 = rent[(rent['상권명'] == area_r) & (rent['연도'] == 2020)]['임대료'].values
        r24 = rent[(rent['상권명'] == area_r) & (rent['연도'] == 2024)]['임대료'].values
        if len(r20) > 0 and len(r24) > 0:
            row['임대료\n변화율'] = (r24[0] - r20[0]) / r20[0]
        else:
            row['임대료\n변화율'] = np.nan
    else:
        row['임대료\n변화율'] = np.nan

    # 6) 공실률 2024 (낮을수록 좋음 → 역수)
    if area_r:
        v24 = rent[(rent['상권명'] == area_r) & (rent['연도'] == 2024)]['공실률'].values
        if len(v24) > 0:
            row['공실률\n(역, 낮을수록↑)'] = 1 - (v24[0] / 20)  # 0~20% 정규화
        else:
            row['공실률\n(역, 낮을수록↑)'] = np.nan
    else:
        row['공실률\n(역, 낮을수록↑)'] = np.nan

    scorecard_data[f'{rank}위 {dong}'] = row

sc_df = pd.DataFrame(scorecard_data).T
sc_cols = list(sc_df.columns)

# 컬럼별 min-max 정규화 (0~1, NaN 유지)
sc_norm = sc_df.copy()
for col in sc_cols:
    col_vals = sc_df[col].dropna()
    if len(col_vals) > 1:
        mn, mx = col_vals.min(), col_vals.max()
        if mx > mn:
            sc_norm[col] = (sc_df[col] - mn) / (mx - mn)

fig, ax = plt.subplots(figsize=(14, 10))

# 히트맵
import matplotlib.colors as mcolors
cmap = plt.cm.RdYlGn

sc_arr = sc_norm.values.astype(float)
im = ax.imshow(sc_arr, cmap=cmap, aspect='auto', vmin=0, vmax=1)

ax.set_xticks(range(len(sc_cols)))
ax.set_xticklabels(sc_cols, fontsize=9)
ax.set_yticks(range(len(sc_df)))
ax.set_yticklabels(sc_df.index, fontsize=9)

# 셀 값 표시
for i in range(len(sc_df)):
    for j in range(len(sc_cols)):
        raw_val = sc_df.iloc[i, j]
        if np.isnan(raw_val):
            txt = '—'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8, color='#888888')
        else:
            # 원래 값으로 표시
            if sc_cols[j] in ['패턴\n유사도', '상승여지\n(1-분위)', '분위\n모멘텀']:
                txt = f'{raw_val:.3f}'
            elif sc_cols[j] == '블로그\n성장률':
                txt = f'{raw_val*100:+.0f}%'
            elif sc_cols[j] == '임대료\n변화율':
                txt = f'{raw_val*100:+.0f}%'
            else:
                txt = f'{raw_val:.2f}'
            norm_v = sc_arr[i, j]
            text_color = 'white' if (norm_v < 0.25 or norm_v > 0.75) else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=7.5, color=text_color)

# 데이터 없는 셀 배경
for i in range(len(sc_df)):
    for j in range(len(sc_cols)):
        if np.isnan(sc_df.iloc[i, j]):
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                        fill=True, color='#f0f0f0', zorder=0))

ax.set_title('차세대 후보 Top 15 — 종합 검증 스코어카드\n'
             '(녹색=강한 신호, 적색=약한 신호, 회색=데이터 없음)',
             fontsize=12, fontweight='bold')

# 컬럼 구분선
for j in [1, 3, 4, 5]:
    ax.axvline(j + 0.5, color='white', linewidth=2)

plt.colorbar(im, ax=ax, shrink=0.6, label='지표별 정규화 점수 (0~1)')
plt.tight_layout()
plt.savefig(IMG_DIR / 'validate_04_scorecard.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: validate_04_scorecard.png')

# ══════════════════════════════════════════════════════════════
# 최종 요약 출력
# ══════════════════════════════════════════════════════════════
print('\n' + '='*65)
print('종합 검증 결과 요약')
print('='*65)

# 검증 데이터별 후보 커버리지
print(f'\n[블로그 데이터 있는 후보]')
for dong, area in BLOG_MAP.items():
    if dong in CANDIDATES_RANKED:
        rank = CANDIDATES_RANKED.index(dong) + 1
        b = get_blog(area)
        if b is not None:
            col_g = 'growth_20_25' if 'growth_20_25' in b.index else 'growth_22_25'
            base  = '2020' if 'growth_20_25' in b.index else '2022'
            print(f'  {rank}위 {dong} ({area}): {base}→2025 성장 {b[col_g]:+.1f}%  총언급 {b["total_vol"]:,}건')

print(f'\n[임대료 데이터 있는 후보]')
for dong in rent_dongs_sorted:
    area = RENT_MAP[dong]
    rank = CANDIDATES_RANKED.index(dong) + 1
    sub = rent[(rent['상권명'] == area) & (rent['연도'].isin([2020,2024]))]
    if len(sub) >= 2:
        r20 = sub[sub['연도']==2020]['임대료'].values[0]
        r24 = sub[sub['연도']==2024]['임대료'].values[0]
        v24 = sub[sub['연도']==2024]['공실률'].values
        v_str = f'공실률 {v24[0]:.1f}%' if len(v24)>0 else ''
        print(f'  {rank}위 {dong} ({area}): 임대료 {r20:.1f}→{r24:.1f}천원/㎡ ({(r24-r20)/r20*100:+.1f}%)  {v_str}')

sc_df.to_csv('data/main_data/validate_scorecard.csv', encoding='utf-8-sig')
print('\n스코어카드 저장: data/main_data/validate_scorecard.csv')
