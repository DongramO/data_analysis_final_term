# -*- coding: utf-8 -*-
"""
두 가지 검증:
1. 인과 사슬 검증 — Δ업종점수(t) → Δ2030비중(t+1) → Δ임대료(t+2)
2. 오피스형 vs 여가형 구분 — 서초4동·역삼1동 vs 트렌드 후보군
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 ──────────────────────────────────────────────────
tsm      = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig')
rental   = pd.read_csv('data/main_data/임대료_공실률.csv', encoding='utf-8-sig')
rmap     = pd.read_csv('data/main_data/rental_vacancy_dongmap.csv', encoding='utf-8-sig')
ml_tr    = pd.read_csv('data/main_data/ml_dataset_quarterly_train.csv', encoding='utf-8-sig')
ml_pr    = pd.read_csv('data/main_data/ml_dataset_quarterly_predict.csv', encoding='utf-8-sig')
ml       = pd.concat([ml_tr, ml_pr], ignore_index=True)

# ── 1. 인과 사슬용 패널 구축 ─────────────────────────────────────
# rental에서 상권 → 대표_행정동 매핑
sang2dong = dict(zip(rmap['상권명'], rmap['대표_행정동']))

# 임대료 연간 패널
rental_panel = rental[['상권명','연도','임대료']].copy()
rental_panel['행정동'] = rental_panel['상권명'].map(sang2dong)
rental_panel = rental_panel.dropna(subset=['행정동'])
rental_ann = rental_panel.groupby(['행정동','연도'])['임대료'].mean().reset_index()

# TSM + 임대료 조인
panel = tsm[['행정동','연도','업종점수','2030비중']].merge(
    rental_ann, on=['행정동','연도'], how='inner')
panel = panel.sort_values(['행정동','연도'])

# YoY 변화량 계산
panel['Δ업종점수'] = panel.groupby('행정동')['업종점수'].diff()
panel['Δ2030비중'] = panel.groupby('행정동')['2030비중'].diff()
panel['Δ임대료']   = panel.groupby('행정동')['임대료'].diff()

# Lag 조합 생성 (동일 행정동 내)
rows_lag = []
for dong, grp in panel.groupby('행정동'):
    grp = grp.sort_values('연도').reset_index(drop=True)
    for i in range(len(grp) - 1):
        t   = grp.iloc[i]
        t1  = grp.iloc[i+1]
        rows_lag.append({
            'dong': dong,
            'year_t': int(t['연도']),
            # 동시 (t)
            'd업종_t':  t['Δ업종점수'],
            'd2030_t':  t['Δ2030비중'],
            'd임대료_t': t['Δ임대료'],
            # t+1 lag
            'd업종_t1':  t1['Δ업종점수'],
            'd2030_t1':  t1['Δ2030비중'],
            'd임대료_t1': t1['Δ임대료'],
        })

lag_df = pd.DataFrame(rows_lag).dropna()

def corr_label(x, y, label=''):
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return 'n/a'
    r, p = pearsonr(x, y)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    return f'r={r:.3f}{sig}  (n={len(x)})'

# ── 2. ML 연간 집계 (오피스 vs 여가 구분) ────────────────────────
def parse_year(s):
    try:
        return int(str(s)[:4])
    except:
        return None

ml['_yr'] = ml.iloc[:, 2].apply(parse_year)
ml_2024 = ml[ml['_yr'] == 2024].copy()

# 연간 평균 (분기 → 연간)
cols_sel = {
    '행정동':      ml.columns[1],
    '주말비중':    ml.columns[18],
    '야간비중':    ml.columns[25],
    '2030소비':    ml.columns[30],
    '금요효과':    ml.columns[31],
    '평일비중':    ml.columns[33],
    '트렌드시간대': ml.columns[34],
    '2030이동':    ml.columns[76],
}

ml_ann_2024 = (ml_2024[[v for v in cols_sel.values()]]
               .rename(columns={v: k for k, v in cols_sel.items()})
               .groupby('행정동').mean().reset_index())

# TSM 2024 업종 정보 추가
tsm24 = tsm[tsm['연도'] == 2024][['행정동','업종점수','2030비중']].copy()
ml_ann_2024 = ml_ann_2024.merge(tsm24, on='행정동', how='left')

# FB카페/식사 비중 추가
cafe_sel = ml_2024[[ml.columns[1], ml.columns[6], ml.columns[7]]].rename(
    columns={ml.columns[1]:'행정동', ml.columns[6]:'FB카페', ml.columns[7]:'FB식사'})
cafe_ann = cafe_sel.groupby('행정동')[['FB카페','FB식사']].mean().reset_index()
cafe_ann['카페_식사_비율'] = cafe_ann['FB카페'] / (cafe_ann['FB식사'] + 1e-6)
ml_ann_2024 = ml_ann_2024.merge(cafe_ann, on='행정동', how='left')

# 오피스 지수: 평일비중 높고 주말/야간 낮으면 오피스
ml_ann_2024['오피스지수'] = ml_ann_2024['평일비중'] / (ml_ann_2024['주말비중'] + 1e-6)

# ── 시각화 ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor('#F8F9FA')
gs = gridspec.GridSpec(3, 3, hspace=0.52, wspace=0.40,
                       left=0.06, right=0.97, top=0.93, bottom=0.05)

fig.suptitle('인과 사슬 검증 + 오피스형/여가형 구분 분석\n(업종 진입 → 2030 유입 → 임대료 상승 / 직장인 2030 vs 여가형 2030)',
             fontsize=13, fontweight='bold', y=0.96)

CONFIRMED = set(tsm[(tsm['연도']==2024) & (tsm['업종점수']>=0.50) & (tsm['2030비중']>=0.35)]['행정동'])

# ── 패널 1: Δ업종점수(t) → Δ2030비중(t+1) ──────────────────────
ax1 = fig.add_subplot(gs[0, 0])

x1 = lag_df['d업종_t'].values
y1 = lag_df['d2030_t1'].values
mask = (~np.isnan(x1)) & (~np.isnan(y1))

ax1.scatter(x1[mask], y1[mask], alpha=0.35, s=20, color='#3498DB')
# 추세선
z = np.polyfit(x1[mask], y1[mask], 1)
xr = np.linspace(x1[mask].min(), x1[mask].max(), 100)
ax1.plot(xr, np.poly1d(z)(xr), color='#1565C0', linewidth=2)
ax1.axhline(0, color='gray', linewidth=0.7, linestyle='--')
ax1.axvline(0, color='gray', linewidth=0.7, linestyle='--')

lbl1 = corr_label(x1[mask], y1[mask])
ax1.set_title(f'① Δ업종점수(t) → Δ2030비중(t+1)\n{lbl1}', fontsize=9, fontweight='bold')
ax1.set_xlabel('Δ업종점수 (해당연도)', fontsize=8)
ax1.set_ylabel('Δ2030비중 (다음연도)', fontsize=8)
ax1.grid(True, alpha=0.3)

# ── 패널 2: Δ2030비중(t) → Δ임대료(t+1) ──────────────────────
ax2 = fig.add_subplot(gs[0, 1])

x2 = lag_df['d2030_t'].values
y2 = lag_df['d임대료_t1'].values
mask2 = (~np.isnan(x2)) & (~np.isnan(y2))

ax2.scatter(x2[mask2], y2[mask2], alpha=0.35, s=20, color='#1565C0')
z2 = np.polyfit(x2[mask2], y2[mask2], 1)
xr2 = np.linspace(x2[mask2].min(), x2[mask2].max(), 100)
ax2.plot(xr2, np.poly1d(z2)(xr2), color='#1565C0', linewidth=2)
ax2.axhline(0, color='gray', linewidth=0.7, linestyle='--')
ax2.axvline(0, color='gray', linewidth=0.7, linestyle='--')

lbl2 = corr_label(x2[mask2], y2[mask2])
ax2.set_title(f'② Δ2030비중(t) → Δ임대료(t+1)\n{lbl2}', fontsize=9, fontweight='bold')
ax2.set_xlabel('Δ2030비중 (해당연도)', fontsize=8)
ax2.set_ylabel('Δ임대료 (다음연도)', fontsize=8)
ax2.grid(True, alpha=0.3)

# ── 패널 3: 동시 vs Lag 상관계수 비교 ──────────────────────────
ax3 = fig.add_subplot(gs[0, 2])

pairs = [
    ('업종→2030 동시', lag_df['d업종_t'], lag_df['d2030_t']),
    ('업종→2030 1년후', lag_df['d업종_t'], lag_df['d2030_t1']),
    ('2030→임대료 동시', lag_df['d2030_t'], lag_df['d임대료_t']),
    ('2030→임대료 1년후', lag_df['d2030_t'], lag_df['d임대료_t1']),
    ('업종→임대료 동시', lag_df['d업종_t'], lag_df['d임대료_t']),
    ('업종→임대료 1년후', lag_df['d업종_t'], lag_df['d임대료_t1']),
]

corrs, labels_c, colors_c = [], [], []
for lbl, x, y in pairs:
    m = (~x.isna()) & (~y.isna())
    if m.sum() < 10:
        corrs.append(0)
    else:
        r, _ = pearsonr(x[m], y[m])
        corrs.append(r)
    labels_c.append(lbl)
    colors_c.append('#1565C0' if '1년후' in lbl else '#BDC3C7')

bars = ax3.barh(range(len(corrs)), corrs, color=colors_c, alpha=0.85)
ax3.axvline(0, color='black', linewidth=0.8)
ax3.set_yticks(range(len(labels_c)))
ax3.set_yticklabels(labels_c, fontsize=8)
ax3.set_xlabel('피어슨 상관계수 r', fontsize=8)
ax3.set_title('③ 동시 상관 vs 1년 후행 상관 비교\n(빨간색 = 시간 지연 효과)', fontsize=9, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# 값 표시
for i, (v, bar) in enumerate(zip(corrs, bars)):
    ax3.text(v + (0.005 if v >= 0 else -0.005), i,
             f'{v:.3f}', va='center', ha='left' if v >= 0 else 'right',
             fontsize=8, fontweight='bold' if colors_c[i] == '#1565C0' else 'normal')

# ── 패널 4: 오피스 vs 여가 — 주말비중 × 야간비중 산점도 ─────────
ax4 = fig.add_subplot(gs[1, :2])

plot_df = ml_ann_2024.dropna(subset=['주말비중','야간비중','2030이동','평일비중'])

# 배경 전체 동
sc = ax4.scatter(plot_df['주말비중'], plot_df['야간비중'],
                 c=plot_df['평일비중'], cmap='RdYlGn_r',
                 s=plot_df['2030이동'] * 3,
                 alpha=0.5, zorder=2, vmin=0.45, vmax=0.70)
plt.colorbar(sc, ax=ax4, label='평일비중 (높을수록 오피스형)', shrink=0.8)

# 임계선
ax4.axvline(0.25, color='green', linewidth=1.2, linestyle='--', alpha=0.7, label='주말비중 0.25')
ax4.axhline(0.35, color='purple', linewidth=1.2, linestyle='--', alpha=0.7, label='야간비중 0.35')

# 구역 라벨
ax4.text(0.18, 0.42, '오피스형\n(평일 직장인 중심)', fontsize=9, color='#0D47A1',
         alpha=0.8, ha='center', style='italic')
ax4.text(0.32, 0.45, '여가형 트렌드\n(주말+야간 소비)', fontsize=9, color='#27AE60',
         alpha=0.8, ha='center', style='italic')
ax4.text(0.18, 0.28, '평일 생활권\n(주민 생활 상권)', fontsize=9, color='gray',
         alpha=0.8, ha='center', style='italic')

# 주목 동 강조
highlight = {
    '서초4동':  ('#0D47A1', '*', 18, '오피스형'),
    '역삼1동':  ('#1565C0', '*', 14, '오피스형'),
    '역삼2동':  ('#1565C0', 'D', 10, '오피스형'),
    '청파동':   ('#2980B9', '*', 18, '전환 후보'),
    '후암동':   ('#3498DB', 's', 12, '전환 후보'),
    '망원1동':  ('#1ABC9C', 's', 12, '전환 후보'),
    '이태원1동':('#9B59B6', '^', 12, '확정'),
    '연남동':   ('#8E44AD', '^', 12, '확정'),
    '성수2가1동':('#6C3483','D', 10, '확정'),
    '합정동':   ('#7D3C98', 'D', 10, '확정'),
}

plotted_labels = {}
for dong, (color, mk, sz, tag) in highlight.items():
    r = ml_ann_2024[ml_ann_2024['행정동'] == dong]
    if len(r) == 0:
        continue
    r = r.iloc[0]
    ax4.scatter(r['주말비중'], r['야간비중'], c=color, marker=mk,
                s=sz**2, zorder=6, edgecolors='white', linewidth=1)
    ax4.annotate(dong, (r['주말비중'], r['야간비중']),
                 xytext=(6, 4), textcoords='offset points',
                 fontsize=9, fontweight='bold', color=color)

ax4.set_title('④ 주말비중 × 야간비중 — 오피스형 vs 여가형 구분\n(점 크기 = 2030이동비율, 색상 = 평일비중)',
              fontsize=10, fontweight='bold')
ax4.set_xlabel('주말비중 (토+일 / 주간 전체)', fontsize=9)
ax4.set_ylabel('야간비중 (17~24시)', fontsize=9)
ax4.legend(fontsize=8, loc='lower right')
ax4.grid(True, alpha=0.3)

# ── 패널 5: 오피스 지수 vs 카페/식사 비율 (여가 정교화) ─────────
ax5 = fig.add_subplot(gs[1, 2])

plot5 = ml_ann_2024.dropna(subset=['오피스지수','카페_식사_비율']).copy()
plot5 = plot5[plot5['카페_식사_비율'] < 1.5]  # 이상값 제거

ax5.scatter(plot5['오피스지수'], plot5['카페_식사_비율'],
            alpha=0.3, s=15, color='lightgray', zorder=1)

for dong, (color, mk, sz, tag) in highlight.items():
    r = plot5[plot5['행정동'] == dong]
    if len(r) == 0:
        continue
    r = r.iloc[0]
    ax5.scatter(r['오피스지수'], r['카페_식사_비율'], c=color, marker=mk,
                s=sz**2, zorder=5, edgecolors='white', linewidth=1)
    ax5.annotate(dong, (r['오피스지수'], r['카페_식사_비율']),
                 xytext=(5, 3), textcoords='offset points',
                 fontsize=8.5, fontweight='bold', color=color)

ax5.axvline(2.5, color='red', linewidth=1, linestyle='--', alpha=0.6, label='오피스지수 2.5')
ax5.axhline(0.30, color='green', linewidth=1, linestyle='--', alpha=0.6, label='카페/식사 비율 0.30')

ax5.set_title('⑤ 오피스지수 × 카페/식사 비율\n(트렌드 상권 = 낮은 오피스지수 + 높은 카페 비중)',
              fontsize=9, fontweight='bold')
ax5.set_xlabel('오피스지수 (평일비중/주말비중)', fontsize=8)
ax5.set_ylabel('카페/식사 매출 비율', fontsize=8)
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# ── 패널 6: 핵심 지표 비교표 ─────────────────────────────────────
ax6 = fig.add_subplot(gs[2, :])
ax6.axis('off')

focus_dongs = ['서초4동', '역삼1동', '청파동', '후암동', '성수2가1동', '합정동', '이태원1동', '연남동']
table_data = []
for dong in focus_dongs:
    r_ml = ml_ann_2024[ml_ann_2024['행정동'] == dong]
    r_tsm = tsm24[tsm24['행정동'] == dong]
    if len(r_ml) == 0 or len(r_tsm) == 0:
        continue
    rm = r_ml.iloc[0]
    rt = r_tsm.iloc[0]

    # 오피스 여부 판단
    office = rm['평일비중'] > 0.55 and rm['주말비중'] < 0.22
    tag = '🔴 오피스형' if office else ('🟢 확정 트렌드' if dong in CONFIRMED else '🔵 전환 후보')

    table_data.append([
        dong, tag,
        f"{rm['주말비중']:.3f}", f"{rm['야간비중']:.3f}",
        f"{rm['평일비중']:.3f}", f"{rm['2030이동']:.1f}%",
        f"{rt['업종점수']:.3f}", f"{rt['2030비중']:.3f}",
        f"{rm['카페_식사_비율']:.3f}",
    ])

col_headers = ['행정동', '분류', '주말비중', '야간비중', '평일비중', '2030이동', '업종점수', '2030소비비중', '카페/식사']

table = ax6.table(
    cellText=table_data,
    colLabels=col_headers,
    cellLoc='center', loc='center',
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(9.5)

# 헤더 스타일
for j in range(len(col_headers)):
    table[0, j].set_facecolor('#2C3E50')
    table[0, j].set_text_props(color='white', fontweight='bold')

# 행 색상
row_colors = {
    '🔴 오피스형': '#FADBD8',
    '🟢 확정 트렌드': '#D5F5E3',
    '🔵 전환 후보': '#D6EAF8',
}
for i, row in enumerate(table_data, start=1):
    tag = row[1]
    bg = row_colors.get(tag, 'white')
    for j in range(len(col_headers)):
        table[i, j].set_facecolor(bg)

ax6.set_title('⑥ 핵심 행정동 비교표 — 오피스형 vs 여가형 vs 확정 트렌드',
              fontsize=10, fontweight='bold', pad=10)

plt.savefig('image/causal_chain_analysis.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print('저장 완료: image/causal_chain_analysis.png')
