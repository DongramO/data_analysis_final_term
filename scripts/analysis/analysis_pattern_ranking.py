# -*- coding: utf-8 -*-
"""
핵심패턴 행정동 순위 비교 시각화
출력: image/pattern_ranking.png
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT    = Path(__file__).resolve().parent
DF_PATH = ROOT / 'data/핵심패턴_행정동_순위비교.csv'
IMG_DIR = ROOT / 'image'

df = pd.read_csv(DF_PATH, encoding='utf-8-sig')

PAT_COLOR = {
    '두지표동시': '#E05C5C',
    '주말만높음': '#3A5CA9',
    '20대만높음': '#27AE60',
}
PAT_LABEL = {
    '두지표동시': '두 지표 동시 상위 (트렌드 상권)',
    '주말만높음': '주말만 높음 (교외·마트형)',
    '20대만높음': '20대만 높음 (대학가 평일형)',
}
TOTAL = 423

fig, axes = plt.subplots(1, 2, figsize=(22, 9),
                         gridspec_kw={'width_ratios': [1, 1.4]})
fig.suptitle('핵심 패턴별 행정동 — 주요 지표 순위 비교 (2024, 전체 423개 행정동)',
             fontsize=14, fontweight='bold')

# ═══════════════════════════════════════════════════════════════
# Panel 1: 산포도 — 주말비중 vs 20대비중 (버블 = 전체매출)
# ═══════════════════════════════════════════════════════════════
ax1 = axes[0]

# 배경 사분면
xmid = df['주말비중(%)'].median()
ymid = df['20대비중(%)'].median()
ax1.axvline(30, color='#DDDDDD', lw=1.2, ls='--', zorder=1)
ax1.axhline(25, color='#DDDDDD', lw=1.2, ls='--', zorder=1)

# 사분면 레이블
ax1.text(43, 38, '주말↑ 20대↑\n(트렌드 핵심)', fontsize=8,
         color='#E05C5C', alpha=0.6, ha='center')
ax1.text(18, 38, '주말↓ 20대↑\n(대학가형)',   fontsize=8,
         color='#27AE60', alpha=0.6, ha='center')
ax1.text(43, 5,  '주말↑ 20대↓\n(마트·교외형)', fontsize=8,
         color='#3A5CA9', alpha=0.6, ha='center')

for _, row in df.iterrows():
    pat   = row['패턴유형']
    color = PAT_COLOR[pat]
    size  = max(row['전체매출(억)'] / 14537 * 2000, 80)   # 서교동 기준 정규화

    ax1.scatter(row['주말비중(%)'], row['20대비중(%)'],
                s=size, color=color, alpha=0.75,
                edgecolors='white', linewidths=1.2, zorder=3)

    # 레이블 위치 미세조정
    dx, dy = 0.4, 0.5
    if row['행정동'] == '노량진2동': dx, dy = 0.4, -1.5
    if row['행정동'] == '서교동':    dx, dy = -5.5, 0.5
    if row['행정동'] == '노량진1동': dx, dy = 0.4, -1.5
    if row['행정동'] == '신촌동':    dx, dy = 0.4, -1.5

    ax1.annotate(row['행정동'],
                 (row['주말비중(%)'], row['20대비중(%)']),
                 xytext=(dx, dy), textcoords='offset points',
                 fontsize=8.5, color=color, fontweight='bold', zorder=4)

# 버블 크기 범례
for amt, label in [(500,'500억'), (5000,'5,000억'), (14537,'14,537억')]:
    s = max(amt / 14537 * 2000, 80)
    ax1.scatter([], [], s=s, color='#AAAAAA', alpha=0.6,
                edgecolors='white', label=label)

# 패턴 범례
pat_handles = [mpatches.Patch(color=c, label=PAT_LABEL[p], alpha=0.8)
               for p, c in PAT_COLOR.items()]
size_handles = [Line2D([0],[0], marker='o', color='w',
                        markerfacecolor='#AAAAAA', markersize=np.sqrt(max(amt/14537*2000,80)/np.pi),
                        label=lbl, alpha=0.7)
                for amt, lbl in [(500,'500억'),(5000,'5,000억'),(14537,'14,537억')]]

leg1 = ax1.legend(handles=pat_handles, fontsize=8.5,
                  loc='upper left', title='패턴유형', title_fontsize=9)
ax1.add_artist(leg1)
ax1.legend(handles=size_handles, fontsize=8, loc='lower right',
           title='전체매출 규모', title_fontsize=9)

ax1.set_xlabel('주말 매출 비중 (%)', fontsize=10)
ax1.set_ylabel('20대 매출 비중 (%)', fontsize=10)
ax1.set_title('주말비중 vs 20대비중\n(버블 크기 = 전체 매출 규모)', fontsize=11)
ax1.set_facecolor('#F8F8F8')
ax1.grid(alpha=0.15)

# ═══════════════════════════════════════════════════════════════
# Panel 2: 순위 히트맵 (낮은 순위 = 좋은 = 진한 색)
# ═══════════════════════════════════════════════════════════════
ax2 = axes[1]

RANK_COLS = [
    ('전체매출_순위(/423)', '전체매출\n순위'),
    ('주말매출_순위(/423)', '주말매출\n순위'),
    ('주말비중_순위(/423)', '주말비중\n순위'),
    ('20대매출_순위(/423)', '20대매출\n순위'),
    ('20대비중_순위(/423)', '20대비중\n순위'),
]

# 정렬: 패턴 → 주말비중순위
pat_order = {'두지표동시': 0, '주말만높음': 1, '20대만높음': 2}
df_sorted = df.copy()
df_sorted['_ord'] = df_sorted['패턴유형'].map(pat_order)
df_sorted = df_sorted.sort_values(['_ord', '주말비중_순위(/423)']).reset_index(drop=True)

ax2.axis('off')

# ── 정렬 ──────────────────────────────────────────────────────
pat_order = {'두지표동시': 0, '주말만높음': 1, '20대만높음': 2}
df_sorted = df.copy()
df_sorted['_ord'] = df_sorted['패턴유형'].map(pat_order)
df_sorted = df_sorted.sort_values(['_ord', '주말비중_순위(/423)']).reset_index(drop=True)

# ── 테이블 데이터 ──────────────────────────────────────────────
col_keys = [
    '패턴유형', '행정동',
    '전체매출(억)', '전체매출_순위(/423)',
    '주말매출(억)', '주말매출_순위(/423)', '주말비중(%)', '주말비중_순위(/423)',
    '20대매출(억)', '20대매출_순위(/423)', '20대비중(%)', '20대비중_순위(/423)',
]
col_headers = [
    '패턴', '행정동',
    '전체매출\n(억)', '전체매출\n순위',
    '주말매출\n(억)', '주말매출\n순위', '주말\n비중%', '주말비중\n순위',
    '20대매출\n(억)', '20대매출\n순위', '20대\n비중%', '20대비중\n순위',
]

cell_text = []
for _, row in df_sorted.iterrows():
    cell_text.append([
        row['패턴유형'].replace('두지표동시','두지표\n동시').replace('주말만높음','주말만\n높음').replace('20대만높음','20대만\n높음'),
        row['행정동'],
        f"{row['전체매출(억)']:,.0f}",  f"{row['전체매출_순위(/423)']:.0f}위",
        f"{row['주말매출(억)']:,.0f}",  f"{row['주말매출_순위(/423)']:.0f}위",
        f"{row['주말비중(%)']:.1f}%",   f"{row['주말비중_순위(/423)']:.0f}위",
        f"{row['20대매출(억)']:,.0f}",  f"{row['20대매출_순위(/423)']:.0f}위",
        f"{row['20대비중(%)']:.1f}%",   f"{row['20대비중_순위(/423)']:.0f}위",
    ])

tbl = ax2.table(cellText=cell_text, colLabels=col_headers,
                cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)

# ── 셀 색상 ───────────────────────────────────────────────────
RANK_COL_IDX = [3, 5, 7, 9, 11]   # 순위 컬럼 인덱스

def rank_color(rank_val):
    if rank_val <= 50:  return '#D5F5E3'   # 연초록 (상위권)
    if rank_val <= 150: return '#FEF9E7'   # 연노랑 (중위권)
    return '#FDECEA'                        # 연분홍 (하위권)

for (ri, ci), cell in tbl.get_celld().items():
    cell.set_edgecolor('#E0E0E0')
    if ri == 0:
        cell.set_facecolor('#3A5CA9')
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_height(0.09)
        continue

    row_data = df_sorted.iloc[ri - 1]
    pat = row_data['패턴유형']

    # 행정동명 · 패턴 컬럼 → 패턴 색 배경
    if ci <= 1:
        cell.set_facecolor('#F7F7F7')
        if ci == 1:
            cell.set_text_props(color=PAT_COLOR[pat], fontweight='bold')
        if ci == 0:
            cell.set_text_props(color=PAT_COLOR[pat], fontsize=7.5)
    elif ci in RANK_COL_IDX:
        rank_val = int(str(cell.get_text().get_text()).replace('위',''))
        cell.set_facecolor(rank_color(rank_val))
        cell.set_text_props(fontweight='bold')
    else:
        cell.set_facecolor('#FFFFFF')

    # 패턴 구분 — 경계선 두껍게
    if ri in [6, 11]:
        cell.visible_edges = 'open'

# 컬럼 너비 조정
col_widths = [0.07, 0.09, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07]
for ci, w in enumerate(col_widths):
    for ri in range(len(cell_text) + 1):
        tbl[ri, ci].set_width(w)

# 범례 (색상 의미)
legend_items = [
    mpatches.Patch(color='#D5F5E3', label='상위권 (50위 이내)'),
    mpatches.Patch(color='#FEF9E7', label='중위권 (51~150위)'),
    mpatches.Patch(color='#FDECEA', label='하위권 (151위~)'),
]
ax2.legend(handles=legend_items, loc='lower center',
           bbox_to_anchor=(0.5, -0.04), ncol=3,
           fontsize=9, framealpha=0.9, edgecolor='#DDDDDD')

ax2.set_title('5개 지표 순위 비교표  (순위셀: 초록=상위 / 노랑=중위 / 분홍=하위)',
              fontsize=11, pad=12)

plt.tight_layout()
out = IMG_DIR / 'pattern_ranking.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[저장] {out.name}")
