# -*- coding: utf-8 -*-
"""
Phase 35 vs Phase 41 후보군 비교 — 보고 자료용
  Phase 35: 레퍼런스 유사도 기반 (업종·소비패턴 현재 상태)
  Phase 41: 20대 이동 성장세 기반 (선행 신호, avg22 ≥ 3% 임계치 적용)

출력:
  image/gen20/10_phase_compare.png  — 비교 종합 차트
  image/gen20/11_compare_map.png    — 지도 비교
"""

import os
import io
import requests
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('image/gen20', exist_ok=True)

# ──────────────────────────────────────────────────────────
# 1. Phase 35 TOP15 (서초4동 오피스형 제외)
# ──────────────────────────────────────────────────────────
print('[1/5] 데이터 로드...')
from report_config import load_v3, phase35_top15, tier1_dongs

_v3 = load_v3()
p35 = _v3[_v3['행정동'].isin(phase35_top15(_v3))].sort_values('코사인유사도', ascending=False).reset_index(drop=True)
p35 = p35.rename(columns={'코사인유사도': '유사도'})
p35['P35순위'] = range(1, len(p35) + 1)

# ──────────────────────────────────────────────────────────
# 2. Phase 41 TOP15 (임계치 적용된 최신 결과)
# ──────────────────────────────────────────────────────────
p41 = pd.read_csv('data/main_data/quadrant_20_result.csv', encoding='utf-8-sig')
p41_top = (p41[p41['사분면'] == '2사분면_유망']
           .nlargest(15, '이동성장세점수')
           .reset_index(drop=True))
p41_top['P41순위'] = range(1, 16)

# ──────────────────────────────────────────────────────────
# 3. 교집합 파악
# ──────────────────────────────────────────────────────────
p35_set  = set(p35['행정동'])
p41_set  = set(p41_top['행정동'])
overlap  = p35_set & p41_set
only_p35 = p35_set - p41_set
only_p41 = p41_set - p35_set

print(f'  Phase 35 후보: {len(p35_set)}개')
print(f'  Phase 41 후보: {len(p41_set)}개')
print(f'  교집합: {len(overlap)}개  → {sorted(overlap)}')
print(f'  Phase35 단독: {sorted(only_p35)}')
print(f'  Phase41 단독: {sorted(only_p41)}')

# ──────────────────────────────────────────────────────────
# 4. 종합 비교 차트  (10_phase_compare.png)
# ──────────────────────────────────────────────────────────
print('[2/5] 종합 비교 차트 생성...')

COLOR_35   = '#1565C0'   # 파란색 — 현재 상태(결과)
COLOR_41   = '#00838F'   # 티얼 — 이동 선행 신호
COLOR_BOTH = '#2E7D32'   # 초록색 — 교집합

fig = plt.figure(figsize=(20, 14))
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
ax_35  = fig.add_subplot(gs[0, 0])
ax_41  = fig.add_subplot(gs[0, 1])
ax_venn = fig.add_subplot(gs[0, 2])
ax_detail = fig.add_subplot(gs[1, :])

# ─── 4-A. Phase 35 가로 막대 ───
p35_s = p35.sort_values('유사도', ascending=True)
colors_35 = [COLOR_BOTH if r in overlap else COLOR_35 for r in p35_s['행정동']]
ax_35.barh(p35_s['행정동'], p35_s['유사도'], color=colors_35, edgecolor='white', height=0.7)
for _, row in p35_s.iterrows():
    c = COLOR_BOTH if row['행정동'] in overlap else COLOR_35
    ax_35.text(row['유사도'] + 0.005, p35_s[p35_s['행정동'] == row['행정동']].index[0],
               f"{row['유사도']:.3f}", va='center', fontsize=8, color='#333')
ax_35.set_xlim(0, 1.05)
ax_35.set_title('Phase 35\n레퍼런스(4) 유사도 TOP15', fontsize=11, fontweight='bold', color=COLOR_35)
ax_35.set_xlabel('코사인 유사도', fontsize=9)
ax_35.tick_params(axis='y', labelsize=9)
ax_35.spines[['top', 'right']].set_visible(False)

# ─── 4-B. Phase 41 가로 막대 ───
p41_s = p41_top.sort_values('이동성장세점수', ascending=True)
colors_41 = [COLOR_BOTH if r in overlap else COLOR_41 for r in p41_s['행정동']]
ax_41.barh(p41_s['행정동'], p41_s['이동성장세점수'], color=colors_41, edgecolor='white', height=0.7)
for _, row in p41_s.iterrows():
    ax_41.text(row['이동성장세점수'] + 0.05,
               p41_s[p41_s['행정동'] == row['행정동']].index[0],
               f"{row['이동성장세점수']:.2f}", va='center', fontsize=8, color='#333')
ax_41.set_title('Phase 41\n20대 이동 성장세 기반 후보', fontsize=11, fontweight='bold', color=COLOR_41)
ax_41.set_xlabel('이동 성장세 점수', fontsize=9)
ax_41.tick_params(axis='y', labelsize=9)
ax_41.spines[['top', 'right']].set_visible(False)

# ─── 4-C. 벤 다이어그램 스타일 요약 ───
ax_venn.axis('off')
from matplotlib.patches import FancyBboxPatch

def draw_candidate_box(ax, x, y, names, title, color, fontsize=8):
    ax.text(x, y + 0.05, title, ha='center', va='bottom', fontsize=9,
            fontweight='bold', color=color,
            transform=ax.transAxes)
    for i, name in enumerate(sorted(names)):
        ax.text(x, y - i * 0.065, f'• {name}', ha='center', va='top',
                fontsize=fontsize, color=color, transform=ax.transAxes)

# Phase 35 전용
draw_candidate_box(ax_venn, 0.18, 0.90,
                   [n for n in p35['행정동'] if n not in overlap],
                   f'Phase 35 전용\n({len(only_p35)}개)', COLOR_35)

# 교집합
if overlap:
    ax_venn.text(0.5, 0.90, f'교집합\n({len(overlap)}개)',
                 ha='center', va='bottom', fontsize=9, fontweight='bold',
                 color=COLOR_BOTH, transform=ax_venn.transAxes)
    for i, name in enumerate(sorted(overlap)):
        ax_venn.text(0.5, 0.83 - i * 0.07, f'★ {name}',
                     ha='center', va='top', fontsize=10, fontweight='bold',
                     color=COLOR_BOTH, transform=ax_venn.transAxes)

# Phase 41 전용
draw_candidate_box(ax_venn, 0.82, 0.90,
                   [n for n in p41_top['행정동'] if n not in overlap],
                   f'Phase 41 전용\n({len(only_p41)}개)', COLOR_41)

ax_venn.set_title('후보군 교집합 분석', fontsize=11, fontweight='bold', pad=10)

# 구분선
from matplotlib.lines import Line2D
for xpos in [0.35, 0.65]:
    line = Line2D([xpos, xpos], [0.05, 0.95], transform=ax_venn.transAxes,
                  color='#dddddd', linewidth=1.5, linestyle='--')
    ax_venn.add_line(line)

# ─── 4-D. 하단 상세 비교 테이블 ───
ax_detail.axis('off')

# 교집합 동에 대해 두 지표를 모두 보여주는 테이블
compare_rows = []
all_dongs = sorted(p35_set | p41_set)

for dong in all_dongs:
    p35_row = p35[p35['행정동'] == dong]
    p41_row = p41_top[p41_top['행정동'] == dong]
    p35_sim  = f"{p35_row['유사도'].values[0]:.3f}" if len(p35_row) else '—'
    p35_ind  = f"#{int(p35_row['P35순위'].values[0])}" if len(p35_row) else '—'
    p41_score = f"{p41_row['이동성장세점수'].values[0]:.2f}" if len(p41_row) else '—'
    p41_ind  = f"#{int(p41_row['P41순위'].values[0])}" if len(p41_row) else '—'
    if dong in overlap:
        tag = '★ 교집합'
    elif dong in only_p35:
        tag = 'Phase 35'
    else:
        tag = 'Phase 41'
    compare_rows.append([dong, tag, p35_ind, p35_sim, p41_ind, p41_score])

col_labels = ['행정동', '분류', 'P35 순위', 'P35 유사도', 'P41 순위', 'P41 성장세점수']
tbl = ax_detail.table(
    cellText=compare_rows,
    colLabels=col_labels,
    loc='center',
    cellLoc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1.0, 1.4)

# 헤더
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#37474F')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

# 행 색상
for i, row in enumerate(compare_rows, 1):
    tag = row[1]
    if tag == '★ 교집합':
        fc = '#E8F5E9'
        tc = COLOR_BOTH
    elif tag == 'Phase 35':
        fc = '#E3F2FD'
        tc = COLOR_35
    else:
        fc = '#E0F7FA'
        tc = COLOR_41
    for j in range(len(col_labels)):
        tbl[i, j].set_facecolor(fc)
        if j == 1:
            tbl[i, j].set_text_props(color=tc, fontweight='bold')

ax_detail.set_title('전체 후보군 지표 비교', fontsize=11, fontweight='bold', pad=8)

# 범례
legend_items = [
    mpatches.Patch(facecolor=COLOR_35, label=f'Phase 35 전용 ({len(only_p35)}개) — 현재 상권 유사도'),
    mpatches.Patch(facecolor=COLOR_41, label=f'Phase 41 전용 ({len(only_p41)}개) — 이동 선행 신호'),
    mpatches.Patch(facecolor=COLOR_BOTH, label=f'교집합 ({len(overlap)}개) — 두 기준 모두 충족'),
]
fig.legend(handles=legend_items, loc='lower center', ncol=3,
           fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

fig.suptitle('Phase 35 vs Phase 41 — 유망 후보군 비교 분석\n'
             'Phase 35: 레퍼런스(4) 코사인 유사도 | Phase 41: 20대 이동 성장세 | 교집합: 숭인2·신당5',
             fontsize=14, fontweight='bold', y=1.01)

plt.savefig('image/gen20/10_phase_compare.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('  저장 완료: image/gen20/10_phase_compare.png')

# ──────────────────────────────────────────────────────────
# 5. 지도 비교  (11_compare_map.png)
# ──────────────────────────────────────────────────────────
print('[3/5] 지도 비교 생성...')

GEOJSON_URL = ('https://raw.githubusercontent.com/vuski/admdongkor/'
               'master/ver20230101/HangJeongDong_ver20230101.geojson')
r = requests.get(GEOJSON_URL, timeout=60)
gdf_all = gpd.read_file(io.BytesIO(r.content))
seoul = gdf_all[gdf_all['sidonm'] == '서울특별시'].copy().to_crs(epsg=5179)
gu = seoul.dissolve(by='sggnm').reset_index()
seoul['dong_name'] = seoul['adm_nm'].apply(lambda x: x.split()[-1])

# 행정동_코드 매핑
master = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig')
code_map = (master[master['연도'] == 2024][['행정동', '행정동_코드']]
            .drop_duplicates()
            .set_index('행정동')['행정동_코드'])

def add_codes(df, name_col):
    df = df.copy()
    df['코드'] = df[name_col].map(code_map).apply(
        lambda x: str(int(x)) if pd.notna(x) else None)
    return df

p35_coded = add_codes(p35, '행정동')
p41_coded = add_codes(p41_top, '행정동')

# GeoJSON 코드 정합
seoul['adm_cd8_s'] = seoul['adm_cd8'].astype(str).str.strip()

def mark_seoul(df_coded, col_name):
    code_set = set(df_coded['코드'].dropna())
    name_set = set(df_coded['행정동'])
    seoul[col_name] = (seoul['adm_cd8_s'].isin(code_set) |
                       seoul['dong_name'].isin(name_set))

mark_seoul(p35_coded, 'is_p35')
mark_seoul(p41_coded, 'is_p41')
seoul['is_overlap'] = seoul['dong_name'].isin(overlap)
seoul['category'] = 'none'
seoul.loc[seoul['is_p35'] & ~seoul['is_p41'],  'category'] = 'p35'
seoul.loc[~seoul['is_p35'] & seoul['is_p41'],  'category'] = 'p41'
seoul.loc[seoul['is_p35'] & seoul['is_p41'],   'category'] = 'both'

fig, axes = plt.subplots(1, 2, figsize=(20, 11))

def plot_map(ax, highlight_cat, title, main_color, alpha_others=0.3):
    # 배경
    seoul[seoul['category'] == 'none'].plot(
        ax=ax, color='#f0f0f0', edgecolor='#d5d5d5', linewidth=0.25)
    # 다른 후보 (연하게)
    other_cats = [c for c in ['p35', 'p41', 'both'] if c != highlight_cat]
    for cat in other_cats:
        c = COLOR_35 if cat == 'p35' else COLOR_41 if cat == 'p41' else COLOR_BOTH
        seoul[seoul['category'] == cat].plot(
            ax=ax, color=c, alpha=alpha_others,
            edgecolor='white', linewidth=0.6)
    # 주요 하이라이트
    seoul[seoul['category'] == highlight_cat].plot(
        ax=ax, color=main_color, edgecolor='white', linewidth=1.0, zorder=3)
    # 교집합은 항상 선명하게
    if highlight_cat != 'both':
        seoul[seoul['category'] == 'both'].plot(
            ax=ax, color=COLOR_BOTH, edgecolor='white', linewidth=1.0, zorder=4)
    # 구 경계
    gu.boundary.plot(ax=ax, color='#888888', linewidth=0.7, zorder=2)

    # 레이블 — 해당 카테고리 + 교집합
    to_label = seoul[seoul['category'].isin([highlight_cat, 'both'])].copy()
    to_label = to_label.drop_duplicates(subset='adm_cd8_s')
    for _, row in to_label.iterrows():
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        is_both = row['category'] == 'both'
        fc = COLOR_BOTH if is_both else main_color
        ax.plot(cx, cy, 'o', markersize=12, color=fc,
                markeredgecolor='white', markeredgewidth=1.5, zorder=5)
        txt = ax.text(cx, cy, row['dong_name'], ha='center', va='center',
                      fontsize=6.5, fontweight='bold', color='white', zorder=6)
        txt.set_path_effects([pe.withStroke(linewidth=2, foreground='#333')])

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.axis('off')

plot_map(axes[0], 'p35',
         f'Phase 35 후보 ({len(p35_set)}개)\n레퍼런스 유사도 기반 (현재 상권 상태)',
         COLOR_35)
plot_map(axes[1], 'p41',
         f'Phase 41 후보 ({len(p41_set)}개)\n20대 이동 성장세 기반 (선행 신호)',
         COLOR_41)

# 공통 범례
legend_items = [
    mpatches.Patch(facecolor=COLOR_35,   label=f'Phase 35 전용 ({len(only_p35)}개)'),
    mpatches.Patch(facecolor=COLOR_41,   label=f'Phase 41 전용 ({len(only_p41)}개)'),
    mpatches.Patch(facecolor=COLOR_BOTH, label=f'교집합 ({len(overlap)}개): {", ".join(sorted(overlap))}'),
]
fig.legend(handles=legend_items, loc='lower center', ncol=3,
           fontsize=10, framealpha=0.95, bbox_to_anchor=(0.5, -0.02))

fig.suptitle('Phase 35 vs Phase 41 — 서울 행정동 지도 비교',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('image/gen20/11_compare_map.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('  저장 완료: image/gen20/11_compare_map.png')

# ──────────────────────────────────────────────────────────
# 6. 결과 요약 출력
# ──────────────────────────────────────────────────────────
print('\n[4/5] 분석 요약')
print('=' * 55)
print(f'  Phase 35 후보 ({len(p35_set)}개):')
for _, r in p35.iterrows():
    mark = '★' if r['행정동'] in overlap else ' '
    print(f'  {mark} {r["P35순위"]:2d}. {r["행정동"]:<12} 유사도={r["유사도"]:.3f}')
print()
print(f'  Phase 41 후보 ({len(p41_set)}개):')
for _, r in p41_top.iterrows():
    mark = '★' if r['행정동'] in overlap else ' '
    print(f'  {mark} {r["P41순위"]:2d}. {r["행정동"]:<12} 성장세={r["이동성장세점수"]:.2f}')
print()
print(f'  ★ 교집합 ({len(overlap)}개): {sorted(overlap)}')
print('  → 두 기준 모두 충족: 가장 신뢰도 높은 후보')
print('=' * 55)
print('[5/5] 완료')
