"""
유망 후보 TOP15 지도 시각화 (Phase 41 — 4분면 분석 2사분면)
출력: image/gen20/09_map_top15.png
"""

import os
import io
import requests
import warnings
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('image/gen20', exist_ok=True)

# ────────────────────────────────────────
# 1. 유망 후보 TOP15 로드
# ────────────────────────────────────────
df_q = pd.read_csv('data/main_data/quadrant_20_result.csv', encoding='utf-8-sig')
top15 = (df_q[df_q['사분면'] == '2사분면_유망']
         .nlargest(15, '이동성장세점수')
         .reset_index(drop=True))
top15['순위'] = range(1, 16)
print(f"[1/5] TOP15 로드: {top15['행정동'].tolist()}")

# ────────────────────────────────────────
# 2. 행정동_코드 보완 (master 파일에서)
# ────────────────────────────────────────
master = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig')
code_map = (master[master['연도'] == 2024][['행정동', '행정동_코드']]
            .drop_duplicates()
            .set_index('행정동'))
top15['행정동_코드'] = top15['행정동'].map(code_map['행정동_코드'])
top15['코드_str'] = top15['행정동_코드'].apply(
    lambda x: str(int(x)) if pd.notna(x) else None
)
print(f"[2/5] 코드 매핑: {top15[['행정동','코드_str']].to_string(index=False)}")

# ────────────────────────────────────────
# 3. 서울 행정동 GeoJSON 다운로드
# ────────────────────────────────────────
GEOJSON_URL = ('https://raw.githubusercontent.com/vuski/admdongkor/'
               'master/ver20230101/HangJeongDong_ver20230101.geojson')
print('[3/5] GeoJSON 다운로드 중...')
r = requests.get(GEOJSON_URL, timeout=60)
gdf_all = gpd.read_file(io.BytesIO(r.content))
seoul = gdf_all[gdf_all['sidonm'] == '서울특별시'].copy()
seoul = seoul.to_crs(epsg=5179)        # 한국 TM 좌표계
print(f"     서울 행정동 수: {len(seoul)}")

# 구 경계 (dissolve)
gu = seoul.dissolve(by='sggnm').reset_index()

# ────────────────────────────────────────
# 4. GeoJSON ↔ TOP15 매핑
# ────────────────────────────────────────
seoul['adm_cd8_s'] = seoul['adm_cd8'].astype(str).str.strip()
seoul['dong_name'] = seoul['adm_nm'].apply(lambda x: x.split()[-1])

# 점수·순위 딕셔너리
score_dict = dict(zip(top15['코드_str'], top15['이동성장세점수']))
rank_dict  = dict(zip(top15['코드_str'], top15['순위']))
name_dict  = dict(zip(top15['코드_str'], top15['행정동']))

# 코드 기반 매핑 (정확)
seoul['성장세점수'] = seoul['adm_cd8_s'].map(score_dict)
seoul['순위']      = seoul['adm_cd8_s'].map(rank_dict)
seoul['행정동명']   = seoul['adm_cd8_s'].map(name_dict)

# 코드 미매핑 동에 대해 이름 보완 (유일 이름만)
name_uniq = top15.groupby('행정동').filter(lambda g: len(g) == 1)
score_by_name = dict(zip(name_uniq['행정동'], name_uniq['이동성장세점수']))
rank_by_name  = dict(zip(name_uniq['행정동'], name_uniq['순위']))
mask_fill = seoul['순위'].isna() & seoul['dong_name'].isin(set(name_uniq['행정동']))
seoul.loc[mask_fill, '성장세점수'] = seoul.loc[mask_fill, 'dong_name'].map(score_by_name)
seoul.loc[mask_fill, '순위']      = seoul.loc[mask_fill, 'dong_name'].map(rank_by_name)
seoul.loc[mask_fill, '행정동명']   = seoul.loc[mask_fill, 'dong_name']

seoul['is_top15'] = seoul['순위'].notna()
matched = seoul[seoul['is_top15']].copy()
print(f"[4/5] 지도 매핑 성공 ({len(matched)}개): {matched['행정동명'].tolist()}")

# ────────────────────────────────────────
# 5. 시각화
# ────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
gs = GridSpec(1, 2, width_ratios=[3, 1.15], figure=fig, wspace=0.05)
ax_map = fig.add_subplot(gs[0])
ax_leg = fig.add_subplot(gs[1])

# ─── 컬러 설정 ───
cmap = LinearSegmentedColormap.from_list(
    'trend', ['#90CAF9', '#1565C0', '#0D47A1'], N=256
)
norm = Normalize(
    vmin=top15['이동성장세점수'].min() - 0.3,
    vmax=top15['이동성장세점수'].max() + 0.3
)

# ─── 지도 패널 ───
# 기본 행정동
seoul[~seoul['is_top15']].plot(
    ax=ax_map, color='#f0f0f0', edgecolor='#d5d5d5', linewidth=0.25
)
# 구 경계
gu.boundary.plot(ax=ax_map, color='#999999', linewidth=0.9, zorder=2)

# TOP15 폴리곤 (중복 제거: 코드 기준 첫 행만)
matched_unique = matched.drop_duplicates(subset='adm_cd8_s')
for _, row in matched_unique.iterrows():
    c = cmap(norm(row['성장세점수']))
    gpd.GeoDataFrame([row], geometry='geometry', crs=seoul.crs).plot(
        ax=ax_map, color=c, edgecolor='white', linewidth=1.0, zorder=3
    )

# 센트로이드에 순위 번호 원형 마커
for _, row in matched_unique.iterrows():
    cx = row.geometry.centroid.x
    cy = row.geometry.centroid.y
    rank = int(row['순위'])
    c = cmap(norm(row['성장세점수']))
    # 원
    ax_map.plot(cx, cy, 'o', markersize=14, color=c,
                markeredgecolor='white', markeredgewidth=1.5, zorder=5)
    # 순위 숫자
    ax_map.text(cx, cy, str(rank), ha='center', va='center',
                fontsize=7, fontweight='bold', color='white', zorder=6)

# 컬러바
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_map, fraction=0.022, pad=0.01, shrink=0.7)
cbar.set_label('이동 성장세 점수', fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax_map.set_title('2사분면 유망 후보 TOP15 — 서울 행정동 지도\n(저매출 활성도 × 고이동 성장세)',
                 fontsize=12, fontweight='bold', pad=10)
ax_map.axis('off')

# ─── 범례/순위표 패널 ───
ax_leg.axis('off')

# 테이블 데이터
top15_sorted = top15.sort_values('순위')
col_labels = ['순위', '행정동', '이동성장세', '매출활성도']
table_data = [
    [f"{'①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮'[int(r['순위'])-1]}",
     r['행정동'],
     f"{r['이동성장세점수']:.2f}",
     f"{r['매출활성도점수']:.2f}"]
    for _, r in top15_sorted.iterrows()
]

tbl = ax_leg.table(
    cellText=table_data,
    colLabels=col_labels,
    loc='center',
    cellLoc='center'
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.0, 1.65)

# 헤더 스타일
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#37474F')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

# 행 스타일
for i, (_, row) in enumerate(top15_sorted.iterrows(), 1):
    c = cmap(norm(row['이동성장세점수']))
    tbl[i, 0].set_facecolor(c)
    tbl[i, 0].set_text_props(color='white', fontweight='bold')
    bg = '#fafafa' if i % 2 == 0 else '#ffffff'
    for j in range(1, 4):
        tbl[i, j].set_facecolor(bg)

ax_leg.set_title('순위표', fontsize=11, fontweight='bold', pad=8)

# 전체 타이틀
fig.suptitle(
    '4분면 분석 — 유망 후보 TOP15\n(2사분면: 매출 활성도 낮음 × 20대 이동 성장세 높음)',
    fontsize=14, fontweight='bold', y=1.01
)

plt.tight_layout()
out = 'image/gen20/09_map_top15.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"[5/5] 저장 완료: {out}")
