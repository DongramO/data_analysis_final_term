# -*- coding: utf-8 -*-
"""
4분면 매트릭스 Q2 (매출↓ 유입↑) TOP15 서울 행정동 지도
출력: image/map_q2_top15.png
"""
import os, io, warnings
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import contextily as ctx
from shapely.ops import unary_union
from shapely.geometry import box as shapely_box

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT = '.'

# ── Q2 TOP15 로드 ─────────────────────────────────────────────
xls = pd.read_excel(f'{ROOT}/data/quadrant_matrix.xlsx', sheet_name='Q2')
top15 = xls[xls['사분면내순위'] <= 15].copy().reset_index(drop=True)
print(f"Q2 TOP15: {top15['행정동'].tolist()}")

# 행정동 코드 보완
master = pd.read_csv(f'{ROOT}/data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig')
code_map = (master[master['연도'] == 2024][['행정동','행정동_코드']]
            .drop_duplicates().set_index('행정동'))
top15['행정동_코드'] = top15['행정동'].map(code_map['행정동_코드'])
top15['코드_str']   = top15['행정동_코드'].apply(
    lambda x: str(int(x)) if pd.notna(x) else None)

# ── GeoJSON 다운로드 ─────────────────────────────────────────
GEOJSON_URL = ('https://raw.githubusercontent.com/vuski/admdongkor/'
               'master/ver20230101/HangJeongDong_ver20230101.geojson')
print('GeoJSON 다운로드 중...')
r = requests.get(GEOJSON_URL, timeout=60)
gdf_all = gpd.read_file(io.BytesIO(r.content))
seoul   = gdf_all[gdf_all['sidonm'] == '서울특별시'].copy()
seoul   = seoul.to_crs(epsg=3857)   # Web Mercator (contextily 호환)
gu      = seoul.dissolve(by='sggnm').reset_index()

# ── 매핑 ─────────────────────────────────────────────────────
seoul['adm_cd8_s'] = seoul['adm_cd8'].astype(str).str.strip()
seoul['dong_name'] = seoul['adm_nm'].apply(lambda x: x.split()[-1])

score_dict = dict(zip(top15['코드_str'], top15['이동성장세점수']))
name_dict  = dict(zip(top15['코드_str'], top15['행정동']))
rank_dict  = dict(zip(top15['코드_str'], top15['사분면내순위']))

seoul['성장세점수'] = seoul['adm_cd8_s'].map(score_dict)
seoul['행정동명']   = seoul['adm_cd8_s'].map(name_dict)
seoul['사분면순위'] = seoul['adm_cd8_s'].map(rank_dict)

# 코드 미매핑 → 이름으로 보완
name_uniq = top15.groupby('행정동').filter(lambda g: len(g) == 1)
s_name = dict(zip(name_uniq['행정동'], name_uniq['이동성장세점수']))
r_name = dict(zip(name_uniq['행정동'], name_uniq['사분면내순위']))
mask   = seoul['성장세점수'].isna() & seoul['dong_name'].isin(set(name_uniq['행정동']))
seoul.loc[mask, '성장세점수'] = seoul.loc[mask, 'dong_name'].map(s_name)
seoul.loc[mask, '행정동명']   = seoul.loc[mask, 'dong_name']
seoul.loc[mask, '사분면순위'] = seoul.loc[mask, 'dong_name'].map(r_name)

seoul['is_top15'] = seoul['성장세점수'].notna()
matched = seoul[seoul['is_top15']].drop_duplicates(subset='adm_cd8_s')
print(f"매핑 성공: {matched['행정동명'].tolist()}")

# ── 시각화 ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 11))

# 배경 행정동 (회색)
seoul[~seoul['is_top15']].plot(
    ax=ax, color='#D9D9D9', edgecolor='#BBBBBB', linewidth=0.25, alpha=0.7)

# 구 경계
gu.boundary.plot(ax=ax, color='#888888', linewidth=0.8, zorder=2)

# Q2 TOP15 하이라이트 (청록색)
matched.plot(ax=ax, color='#00BCD4', edgecolor='white',
             linewidth=1.0, alpha=0.85, zorder=3)

# 행정동 이름 라벨
for _, row in matched.iterrows():
    cx = row.geometry.centroid.x
    cy = row.geometry.centroid.y
    txt = ax.text(cx, cy, row['행정동명'],
                  ha='center', va='center',
                  fontsize=11, fontweight='bold', color='#0D1B2A', zorder=6)
    txt.set_path_effects([
        pe.Stroke(linewidth=4.0, foreground='white'),
        pe.Normal()
    ])

# OpenStreetMap 타일 배경
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=11, alpha=0.45)

# 서울 외부 블러(화이트 오버레이)
xlim, ylim = ax.get_xlim(), ax.get_ylim()
bbox_geom = shapely_box(xlim[0] - 5000, ylim[0] - 5000,
                        xlim[1] + 5000, ylim[1] + 5000)
outside_geom = bbox_geom.difference(unary_union(seoul.geometry))
gpd.GeoDataFrame(geometry=[outside_geom], crs=seoul.crs).plot(
    ax=ax, color='white', alpha=0.80, zorder=4)

# 서울 외곽선 강조
seoul.dissolve().boundary.plot(ax=ax, color='#666666', linewidth=1.0, zorder=5)

ax.set_title(
    '4분면 매트릭스 Q2 TOP 15  —  서울 행정동 지도\n'
    '(매출활성도↓  ×  이동성장세↑  |  유입 선행 유망 지역)',
    fontsize=12, fontweight='bold', pad=12)
ax.axis('off')

plt.tight_layout()
out = f'{ROOT}/image/map_q2_top15.png'
fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'[저장] {out}')
