# -*- coding: utf-8 -*-
"""
4분면 분석: 20대 매출 활성도 × 이동 성장세
  X축: 매출 활성도 점수 = z(20대매출규모) + z(20대매출비중) + z(YoY성장률)
  Y축: 이동 성장세 점수 = z(장기증가율) + z(선형기울기) + z(최근급등_lag1)

  2사분면 (저활성도 × 고성장세) = 유망 후보
  1사분면 (고활성도 × 고성장세) = 이미 뜬 곳

출력: image/gen20/07_quadrant_analysis.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import linregress

from report_config import REFERENCE_DONGS, load_v3, tier1_dongs

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = 'image/gen20'
os.makedirs(OUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────
# 1. 매출 활성도 점수
# ────────────────────────────────────────────────────────────────
print('[1/4] 매출 활성도 점수 계산...')
raw = pd.read_csv('data/archive/서울시_상권분석서비스(추정매출-행정동)_2020_2024.csv',
                  encoding='utf-8-sig')
raw['연도'] = raw['기준_년분기_코드'] // 10
grp = raw.groupby(['행정동_코드_명', '연도'])[['당월_매출_금액', '연령대_20_매출_금액']].sum().reset_index()
grp.columns = ['행정동', '연도', '총매출', '20대매출']
grp['20대비중'] = grp['20대매출'] / grp['총매출'].replace(0, np.nan)

d24 = grp[grp['연도'] == 2024][['행정동', '20대매출', '20대비중']].copy()
d23 = grp[grp['연도'] == 2023][['행정동', '20대매출']].rename(columns={'20대매출': '20대매출_2023'})
d24 = d24.merge(d23, on='행정동', how='left')
d24['YoY성장률'] = (d24['20대매출'] - d24['20대매출_2023']) / d24['20대매출_2023'].replace(0, np.nan)

def zscore(s):
    return (s - s.mean()) / s.std()

d24['z_규모']  = zscore(d24['20대매출'])
d24['z_비중']  = zscore(d24['20대비중'])
d24['z_성장률'] = zscore(d24['YoY성장률'])
d24['매출활성도점수'] = d24[['z_규모', 'z_비중', 'z_성장률']].mean(axis=1)
print(f'  매출 활성도 계산 완료: {len(d24)}개 행정동')

# ────────────────────────────────────────────────────────────────
# 2. 이동 성장세 점수
# ────────────────────────────────────────────────────────────────
print('[2/4] 이동 성장세 점수 계산...')
mig_raw = pd.read_csv('data/main_data/분기별_주말유입인구.csv', encoding='utf-8-sig')
_kt = pd.read_excel('data/archive/KT_서울생활이동데이터_행정동코드_20210907.xlsx')
_kt.columns = ['시도', '시군구', '행정동코드', '행정동명', '전체명']
kt_map = dict(zip(_kt['행정동코드'], _kt['행정동명']))

mig20    = mig_raw[mig_raw['연령대'] == 20].copy()
mig_all  = mig_raw.groupby(['도착_행정동코드', '연도', '분기'])['합계_이동인구'].sum().reset_index()
mig20_q  = mig20.groupby(['도착_행정동코드', '연도', '분기'])['합계_이동인구'].sum().reset_index()
mig20_q  = mig20_q.merge(mig_all, on=['도착_행정동코드', '연도', '분기'], suffixes=('_20대', '_전체'))
mig20_q['20대_이동비율'] = mig20_q['합계_이동인구_20대'] / mig20_q['합계_이동인구_전체'].replace(0, np.nan)
mig20_q['행정동'] = mig20_q['도착_행정동코드'].map(kt_map)
mig20_q = mig20_q.dropna(subset=['행정동', '20대_이동비율'])
# 시계열 인덱스 (2020Q1=1 ~ 2024Q4=20)
mig20_q['YQ'] = (mig20_q['연도'] - 2020) * 4 + mig20_q['분기']

growth_records = []
for dong, d in mig20_q.groupby('행정동'):
    d = d.sort_values('YQ')

    # ① 장기 증가율: 2022 평균 → 2024 평균
    avg22 = d[d['연도'] == 2022]['20대_이동비율'].mean()
    avg24 = d[d['연도'] == 2024]['20대_이동비율'].mean()
    long_growth = (avg24 - avg22) / avg22 if (pd.notna(avg22) and avg22 > 0 and pd.notna(avg24)) else np.nan

    # ② 선형 기울기 (OLS) — 일관된 우상향 여부
    clean = d.dropna(subset=['20대_이동비율'])
    if len(clean) >= 4:
        slope, _, _, _, _ = linregress(clean['YQ'].values, clean['20대_이동비율'].values)
    else:
        slope = np.nan

    # ③ 최근 급등 (lag1 quarterly Q-o-Q): 최근 4개 분기 Q-o-Q 평균
    #    각 분기의 직전 분기 대비 성장률을 계산해 평균 → 단일 포인트보다 안정적
    #    분모 임계치 ≥ 5%: 기저값 낮은 분기는 중립(0.0) 처리
    max_yq = d['YQ'].max()
    recent_n = 4
    lag_changes = []
    for yq in range(max_yq - recent_n + 1, max_yq + 1):
        r_cur = d[d['YQ'] == yq]['20대_이동비율']
        r_lag = d[d['YQ'] == yq - 1]['20대_이동비율']
        vc = r_cur.values[0] if len(r_cur) > 0 else np.nan
        vl = r_lag.values[0] if len(r_lag) > 0 else np.nan
        if pd.notna(vl) and vl >= 0.05 and pd.notna(vc):
            lag_changes.append((vc - vl) / vl)
        else:
            lag_changes.append(0.0)
    recent_surge = float(np.mean(lag_changes)) if lag_changes else 0.0

    growth_records.append({
        '행정동': dong,
        '장기증가율': long_growth,
        '기울기': slope,
        '최근급등': recent_surge,
        'avg22': avg22,
        'avg24': avg24,
    })

gdf = pd.DataFrame(growth_records).dropna(subset=['장기증가율', '기울기', '최근급등'])

# ── 최소 임계치: 2022년 평균 20대 이동비율 ≥ 3% ──────────────────────
# 기저값이 너무 낮은 동은 작은 절댓값 변화도 성장률이 폭등 → artifact 제거
before = len(gdf)
gdf = gdf[gdf['avg22'].fillna(0) >= 0.03].copy()
gdf = gdf[gdf['기울기'] > 0].copy()
print(f'  임계치 필터 (avg22 ≥ 3% + OLS기울기 > 0): {before}개 → {len(gdf)}개 행정동')

gdf['z_장기']  = zscore(gdf['장기증가율'])
gdf['z_기울기'] = zscore(gdf['기울기'])
gdf['z_급등']  = zscore(gdf['최근급등'])
gdf['이동성장세점수'] = gdf[['z_장기', 'z_기울기', 'z_급등']].mean(axis=1)
print(f'  이동 성장세 계산 완료: {len(gdf)}개 행정동')

# ────────────────────────────────────────────────────────────────
# 3. 통합 & 사분면 분류
# ────────────────────────────────────────────────────────────────
print('[3/4] 사분면 분류...')
combined = d24[['행정동', '매출활성도점수', '20대매출', '20대비중', 'YoY성장률']].merge(
    gdf[['행정동', '이동성장세점수', '장기증가율', '최근급등']], on='행정동', how='inner')

def quadrant(row):
    if row['매출활성도점수'] < 0 and row['이동성장세점수'] > 0:
        return '2사분면_유망'
    elif row['매출활성도점수'] > 0 and row['이동성장세점수'] > 0:
        return '1사분면_핫'
    elif row['매출활성도점수'] > 0 and row['이동성장세점수'] < 0:
        return '4사분면_정체'
    else:
        return '3사분면_하락'

combined['사분면'] = combined.apply(quadrant, axis=1)
q_counts = combined['사분면'].value_counts()
print('  사분면별 행정동 수:')
for q, n in q_counts.items():
    print(f'    {q}: {n}개')

# 유망 후보 (2사분면) TOP15 — 이동 성장세 점수 기준
top_promising = combined[combined['사분면'] == '2사분면_유망'].nlargest(15, '이동성장세점수')
# 이미 뜬 곳 (1사분면) TOP10
top_hot = combined[combined['사분면'] == '1사분면_핫'].nlargest(10, '매출활성도점수')

refs = REFERENCE_DONGS
_t1 = set(tier1_dongs(load_v3()))

combined.to_csv('data/main_data/quadrant_20_result.csv', encoding='utf-8-sig', index=False)

# ────────────────────────────────────────────────────────────────
# 4. 시각화
# ────────────────────────────────────────────────────────────────
print('[4/4] 시각화 생성...')
fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('#F8F9FA')
fig.suptitle('20대 매출 활성도 × 이동 성장세 4분면 분석', fontsize=15, fontweight='bold', y=0.98)

gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)
ax_main  = fig.add_subplot(gs[:, :2])   # 메인 4분면 (왼쪽 2/3 전체)
ax_top   = fig.add_subplot(gs[0, 2])    # 유망 TOP15
ax_comp  = fig.add_subplot(gs[1, 2])    # 1사분면 vs 2사분면 비교

# ── 메인 4분면 ──────────────────────────────────────────────────
color_map = {
    '2사분면_유망':  '#1565C0',
    '1사분면_핫':    '#00838F',
    '4사분면_정체':  '#95A5A6',
    '3사분면_하락':  '#BDC3C7',
}
label_map = {
    '2사분면_유망':  f'② 유망 후보 ({q_counts.get("2사분면_유망",0)}개)',
    '1사분면_핫':    f'① 이미 뜬 곳 ({q_counts.get("1사분면_핫",0)}개)',
    '4사분면_정체':  f'④ 정체 ({q_counts.get("4사분면_정체",0)}개)',
    '3사분면_하락':  f'③ 하락 ({q_counts.get("3사분면_하락",0)}개)',
}

for quad, grp_df in combined.groupby('사분면'):
    c = color_map.get(quad, '#7F8C8D')
    sz = 55 if '유망' in quad else (45 if '핫' in quad else 18)
    al = 0.85 if '유망' in quad else (0.7 if '핫' in quad else 0.25)
    ax_main.scatter(grp_df['매출활성도점수'], grp_df['이동성장세점수'],
                    color=c, s=sz, alpha=al, label=label_map.get(quad, quad), zorder=3)

# 유망 TOP15 레이블
for _, row in top_promising.iterrows():
    ax_main.annotate(row['행정동'],
                     (row['매출활성도점수'], row['이동성장세점수']),
                     fontsize=7.5, color='#0D47A1', fontweight='bold',
                     xytext=(4, 3), textcoords='offset points')

# 레퍼런스 4개 · Tier1 교집합
ref_data = combined[combined['행정동'].isin(refs)]
ax_main.scatter(ref_data['매출활성도점수'], ref_data['이동성장세점수'],
                color='#0D47A1', s=100, marker='D', zorder=6, alpha=0.95, label='레퍼런스(4)')
for _, row in ref_data.iterrows():
    ax_main.annotate(row['행정동'],
                     (row['매출활성도점수'], row['이동성장세점수']),
                     fontsize=7.5, color='#0D47A1', fontweight='bold',
                     xytext=(4, 3), textcoords='offset points')
t1_data = combined[combined['행정동'].isin(_t1)]
if len(t1_data):
    ax_main.scatter(t1_data['매출활성도점수'], t1_data['이동성장세점수'],
                    color='#2E7D32', s=120, marker='*', zorder=7, label='Tier1 교집합')

# 사분면 구분선 & 배경
ax_main.axhline(0, color='#555', linewidth=1.2, alpha=0.7)
ax_main.axvline(0, color='#555', linewidth=1.2, alpha=0.7)

xlim = ax_main.get_xlim(); ylim = ax_main.get_ylim()
ax_main.fill_between([xlim[0], 0], 0, ylim[1] if ylim[1] > 0 else 3,
                      color='#E3F2FD', alpha=0.18, zorder=1)  # 2사분면
ax_main.fill_between([0, xlim[1]], 0, ylim[1] if ylim[1] > 0 else 3,
                      color='#E0F7FA', alpha=0.18, zorder=1)  # 1사분면

ax_main.text(xlim[0]*0.85, (ylim[1] if ylim[1]>0 else 3)*0.88,
             '② 유망 후보\n(저활성도 × 고성장세)', fontsize=10, color='#0D47A1',
             fontweight='bold', alpha=0.7)
ax_main.text((xlim[1] if xlim[1]>0 else 3)*0.45, (ylim[1] if ylim[1]>0 else 3)*0.88,
             '① 이미 뜬 곳\n(고활성도 × 고성장세)', fontsize=10, color='#42A5F5',
             fontweight='bold', alpha=0.7)

ax_main.set_xlabel('매출 활성도 점수  (z규모 + z비중 + zYoY성장률)', fontsize=11)
ax_main.set_ylabel('이동 성장세 점수  (z장기증가율 + z기울기 + z최근급등)', fontsize=11)
ax_main.set_title('20대 소비 활성도 vs 이동 성장세 4분면', fontsize=12, fontweight='bold')
ax_main.legend(fontsize=9, loc='lower right')
ax_main.grid(alpha=0.25)

# ── 유망 후보 TOP15 ─────────────────────────────────────────────
colors_top = plt.cm.Blues(np.linspace(0.45, 0.85, len(top_promising)))
bars = ax_top.barh(range(len(top_promising)),
                   top_promising['이동성장세점수'].values,
                   color=colors_top[::-1], edgecolor='white', linewidth=0.8)
ax_top.set_yticks(range(len(top_promising)))
ax_top.set_yticklabels(top_promising['행정동'].values, fontsize=8.5)
ax_top.set_title('② 유망 후보 TOP 15\n(저활성도 × 이동 성장세 상위)', fontsize=10, fontweight='bold')
ax_top.set_xlabel('이동 성장세 점수', fontsize=9)
ax_top.grid(alpha=0.3, axis='x')
# 최근급등 값 표기
for i, (_, row) in enumerate(top_promising.iterrows()):
    ax_top.text(row['이동성장세점수'] + 0.02,
                i, f'+{row["최근급등"]*100:.0f}%', va='center', fontsize=7.5, color='#0D47A1')

# ── 1사분면 vs 2사분면 비교 ─────────────────────────────────────
compare = pd.concat([
    top_promising.head(8).assign(구분='유망(2사분면)'),
    top_hot.head(8).assign(구분='핫(1사분면)')
])
colors_comp = ['#1565C0' if x == '유망(2사분면)' else '#00838F' for x in compare['구분']]
ax_comp.barh(range(len(compare)),
             compare['이동성장세점수'].values,
             color=colors_comp, alpha=0.85, edgecolor='white')
ax_comp.set_yticks(range(len(compare)))
ax_comp.set_yticklabels(compare['행정동'].values, fontsize=8.5)
ax_comp.axvline(0, color='black', linewidth=0.8)
ax_comp.set_title('① 핫 vs ② 유망 이동 성장세 비교\n(각 상위 8개)', fontsize=10, fontweight='bold')
ax_comp.set_xlabel('이동 성장세 점수', fontsize=9)
from matplotlib.patches import Patch
ax_comp.legend(handles=[Patch(color='#1565C0', label='유망(2사분면)'),
                         Patch(color='#00838F', label='핫(1사분면)')], fontsize=8)
ax_comp.grid(alpha=0.3, axis='x')

plt.savefig(f'{OUT_DIR}/07_quadrant_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  저장 완료: {OUT_DIR}/07_quadrant_analysis.png')

# ────────────────────────────────────────────────────────────────
# 유망 후보 상세 출력
# ────────────────────────────────────────────────────────────────
print('\n=== 유망 후보 TOP 15 (2사분면) ===')
show = top_promising[['행정동','이동성장세점수','장기증가율','최근급등','매출활성도점수']].copy()
show['장기증가율'] = show['장기증가율'] * 100
show['최근급등']  = show['최근급등']  * 100
show.columns = ['행정동','성장세점수','장기증가율(%)','최근급등(%)','매출활성도점수']
print(show.to_string(index=False, float_format=lambda x: f'{x:.2f}'))

print('\n=== 이미 뜬 곳 TOP 10 (1사분면) ===')
print(top_hot[['행정동','매출활성도점수','이동성장세점수']].to_string(index=False, float_format=lambda x: f'{x:.2f}'))

# ────────────────────────────────────────────────────────────────
# 5. 유망 후보 시계열 차트
# ────────────────────────────────────────────────────────────────
print('\n[5/5] 유망 후보 시계열 차트 생성...')

top_dongs = top_promising['행정동'].tolist()

# 분기 레이블 생성 (YQ 1~20 → "2020Q1" 형식)
def yq_to_label(yq):
    yr = 2020 + (yq - 1) // 4
    q  = (yq - 1) % 4 + 1
    return f'{yr}Q{q}'

fig2, axes2 = plt.subplots(3, 5, figsize=(22, 12))
fig2.patch.set_facecolor('#F8F9FA')
fig2.suptitle('유망 후보 TOP 15 — 20대 이동비율 분기별 시계열\n(저활성도 × 이동 성장세 상위)',
              fontsize=14, fontweight='bold', y=0.99)

for idx, dong in enumerate(top_dongs):
    ax = axes2[idx // 5][idx % 5]
    d = mig20_q[mig20_q['행정동'] == dong].sort_values('YQ')

    if d.empty:
        ax.set_visible(False)
        continue

    xs = d['YQ'].values
    ys = d['20대_이동비율'].values * 100  # %로 변환

    # 기간 구분 색상
    colors_bar = ['#1565C0' if x >= 13 else '#E3F2FD' for x in xs]  # 2023Q1(YQ=13)부터 강조

    ax.bar(xs, ys, color=colors_bar, edgecolor='white', linewidth=0.5, width=0.85)

    # 선형 추세선
    if len(xs) >= 4:
        slope, intercept, _, _, _ = linregress(xs, ys)
        trend_y = slope * xs + intercept
        ax.plot(xs, trend_y, color='#1565C0', linewidth=1.5, linestyle='--', alpha=0.8)

    # 2023 구분선
    ax.axvline(12.5, color='#1565C0', linewidth=1.2, alpha=0.6, linestyle=':')

    # 최근급등 값
    row = top_promising[top_promising['행정동'] == dong].iloc[0]
    surge_sign = '+' if row['최근급등'] >= 0 else ''
    ax.set_title(f"{dong}\n성장세 {row['이동성장세점수']:.2f} | 최근급등 {surge_sign}{row['최근급등']*100:.0f}%",
                 fontsize=8.5, fontweight='bold', pad=3)

    # x축 레이블: 연도만 표시
    yr_ticks = [yq for yq in xs if (yq - 1) % 4 == 0]  # Q1만
    ax.set_xticks(yr_ticks)
    ax.set_xticklabels([str(2020 + (t-1)//4) for t in yr_ticks], fontsize=7.5)
    ax.set_ylabel('%', fontsize=7.5)
    ax.tick_params(axis='y', labelsize=7.5)
    ax.grid(alpha=0.25, axis='y')
    ax.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)

# 범례 설명
from matplotlib.patches import Patch as MPatch
legend_els = [
    MPatch(facecolor='#E3F2FD', edgecolor='gray', label='~2022년'),
    MPatch(facecolor='#1565C0', edgecolor='gray', label='2023년~'),
    plt.Line2D([0], [0], color='#1565C0', linewidth=1.5, linestyle='--', label='추세선'),
    plt.Line2D([0], [0], color='#1565C0', linewidth=1.2, linestyle=':', label='2023 기준선'),
]
fig2.legend(handles=legend_els, loc='lower center', ncol=4, fontsize=9,
            bbox_to_anchor=(0.5, -0.01), framealpha=0.9)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig(f'{OUT_DIR}/08_quadrant_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  저장 완료: {OUT_DIR}/08_quadrant_timeseries.png')
