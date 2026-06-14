# -*- coding: utf-8 -*-
"""
방법론 비교: 우리 방식 vs 팀원 방식
  - 우리: 장기(2022→2024) + OLS기울기 + 최근4분기 Q-o-Q평균 / 필터(avg22≥3%, 기울기>0)
  - 팀원: 장기(2023→2024) + OLS기울기 + window(최근4분기 vs 이전4분기) / 필터 없음

출력: image/gen20/12_method_compare.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import linregress

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = 'image/gen20'
os.makedirs(OUT_DIR, exist_ok=True)

TEAM_DONGS = [
    '고덕2동','방화1동','암사1동','충현동','창2동','화곡8동','수유1동','은천동',
    '신대방2동','종로5·6가동','다산동','상도3동','대방동','신길6동','서초3동'
]

# ────────────────────────────────────────────────────────────────
# 공통: 매출 활성도 점수
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
d24['매출활성도'] = d24[['z_규모', 'z_비중', 'z_성장률']].mean(axis=1)

# ────────────────────────────────────────────────────────────────
# 공통: 이동 데이터 로드
# ────────────────────────────────────────────────────────────────
print('[2/4] 이동 데이터 로드...')
mig_raw = pd.read_csv('data/main_data/분기별_주말유입인구.csv', encoding='utf-8-sig')
_kt = pd.read_excel('data/archive/KT_서울생활이동데이터_행정동코드_20210907.xlsx')
_kt.columns = ['시도', '시군구', '행정동코드', '행정동명', '전체명']
kt_map = dict(zip(_kt['행정동코드'], _kt['행정동명']))

mig20   = mig_raw[mig_raw['연령대'] == 20].copy()
mig_all = mig_raw.groupby(['도착_행정동코드', '연도', '분기'])['합계_이동인구'].sum().reset_index()
mig20_q = mig20.groupby(['도착_행정동코드', '연도', '분기'])['합계_이동인구'].sum().reset_index()
mig20_q = mig20_q.merge(mig_all, on=['도착_행정동코드', '연도', '분기'], suffixes=('_20대', '_전체'))
mig20_q['20대_이동비율'] = mig20_q['합계_이동인구_20대'] / mig20_q['합계_이동인구_전체'].replace(0, np.nan)
mig20_q['행정동'] = mig20_q['도착_행정동코드'].map(kt_map)
mig20_q = mig20_q.dropna(subset=['행정동', '20대_이동비율'])
mig20_q['YQ'] = (mig20_q['연도'] - 2020) * 4 + mig20_q['분기']

# ────────────────────────────────────────────────────────────────
# 이동 성장세: 두 방식으로 계산
# ────────────────────────────────────────────────────────────────
print('[3/4] 이동 성장세 계산 (우리 방식 + 팀원 방식)...')

ours_records = []
team_records = []

for dong, d in mig20_q.groupby('행정동'):
    d = d.sort_values('YQ')

    # ── 공통: OLS 기울기 ──────────────────────────────────
    clean = d.dropna(subset=['20대_이동비율'])
    slope = linregress(clean['YQ'].values, clean['20대_이동비율'].values)[0] if len(clean) >= 4 else np.nan

    avg22 = d[d['연도'] == 2022]['20대_이동비율'].mean()
    avg23 = d[d['연도'] == 2023]['20대_이동비율'].mean()
    avg24 = d[d['연도'] == 2024]['20대_이동비율'].mean()

    # ── 우리 방식 ─────────────────────────────────────────
    # 장기: 2022→2024
    long_ours = (avg24 - avg22) / avg22 if (pd.notna(avg22) and avg22 > 0 and pd.notna(avg24)) else np.nan
    # 최근 모멘텀: 최근 4개 분기 Q-o-Q 평균
    max_yq = d['YQ'].max()
    lag_changes = []
    for yq in range(max_yq - 3, max_yq + 1):
        r_cur = d[d['YQ'] == yq]['20대_이동비율']
        r_lag = d[d['YQ'] == yq - 1]['20대_이동비율']
        vc = r_cur.values[0] if len(r_cur) > 0 else np.nan
        vl = r_lag.values[0] if len(r_lag) > 0 else np.nan
        if pd.notna(vl) and vl >= 0.05 and pd.notna(vc):
            lag_changes.append((vc - vl) / vl)
        else:
            lag_changes.append(0.0)
    momentum_ours = float(np.mean(lag_changes)) if lag_changes else 0.0

    ours_records.append({
        '행정동': dong, '장기증가율': long_ours, '기울기': slope,
        '최근모멘텀': momentum_ours, 'avg22': avg22,
    })

    # ── 팀원 방식 ─────────────────────────────────────────
    # 장기: 2023→2024
    long_team = (avg24 - avg23) / avg23 if (pd.notna(avg23) and avg23 > 0 and pd.notna(avg24)) else np.nan
    # 최근 모멘텀: 최근 4분기(YQ 17~20) vs 이전 4분기(YQ 13~16)
    recent4 = d[d['YQ'].between(17, 20)]['20대_이동비율'].mean()
    prev4   = d[d['YQ'].between(13, 16)]['20대_이동비율'].mean()
    momentum_team = (recent4 - prev4) / prev4 if (pd.notna(prev4) and prev4 > 0 and pd.notna(recent4)) else np.nan

    team_records.append({
        '행정동': dong, '장기증가율': long_team, '기울기': slope,
        '최근모멘텀': momentum_team,
    })

# ── 우리 방식: 필터 + z-score ──────────────────────────────────
gdf_ours = pd.DataFrame(ours_records).dropna(subset=['장기증가율', '기울기'])
gdf_ours = gdf_ours[gdf_ours['avg22'].fillna(0) >= 0.03].copy()
gdf_ours = gdf_ours[gdf_ours['기울기'] > 0].copy()

gdf_ours['z_장기']    = zscore(gdf_ours['장기증가율'])
gdf_ours['z_기울기']  = zscore(gdf_ours['기울기'])
gdf_ours['z_모멘텀']  = zscore(gdf_ours['최근모멘텀'])
gdf_ours['이동성장세'] = gdf_ours[['z_장기', 'z_기울기', 'z_모멘텀']].mean(axis=1)
gdf_ours = gdf_ours.merge(d24[['행정동', '매출활성도']], on='행정동', how='inner')

# ── 팀원 방식: 필터 없음 + z-score ────────────────────────────
gdf_team = pd.DataFrame(team_records).dropna(subset=['장기증가율', '기울기', '최근모멘텀'])
gdf_team['z_장기']    = zscore(gdf_team['장기증가율'])
gdf_team['z_기울기']  = zscore(gdf_team['기울기'])
gdf_team['z_모멘텀']  = zscore(gdf_team['최근모멘텀'])
gdf_team['이동성장세'] = gdf_team[['z_장기', 'z_기울기', 'z_모멘텀']].mean(axis=1)
gdf_team = gdf_team.merge(d24[['행정동', '매출활성도']], on='행정동', how='inner')

# 2사분면 TOP15 추출
ours_q2 = gdf_ours[(gdf_ours['이동성장세'] > 0) & (gdf_ours['매출활성도'] < 0)].nlargest(15, '이동성장세')
team_q2 = gdf_team[(gdf_team['이동성장세'] > 0) & (gdf_team['매출활성도'] < 0)].nlargest(15, '이동성장세')

ours_set = set(ours_q2['행정동'])
team_set = set(team_q2['행정동'])
both_set = ours_set & team_set

print(f'  우리 방식 전체: {len(gdf_ours)}개 행정동')
print(f'  팀원 방식 전체: {len(gdf_team)}개 행정동')
print(f'  우리 2사분면 TOP15: {sorted(ours_set)}')
print(f'  팀원 2사분면 TOP15: {sorted(team_set)}')
print(f'  교집합({len(both_set)}개): {sorted(both_set)}')

# ────────────────────────────────────────────────────────────────
# 시각화
# ────────────────────────────────────────────────────────────────
print('[4/4] 시각화 생성...')

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
fig.suptitle('방법론 비교: 우리 방식 vs 팀원 방식\n20대 매출활성도 × 이동성장세 4분면', fontsize=15, fontweight='bold', y=1.01)

METHOD_INFO = [
    ('우리 방식', gdf_ours, ours_q2, ours_set, '#1565C0',
     '• 장기: 2022→2024\n• 최근: 최근 4개 Q-o-Q 평균\n• 필터: avg22≥3%, OLS기울기>0'),
    ('팀원 방식', gdf_team, team_q2, team_set, '#5C6BC0',
     '• 장기: 2023→2024\n• 최근: 최근4분기 vs 이전4분기 window\n• 필터: 없음'),
]

for ax, (title, gdf, q2, q2_set, color, note) in zip(axes, METHOD_INFO):
    x = gdf['매출활성도']
    y = gdf['이동성장세']

    # 배경 사분면 색상
    ax.axhspan(0, y.max() * 1.2, xmin=0, xmax=0.5, alpha=0.04, color='#1565C0')  # 2사분면
    ax.axhspan(0, y.max() * 1.2, xmin=0.5, xmax=1,  alpha=0.04, color='#00838F')  # 1사분면

    # 전체 산점도
    ax.scatter(x, y, s=18, c='#BDBDBD', alpha=0.5, zorder=2)

    # 팀원이 뽑은 동 표시 (우리 방식 그래프에만)
    if title == '우리 방식':
        for dong in TEAM_DONGS:
            row = gdf[gdf['행정동'] == dong]
            if len(row):
                r = row.iloc[0]
                ax.scatter(r['매출활성도'], r['이동성장세'],
                           s=80, c='#42A5F5', marker='D', zorder=4, alpha=0.85)
                ax.annotate(dong, (r['매출활성도'], r['이동성장세']),
                            fontsize=6.5, color='#00838F',
                            xytext=(4, 3), textcoords='offset points')

    # 2사분면 TOP15 강조
    for _, r in q2.iterrows():
        dong = r['행정동']
        mk = '*' if dong in both_set else 'o'
        c2 = '#2E7D32' if dong in both_set else color
        ax.scatter(r['매출활성도'], r['이동성장세'],
                   s=130 if dong in both_set else 80,
                   c=c2, marker=mk, zorder=5, edgecolors='white', linewidths=0.5)
        ax.annotate(dong, (r['매출활성도'], r['이동성장세']),
                    fontsize=7.5, fontweight='bold', color=c2,
                    xytext=(5, 4), textcoords='offset points')

    ax.axhline(0, color='#555', lw=1.0, ls='--')
    ax.axvline(0, color='#555', lw=1.0, ls='--')

    ax.set_xlabel('매출 활성도 점수 →', fontsize=10)
    ax.set_ylabel('이동 성장세 점수 →', fontsize=10)
    ax.set_title(f'{title}\n(전체 {len(gdf)}개 행정동 | 2사분면 TOP15 표시)', fontsize=11, fontweight='bold')

    # 사분면 레이블
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.text(xlim[0]+0.05, ylim[1]-0.05, '② 유망 후보\n저활성도 × 고성장세', fontsize=8, color='#1565C0', va='top', ha='left', alpha=0.7)
    ax.text(xlim[1]-0.05, ylim[1]-0.05, '① 이미 뜬 곳\n고활성도 × 고성장세', fontsize=8, color='#00838F', va='top', ha='right', alpha=0.7)
    ax.text(xlim[0]+0.05, ylim[0]+0.05, '③ 하락\n저활성도 × 저성장세', fontsize=8, color='gray', va='bottom', ha='left', alpha=0.5)
    ax.text(xlim[1]-0.05, ylim[0]+0.05, '④ 정체\n고활성도 × 저성장세', fontsize=8, color='gray', va='bottom', ha='right', alpha=0.5)

    # 방법론 노트
    ax.text(0.02, 0.02, note, transform=ax.transAxes,
            fontsize=8, va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', alpha=0.8))

    ax.grid(True, alpha=0.2)

# 범례 (우리 방식 그래프 기준)
legend_elements = [
    mpatches.Patch(facecolor='#1565C0', label=f'우리 TOP15 ({len(ours_set)}개)'),
    mpatches.Patch(facecolor='#5C6BC0', label=f'팀원 TOP15 ({len(team_set)}개)'),
    mpatches.Patch(facecolor='#2E7D32', label=f'교집합 ({len(both_set)}개): {", ".join(sorted(both_set)) if both_set else "없음"}'),
    plt.scatter([], [], c='#42A5F5', marker='D', s=60, label='팀원이 뽑은 동 (우리 방식 위치)'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.04), frameon=True)

plt.tight_layout()
out = f'{OUT_DIR}/12_method_compare.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'  저장 완료: {out}')

# 텍스트 비교표 출력
print('\n' + '='*60)
print('우리 TOP15:')
for i, r in enumerate(ours_q2.itertuples(), 1):
    mark = '★' if r.행정동 in both_set else ' '
    print(f'  {i:2}. {mark}{r.행정동:8s}  성장세={r.이동성장세:+.3f}')
print('\n팀원 TOP15:')
for i, r in enumerate(team_q2.itertuples(), 1):
    mark = '★' if r.행정동 in both_set else ' '
    print(f'  {i:2}. {mark}{r.행정동:8s}  성장세={r.이동성장세:+.3f}')
print(f'\n교집합 ({len(both_set)}개): {sorted(both_set)}')
