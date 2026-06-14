# -*- coding: utf-8 -*-
"""
20대 단독 기준 재분석
2030 복합 지표를 20대(20~29세)만으로 재정의하여 전체 분석 재수행.

출력 디렉토리: image/gen20/
  01_consumption_trend.png  — 20대 소비비중 서울 전체 트렌드 (2020~2024)
  02_gen20_vs_30s.png       — 20대 vs 30대 소비 패턴 비교
  03_migration.png          — 20대 이동비율 상위 지역 및 패턴
  04_weekend_profile.png    — 20대 주말집중도 (생활인구 재계산)
  05_candidates.png         — 20대 기준 차세대 후보군
  06_reference_trajectory.png — 레퍼런스 지역 20대 궤적

데이터 소스:
  - 소비 20대비중 : data/archive/서울시_상권분석서비스(추정매출-행정동)_2020_2024.csv
  - 이동 20대      : data/main_data/분기별_주말유입인구.csv  (연령대==20)
  - 생활인구 20대  : data/population/LOCAL_PEOPLE_DONG_2024{07~09}.csv
                     iloc[:, [6,7,20,21]] = 남20-24, 남25-29, 여20-24, 여25-29
"""

import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

from report_config import (
    REFERENCE_DONGS,
    load_v3,
    phase35_top15,
    tier1_dongs,
    tier2_dongs,
    tier3_dongs,
)

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = 'image/gen20'
os.makedirs(OUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────
# 1. 소비 데이터 — 20대비중 계산
# ────────────────────────────────────────────────────────────────
print('[1/6] 소비 데이터 로드 중...')
raw = pd.read_csv('data/archive/서울시_상권분석서비스(추정매출-행정동)_2020_2024.csv',
                  encoding='utf-8-sig')

# 컬럼 정의 (실제 컬럼명 기준)
# 기준_년분기_코드: 20201 = 2020년 1분기 (YYYYQ 형식)
raw['연도'] = raw['기준_년분기_코드'] // 10
dong_col = '행정동_코드_명'
tot_col  = '당월_매출_금액'
a20_col  = '연령대_20_매출_금액'
a30_col  = '연령대_30_매출_금액'

# 행정동코드 → 행정동명 매핑 (이동 데이터 조인용)
dong_name_map = raw.drop_duplicates('행정동_코드').set_index('행정동_코드')['행정동_코드_명'].to_dict()

# 행정동×연도 집계 (전업종 합산)
grp = raw.groupby([dong_col, '연도'])[[tot_col, a20_col, a30_col]].sum().reset_index()
grp.columns = ['행정동', '연도', '총매출', '20대매출', '30대매출']
grp['20대비중'] = grp['20대매출'] / grp['총매출'].replace(0, np.nan)
grp['30대비중'] = grp['30대매출'] / grp['총매출'].replace(0, np.nan)
grp['2030비중'] = (grp['20대매출'] + grp['30대매출']) / grp['총매출'].replace(0, np.nan)

# trend_sales_master 병합 (업종점수, 복합점수_분위)
tsm = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig')
grp = grp.merge(tsm[['행정동','연도','업종점수','복합점수_분위','상권티어']], on=['행정동','연도'], how='left')

# ────────────────────────────────────────────────────────────────
# 2. 이동 데이터 — 20대만 필터
# ────────────────────────────────────────────────────────────────
print('[2/6] 이동 데이터 로드 중...')
mig_raw = pd.read_csv('data/main_data/분기별_주말유입인구.csv', encoding='utf-8-sig')

# KT 7자리 코드 → 행정동명 매핑 (migration 데이터는 7자리 코드 사용)
_kt = pd.read_excel('data/archive/KT_서울생활이동데이터_행정동코드_20210907.xlsx')
_kt.columns = ['시도','시군구','행정동코드','행정동명','전체명']
kt_dong_map = dict(zip(_kt['행정동코드'], _kt['행정동명']))

# 20대만 필터 (연령대==20)
mig20 = mig_raw[mig_raw['연령대'] == 20].copy()
mig_all = mig_raw.groupby(['도착_행정동코드','연도','분기'])['합계_이동인구'].sum().reset_index()
mig20_grp = mig20.groupby(['도착_행정동코드','연도','분기'])['합계_이동인구'].sum().reset_index()
mig20_grp = mig20_grp.merge(mig_all, on=['도착_행정동코드','연도','분기'], suffixes=('_20대','_전체'))
mig20_grp['20대_이동비율'] = mig20_grp['합계_이동인구_20대'] / mig20_grp['합계_이동인구_전체'].replace(0, np.nan)

# KT 7자리 코드 → 행정동명 매핑
mig20_grp['행정동'] = mig20_grp['도착_행정동코드'].map(kt_dong_map)

# 연도별 평균
mig20_annual = mig20_grp.groupby(['행정동','연도'])['20대_이동비율'].mean().reset_index()

# ────────────────────────────────────────────────────────────────
# 3. 생활인구 — 20대 주말집중도 재계산 (2024년 7~9월)
# ────────────────────────────────────────────────────────────────
print('[3/6] 생활인구 20대 주말집중도 계산 중...')
POP_COLS_20 = [6, 7, 20, 21]   # 남20-24, 남25-29, 여20-24, 여25-29 (index after date→index)
POP_MONTHS  = ['202407','202408','202409']

pop_records = []
for ym in POP_MONTHS:
    fpath = f'data/population/LOCAL_PEOPLE_DONG_{ym}.csv'
    if not os.path.exists(fpath):
        continue
    df = pd.read_csv(fpath, encoding='utf-8-sig', index_col=0)
    dates   = df.index.astype(str)
    hours   = df.iloc[:, 0].values
    codes   = df.iloc[:, 1].values
    pop_tot = df.iloc[:, 2].values
    pop_20  = df.iloc[:, POP_COLS_20].sum(axis=1).values

    for date, hour, code, tot, p20 in zip(dates, hours, codes, pop_tot, pop_20):
        try:
            d = pd.to_datetime(str(int(date)), format='%Y%m%d')
            is_weekend = d.dayofweek >= 5
            pop_records.append({'코드': int(code), '주말여부': is_weekend,
                                 '총인구': tot, '20대인구': p20})
        except:
            continue

pop_df = pd.DataFrame(pop_records)
profile20 = pop_df.groupby(['코드','주말여부'])[['총인구','20대인구']].mean().reset_index()
weekend20 = profile20[profile20['주말여부']==True].rename(
    columns={'총인구':'주말_총인구','20대인구':'주말_20대인구'})
weekday20 = profile20[profile20['주말여부']==False].rename(
    columns={'총인구':'평일_총인구','20대인구':'평일_20대인구'})
prof20 = weekend20[['코드','주말_총인구','주말_20대인구']].merge(
         weekday20[['코드','평일_총인구','평일_20대인구']], on='코드')
prof20['주말집중도']    = prof20['주말_총인구']   / prof20['평일_총인구'].replace(0, np.nan)
prof20['20대주말집중도'] = prof20['주말_20대인구'] / prof20['평일_20대인구'].replace(0, np.nan)
prof20['행정동'] = prof20['코드'].map(dong_name_map)
prof20 = prof20.dropna(subset=['행정동'])

# 구역유형 (주말집중도 기준)
def classify_zone(v):
    if v < 0.80: return '오피스형'
    if v < 0.95: return '혼합형'
    if v <= 1.10: return '주거형'
    return '상권형'
prof20['구역유형'] = prof20['주말집중도'].apply(classify_zone)

prof20.to_csv('data/main_data/area_type_profile_gen20.csv', encoding='utf-8-sig', index=False)
print(f'  생활인구 프로파일 저장: {len(prof20)}개 행정동')

# ────────────────────────────────────────────────────────────────
# 4. 분석용 병합 데이터셋
# ────────────────────────────────────────────────────────────────
anal = grp.merge(
    mig20_annual[['행정동','연도','20대_이동비율']], on=['행정동','연도'], how='left'
).merge(
    prof20[['행정동','20대주말집중도','구역유형']], on='행정동', how='left'
)

# ════════════════════════════════════════════════════════════════
# 시각화 01: 20대 소비비중 서울 전체 트렌드
# ════════════════════════════════════════════════════════════════
print('[4/6] 시각화 01~02 생성 중...')
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor('#F8F9FA')
fig.suptitle('20대 소비비중 서울 전체 트렌드 (2020~2024)', fontsize=14, fontweight='bold')

yrs = sorted(anal['연도'].dropna().unique())

# ① 서울 전체 평균 20대비중 vs 30대비중 연도별
ax = axes[0,0]
mean20 = anal.groupby('연도')['20대비중'].mean()
mean30 = anal.groupby('연도')['30대비중'].mean()
mean2030 = anal.groupby('연도')['2030비중'].mean()
ax.plot(yrs, mean20.reindex(yrs)*100, 'o-', color='#1565C0', linewidth=2.5, markersize=7, label='20대')
ax.plot(yrs, mean30.reindex(yrs)*100, 's-', color='#3498DB', linewidth=2.5, markersize=7, label='30대')
ax.plot(yrs, mean2030.reindex(yrs)*100, '^--', color='#8E44AD', linewidth=1.5, markersize=6, label='2030합산', alpha=0.7)
ax.set_title('① 서울 전체 평균 연령대별 소비비중', fontsize=10, fontweight='bold')
ax.set_ylabel('평균 비중 (%)')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
for yr, v in mean20.reindex(yrs).items():
    ax.annotate(f'{v*100:.1f}%', (yr, v*100+0.2), ha='center', fontsize=8, color='#1565C0')

# ② 20대비중 분포 변화 (박스플롯)
ax = axes[0,1]
data_by_yr = [anal[anal['연도']==yr]['20대비중'].dropna()*100 for yr in yrs]
bp = ax.boxplot(data_by_yr, labels=yrs, patch_artist=True, medianprops=dict(color='white', linewidth=2))
colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(yrs)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_title('② 20대비중 분포 변화 (박스플롯)', fontsize=10, fontweight='bold')
ax.set_ylabel('20대 소비비중 (%)')
ax.grid(alpha=0.3, axis='y')

# ③ 20대비중 상위 동 TOP 15 (2024)
ax = axes[0,2]
top15_20 = anal[anal['연도']==2024].nlargest(15, '20대비중')[['행정동','20대비중','2030비중']]
y_pos = range(len(top15_20))
ax.barh(y_pos, top15_20['20대비중'].values*100, color='#1565C0', alpha=0.8, label='20대')
ax.barh(y_pos, (top15_20['2030비중']-top15_20['20대비중']).values*100,
        left=top15_20['20대비중'].values*100, color='#3498DB', alpha=0.6, label='30대(추가)')
ax.set_yticks(list(y_pos)); ax.set_yticklabels(top15_20['행정동'].values, fontsize=8)
ax.set_title('③ 20대비중 TOP 15 (2024)', fontsize=10, fontweight='bold')
ax.set_xlabel('소비비중 (%)'); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='x')

# ④ 20대비중 변화량 TOP 15 (2020→2024)
ax = axes[1,0]
pivot = anal.pivot_table(index='행정동', columns='연도', values='20대비중')
if 2020 in pivot.columns and 2024 in pivot.columns:
    change = (pivot[2024] - pivot[2020]).dropna().sort_values(ascending=False)
    top15c = change.head(15)
    bottom15c = change.tail(15)
    ax.barh(range(15), top15c.values*100, color='#1565C0', alpha=0.85)
    ax.set_yticks(range(15)); ax.set_yticklabels(top15c.index, fontsize=8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title('④ 20대비중 상승폭 TOP 15\n(2020→2024)', fontsize=10, fontweight='bold')
    ax.set_xlabel('변화량 (%p)')
    ax.grid(alpha=0.3, axis='x')

# ⑤ 20대 이동비율 상위 동 (2024)
ax = axes[1,1]
mig24 = mig20_annual[mig20_annual['연도']==2024].dropna(subset=['행정동']).nlargest(15, '20대_이동비율')
ax.barh(range(len(mig24)), mig24['20대_이동비율'].values*100, color='#1565C0', alpha=0.85)
ax.set_yticks(range(len(mig24))); ax.set_yticklabels(mig24['행정동'].values, fontsize=8)
ax.set_title('⑤ 20대 이동비율 TOP 15 (2024)', fontsize=10, fontweight='bold')
ax.set_xlabel('20대 이동비율 (%)'); ax.grid(alpha=0.3, axis='x')

# ⑥ 20대 소비비중 × 업종점수 산점도 (2024)
ax = axes[1,2]
d24 = anal[anal['연도']==2024].dropna(subset=['20대비중','업종점수'])
ax.scatter(d24['업종점수'], d24['20대비중']*100, alpha=0.35, s=20, color='#95A5A6')
# 상위 동 강조
top_dong = d24.nlargest(10, '20대비중')
ax.scatter(top_dong['업종점수'], top_dong['20대비중']*100, color='#1565C0', s=60, zorder=5)
for _, row in top_dong.iterrows():
    ax.annotate(row['행정동'], (row['업종점수'], row['20대비중']*100+0.3), fontsize=7, color='#0D47A1')
ax.axvline(0.50, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='업종점수 0.50')
ax.set_title('⑥ 업종점수 × 20대 소비비중 (2024)', fontsize=10, fontweight='bold')
ax.set_xlabel('업종점수'); ax.set_ylabel('20대 소비비중 (%)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{OUT_DIR}/01_consumption_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print('  01_consumption_trend.png 저장')

# ════════════════════════════════════════════════════════════════
# 시각화 02: 20대 vs 30대 소비 패턴 비교
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor('#F8F9FA')
fig.suptitle('20대 vs 30대 소비 패턴 비교 분석', fontsize=14, fontweight='bold')

d24 = anal[anal['연도']==2024].dropna(subset=['20대비중','30대비중'])

# ① 20대비중 vs 30대비중 산점도
ax = axes[0,0]
ax.scatter(d24['20대비중']*100, d24['30대비중']*100, alpha=0.4, s=25, color='#7F8C8D')
# 사분면 구분
med20, med30 = d24['20대비중'].median()*100, d24['30대비중'].median()*100
ax.axvline(med20, color='#1565C0', linestyle='--', alpha=0.6, linewidth=1.5, label=f'20대 중위 {med20:.1f}%')
ax.axhline(med30, color='#3498DB', linestyle='--', alpha=0.6, linewidth=1.5, label=f'30대 중위 {med30:.1f}%')
# 20대 > 중위, 30대 < 중위 = "20대 특화" 동 강조
q1 = d24[(d24['20대비중']*100 > med20) & (d24['30대비중']*100 < med30)]
ax.scatter(q1['20대비중']*100, q1['30대비중']*100, color='#1565C0', s=40, alpha=0.7, label=f'20대 특화 ({len(q1)}개)')
ax.set_title('① 20대비중 vs 30대비중 (2024)', fontsize=10, fontweight='bold')
ax.set_xlabel('20대 소비비중 (%)'); ax.set_ylabel('30대 소비비중 (%)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ② 20대/30대 비율 (20대 선호도) TOP15
ax = axes[0,1]
d24 = d24.copy()
d24['20대_우세도'] = d24['20대비중'] / d24['30대비중'].replace(0, np.nan)
top15_ratio = d24.nlargest(15, '20대_우세도')
ax.barh(range(len(top15_ratio)), top15_ratio['20대_우세도'].values,
        color='#1565C0', alpha=0.85)
ax.axvline(1.0, color='black', linewidth=1, linestyle='--')
ax.set_yticks(range(len(top15_ratio)))
ax.set_yticklabels(top15_ratio['행정동'].values, fontsize=8)
ax.set_title('② 20대/30대 비율 TOP 15\n(1.0 이상 = 20대가 30대보다 많이 소비)', fontsize=10, fontweight='bold')
ax.set_xlabel('20대/30대 소비비중 비율')
ax.grid(alpha=0.3, axis='x')

# ③ 5년간 20대-30대 갭 변화 (상위 10개 동)
ax = axes[0,2]
pivot20 = anal.pivot_table(index='행정동', columns='연도', values='20대비중')
pivot30 = anal.pivot_table(index='행정동', columns='연도', values='30대비중')
gap = (pivot20 - pivot30).dropna()
gap_change = gap[2024] - gap[2020] if 2024 in gap.columns and 2020 in gap.columns else pd.Series()
if len(gap_change):
    top10_gap = gap_change.sort_values(ascending=False).head(10)
    colors_gap = ['#1565C0' if v > 0 else '#3498DB' for v in top10_gap.values]
    ax.barh(range(len(top10_gap)), top10_gap.values*100, color=colors_gap, alpha=0.85)
    ax.set_yticks(range(len(top10_gap)))
    ax.set_yticklabels(top10_gap.index, fontsize=8)
    ax.axvline(0, color='black', linewidth=1)
ax.set_title('③ 20대-30대 갭 변화 TOP10\n(양수=20대 우세 강화)', fontsize=10, fontweight='bold')
ax.set_xlabel('갭 변화량 (%p)')
ax.grid(alpha=0.3, axis='x')

# ④ 구역유형별 20대/30대 비중
ax = axes[1,0]
zone_data = anal[anal['연도']==2024].copy()
# anal에 이미 구역유형이 있으면 그대로, 없으면 prof20에서 조인
if '구역유형' not in zone_data.columns:
    zone_data = zone_data.merge(prof20[['행정동','구역유형']], on='행정동', how='left')
zone_order = ['오피스형','혼합형','주거형','상권형']
zone_mean = zone_data.groupby('구역유형')[['20대비중','30대비중']].mean() * 100
zone_mean = zone_mean.reindex([z for z in zone_order if z in zone_mean.index])
x = np.arange(len(zone_mean))
w = 0.35
ax.bar(x - w/2, zone_mean['20대비중'], w, label='20대', color='#1565C0', alpha=0.85)
ax.bar(x + w/2, zone_mean['30대비중'], w, label='30대', color='#3498DB', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(zone_mean.index)
ax.set_title('④ 구역유형별 20대/30대 소비비중', fontsize=10, fontweight='bold')
ax.set_ylabel('평균 소비비중 (%)')
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

# ⑤ 20대비중 × 이동비율 관계 (2024)
ax = axes[1,1]
# anal already has 20대_이동비율 from its initial merge
merge_for_scatter = anal[anal['연도']==2024].dropna(subset=['20대비중','20대_이동비율'])
ax.scatter(merge_for_scatter['20대_이동비율']*100, merge_for_scatter['20대비중']*100,
           alpha=0.4, s=25, color='#7F8C8D')
if len(merge_for_scatter) > 5:
    r, p = pearsonr(merge_for_scatter['20대_이동비율'], merge_for_scatter['20대비중'])
    ax.set_title(f'⑤ 20대 이동비율 × 소비비중\nr={r:.3f} (p={p:.3f})', fontsize=10, fontweight='bold')
else:
    ax.set_title('⑤ 20대 이동비율 × 소비비중', fontsize=10, fontweight='bold')
ax.set_xlabel('20대 이동비율 (%)'); ax.set_ylabel('20대 소비비중 (%)')
ax.grid(alpha=0.3)

# ⑥ 20대비중 vs 복합점수_분위 (트렌드 상권과 관계)
ax = axes[1,2]
d24v = anal[anal['연도']==2024].dropna(subset=['20대비중','복합점수_분위'])
ax.scatter(d24v['복합점수_분위'], d24v['20대비중']*100, alpha=0.35, s=20, color='#7F8C8D')
if len(d24v) > 5:
    r2, p2 = pearsonr(d24v['복합점수_분위'], d24v['20대비중'])
    ax.set_title(f'⑥ 복합점수_분위 × 20대비중\nr={r2:.3f} (p={p2:.3f})', fontsize=10, fontweight='bold')
else:
    ax.set_title('⑥ 복합점수_분위 × 20대비중', fontsize=10, fontweight='bold')
# 상위 분위 강조
high_tier = d24v[d24v['복합점수_분위'] >= 0.80]
ax.scatter(high_tier['복합점수_분위'], high_tier['20대비중']*100,
           color='#1565C0', s=40, alpha=0.7, label='분위 상위 20%')
ax.set_xlabel('복합점수 분위'); ax.set_ylabel('20대 소비비중 (%)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{OUT_DIR}/02_gen20_vs_30s.png', dpi=150, bbox_inches='tight')
plt.close()
print('  02_gen20_vs_30s.png 저장')

# ════════════════════════════════════════════════════════════════
# 시각화 03: 20대 이동비율 분석
# ════════════════════════════════════════════════════════════════
print('[5/6] 시각화 03~04 생성 중...')
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor('#F8F9FA')
fig.suptitle('20대 이동비율 분석 (KT 생활이동 기반)', fontsize=14, fontweight='bold')

# ① 서울 전체 연도별 20대 이동비율 트렌드
ax = axes[0,0]
yr_mean = mig20_annual.groupby('연도')['20대_이동비율'].mean()
ax.bar(yr_mean.index, yr_mean.values*100, color='#1565C0', alpha=0.85, edgecolor='white')
ax.set_title('① 서울 전체 연도별 20대 이동비율', fontsize=10, fontweight='bold')
ax.set_ylabel('평균 20대 이동비율 (%)')
for yr, v in yr_mean.items():
    ax.text(yr, v*100+0.1, f'{v*100:.1f}%', ha='center', fontsize=9)
ax.grid(alpha=0.3, axis='y')

# ② 2024년 20대 이동비율 TOP20
ax = axes[0,1]
top20_mig = mig20_annual[mig20_annual['연도']==2024].dropna(subset=['행정동']).nlargest(20,'20대_이동비율')
ax.barh(range(len(top20_mig)), top20_mig['20대_이동비율'].values*100,
        color='#1565C0', alpha=0.85, edgecolor='white')
ax.set_yticks(range(len(top20_mig)))
ax.set_yticklabels(top20_mig['행정동'].values, fontsize=7.5)
ax.set_title('② 2024년 20대 이동비율 TOP 20', fontsize=10, fontweight='bold')
ax.set_xlabel('20대 이동비율 (%)'); ax.grid(alpha=0.3, axis='x')

# ③ 20대 이동비율 변화 (2020→2024)
ax = axes[0,2]
mig_change = pd.Series(dtype=float)  # default; overwritten below if data available
pivot_mig = mig20_annual.pivot_table(index='행정동', columns='연도', values='20대_이동비율')
if 2020 in pivot_mig.columns and 2024 in pivot_mig.columns:
    mig_change = (pivot_mig[2024] - pivot_mig[2020]).dropna().sort_values(ascending=False)
    top15mc = mig_change.head(15)
    colors_mc = ['#1565C0' if v > 0 else '#3498DB' for v in top15mc.values]
    ax.barh(range(len(top15mc)), top15mc.values*100, color=colors_mc, alpha=0.85)
    ax.set_yticks(range(len(top15mc)))
    ax.set_yticklabels(top15mc.index, fontsize=8)
    ax.axvline(0, color='black', linewidth=1)
ax.set_title('③ 20대 이동비율 상승폭 TOP 15\n(2020→2024)', fontsize=10, fontweight='bold')
ax.set_xlabel('변화량 (%p)'); ax.grid(alpha=0.3, axis='x')

# ④ 20대 이동비율 × 20대 소비비중 상관 (연도별)
ax = axes[1,0]
colors_yr = plt.cm.Blues(np.linspace(0.3, 0.9, len(yrs)))
for yr, color in zip(yrs, colors_yr):
    # anal already has 20대_이동비율
    d = anal[anal['연도']==yr].dropna(subset=['20대비중','20대_이동비율'])
    if len(d) > 10:
        r, _ = pearsonr(d['20대_이동비율'], d['20대비중'])
        ax.scatter(d['20대_이동비율']*100, d['20대비중']*100,
                   alpha=0.3, s=15, color=color, label=f'{int(yr)} r={r:.2f}')
ax.set_title('④ 20대 이동비율 × 소비비중 (연도별)', fontsize=10, fontweight='bold')
ax.set_xlabel('20대 이동비율 (%)'); ax.set_ylabel('20대 소비비중 (%)')
ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

# ⑤ 20대 이동 증가 vs 소비 증가 동반 여부 (2020→2024)
ax = axes[1,1]
pivot_con = anal.pivot_table(index='행정동', columns='연도', values='20대비중')
if 2020 in pivot_con.columns and 2024 in pivot_con.columns:
    con_change = (pivot_con[2024] - pivot_con[2020]).dropna()
    both = pd.DataFrame({'이동변화': mig_change*100, '소비변화': con_change*100}).dropna()
    ax.scatter(both['이동변화'], both['소비변화'], alpha=0.4, s=25, color='#7F8C8D')
    ax.axhline(0, color='black', linewidth=0.8); ax.axvline(0, color='black', linewidth=0.8)
    # 동반 상승 강조
    both_up = both[(both['이동변화']>0) & (both['소비변화']>0)]
    ax.scatter(both_up['이동변화'], both_up['소비변화'], color='#1565C0', s=35, alpha=0.7,
               label=f'동반상승 {len(both_up)}개')
    if len(both) > 5:
        r3, p3 = pearsonr(both['이동변화'], both['소비변화'])
        ax.set_title(f'⑤ 이동↑ × 소비↑ 동반 여부\nr={r3:.3f} (2020→2024)', fontsize=10, fontweight='bold')
    ax.set_xlabel('20대 이동비율 변화 (%p)'); ax.set_ylabel('20대 소비비중 변화 (%p)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ⑥ 주요 후보 동 20대 이동 시계열
ax = axes[1,2]
candidates = ['청파동','후암동','연희동','망원1동','다산동']
cand_colors = ['#1565C0','#42A5F5','#27AE60','#2980B9','#8E44AD']
for dong, color in zip(candidates, cand_colors):
    d = mig20_annual[mig20_annual['행정동']==dong].sort_values('연도')
    if len(d):
        ax.plot(d['연도'], d['20대_이동비율']*100, 'o-', color=color,
                linewidth=2, markersize=6, label=dong)
ax.set_title('⑥ 차세대 후보 주요 동 20대 이동 시계열', fontsize=10, fontweight='bold')
ax.set_ylabel('20대 이동비율 (%)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{OUT_DIR}/03_migration.png', dpi=150, bbox_inches='tight')
plt.close()
print('  03_migration.png 저장')

# ════════════════════════════════════════════════════════════════
# 시각화 04: 20대 주말집중도 프로파일
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor('#F8F9FA')
fig.suptitle('20대 주말집중도 프로파일 (생활인구 2024.7~9 기반)', fontsize=14, fontweight='bold')

# ① 구역유형 분포
ax = axes[0,0]
zone_cnt = prof20['구역유형'].value_counts().reindex(['오피스형','혼합형','주거형','상권형'])
bar_colors = ['#1565C0','#00838F','#27AE60','#9B59B6']
bars = ax.bar(zone_cnt.index, zone_cnt.values, color=bar_colors, edgecolor='white', linewidth=1.2)
ax.set_title('① 구역유형 분포 (20대 기준)', fontsize=10, fontweight='bold')
ax.set_ylabel('행정동 수')
for bar, v in zip(bars, zone_cnt.values):
    ax.text(bar.get_x()+bar.get_width()/2, v+2, str(v), ha='center', fontsize=10, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# ② 20대주말집중도 vs 전체주말집중도 비교
ax = axes[0,1]
ax.scatter(prof20['주말집중도'], prof20['20대주말집중도'], alpha=0.4, s=20, color='#7F8C8D')
lim_min = min(prof20['주말집중도'].min(), prof20['20대주말집중도'].min()) * 0.95
lim_max = max(prof20['주말집중도'].max(), prof20['20대주말집중도'].max()) * 1.05
ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1, alpha=0.5, label='y=x')
# 20대가 전체보다 높은 동 강조
above = prof20[prof20['20대주말집중도'] > prof20['주말집중도'] + 0.05]
ax.scatter(above['주말집중도'], above['20대주말집중도'], color='#1565C0', s=30, alpha=0.7,
           label=f'20대↑ ({len(above)}개)')
ax.set_title('② 전체 vs 20대 주말집중도', fontsize=10, fontweight='bold')
ax.set_xlabel('전체 주말집중도'); ax.set_ylabel('20대 주말집중도')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ③ 20대주말집중도 상위 동 TOP20
ax = axes[0,2]
top20_wknd = prof20.nlargest(20, '20대주말집중도')
zone_color_map = {'오피스형':'#1565C0','혼합형':'#00838F','주거형':'#27AE60','상권형':'#9B59B6'}
bar_colors_wknd = [zone_color_map.get(z, '#7F8C8D') for z in top20_wknd['구역유형'].fillna('기타')]
ax.barh(range(len(top20_wknd)), top20_wknd['20대주말집중도'].values,
        color=bar_colors_wknd, alpha=0.85, edgecolor='white')
ax.axvline(1.0, color='black', linewidth=1, linestyle='--')
ax.set_yticks(range(len(top20_wknd)))
ax.set_yticklabels(top20_wknd['행정동'].values, fontsize=7.5)
ax.set_title('③ 20대 주말집중도 TOP 20', fontsize=10, fontweight='bold')
ax.set_xlabel('20대 주말집중도')
# 범례
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=z) for z, c in zone_color_map.items()]
ax.legend(handles=legend_elements, fontsize=7.5, loc='lower right')
ax.grid(alpha=0.3, axis='x')

# ④ 주요 동 비교 (차세대 후보 + 레퍼런스)
ax = axes[1,0]
key_dongs = ['청파동','후암동','연희동','이태원1동','합정동','성수2가1동','역삼1동','서초4동']
key_data = prof20[prof20['행정동'].isin(key_dongs)].set_index('행정동')
key_data = key_data.reindex([d for d in key_dongs if d in key_data.index])
x2 = np.arange(len(key_data))
w2 = 0.35
ax.bar(x2 - w2/2, key_data['주말집중도'].values, w2, label='전체', color='#3498DB', alpha=0.7)
ax.bar(x2 + w2/2, key_data['20대주말집중도'].values, w2, label='20대', color='#1565C0', alpha=0.85)
ax.axhline(1.0, color='black', linewidth=1, linestyle='--')
ax.set_xticks(x2); ax.set_xticklabels(key_data.index, rotation=30, ha='right', fontsize=8)
ax.set_title('④ 주요 동 전체/20대 주말집중도 비교', fontsize=10, fontweight='bold')
ax.set_ylabel('주말집중도'); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

# ⑤ 구역유형별 20대 소비비중 (소비×생활인구 교차)
ax = axes[1,1]
# anal already has 구역유형 and 20대주말집중도 from the initial merge
cross = anal[anal['연도']==2024].copy()
zone_box = [cross[cross['구역유형']==z]['20대비중'].dropna()*100 for z in zone_order if z in cross['구역유형'].values]
zone_labels = [z for z in zone_order if z in cross['구역유형'].values]
bp2 = ax.boxplot(zone_box, labels=zone_labels, patch_artist=True,
                 medianprops=dict(color='white', linewidth=2))
for patch, color in zip(bp2['boxes'], bar_colors[:len(zone_labels)]):
    patch.set_facecolor(color)
ax.set_title('⑤ 구역유형별 20대 소비비중 분포', fontsize=10, fontweight='bold')
ax.set_ylabel('20대 소비비중 (%)'); ax.grid(alpha=0.3, axis='y')

# ⑥ 20대주말집중도 히스토그램
ax = axes[1,2]
ax.hist(prof20['주말집중도'], bins=30, alpha=0.5, color='#3498DB', label='전체', density=True)
ax.hist(prof20['20대주말집중도'], bins=30, alpha=0.6, color='#1565C0', label='20대', density=True)
ax.axvline(0.80, color='#1565C0', linestyle=':', linewidth=1.5, alpha=0.8)
ax.axvline(0.95, color='#00838F', linestyle=':', linewidth=1.5, alpha=0.8)
ax.axvline(1.10, color='#27AE60', linestyle=':', linewidth=1.5, alpha=0.8)
ax.set_title('⑥ 주말집중도 분포 (전체 vs 20대)', fontsize=10, fontweight='bold')
ax.set_xlabel('주말집중도'); ax.set_ylabel('밀도'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{OUT_DIR}/04_weekend_profile.png', dpi=150, bbox_inches='tight')
plt.close()
print('  04_weekend_profile.png 저장')

# ════════════════════════════════════════════════════════════════
# 시각화 05: 20대 기준 차세대 후보군
# ════════════════════════════════════════════════════════════════
print('[6/6] 시각화 05~06 생성 중...')

# 20대 임계값 결정
TH_20 = anal[anal['연도']==2024]['20대비중'].quantile(0.70)  # 상위 30% 기준
TH_IND = 0.50
print(f'  20대비중 임계값 (상위30%): {TH_20:.3f}')

_v3 = load_v3()
p35_list = phase35_top15(_v3)
t1_list = tier1_dongs(_v3)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor('#F8F9FA')
fig.suptitle('차세대 후보군 (레퍼런스 4개 · trend_candidates_v3)', fontsize=14, fontweight='bold')

d24_cand = anal[anal['연도'] == 2024].dropna(subset=['20대비중', '업종점수']).copy()

# ① Tier별 동 수
ax = axes[0, 0]
tier_counts = [
    len(tier1_dongs(_v3)),
    len(tier2_dongs(_v3)),
    len(tier3_dongs(_v3)),
]
ax.bar(['Tier1\n교집합', 'Tier2\n업종전환', 'Tier3\n이동선행'], tier_counts,
       color=['#2E7D32', '#1565C0', '#00838F'], alpha=0.88, edgecolor='white')
ax.set_title('① Tier 분류 (28개 동)', fontsize=10, fontweight='bold')
ax.set_ylabel('행정동 수')
for i, v in enumerate(tier_counts):
    ax.text(i, v + 0.3, str(v), ha='center', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

# ② Phase35 코사인 유사도 TOP15
ax = axes[0, 1]
p35_df = _v3[_v3['행정동'].isin(p35_list)].sort_values('코사인유사도', ascending=True)
colors_p35 = ['#2E7D32' if d in t1_list else '#1565C0' for d in p35_df['행정동']]
ax.barh(range(len(p35_df)), p35_df['코사인유사도'], color=colors_p35, alpha=0.88, edgecolor='white')
ax.set_yticks(range(len(p35_df)))
ax.set_yticklabels(p35_df['행정동'], fontsize=8)
ax.set_title('② Phase35 유사도 TOP15\n(★=Tier1)', fontsize=10, fontweight='bold')
ax.set_xlabel('코사인 유사도'); ax.grid(alpha=0.3, axis='x')

# ③ Tier1 vs 레퍼런스 (2024 지표)
ax = axes[0, 2]
cmp_dongs = REFERENCE_DONGS + t1_list
cmp = d24_cand[d24_cand['행정동'].isin(cmp_dongs)].set_index('행정동').reindex(
    [d for d in cmp_dongs if d in d24_cand['행정동'].values])
x = np.arange(len(cmp))
w = 0.35
ax.bar(x - w / 2, cmp['업종점수'], w, label='업종점수', color='#1565C0', alpha=0.85)
ax.bar(x + w / 2, cmp['20대비중'] * 100, w, label='20대비중(%)', color='#00838F', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(cmp.index, rotation=35, ha='right', fontsize=8)
ax.set_title('③ 레퍼런스(4) + Tier1\n2024 업종·20대비중', fontsize=10, fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

# ④ Tier3 이동성장세 (quadrant)
ax = axes[1, 0]
q = pd.read_csv('data/main_data/quadrant_20_result.csv', encoding='utf-8-sig')
t3_q = q[q['행정동'].isin(tier3_dongs(_v3))].nlargest(10, '이동성장세점수')
ax.barh(range(len(t3_q)), t3_q['이동성장세점수'], color='#00838F', alpha=0.88)
ax.set_yticks(range(len(t3_q)))
ax.set_yticklabels(t3_q['행정동'], fontsize=8)
ax.set_title('④ Tier3 이동성장세 TOP10', fontsize=10, fontweight='bold')
ax.set_xlabel('이동성장세 점수'); ax.grid(alpha=0.3, axis='x')

# ⑤ Tier1 이동비율
ax = axes[1, 1]
t1_mig = d24_cand[d24_cand['행정동'].isin(t1_list)].dropna(subset=['20대_이동비율']).sort_values('20대_이동비율')
ax.barh(range(len(t1_mig)), t1_mig['20대_이동비율'] * 100, color='#2E7D32', alpha=0.88)
ax.set_yticks(range(len(t1_mig)))
ax.set_yticklabels(t1_mig['행정동'], fontsize=9)
ax.set_title('⑤ Tier1 — 20대 이동비율', fontsize=10, fontweight='bold')
ax.set_xlabel('20대 이동비율 (%)'); ax.grid(alpha=0.3, axis='x')

# ⑥ 후보 주말집중도 (v3 전체)
ax = axes[1, 2]
cand_all = d24_cand[d24_cand['행정동'].isin(_v3['행정동'])].merge(
    prof20[['행정동', '주말집중도', '구역유형']], on='행정동', how='left')
zone_colors_map = {'오피스형': '#1565C0', '혼합형': '#00838F', '주거형': '#27AE60', '상권형': '#9B59B6'}
for zone, grp_df in cand_all.groupby('구역유형'):
    ax.scatter(grp_df['주말집중도'], grp_df['20대비중'] * 100,
               color=zone_colors_map.get(zone, '#7F8C8D'), s=40, alpha=0.7, label=zone)
for d in t1_list:
    r = cand_all[cand_all['행정동'] == d]
    if len(r):
        ax.scatter(r['주말집중도'], r['20대비중'] * 100, s=120, marker='*', color='#2E7D32', zorder=5)
ax.axvline(0.80, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='오피스 제외(0.80)')
ax.set_title('⑥ v3 후보 — 주말집중도×20대비중', fontsize=10, fontweight='bold')
ax.set_xlabel('주말집중도'); ax.set_ylabel('20대 소비비중 (%)'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{OUT_DIR}/05_candidates.png', dpi=150, bbox_inches='tight')
plt.close()
print('  05_candidates.png 저장')

# ════════════════════════════════════════════════════════════════
# 시각화 06: 레퍼런스 지역 20대 궤적
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor('#F8F9FA')
fig.suptitle('레퍼런스 4개 동 — 20대 소비·이동 궤적 (신규 기준)', fontsize=14, fontweight='bold')

refs_confirmed = REFERENCE_DONGS
top_cands = p35_list[:8]
ref_colors = plt.cm.tab10(np.linspace(0, 0.9, len(refs_confirmed)))

# ① 레퍼런스 20대비중 시계열
ax = axes[0, 0]
for dong, color in zip(refs_confirmed, ref_colors):
    d = grp[grp['행정동']==dong].sort_values('연도')
    if len(d):
        ax.plot(d['연도'], d['20대비중']*100, 'o-', color=color, linewidth=2, markersize=6, label=dong)
ax.axhline(TH_20*100, color='gray', linestyle='--', linewidth=1, alpha=0.8, label=f'임계값 {TH_20*100:.1f}%')
ax.set_title('① 레퍼런스(4) 20대비중 시계열', fontsize=10, fontweight='bold')
ax.set_ylabel('20대 소비비중 (%)'); ax.legend(fontsize=7.5, ncol=2); ax.grid(alpha=0.3)

# ② 확정 레퍼런스 2020→2024 변화 (20대 vs 2030)
ax = axes[0,1]
ref_change = grp[grp['행정동'].isin(refs_confirmed)].pivot_table(index='행정동', columns='연도', values='20대비중')
if 2020 in ref_change.columns and 2024 in ref_change.columns:
    rc = (ref_change[2024] - ref_change[2020]).dropna().sort_values(ascending=False) * 100
    ax.barh(range(len(rc)), rc.values,
            color=['#1565C0' if v>0 else '#3498DB' for v in rc.values], alpha=0.85)
    ax.set_yticks(range(len(rc))); ax.set_yticklabels(rc.index, fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('② 레퍼런스(4) 20대비중 변화\n(2020→2024)', fontsize=10, fontweight='bold')
ax.set_xlabel('변화량 (%p)'); ax.grid(alpha=0.3, axis='x')

# ③ 레퍼런스 vs 차세대 후보 20대비중 비교 (2024)
ax = axes[0,2]
compare_dongs = refs_confirmed + top_cands[:4]
compare_data = grp[(grp['행정동'].isin(compare_dongs)) & (grp['연도']==2024)]
compare_data = compare_data.set_index('행정동').reindex([d for d in compare_dongs if d in compare_data['행정동'].values])
ref_mask = compare_data.index.isin(refs_confirmed)
ax.bar(range(len(compare_data)),
       compare_data['20대비중'].values*100,
       color=['#1565C0' if r else '#2980B9' for r in ref_mask],
       alpha=0.85, edgecolor='white')
ax.set_xticks(range(len(compare_data)))
ax.set_xticklabels(compare_data.index, rotation=30, ha='right', fontsize=8)
ax.axhline(TH_20*100, color='gray', linestyle='--', linewidth=1)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#1565C0', label='레퍼런스'),
                   Patch(color='#2980B9', label='차세대 후보')], fontsize=8)
ax.set_title('③ 레퍼런스 vs 차세대 후보\n20대비중 비교 (2024)', fontsize=10, fontweight='bold')
ax.set_ylabel('20대 소비비중 (%)'); ax.grid(alpha=0.3, axis='y')

# ④ 20대 이동비율 시계열 (레퍼런스)
ax = axes[1,0]
for dong, color in zip(refs_confirmed, ref_colors):
    d = mig20_annual[mig20_annual['행정동']==dong].sort_values('연도')
    if len(d):
        ax.plot(d['연도'], d['20대_이동비율']*100, 'o-', color=color, linewidth=2, markersize=6, label=dong)
ax.set_title('④ 레퍼런스(4) 20대 이동비율 시계열', fontsize=10, fontweight='bold')
ax.set_ylabel('20대 이동비율 (%)'); ax.legend(fontsize=7.5, ncol=2); ax.grid(alpha=0.3)

# ⑤ 레퍼런스 20대주말집중도 vs 소비비중
ax = axes[1,1]
ref_prof = grp[grp['연도']==2024].merge(
    prof20[['행정동','20대주말집중도','구역유형']], on='행정동', how='left')
ref_prof_ref = ref_prof[ref_prof['행정동'].isin(refs_confirmed)]
ref_prof_cand = ref_prof[ref_prof['행정동'].isin(top_cands)]
ax.scatter(ref_prof_ref['20대주말집중도'], ref_prof_ref['20대비중']*100,
           s=80, color='#1565C0', alpha=0.8, label='레퍼런스', zorder=5)
ax.scatter(ref_prof_cand['20대주말집중도'], ref_prof_cand['20대비중']*100,
           s=80, color='#2980B9', alpha=0.8, label='차세대 후보', zorder=5, marker='s')
for _, row in ref_prof_ref.iterrows():
    if pd.notna(row['20대주말집중도']):
        ax.annotate(row['행정동'], (row['20대주말집중도'], row['20대비중']*100+0.2), fontsize=7, color='#0D47A1')
ax.axvline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax.set_title('⑤ 20대 주말집중도 × 소비비중', fontsize=10, fontweight='bold')
ax.set_xlabel('20대 주말집중도'); ax.set_ylabel('20대 소비비중 (%)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ⑥ 선행성 검증: 20대비중(t) → 복합점수_분위(t+1)
ax = axes[1,2]
lag_data = []
for dong in grp['행정동'].unique():
    d = grp[grp['행정동']==dong].sort_values('연도')
    for i in range(len(d)-1):
        lag_data.append({
            '20대비중_t':   d.iloc[i]['20대비중'],
            '분위_t1':      d.iloc[i+1]['복합점수_분위'],
            '분위_t':       d.iloc[i]['복합점수_분위'],
        })
lag_df = pd.DataFrame(lag_data).dropna()
ax.scatter(lag_df['20대비중_t']*100, lag_df['분위_t1'],
           alpha=0.2, s=10, color='#7F8C8D')
if len(lag_df) > 10:
    r_lag, p_lag = pearsonr(lag_df['20대비중_t'], lag_df['분위_t1'])
    r_sim, p_sim = pearsonr(lag_df['20대비중_t'], lag_df['분위_t'])
    ax.set_title(f'⑥ 20대비중(t) → 복합점수분위(t+1)\n선행 r={r_lag:.3f}  동시 r={r_sim:.3f}', fontsize=10, fontweight='bold')
ax.set_xlabel('20대 소비비중 (%t)'); ax.set_ylabel('복합점수 분위 (t+1)'); ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f'{OUT_DIR}/06_reference_trajectory.png', dpi=150, bbox_inches='tight')
plt.close()
print('  06_reference_trajectory.png 저장')

print('\n=== 모든 시각화 완료 ===')
print(f'저장 경로: {OUT_DIR}/')
print('  01_consumption_trend.png')
print('  02_gen20_vs_30s.png')
print('  03_migration.png')
print('  04_weekend_profile.png')
print('  05_candidates.png')
print('  06_reference_trajectory.png')
