# -*- coding: utf-8 -*-
"""
핵심 컬럼 통합 분석 데이터셋 생성
==================================
분석 전반에서 사용한 모든 핵심 지표를 하나로 통합

Output:
  data/master_analysis_long.csv         — 행정동 × 연도 (2020~2024) 시계열 완전판
  data/master_analysis_long_quarterly.csv — 행정동 × 분기 (2020Q1~2024Q4) 시계열 완전판
  data/master_analysis_summary.csv      — 행정동 단위 요약 (2024 스냅샷 + 변화량 + 임대료)
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

data_dir = 'data'
YEARS = [2020, 2021, 2022, 2023, 2024]

# ════════════════════════════════════════════════════════════
# 1. 원본 로드
# ════════════════════════════════════════════════════════════
master  = pd.read_csv(os.path.join(data_dir, '트렌드매출_마스터.csv'),     encoding='utf-8-sig')
rental  = pd.read_csv(os.path.join(data_dir, '임대료_공실률.csv'),   encoding='utf-8-sig')
dongmap = pd.read_csv(os.path.join(data_dir, 'rental_vacancy_dongmap.csv'), encoding='utf-8-sig')

# ════════════════════════════════════════════════════════════
# 2. 행정동별 궤적 지표 계산
#    (momentum, slope, r2 는 master에 없으므로 재계산)
# ════════════════════════════════════════════════════════════
def calc_traj(grp):
    grp = grp.sort_values('연도')
    yrs = grp['연도'].values
    rnk = grp['복합점수_분위'].values
    if len(yrs) < 3:
        return pd.Series({'momentum': np.nan, 'slope': np.nan, 'r2': np.nan})
    sl, ic, r, p, se = sp_stats.linregress(yrs, rnk)
    return pd.Series({'momentum': float(rnk[-1] - rnk[0]),
                      'slope':    float(sl),
                      'r2':       float(r**2)})

traj = (master.groupby('행정동', group_keys=False)
              .apply(calc_traj)
              .reset_index())

# ════════════════════════════════════════════════════════════
# 3. 임대료·공실률 → 행정동 단위로 변환
#    (상권명 → 대표_행정동 via dongmap, 중복 시 평균)
# ════════════════════════════════════════════════════════════
rental_dong = (
    rental.merge(dongmap[['상권명', '대표_행정동']], on='상권명', how='left')
    .groupby(['대표_행정동', '연도'])[['임대료', '공실률', '투자수익률', '소득수익률', '자본수익률']]
    .mean()
    .reset_index()
    .rename(columns={'대표_행정동': '행정동'})
)

# ════════════════════════════════════════════════════════════
# 4. Long format 통합 — 행정동 × 연도
# ════════════════════════════════════════════════════════════
LONG_COLS = ['행정동_코드', '행정동', '연도', '상권티어',
             '복합점수_분위', '복합점수',
             '트렌드_대표_매출', '총_매출', '트렌드매출_비중',
             '업종점수', '금요효과', '주말비중', '야간비중', '2030비중']

long = (
    master[master['연도'].isin(YEARS)][LONG_COLS]
    .merge(traj[['행정동', 'momentum', 'slope', 'r2']], on='행정동', how='left')
    .merge(rental_dong[['행정동', '연도', '임대료', '공실률']],
           on=['행정동', '연도'], how='left')
    .sort_values(['행정동', '연도'])
    .reset_index(drop=True)
)

# 컬럼 순서 정리
long = long[['행정동_코드', '행정동', '연도', '상권티어',
             'momentum', 'slope', 'r2',
             '복합점수_분위', '복합점수',
             '트렌드_대표_매출', '총_매출', '트렌드매출_비중',
             '업종점수', '금요효과', '주말비중', '야간비중', '2030비중',
             '임대료', '공실률']]

long.to_csv(os.path.join(data_dir, 'master_analysis_long.csv'), index=False, encoding='utf-8-sig')
print(f"저장: data/master_analysis_long.csv")
print(f"      {len(long):,}행 × {len(long.columns)}컬럼  ({long['행정동'].nunique()}개 동 × 5년)")

# ════════════════════════════════════════════════════════════
# 4b. 분기 Long format — 행정동 × 분기
# ════════════════════════════════════════════════════════════
quarterly = pd.read_csv(os.path.join(data_dir, 'trend_sales_master_quarterly.csv'), encoding='utf-8-sig')

QLONG_COLS = ['행정동_코드', '행정동', '기준_년분기_코드', '연도', '분기', '연도분기', '상권티어', '상권유형',
              '복합점수_분위', '복합점수',
              '트렌드_대표_매출', '총_매출', '트렌드매출_비중',
              '업종점수', '금요효과', '주말비중', '야간비중', '2030비중']

long_q = (
    quarterly[QLONG_COLS]
    .merge(traj[['행정동', 'momentum', 'slope', 'r2']], on='행정동', how='left')
    .merge(rental_dong[['행정동', '연도', '임대료', '공실률']],
           on=['행정동', '연도'], how='left')
    .sort_values(['행정동_코드', '기준_년분기_코드'])
    .reset_index(drop=True)
)

long_q = long_q[['행정동_코드', '행정동', '기준_년분기_코드', '연도', '분기', '연도분기', '상권티어', '상권유형',
                  'momentum', 'slope', 'r2',
                  '복합점수_분위', '복합점수',
                  '트렌드_대표_매출', '총_매출', '트렌드매출_비중',
                  '업종점수', '금요효과', '주말비중', '야간비중', '2030비중',
                  '임대료', '공실률']]

long_q.to_csv(os.path.join(data_dir, 'master_analysis_long_quarterly.csv'), index=False, encoding='utf-8-sig')
print(f"\n저장: data/master_analysis_long_quarterly.csv")
print(f"      {len(long_q):,}행 × {len(long_q.columns)}컬럼  ({long_q['행정동'].nunique()}개 동 × 20분기)")

# ════════════════════════════════════════════════════════════
# 5. Summary format — 행정동 단위 (1행 = 1개 행정동)
# ════════════════════════════════════════════════════════════
m20 = master[master['연도'] == 2020].set_index('행정동')
m24 = master[master['연도'] == 2024].set_index('행정동')
r20 = rental_dong[rental_dong['연도'] == 2020].set_index('행정동')
r24 = rental_dong[rental_dong['연도'] == 2024].set_index('행정동')

summ = traj.set_index('행정동').copy()

# 행정동_코드 (2024 기준 코드 사용)
summ['행정동_코드'] = m24['행정동_코드']

# 티어 및 지역 분류
summ['상권티어']  = m24['상권티어']
summ['상권유형']  = m24['상권유형']   # 기존/신흥/쇠퇴/일반

# ── 2024 스냅샷 ──────────────────────────────────────────
for col in ['복합점수_분위', '복합점수',
            '트렌드매출_비중', '업종점수',
            '금요효과', '주말비중', '야간비중', '2030비중']:
    summ[f'{col}_2024'] = m24[col]

# ── 2020 스냅샷 ──────────────────────────────────────────
for col in ['복합점수_분위', '트렌드매출_비중', '업종점수', '2030비중']:
    summ[f'{col}_2020'] = m20[col]

# ── 2020→2024 변화량 ────────────────────────────────────
summ['트렌드매출비중_변화'] = summ['트렌드매출_비중_2024'] - summ['트렌드매출_비중_2020']
summ['업종점수_변화']      = summ['업종점수_2024']       - summ['업종점수_2020']
summ['2030비중_변화']      = summ['2030비중_2024']       - summ['2030비중_2020']

# ── 임대료·공실률 ────────────────────────────────────────
for col in ['임대료', '공실률']:
    summ[f'{col}_2020'] = r20[col].reindex(summ.index)
    summ[f'{col}_2024'] = r24[col].reindex(summ.index)

summ['임대료_변화율(%)'] = ((summ['임대료_2024'] / summ['임대료_2020']) - 1) * 100
summ['공실률_변화(pp)']  = summ['공실률_2024'] - summ['공실률_2020']

# ── 컬럼 순서 ────────────────────────────────────────────
COL_ORDER = [
    # 식별자
    '행정동_코드',
    # 분류
    '상권티어', '상권유형',
    # 궤적 지표
    'momentum', 'slope', 'r2',
    # 2024 스냅샷
    '복합점수_분위_2024', '복합점수_2024',
    '트렌드매출_비중_2024', '업종점수_2024',
    '금요효과_2024', '주말비중_2024', '야간비중_2024', '2030비중_2024',
    # 2020 스냅샷
    '복합점수_분위_2020', '트렌드매출_비중_2020', '업종점수_2020', '2030비중_2020',
    # 변화량
    '트렌드매출비중_변화', '업종점수_변화', '2030비중_변화',
    # 임대료·공실률
    '임대료_2020', '임대료_2024', '임대료_변화율(%)',
    '공실률_2020', '공실률_2024', '공실률_변화(pp)',
]

summ = (summ[[c for c in COL_ORDER if c in summ.columns]]
        .reset_index()
        .rename(columns={'행정동': '행정동'})
        .sort_values(['상권티어', '복합점수_분위_2024'], ascending=[True, False])
        .reset_index(drop=True))

summ.to_csv(os.path.join(data_dir, 'master_analysis_summary.csv'), index=False, encoding='utf-8-sig')
print(f"\n저장: data/master_analysis_summary.csv")
print(f"      {len(summ):,}행 × {len(summ.columns)}컬럼")

# ════════════════════════════════════════════════════════════
# 6. 데이터셋 현황 요약 출력
# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  통합 데이터셋 컬럼 가이드 (master_analysis_summary.csv)")
print("="*60)
groups = {
    "[ 분류 ]":       ['상권티어', '상권유형'],
    "[ 궤적 지표 ]":  ['momentum', 'slope', 'r2'],
    "[ 2024 스냅샷 ]": ['복합점수_분위_2024', '복합점수_2024', '트렌드매출_비중_2024',
                        '업종점수_2024', '금요효과_2024', '주말비중_2024',
                        '야간비중_2024', '2030비중_2024'],
    "[ 2020 스냅샷 ]": ['복합점수_분위_2020', '트렌드매출_비중_2020',
                        '업종점수_2020', '2030비중_2020'],
    "[ 변화량 ]":     ['트렌드매출비중_변화', '업종점수_변화', '2030비중_변화'],
    "[ 임대료·공실률 ]": ['임대료_2020', '임대료_2024', '임대료_변화율(%)',
                           '공실률_2020', '공실률_2024', '공실률_변화(pp)'],
}
for grp_name, cols in groups.items():
    present = [c for c in cols if c in summ.columns]
    print(f"\n{grp_name}")
    for c in present:
        n = summ[c].notna().sum()
        coverage = f"{n}/{len(summ)} ({n/len(summ)*100:.0f}%)"
        print(f"  {c:<28} {coverage}")

print("\n" + "="*60)
print("  티어별 임대료 데이터 커버리지")
print("="*60)
tier_order = ['① 레전드','② 라이징스타','③ 급성장 신흥','④ 잠재 후보',
              '⑤ 성숙·정체','⑥ 쇠퇴 주의','⑦ 일반']
for tier in tier_order:
    sub = summ[summ['상권티어'] == tier]
    n_rent = sub['임대료_2024'].notna().sum()
    bar = '█' * n_rent + '░' * (len(sub) - n_rent)
    print(f"  {tier:<12}  {len(sub):3d}개 중 임대료 {n_rent:2d}개  [{bar[:20]}]")

print("\n완료!")
