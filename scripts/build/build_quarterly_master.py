# -*- coding: utf-8 -*-
"""
분기별 트렌드 대표 매출 마스터 파일 생성
=========================================
build_trend_master.py 의 분기 버전
  연간 집계(year) 대신 분기(기준_년분기_코드) 단위로 집계
  복합점수_분위도 분기 내 표준화

Output:
  data/trend_sales_master_quarterly.csv
  — 행정동 × 분기 (423개 동 × 20분기 = 8,460행)
"""

import pandas as pd
import numpy as np
from scipy import stats as sp_stats
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

data_dir = 'data'

CATEG_MAP = {
    'F&B (카페/디저트)': ['커피-음료', '제과점'],
    'F&B (식사)':       ['한식음식점', '일식음식점', '양식음식점', '중식음식점',
                         '분식전문점', '패스트푸드점', '치킨전문점'],
    '주류/유흥':        ['호프-간이주점', '노래방'],
    '라이프스타일':     ['일반의류', '편의점', '애완동물', '화초', '인테리어'],
}
TREND_CATEGS = list(CATEG_MAP.keys())

DAY_COLS  = ['월요일_매출_금액','화요일_매출_금액','수요일_매출_금액',
             '목요일_매출_금액','금요일_매출_금액','토요일_매출_금액','일요일_매출_금액']
TIME_COLS = ['시간대_00~06_매출_금액','시간대_06~11_매출_금액','시간대_11~14_매출_금액',
             '시간대_14~17_매출_금액','시간대_17~21_매출_금액','시간대_21~24_매출_금액']

# ── 1. 로드 ────────────────────────────────────────────────────
print("데이터 로드 중...")
df = pd.read_csv(
    os.path.join(data_dir, '서울시_상권분석서비스(추정매출-행정동)_2020_2024.csv'),
    encoding='utf-8-sig',
)

# 연도·분기 파생
df['연도']  = df['기준_년분기_코드'] // 10
df['분기']  = df['기준_년분기_코드'] %  10
df['연도분기'] = df['연도'].astype(str) + 'Q' + df['분기'].astype(str)  # 예: "2020Q1"

df['category'] = df['서비스_업종_코드_명'].map(
    lambda x: next((c for c, items in CATEG_MAP.items() if x in items), '기타')
)

# ── 2. 행정동 × 분기 집계 ─────────────────────────────────────
GRP = ['행정동_코드', '행정동_코드_명', '기준_년분기_코드', '연도', '분기', '연도분기']

agg_dict = {c: 'sum' for c in DAY_COLS + TIME_COLS}
agg_dict.update({'당월_매출_금액': 'sum',
                 '연령대_20_매출_금액': 'sum',
                 '연령대_30_매출_금액': 'sum'})

base = df.groupby(GRP).agg(agg_dict).reset_index()

# 업종점수용 트렌드 업종 매출 합산
categ_agg = (df.groupby(GRP + ['category'])['당월_매출_금액']
               .sum().reset_index())
trend_sum = (categ_agg[categ_agg['category'].isin(TREND_CATEGS)]
             .groupby(GRP)['당월_매출_금액'].sum().reset_index()
             .rename(columns={'당월_매출_금액': '트렌드업종_매출'}))
base = base.merge(trend_sum, on=GRP, how='left')

# ── 3. 핵심 지표 계산 ─────────────────────────────────────────
day_total  = base[DAY_COLS].sum(axis=1)
time_total = base[TIME_COLS].sum(axis=1)

base['야간비중']        = (base['시간대_17~21_매출_금액'] + base['시간대_21~24_매출_금액']) / time_total
base['트렌드_대표_매출'] = (base['금요일_매출_금액'] * base['야간비중']
                           + base['토요일_매출_금액']
                           + base['일요일_매출_금액'])
base['총_매출']         = base['당월_매출_금액']
base['트렌드매출_비중']  = base['트렌드_대표_매출'] / base['총_매출']
base['업종점수']        = base['트렌드업종_매출'] / base['총_매출']
base['2030비중']        = (base['연령대_20_매출_금액'] + base['연령대_30_매출_금액']) / base['총_매출']
base['금요효과']        = base['금요일_매출_금액'] / base['목요일_매출_금액']
base['주말비중']        = (base['토요일_매출_금액'] + base['일요일_매출_금액']) / day_total

# ── 4. 분기 내 복합점수 표준화 (연간과 동일 방법, 단위만 분기) ──
def zscore(s):
    std = s.std()
    return (s - s.mean()) / std if std > 0 else s * 0

records = []
for qcode, gdf in base.groupby('기준_년분기_코드'):
    tmp = gdf.copy()
    tmp['temporal_score'] = (zscore(tmp['금요효과']) +
                              zscore(tmp['주말비중']) +
                              zscore(tmp['야간비중']))
    tmp['복합점수']      = zscore(tmp['업종점수']) + zscore(tmp['temporal_score'])
    tmp['복합점수_분위'] = tmp['복합점수'].rank(pct=True)
    records.append(tmp)

base = pd.concat(records, ignore_index=True)

# ── 5. 상권티어 연결 (연간 master 기준 2024년 티어 사용) ──────
annual_tier = (pd.read_csv(os.path.join(data_dir, '트렌드매출_마스터.csv'),
                            encoding='utf-8-sig')
               .query('연도 == 2024')[['행정동', '상권티어', '상권유형']]
               .drop_duplicates('행정동'))

base = base.merge(annual_tier, left_on='행정동_코드_명', right_on='행정동', how='left')

# ── 6. 저장 ────────────────────────────────────────────────────
SAVE_COLS = ['행정동_코드', '행정동_코드_명', '기준_년분기_코드', '연도', '분기', '연도분기',
             '상권티어', '상권유형',
             '트렌드_대표_매출', '총_매출', '트렌드매출_비중',
             '업종점수', '금요효과', '주말비중', '야간비중', '2030비중',
             '복합점수', '복합점수_분위']

out = base[SAVE_COLS].copy()
out.columns = ['행정동_코드', '행정동', '기준_년분기_코드', '연도', '분기', '연도분기',
               '상권티어', '상권유형',
               '트렌드_대표_매출', '총_매출', '트렌드매출_비중',
               '업종점수', '금요효과', '주말비중', '야간비중', '2030비중',
               '복합점수', '복합점수_분위']

out = out.sort_values(['행정동', '기준_년분기_코드']).reset_index(drop=True)
out.to_csv(os.path.join(data_dir, 'trend_sales_master_quarterly.csv'),
           index=False, encoding='utf-8-sig')

print(f"저장: data/trend_sales_master_quarterly.csv")
print(f"      {len(out):,}행 × {len(out.columns)}컬럼")
print(f"      행정동 {out['행정동'].nunique()}개 × 분기 {out['기준_년분기_코드'].nunique()}개")

# 분기 커버리지 확인
q_counts = out.groupby('기준_년분기_코드')['행정동'].count()
print("\n=== 분기별 행정동 수 ===")
for qcode, n in q_counts.items():
    yr, q = qcode // 10, qcode % 10
    print(f"  {yr}Q{q}  {n}개 동")

print("\n완료!")
