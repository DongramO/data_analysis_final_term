# -*- coding: utf-8 -*-
"""
모델링용 Raw Feature 데이터셋 생성
====================================
파생 지표(복합점수, 업종점수 등)를 사용하지 않고
원본 수치를 최대한 보존한 feature matrix 생성

단위: 행정동 × 분기  (~8,460행)

컬럼 구성:
  [식별자]    행정동_코드, 행정동, 기준_년분기_코드, 연도, 분기, 연도분기
  [레이블]    상권티어, 상권유형  ← 마스터에서 join
  [업종카테고리] FB카페/FB식사/주류유흥/라이프스타일/기타 × (매출금액, 매출건수)
  [요일별]    월~일 × (매출금액, 매출건수)
  [시간대별]  6구간 × (매출금액, 매출건수)
  [성별]      남성/여성 × (매출금액, 매출건수)
  [연령대별]  10~60이상 × (매출금액, 매출건수)
  [주중/주말] (매출금액, 매출건수)
  [합계]      총_매출금액, 총_매출건수

Output:
  data/분기별_원본피처.csv
"""

import pandas as pd
import numpy as np
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

data_dir = 'data'

CATEG_MAP = {
    'FB카페':     ['커피-음료', '제과점'],
    'FB식사':     ['한식음식점', '일식음식점', '양식음식점', '중식음식점',
                   '분식전문점', '패스트푸드점', '치킨전문점'],
    '주류유흥':   ['호프-간이주점', '노래방'],
    '라이프스타일': ['일반의류', '편의점', '애완동물', '화초', '인테리어'],
}

DAY_COLS_AMT  = ['월요일_매출_금액','화요일_매출_금액','수요일_매출_금액',
                 '목요일_매출_금액','금요일_매출_금액','토요일_매출_금액','일요일_매출_금액']
DAY_COLS_CNT  = ['월요일_매출_건수','화요일_매출_건수','수요일_매출_건수',
                 '목요일_매출_건수','금요일_매출_건수','토요일_매출_건수','일요일_매출_건수']

TIME_COLS_AMT = ['시간대_00~06_매출_금액','시간대_06~11_매출_금액','시간대_11~14_매출_금액',
                 '시간대_14~17_매출_금액','시간대_17~21_매출_금액','시간대_21~24_매출_금액']
TIME_COLS_CNT = ['시간대_건수~06_매출_건수','시간대_건수~11_매출_건수','시간대_건수~14_매출_건수',
                 '시간대_건수~17_매출_건수','시간대_건수~21_매출_건수','시간대_건수~24_매출_건수']

AGE_COLS_AMT  = ['연령대_10_매출_금액','연령대_20_매출_금액','연령대_30_매출_금액',
                 '연령대_40_매출_금액','연령대_50_매출_금액','연령대_60_이상_매출_금액']
AGE_COLS_CNT  = ['연령대_10_매출_건수','연령대_20_매출_건수','연령대_30_매출_건수',
                 '연령대_40_매출_건수','연령대_50_매출_건수','연령대_60_이상_매출_건수']

# ── 1. 로드 ────────────────────────────────────────────────
print("원본 데이터 로드 중...")
df = pd.read_csv(
    os.path.join(data_dir, '서울시_상권분석서비스(추정매출-행정동)_2020_2024.csv'),
    encoding='utf-8-sig',
)

df['연도']    = df['기준_년분기_코드'] // 10
df['분기']    = df['기준_년분기_코드'] %  10
df['연도분기'] = df['연도'].astype(str) + 'Q' + df['분기'].astype(str)

df['카테고리'] = df['서비스_업종_코드_명'].map(
    lambda x: next((c for c, items in CATEG_MAP.items() if x in items), '기타')
)

GRP = ['행정동_코드', '행정동_코드_명', '기준_년분기_코드', '연도', '분기', '연도분기']

# ── 2. 공통 집계 (요일/시간대/연령대/성별/주중주말) ──────────
print("공통 지표 집계 중...")
SUM_COLS = (DAY_COLS_AMT + DAY_COLS_CNT +
            TIME_COLS_AMT + TIME_COLS_CNT +
            AGE_COLS_AMT  + AGE_COLS_CNT +
            ['남성_매출_금액','여성_매출_금액','남성_매출_건수','여성_매출_건수',
             '주중_매출_금액','주말_매출_금액','주중_매출_건수','주말_매출_건수',
             '당월_매출_금액','당월_매출_건수'])

base = df.groupby(GRP)[SUM_COLS].sum().reset_index()

# ── 3. 업종 카테고리별 매출금액 + 매출건수 ────────────────
print("업종 카테고리 집계 중...")
categ_agg = (
    df.groupby(GRP + ['카테고리'])[['당월_매출_금액','당월_매출_건수']]
    .sum()
    .reset_index()
)

categ_wide = categ_agg.pivot_table(
    index=GRP,
    columns='카테고리',
    values=['당월_매출_금액','당월_매출_건수'],
    aggfunc='sum',
    fill_value=0,
).reset_index()

# 멀티인덱스 컬럼 → 단일 컬럼명 정리
# pivot_table: level0=metric(당월_매출_금액), level1=카테고리(FB카페 등)
# GRP 컬럼은 level0=컬럼명, level1='' 형태
categ_wide.columns = [
    f'{b}_매출금액' if a == '당월_매출_금액' else
    f'{b}_매출건수' if a == '당월_매출_건수' else
    a  # GRP 식별자 컬럼
    for a, b in categ_wide.columns
]

base = base.merge(categ_wide, on=GRP, how='left')

# ── 4. 시간대 건수 컬럼명 정리 (원본이 비직관적) ──────────
time_cnt_rename = {
    '시간대_건수~06_매출_건수': '시간대_00~06_매출_건수',
    '시간대_건수~11_매출_건수': '시간대_06~11_매출_건수',
    '시간대_건수~14_매출_건수': '시간대_11~14_매출_건수',
    '시간대_건수~17_매출_건수': '시간대_14~17_매출_건수',
    '시간대_건수~21_매출_건수': '시간대_17~21_매출_건수',
    '시간대_건수~24_매출_건수': '시간대_21~24_매출_건수',
}
base = base.rename(columns=time_cnt_rename)

# ── 5. 상권티어·상권유형 join (분기 마스터에서) ───────────
print("티어 레이블 join 중...")
tier = (pd.read_csv(os.path.join(data_dir, 'trend_sales_master_quarterly.csv'),
                    encoding='utf-8-sig')
        [['행정동_코드', '기준_년분기_코드', '상권티어', '상권유형']])

base = base.merge(tier, on=['행정동_코드', '기준_년분기_코드'], how='left')

# ── 6. 컬럼 순서 정리 및 이름 통일 ──────────────────────
base = base.rename(columns={
    '행정동_코드_명': '행정동',
    '당월_매출_금액': '총_매출금액',
    '당월_매출_건수': '총_매출건수',
})

CATEG_NAMES = ['FB카페', 'FB식사', '주류유흥', '라이프스타일', '기타']
CATEG_AMT = [f'{c}_매출금액' for c in CATEG_NAMES]
CATEG_CNT = [f'{c}_매출건수' for c in CATEG_NAMES]

DAY_CNT_CLEAN = [c.replace('금액', '건수') for c in DAY_COLS_AMT]
TIME_CNT_CLEAN = [c.replace('금액', '건수') for c in TIME_COLS_AMT]
AGE_CNT_CLEAN  = [c.replace('금액', '건수') for c in AGE_COLS_AMT]

COL_ORDER = (
    # 식별자
    ['행정동_코드', '행정동', '기준_년분기_코드', '연도', '분기', '연도분기',
     '상권티어', '상권유형'] +
    # 업종 카테고리
    CATEG_AMT + CATEG_CNT +
    # 합계
    ['총_매출금액', '총_매출건수'] +
    # 요일별
    DAY_COLS_AMT + DAY_CNT_CLEAN +
    # 시간대별
    TIME_COLS_AMT + [c.replace('금액', '건수') for c in TIME_COLS_AMT] +
    # 성별
    ['남성_매출_금액', '여성_매출_금액', '남성_매출_건수', '여성_매출_건수'] +
    # 연령대별
    AGE_COLS_AMT + AGE_COLS_CNT +
    # 주중/주말
    ['주중_매출_금액', '주말_매출_금액', '주중_매출_건수', '주말_매출_건수']
)

present = [c for c in COL_ORDER if c in base.columns]
out = base[present].sort_values(['행정동_코드', '기준_년분기_코드']).reset_index(drop=True)

# ── 7. 저장 ────────────────────────────────────────────────
out_path = os.path.join(data_dir, '분기별_원본피처.csv')
out.to_csv(out_path, index=False, encoding='utf-8-sig')

print(f"\n저장: data/분기별_원본피처.csv")
print(f"      {len(out):,}행 × {len(out.columns)}컬럼")
print(f"      행정동 {out['행정동'].nunique()}개 × 분기 {out['기준_년분기_코드'].nunique()}개")

print("\n=== 컬럼 그룹별 현황 ===")
groups = {
    '식별자·레이블': ['행정동_코드','행정동','기준_년분기_코드','연도','분기','연도분기','상권티어','상권유형'],
    '업종카테고리(금액)': CATEG_AMT,
    '업종카테고리(건수)': CATEG_CNT,
    '요일별(금액)': DAY_COLS_AMT,
    '시간대별(금액)': TIME_COLS_AMT,
    '연령대별(금액)': AGE_COLS_AMT,
    '성별': ['남성_매출_금액','여성_매출_금액'],
    '주중/주말': ['주중_매출_금액','주말_매출_금액'],
}
for name, cols in groups.items():
    present_cols = [c for c in cols if c in out.columns]
    print(f"  {name}: {len(present_cols)}개")

print("\n완료!")
