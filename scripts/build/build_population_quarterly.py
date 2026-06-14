import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path('data/population')
OUTPUT_FILE = 'data/population_quarterly.csv'

# 컬럼 위치 (파일 구조 고정)
# [0] 기준일ID  [1] 시간대구분  [2] 행정동코드  [3] 한국생활인구
# [4~17]  남성 14개 (0-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59, 60-64, 65-69, 70이상)
# [18~31] 여성 14개 (동일 순서)
MALE_IDX  = list(range(4, 18))
FEMALE_IDX = list(range(18, 32))

# 5세 단위 인덱스 → 10세 연령대 그룹
AGE_MAP = {
    '10대미만': [0],
    '10대':    [1, 2],
    '20대':    [3, 4],
    '30대':    [5, 6],
    '40대':    [7, 8],
    '50대':    [9, 10],
    '60대':    [11, 12],
    '70이상':  [13],
}

def month_to_quarter(month: int) -> int:
    return (month - 1) // 3 + 1


def process_file(file: Path) -> pd.DataFrame:
    """월별 파일 → 행정동별 월합계 DataFrame"""
    year  = int(file.stem[-6:-2])
    month = int(file.stem[-2:])

    df = pd.read_csv(file, encoding='utf-8-sig', header=0,
                     index_col=False, usecols=range(32))

    cols = df.columns.tolist()
    dong_col  = cols[2]
    total_col = cols[3]
    male_cols   = [cols[i] for i in MALE_IDX]
    female_cols = [cols[i] for i in FEMALE_IDX]

    # 수치형 변환
    for c in [total_col] + male_cols + female_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # 연령대 10세 단위 합산 (남+여)
    for age_name, idxs in AGE_MAP.items():
        m = [male_cols[i] for i in idxs]
        f = [female_cols[i] for i in idxs]
        df[age_name] = df[m + f].sum(axis=1)

    df['남성_유동인구'] = df[male_cols].sum(axis=1)
    df['여성_유동인구'] = df[female_cols].sum(axis=1)

    agg_cols = [total_col] + list(AGE_MAP.keys()) + ['남성_유동인구', '여성_유동인구']

    monthly = df.groupby(dong_col)[agg_cols].sum().reset_index()
    monthly.columns = ['행정동코드', '총_유동인구'] + list(AGE_MAP.keys()) + ['남성_유동인구', '여성_유동인구']
    monthly['연도']  = year
    monthly['분기']  = month_to_quarter(month)
    monthly['연도분기'] = f'{year}Q{month_to_quarter(month)}'

    return monthly


def main():
    files = sorted(DATA_DIR.glob('LOCAL_PEOPLE_DONG_*.csv'))
    print(f'처리 대상 파일: {len(files)}개\n')

    chunks = []
    for file in files:
        year  = int(file.stem[-6:-2])
        month = int(file.stem[-2:])
        print(f'  {year}-{month:02d} 처리 중...', end=' ', flush=True)
        m = process_file(file)
        chunks.append(m)
        print(f'{len(m)}개 동 완료')

    print('\n분기별 합산 중...')
    all_monthly = pd.concat(chunks, ignore_index=True)

    agg_sum_cols = ['총_유동인구'] + list(AGE_MAP.keys()) + ['남성_유동인구', '여성_유동인구']
    quarterly = (
        all_monthly
        .groupby(['행정동코드', '연도', '분기', '연도분기'])[agg_sum_cols]
        .sum()
        .reset_index()
    )

    # 파생 지표
    quarterly['2030_유동인구'] = quarterly['20대'] + quarterly['30대']
    quarterly['2030_유동인구_비중'] = (
        quarterly['2030_유동인구'] / quarterly['총_유동인구'].replace(0, np.nan)
    ).round(6)

    # 컬럼 순서 정리
    col_order = [
        '행정동코드', '연도', '분기', '연도분기',
        '총_유동인구', '남성_유동인구', '여성_유동인구',
        '10대미만', '10대', '20대', '30대', '40대', '50대', '60대', '70이상',
        '2030_유동인구', '2030_유동인구_비중',
    ]
    quarterly = quarterly[col_order].sort_values(['행정동코드', '연도', '분기']).reset_index(drop=True)

    quarterly.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print(f'\n=== 완료 ===')
    print(f'저장 경로  : {OUTPUT_FILE}')
    print(f'총 행수    : {len(quarterly):,}')
    print(f'고유 행정동: {quarterly["행정동코드"].nunique()}개')
    print(f'연도분기   : {quarterly["연도분기"].min()} ~ {quarterly["연도분기"].max()}')
    print(f'\n샘플 (첫 3행):')
    print(quarterly.head(3).to_string())


if __name__ == '__main__':
    main()
