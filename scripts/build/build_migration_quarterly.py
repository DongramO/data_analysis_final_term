"""
서울 생활이동 데이터 전처리 스크립트
: 도착 행정동 × 연도 × 분기 × 연령대 기준 이동유형별 집계

실행:
    python build_migration_quarterly.py            # 전체 실행 (멀티프로세싱)
    python build_migration_quarterly.py --test     # 2020.02 한 달만 테스트
    python build_migration_quarterly.py --workers 4  # 워커 수 지정 (기본: CPU 코어 수)
"""

import sys
import io
import argparse
import re
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

DATA_DIR = Path('data/living_migration')
OUTPUT_FULL = Path('data/main_data/분기별_주말유입인구.csv')
OUTPUT_TEST = Path('data/migration_quarterly_test.csv')

# 파일명 패턴: 생활이동_행정동_2020.02_17시.csv
FILE_RE = re.compile(r'생활이동_행정동_(\d{4})\.(\d{2})_(\d{2})시\.csv')

# 읽을 컬럼 인덱스만 지정 (전체 컬럼 파싱 생략 → 속도 향상)
# [0]기준연월 [1]요일 [2]출발시간 [3]출발행정동 [4]도착행정동 [5]성별 [6]연령 [7]이동유형 [8]평균이동시간 [9]이동인구
USE_COLS = [1, 3, 4, 6, 7, 9]  # 요일, 출발, 도착, 연령, 이동유형, 이동인구

DTYPES = {1: str, 3: str, 4: str, 6: str, 7: str, 9: str}


def to_quarter(month: int) -> int:
    return (month - 1) // 3 + 1


def is_trend_time(weekday: str, hour: int) -> bool:
    return (weekday == '금' and hour >= 17) or weekday in ('토', '일')


def process_file(path: Path) -> pd.DataFrame | None:
    m = FILE_RE.search(path.name)
    if not m:
        return None

    year, month, hour = int(m.group(1)), int(m.group(2)), int(m.group(3))
    quarter = to_quarter(month)
    yq = f'{year}Q{quarter}'

    # 인코딩 자동 감지 (UTF-8 BOM 있으면 utf-8-sig, 없으면 euc-kr)
    with open(path, 'rb') as fp:
        enc = 'utf-8-sig' if fp.read(3) == b'\xef\xbb\xbf' else 'euc-kr'

    try:
        df = pd.read_csv(
            path,
            encoding=enc,
            dtype=str,
            usecols=USE_COLS,
            header=0,
            engine='c',
        )
    except Exception:
        return None

    # 컬럼 위치 기반 이름 지정 (usecols 후 순서 유지됨)
    df.columns = ['요일', '출발_행정동', '도착_행정동', '연령', '이동유형', '이동인구']

    # 요일 × 시간 필터
    df = df[df['요일'].isin(['금', '토', '일'])]
    df = df[df['요일'].apply(lambda w: is_trend_time(w, hour))]
    if df.empty:
        return None

    # 동일 행정동 이동 제외
    df = df[df['출발_행정동'] != df['도착_행정동']]

    # 이동유형 필터
    df = df[df['이동유형'].isin(['HE', 'WE', 'EE'])]

    # 이동인구: '*' → 0
    df['이동인구'] = pd.to_numeric(df['이동인구'], errors='coerce').fillna(0)

    # 연령 정리 + 10대 단위 그루핑 (5세 구간 → 10, 20, 30...)
    df['연령'] = pd.to_numeric(df['연령'], errors='coerce')
    df = df.dropna(subset=['연령'])
    df['연령'] = (df['연령'].astype(int) // 10) * 10  # 0,5→0 / 10,15→10 / 20,25→20 ...

    # 서울 행정동만 (도착 기준: 코드가 11로 시작하는 7자리)
    df = df[df['도착_행정동'].str.match(r'^11\d{5}$', na=False)]

    if df.empty:
        return None

    grp = df.groupby(['도착_행정동', '연령', '이동유형'])['이동인구'].sum().reset_index()
    grp.columns = ['도착_행정동코드', '연령대', '이동유형', '이동인구']
    grp['연도'] = year
    grp['분기'] = quarter
    grp['연도분기'] = yq

    return grp


def aggregate(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    all_data = pd.concat(chunks, ignore_index=True)

    pivoted = (
        all_data
        .groupby(['도착_행정동코드', '연도', '분기', '연도분기', '연령대', '이동유형'])['이동인구']
        .sum()
        .unstack('이동유형', fill_value=0)
        .reset_index()
    )

    for t in ['HE', 'WE', 'EE']:
        if t not in pivoted.columns:
            pivoted[t] = 0
    pivoted = pivoted.rename(columns={'HE': 'HE_이동인구', 'WE': 'WE_이동인구', 'EE': 'EE_이동인구'})
    pivoted['합계_이동인구'] = pivoted['HE_이동인구'] + pivoted['WE_이동인구'] + pivoted['EE_이동인구']

    col_order = ['도착_행정동코드', '연도', '분기', '연도분기', '연령대',
                 'HE_이동인구', 'WE_이동인구', 'EE_이동인구', '합계_이동인구']
    return pivoted[col_order].sort_values(['도착_행정동코드', '연도', '분기', '연령대']).reset_index(drop=True)


def _worker(path: Path) -> pd.DataFrame | None:
    """멀티프로세싱 워커 (모듈 최상위에 있어야 pickle 가능)"""
    return process_file(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help='2020.02 한 달만 테스트 실행')
    parser.add_argument('--workers', type=int, default=None, help='병렬 워커 수 (기본: CPU 코어 수)')
    args = parser.parse_args()

    all_files = sorted(DATA_DIR.glob('생활이동_행정동_*.csv'))

    if args.test:
        files = [f for f in all_files if '2020.02' in f.name]
        output = OUTPUT_TEST
        print(f'[테스트 모드] 2020.02 파일 {len(files)}개 처리\n')
    else:
        files = all_files
        output = OUTPUT_FULL
        n_workers = args.workers or cpu_count()
        print(f'[전체 모드] {len(files)}개 파일 / 워커 {n_workers}개\n')

    t0 = time.time()

    if args.test:
        # 테스트는 싱글 프로세스 (디버깅 용이)
        chunks = []
        for i, f in enumerate(files, 1):
            result = process_file(f)
            if result is not None and not result.empty:
                chunks.append(result)
            print(f'  [{i}/{len(files)}] {f.name}')
    else:
        # 전체는 멀티프로세싱
        n_workers = args.workers or cpu_count()
        chunks = []
        done = 0
        with Pool(processes=n_workers) as pool:
            for result in pool.imap_unordered(_worker, files, chunksize=4):
                done += 1
                if result is not None and not result.empty:
                    chunks.append(result)
                if done % 48 == 0 or done == len(files):
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    remaining = (len(files) - done) / rate if rate > 0 else 0
                    print(f'  [{done}/{len(files)}] 경과 {elapsed/60:.1f}분 / 잔여 약 {remaining/60:.1f}분')

    if not chunks:
        print('처리된 데이터 없음')
        return

    print('\n집계 중...')
    final = aggregate(chunks)
    final.to_csv(output, index=False, encoding='utf-8-sig')

    elapsed = time.time() - t0
    print(f'\n=== 완료 ({elapsed/60:.1f}분) ===')
    print(f'저장: {output}')
    print(f'총 행수: {len(final):,}')
    print(f'고유 도착 행정동: {final["도착_행정동코드"].nunique()}개')
    print(f'연도분기 범위: {final["연도분기"].min()} ~ {final["연도분기"].max()}')
    print(f'\n샘플 (5행):')
    print(final.head(5).to_string())


if __name__ == '__main__':
    main()
