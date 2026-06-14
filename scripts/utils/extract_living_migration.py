"""
생활이동 ZIP 파일 자동 압축해제 + CSV 이동 스크립트

사용법:
    python extract_living_migration.py
    python extract_living_migration.py --src "C:/다른/경로"

기본 동작:
    1. --src 폴더(기본: 내 다운로드 폴더)에서 ZIP 파일을 스캔
    2. 각 ZIP 압축 해제
    3. 내부 CSV 파일을 data/living_migration/ 로 이동
    4. 이미 처리된 파일은 건너뜀
"""

import argparse
import zipfile
import shutil
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SRC_DEFAULT = Path.home() / 'Downloads'
DEST = Path('data/living_migration')


def extract_and_move(zip_path: Path, dest: Path, processed: set) -> list[str]:
    if zip_path.name in processed:
        return []

    moved = []
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for member in zf.namelist():
                if not member.lower().endswith('.csv'):
                    continue

                csv_name = Path(member).name
                target = dest / csv_name

                if target.exists():
                    print(f'  [SKIP] 이미 존재: {csv_name}')
                    continue

                # 임시 디렉토리에 추출 후 이동
                tmp = dest / '__tmp__'
                tmp.mkdir(exist_ok=True)
                zf.extract(member, tmp)

                extracted = tmp / member
                shutil.move(str(extracted), str(target))
                moved.append(csv_name)
                print(f'  [OK]   {csv_name}')

            # 임시 폴더 정리
            tmp = dest / '__tmp__'
            if tmp.exists():
                shutil.rmtree(tmp)

    except zipfile.BadZipFile:
        print(f'  [ERR]  ZIP 파일 손상: {zip_path.name}')

    return moved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default=str(SRC_DEFAULT),
                        help=f'ZIP 파일이 있는 폴더 (기본: {SRC_DEFAULT})')
    args = parser.parse_args()

    src = Path(args.src)
    DEST.mkdir(parents=True, exist_ok=True)

    # 이미 이동된 파일 목록 (중복 방지용 로그)
    log_file = DEST / '.processed.txt'
    processed = set()
    if log_file.exists():
        processed = set(log_file.read_text(encoding='utf-8').splitlines())

    all_zips = sorted(src.glob('*.zip'))
    # 생활이동 관련 파일만 필터 (키워드 없으면 전체 처리)
    KEYWORDS = ['이동', 'migration', 'living', 'LOCAL_PEOPLE_DONG', 'LIVING']
    zip_files = [z for z in all_zips
                 if any(k.lower() in z.name.lower() for k in KEYWORDS)]
    if not zip_files:
        zip_files = all_zips  # 매칭 없으면 전체 처리

    print(f'스캔 경로: {src}')
    print(f'저장 경로: {DEST.resolve()}')
    print(f'발견된 ZIP: {len(zip_files)}개\n')

    total_moved = []
    for zp in zip_files:
        print(f'처리 중: {zp.name}')
        moved = extract_and_move(zp, DEST, processed)
        total_moved.extend(moved)
        if moved:
            processed.add(zp.name)

    # 처리 로그 업데이트
    log_file.write_text('\n'.join(sorted(processed)), encoding='utf-8')

    print(f'\n=== 완료 ===')
    print(f'새로 이동한 CSV: {len(total_moved)}개')
    print(f'저장 위치: {DEST.resolve()}')

    if total_moved:
        print('\n이동한 파일 목록:')
        for f in total_moved:
            print(f'  - {f}')

    # 현재 저장 폴더 현황
    csvs = sorted(DEST.glob('*.csv'))
    print(f'\n현재 data/living_migration/ 내 CSV: {len(csvs)}개')


if __name__ == '__main__':
    main()
