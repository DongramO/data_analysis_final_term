"""
생활이동 CSV 파일 인코딩 일괄 변환: EUC-KR → UTF-8-SIG
"""

import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

TARGET_DIR = Path('data/living_migration')


def is_utf8(path: Path) -> bool:
    """파일이 이미 UTF-8인지 확인 (BOM 또는 디코딩 시도)"""
    with open(path, 'rb') as f:
        header = f.read(3)
    if header[:3] == b'\xef\xbb\xbf':  # UTF-8 BOM
        return True
    # BOM 없으면 앞 1024바이트로 UTF-8 디코딩 시도
    with open(path, 'rb') as f:
        chunk = f.read(1024)
    try:
        chunk.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def convert_file(path: Path) -> str:
    """EUC-KR 파일을 UTF-8-SIG로 변환 (스트리밍, 임시파일 경유)"""
    tmp = path.with_suffix('.tmp')
    try:
        with open(path, encoding='euc-kr', errors='replace') as src, \
             open(tmp, 'w', encoding='utf-8-sig') as dst:
            for line in src:
                dst.write(line)
        tmp.replace(path)  # 원본 덮어쓰기
        return 'ok'
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        return f'err: {e}'


def main():
    files = sorted(TARGET_DIR.glob('생활이동_행정동_*.csv'))
    total = len(files)
    print(f'대상 파일: {total}개\n')

    done = skip = error = 0
    for i, f in enumerate(files, 1):
        if is_utf8(f):
            skip += 1
            if i % 100 == 0:
                print(f'[{i}/{total}] 진행 중... (변환 {done} / 스킵 {skip})')
            continue

        result = convert_file(f)
        if result == 'ok':
            done += 1
        else:
            error += 1
            print(f'  [ERR] {f.name}: {result}')

        if i % 50 == 0 or i == total:
            print(f'[{i}/{total}] 진행 중... (변환 {done} / 스킵 {skip} / 오류 {error})')

    print(f'\n=== 완료 ===')
    print(f'변환: {done}개 / 스킵(이미 UTF-8): {skip}개 / 오류: {error}개')


if __name__ == '__main__':
    main()
