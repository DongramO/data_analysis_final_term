# -*- coding: utf-8 -*-
"""
네이버 DataLab 검색어 트렌드 수집기
- 기간: 2020-01 ~ 현재 (기존 2022~ 데이터와 병합)
- 키워드 전략: 행정동명 단일어 + 소비 의도 키워드 묶음으로 정량화 개선
- API 제한: 1회 요청당 최대 5개 그룹, 배치 처리
"""

import os
import json
import time
import requests
import pandas as pd
from datetime import datetime

# ── API 인증 정보 ─────────────────────────────────────────────
# 실행 전 환경변수로 설정하거나 아래에 직접 입력
# 터미널: set NAVER_CLIENT_ID=your_id && set NAVER_CLIENT_SECRET=your_secret
CLIENT_ID     = os.environ.get('NAVER_CLIENT_ID',     '8Bx6RLqBs8X32IhrACqU')
CLIENT_SECRET = os.environ.get('NAVER_CLIENT_SECRET', '6iw3Sfw8WM')

API_URL = 'https://openapi.naver.com/v1/datalab/search'

# ── 수집 기간 ─────────────────────────────────────────────────
START_DATE = '2020-01-01'
END_DATE   = '2024-12-31'
TIME_UNIT  = 'month'  # month / week / date

# ── 행정동별 키워드 그룹 정의 ─────────────────────────────────
# 단일 동명만 쓰면 여행/부동산 검색 노이즈가 섞임
# 소비 의도 키워드(맛집/카페/팝업)를 묶어서 하나의 그룹으로 처리
DONG_KEYWORDS = {
    '가락':  ['가락동', '가락 맛집', '가락 카페'],
    '개포':  ['개포동', '개포 맛집', '개포 카페'],
    '금호':  ['금호동', '금호 맛집', '금호 카페', '금호 팝업'],
    '난곡':  ['난곡동', '난곡 맛집', '난곡 카페'],
    '노량진': ['노량진', '노량진 맛집', '노량진 카페', '노량진 팝업'],
    '당산':  ['당산동', '당산 맛집', '당산 카페'],
    '대학로': ['대학로', '대학로 맛집', '대학로 카페', '대학로 팝업'],
    '도봉':  ['도봉동', '도봉 맛집', '도봉 카페'],
    '독립문': ['독립문', '독립문 맛집', '독립문 카페'],
    '둔촌':  ['둔촌동', '둔촌 맛집', '둔촌 카페'],
    '명일':  ['명일동', '명일 맛집', '명일 카페'],
    '방화':  ['방화동', '방화 맛집', '방화 카페'],
    '보라매': ['보라매', '보라매 맛집', '보라매 카페'],
    '신길':  ['신길동', '신길 맛집', '신길 카페'],
    '신림':  ['신림동', '신림 맛집', '신림 카페'],
    '신월':  ['신월동', '신월 맛집', '신월 카페'],
    '연희동': ['연희동', '연희 맛집', '연희 카페', '연희 팝업'],
    '왕십리': ['왕십리', '왕십리 맛집', '왕십리 카페', '왕십리 팝업'],
    '을지로': ['을지로', '을지로 맛집', '을지로 카페', '을지로 팝업', '을지로 힙'],
    '중계':  ['중계동', '중계 맛집', '중계 카페'],
    '청구':  ['청구동', '청구 맛집', '청구 카페'],
    '청량리': ['청량리', '청량리 맛집', '청량리 카페'],
    '화곡':  ['화곡동', '화곡 맛집', '화곡 카페'],
    '회기':  ['회기동', '회기 맛집', '회기 카페'],
    '후암동': ['후암동', '후암 맛집', '후암 카페'],
}


def fetch_trend(keyword_groups: list, start: str, end: str) -> dict:
    """최대 5개 그룹을 API에 요청하고 결과 반환"""
    payload = {
        'startDate': start,
        'endDate':   end,
        'timeUnit':  TIME_UNIT,
        'keywordGroups': [
            {'groupName': g['name'], 'keywords': g['keywords']}
            for g in keyword_groups
        ],
    }
    headers = {
        'X-Naver-Client-Id':     CLIENT_ID,
        'X-Naver-Client-Secret': CLIENT_SECRET,
        'Content-Type':          'application/json',
    }
    resp = requests.post(API_URL, headers=headers, data=json.dumps(payload))
    resp.raise_for_status()
    return resp.json()


def collect_all(dong_keywords: dict, start: str, end: str, batch_size: int = 5) -> pd.DataFrame:
    """전체 행정동을 배치 처리하여 수집"""
    items = list(dong_keywords.items())
    all_rows = []

    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        groups = [{'name': name, 'keywords': kws} for name, kws in batch]

        print(f"요청 중: {[g['name'] for g in groups]}")
        try:
            result = fetch_trend(groups, start, end)
            for series in result.get('results', []):
                dong = series['title']
                for pt in series['data']:
                    all_rows.append({
                        'period': pt['period'],
                        'group':  dong,
                        'ratio':  pt['ratio'],
                    })
        except requests.HTTPError as e:
            print(f"  오류: {e}")

        time.sleep(0.5)  # API 과부하 방지

    return pd.DataFrame(all_rows)


def main():
    if CLIENT_ID == 'YOUR_CLIENT_ID':
        print("ERROR: API 키를 설정하세요.")
        print("  방법 1 - 환경변수:")
        print("    set NAVER_CLIENT_ID=your_id")
        print("    set NAVER_CLIENT_SECRET=your_secret")
        print("  방법 2 - 이 파일 상단 CLIENT_ID / CLIENT_SECRET 직접 입력")
        return

    print(f"수집 기간: {START_DATE} ~ {END_DATE}")
    print(f"행정동 수: {len(DONG_KEYWORDS)}")

    df_new = collect_all(DONG_KEYWORDS, START_DATE, END_DATE)

    if df_new.empty:
        print("수집 결과 없음")
        return

    # 기존 보정 파일과 병합 (2022~)
    existing_path = 'data/naver_trend_행정동보정_20260510.csv'
    if os.path.exists(existing_path):
        df_old = pd.read_csv(existing_path, encoding='utf-8-sig')
        # 새로 수집한 데이터 중 2022 이전만 추가 (중복 방지)
        df_fill = df_new[df_new['period'] < '2022-01-01']
        df_merged = pd.concat([df_fill[['period', 'group', 'ratio']], df_old], ignore_index=True)
        df_merged = df_merged.sort_values(['group', 'period']).reset_index(drop=True)
        print(f"\n기존 데이터와 병합: {df_old.shape} + {df_fill.shape} = {df_merged.shape}")
    else:
        df_merged = df_new

    today = datetime.now().strftime('%Y%m%d')
    out = f'data/naver_trend_행정동_소비키워드_{today}.csv'
    df_merged.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n저장 완료: {out}")
    print(df_merged.groupby('group')['period'].agg(['min', 'max']))


if __name__ == '__main__':
    main()
