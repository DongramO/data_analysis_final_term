import os
import json
import urllib.request
import pandas as pd
from datetime import datetime

CLIENT_ID = "8Bx6RLqBs8X32IhrACqU"
CLIENT_SECRET = "6iw3Sfw8WM"

DATALAB_URL = "https://openapi.naver.com/v1/datalab/search"


def call_datalab(start_date: str, end_date: str, keyword_groups: list,
                 time_unit: str = "month", device: str = "", ages: list = None, gender: str = "") -> dict:
    body = {
        "startDate": start_date,
        "endDate": end_date,
        "timeUnit": time_unit,
        "keywordGroups": keyword_groups,
    }
    if device:
        body["device"] = device
    if ages:
        body["ages"] = ages
    if gender:
        body["gender"] = gender

    request = urllib.request.Request(DATALAB_URL)
    request.add_header("X-Naver-Client-Id", CLIENT_ID)
    request.add_header("X-Naver-Client-Secret", CLIENT_SECRET)
    request.add_header("Content-Type", "application/json")

    response = urllib.request.urlopen(request, data=json.dumps(body).encode("utf-8"))
    if response.getcode() != 200:
        raise RuntimeError(f"API 오류: {response.getcode()}")

    return json.loads(response.read().decode("utf-8"))


def parse_to_dataframe(result: dict) -> pd.DataFrame:
    rows = []
    for group in result.get("results", []):
        group_name = group["title"]
        for point in group["data"]:
            rows.append({
                "group": group_name,
                "period": point["period"],
                "ratio": point["ratio"],
            })
    return pd.DataFrame(rows)


ANCHOR = {"groupName": "서울", "keywords": ["서울"]}

if __name__ == "__main__":
    # 행정동별 상권발달 지수 상위 25개, 4개씩 묶어 서울 앵커 포함 7회 호출
    dong_groups = [
        [
            {"groupName": "화곡",  "keywords": ["화곡동", "화곡역", "화곡시장"]},
            {"groupName": "당산",  "keywords": ["당산동", "당산역", "당산"]},
            {"groupName": "신길",  "keywords": ["신길동", "신길역"]},
            {"groupName": "방화",  "keywords": ["방화동", "방화역"]},
        ],
        [
            {"groupName": "신월",  "keywords": ["신월동", "신월"]},
            {"groupName": "을지로", "keywords": ["을지로", "을지로입구", "을지로3가"]},
            {"groupName": "대학로", "keywords": ["대학로", "혜화동", "혜화역"]},
            {"groupName": "후암동", "keywords": ["후암동", "후암"]},
        ],
        [
            {"groupName": "독립문", "keywords": ["교남동", "독립문", "교남"]},
            {"groupName": "청구",   "keywords": ["청구동", "청구역"]},
            {"groupName": "중계",   "keywords": ["중계동", "중계역", "중계"]},
            {"groupName": "도봉",   "keywords": ["도봉동", "도봉역"]},
        ],
        [
            {"groupName": "청량리", "keywords": ["청량리", "제기동", "청량리역"]},
            {"groupName": "회기",   "keywords": ["회기동", "회기역"]},
            {"groupName": "왕십리", "keywords": ["왕십리", "왕십리역"]},
            {"groupName": "둔촌",   "keywords": ["둔촌동", "둔촌올림픽파크"]},
        ],
        [
            {"groupName": "가락",   "keywords": ["가락동", "가락시장", "가락역"]},
            {"groupName": "명일",   "keywords": ["명일동", "명일역"]},
            {"groupName": "금호",   "keywords": ["금호동", "금호역"]},
            {"groupName": "노량진", "keywords": ["노량진", "노량진역", "노량진수산시장"]},
        ],
        [
            {"groupName": "신림",   "keywords": ["신림", "신림역", "관악"]},
            {"groupName": "보라매", "keywords": ["보라매", "보라매공원", "보라매역"]},
            {"groupName": "개포",   "keywords": ["개포동", "개포역"]},
            {"groupName": "연희동", "keywords": ["연희동", "연희"]},
        ],
        [
            {"groupName": "난곡",   "keywords": ["난향동", "난곡", "난곡역"]},
        ],
    ]

    all_frames = []
    total = len(dong_groups)
    for i, groups in enumerate(dong_groups, 1):
        keyword_groups = [ANCHOR] + groups
        names = [g['groupName'] for g in groups]
        print(f"[{i}/{total}] 호출 중: {names}")
        result = call_datalab(
            start_date="2022-01-01",
            end_date="2024-12-31",
            keyword_groups=keyword_groups,
            time_unit="month",
        )
        frame = parse_to_dataframe(result)
        frame['call_id'] = i
        all_frames.append(frame)

    df_raw = pd.concat(all_frames, ignore_index=True)

    # 앵커 보정: 같은 호출의 서울 ratio로 나누기
    anchor_df = (df_raw[df_raw['group'] == '서울'][['call_id', 'period', 'ratio']]
                 .rename(columns={'ratio': 'anchor_ratio'}))
    df_dong = df_raw[df_raw['group'] != '서울'].copy()
    df = df_dong.merge(anchor_df, on=['call_id', 'period'])
    df['adjusted_ratio'] = (df['ratio'] / df['anchor_ratio'] * 100).round(2)

    pivot = df.pivot(index='period', columns='group', values='adjusted_ratio')
    print("\n=== 행정동별 보정 검색 트렌드 (서울 기준 정규화) ===")
    print(pivot.to_string())

    output_path = f"data/naver_trend_행정동_보정_{datetime.today().strftime('%Y%m%d')}.csv"
    df[['period', 'group', 'ratio', 'anchor_ratio', 'adjusted_ratio']].to_csv(
        output_path, index=False, encoding="utf-8-sig"
    )
    print(f"\n저장 완료: {output_path}")
