# -*- coding: utf-8 -*-
"""
트렌드 상권 Top 15 네이버 DataLab 수집
2020-01 ~ 2024-12, 월별
"""

import requests, json, time, os, sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

CLIENT_ID     = "8Bx6RLqBs8X32IhrACqU"
CLIENT_SECRET = "6iw3Sfw8WM"
URL           = "https://openapi.naver.com/v1/datalab/search"

START_DATE = '2020-01-01'
END_DATE   = '2024-12-31'
TIME_UNIT  = 'month'

# 트렌드 상권 Top 15 — 소비 의도 키워드 그룹
TREND_GROUPS = [
    # Batch 1
    {"groupName": "연남동",
     "keywords": ["연남동", "연남동 맛집", "연남동 카페", "연남동 술집", "연남동 팝업"]},
    {"groupName": "이태원",
     "keywords": ["이태원", "이태원 맛집", "이태원 카페", "이태원 클럽", "경리단길"]},
    {"groupName": "홍대",
     "keywords": ["홍대", "홍대 맛집", "홍대 카페", "홍대 클럽", "홍대입구"]},
    {"groupName": "삼청동",
     "keywords": ["삼청동", "삼청동 카페", "삼청동 맛집", "북촌", "안국역 카페"]},
    {"groupName": "화양동",
     "keywords": ["건대", "건대 맛집", "건대 카페", "건대 클럽", "화양동"]},
    # Batch 2
    {"groupName": "서촌",
     "keywords": ["서촌", "서촌 카페", "서촌 맛집", "경복궁역 카페", "통인시장"]},
    {"groupName": "한남동",
     "keywords": ["한남동", "한남동 카페", "한남동 맛집", "리움미술관", "한남대로"]},
    {"groupName": "북촌",
     "keywords": ["북촌", "가회동", "북촌 한옥마을", "계동 카페", "안국 카페"]},
    {"groupName": "합정망원",
     "keywords": ["합정", "망원", "망원동 맛집", "합정 카페", "망원 카페"]},
    {"groupName": "대학로",
     "keywords": ["대학로", "혜화", "대학로 공연", "혜화역 카페", "대학로 맛집"]},
    # Batch 3
    {"groupName": "잠실",
     "keywords": ["잠실", "잠실 맛집", "잠실 카페", "롯데월드몰", "잠실 팝업"]},
    {"groupName": "낙성대",
     "keywords": ["낙성대", "서울대입구", "낙성대 카페", "서울대 맛집", "낙성대역"]},
    {"groupName": "방배",
     "keywords": ["방배동", "방배 카페골목", "서래마을", "방배카페골목", "방배역"]},
    {"groupName": "상봉",
     "keywords": ["상봉", "상봉 맛집", "상봉 술집", "중랑 맛집", "상봉역"]},
    {"groupName": "용산해방촌",
     "keywords": ["해방촌", "용산 카페", "후커힐", "해방촌 맛집", "용산 술집"]},
]


def fetch_batch(groups):
    body = {
        "startDate": START_DATE,
        "endDate":   END_DATE,
        "timeUnit":  TIME_UNIT,
        "keywordGroups": groups,
    }
    headers = {
        "X-Naver-Client-Id":     CLIENT_ID,
        "X-Naver-Client-Secret": CLIENT_SECRET,
        "Content-Type":          "application/json",
    }
    resp = requests.post(URL, headers=headers, data=json.dumps(body))
    resp.raise_for_status()
    return resp.json()


rows = []
for i in range(0, len(TREND_GROUPS), 5):
    batch = TREND_GROUPS[i:i+5]
    names = [g['groupName'] for g in batch]
    print(f"수집 중: {names}")
    data = fetch_batch(batch)
    for item in data['results']:
        grp = item['title']
        for pt in item['data']:
            rows.append({'period': pt['period'], 'group': grp, 'ratio': pt['ratio']})
    time.sleep(0.6)

df_new = pd.DataFrame(rows)
print(f"\n수집 완료: {df_new.shape}")
print(df_new.groupby('group')['period'].agg(['min','max']))

out_path = os.path.join('data', '네이버트렌드_트렌드상권_20260517.csv')
df_new.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"\n저장: {out_path}")
