# -*- coding: utf-8 -*-
"""
패턴매칭 상위 후보군 네이버 DataLab 수집 (월별, 2020-01 ~ 2024-12)
레퍼런스 상권(성수·문래) 포함 → 급등 시점 비교 분석용
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

# 레퍼런스 상권 (비교 기준)
REFERENCE_GROUPS = [
    {"groupName": "성수",
     "keywords": ["성수동", "성수동 카페", "성수 팝업", "성수 맛집", "성수동 팝업스토어"]},
    {"groupName": "문래",
     "keywords": ["문래동", "문래창작촌", "문래동 카페", "문래동 맛집", "문래 술집"]},
    {"groupName": "신대방",
     "keywords": ["신대방동", "신대방 카페", "보라매 카페", "신대방 맛집", "보라매공원 카페"]},
    {"groupName": "여의",
     "keywords": ["여의도", "여의도 카페", "여의도 맛집", "더현대서울", "여의도 팝업"]},
    {"groupName": "동화",
     "keywords": ["동화동", "동작 카페", "동작구 카페", "동화동 맛집", "노들역 카페"]},
]

# 후보군 (패턴 유사도 순)
CANDIDATE_GROUPS = [
    {"groupName": "역삼2동",
     "keywords": ["역삼2동", "역삼 카페", "강남 역삼 맛집", "역삼역 카페", "역삼동 술집"]},
    {"groupName": "문래동",
     "keywords": ["문래동", "문래창작촌", "문래동 카페", "문래동 맛집", "문래 술집"]},
    {"groupName": "서초동",
     "keywords": ["서초동", "서초 카페", "서초 맛집", "서초역 카페", "교대역 카페"]},
    {"groupName": "장안동",
     "keywords": ["장안동", "장안동 카페", "장안동 맛집", "장한평 카페", "장안동 술집"]},
    {"groupName": "아현동",
     "keywords": ["아현동", "아현 카페", "공덕 아현", "아현동 맛집", "이대입구 카페"]},
    {"groupName": "효창동",
     "keywords": ["효창동", "효창공원 카페", "숙명여대 카페", "효창동 맛집", "용산 효창"]},
    {"groupName": "교남동",
     "keywords": ["교남동", "독립문 카페", "교남동 맛집", "무악재 카페", "독립문역 맛집"]},
    {"groupName": "도곡동",
     "keywords": ["도곡동", "도곡 카페", "도곡역 카페", "도곡동 맛집", "한티역 카페"]},
    {"groupName": "양재동",
     "keywords": ["양재2동", "양재동 카페", "양재역 카페", "양재동 맛집", "말죽거리 카페"]},
    {"groupName": "풍납동",
     "keywords": ["풍납동", "풍납 카페", "풍납동 맛집", "천호 풍납", "풍납토성 카페"]},
]

ALL_GROUPS = REFERENCE_GROUPS + CANDIDATE_GROUPS


def fetch_batch(groups):
    body = {
        "startDate":    START_DATE,
        "endDate":      END_DATE,
        "timeUnit":     TIME_UNIT,
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
errors = []

for i in range(0, len(ALL_GROUPS), 5):
    batch = ALL_GROUPS[i:i+5]
    names = [g['groupName'] for g in batch]
    print(f"수집 중 [{i//5+1}/{(len(ALL_GROUPS)+4)//5}]: {names}")
    try:
        data = fetch_batch(batch)
        for item in data['results']:
            grp = item['title']
            for pt in item['data']:
                rows.append({'period': pt['period'], 'group': grp,
                             'ratio': pt['ratio'],
                             'type': 'reference' if any(g['groupName']==grp for g in REFERENCE_GROUPS) else 'candidate'})
        print(f"  → {len(data['results'])}개 그룹 수집 완료")
    except Exception as e:
        print(f"  → 오류: {e}")
        errors.append({'batch': names, 'error': str(e)})
    time.sleep(0.6)

df = pd.DataFrame(rows)
print(f"\n수집 완료: {df.shape[0]}행, {df['group'].nunique()}개 그룹")
print(df.groupby('group')['period'].agg(['min','max','count']))

if errors:
    print(f"\n오류 발생: {len(errors)}건")
    for e in errors:
        print(f"  {e}")

out_path = os.path.join('data', 'main_data', 'naver_trend_candidates.csv')
df.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"\n저장 완료: {out_path}")
