"""
블로그 언급량 2020·2021 추가 수집
- 기존 data/main_data/블로그트렌드_상권.csv (2022~2025) 유지
- 2020, 2021 컬럼만 추가로 API 호출
- 성장률 지표 재계산 (2020→2025 기준)
"""
import urllib.request
import urllib.parse
import json
import time
import pandas as pd

CLIENT_ID     = "8Bx6RLqBs8X32IhrACqU"
CLIENT_SECRET = "6iw3Sfw8WM"

EXISTING_PATH = "data/main_data/블로그트렌드_상권.csv"
ADD_YEARS     = ["2020", "2021"]
KEYWORD       = "카페"


def search_blog(query: str) -> int:
    encoded = urllib.parse.quote(query)
    url = f"https://openapi.naver.com/v1/search/blog?query={encoded}&display=1&sort=sim"
    req = urllib.request.Request(url)
    req.add_header("X-Naver-Client-Id", CLIENT_ID)
    req.add_header("X-Naver-Client-Secret", CLIENT_SECRET)
    try:
        res = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
        return res.get("total", 0)
    except Exception as e:
        print(f"  [오류] {e}")
        return 0


# ── 기존 데이터 로드 ──────────────────────────────────────────
df = pd.read_csv(EXISTING_PATH, encoding="utf-8-sig")
areas = df["area"].tolist()
print(f"기존 데이터: {len(areas)}개 상권, {list(df.columns)}")

# ── 2020·2021 수집 ────────────────────────────────────────────
total_calls = len(areas) * len(ADD_YEARS)
done = 0

print(f"\n총 {total_calls}회 API 호출 시작 ({', '.join(ADD_YEARS)})...\n")

new_cols = {yr: [] for yr in ADD_YEARS}

for area in areas:
    for year in ADD_YEARS:
        count = search_blog(f"{area} {KEYWORD} {year}")
        new_cols[year].append(count)
        done += 1
        print(f"  [{done:3d}/{total_calls}] {area} {year}: {count:,}")
        time.sleep(0.15)

# ── 기존 데이터에 컬럼 추가 ──────────────────────────────────
for yr in ADD_YEARS:
    df[yr] = new_cols[yr]

# ── 컬럼 순서 재정렬 ─────────────────────────────────────────
year_cols = ["2020", "2021", "2022", "2023", "2024", "2025"]
year_cols_avail = [y for y in year_cols if y in df.columns]

# 성장률 재계산
df["growth_20_25"] = ((df["2025"] - df["2020"]) / (df["2020"] + 1) * 100).round(1)
df["growth_22_25"] = ((df["2025"] - df["2022"]) / (df["2022"] + 1) * 100).round(1)
df["growth_23_25"] = ((df["2025"] - df["2023"]) / (df["2023"] + 1) * 100).round(1)
df["total_vol"]    = df[year_cols_avail].sum(axis=1)

# 컬럼 순서 정리
stat_cols = ["growth_20_25", "growth_22_25", "growth_23_25", "total_vol"]
df = df[["area"] + year_cols_avail + stat_cols]
df = df.sort_values("growth_20_25", ascending=False).reset_index(drop=True)

# ── 저장 ──────────────────────────────────────────────────────
df.to_csv(EXISTING_PATH, index=False, encoding="utf-8-sig")
print(f"\n저장 완료: {EXISTING_PATH}")
print(f"\n{'상권':10} " + " ".join(f"{y:>7}" for y in year_cols_avail) + f" {'20→25성장률':>10}")
print("-" * 75)
for _, row in df.head(20).iterrows():
    vals = "  ".join(f"{int(row[y]):>7,}" for y in year_cols_avail)
    print(f"{row['area']:10} {vals}  {row['growth_20_25']:>+8.1f}%")
