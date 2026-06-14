import urllib.request
import urllib.parse
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Nanum Gothic'
plt.rcParams['axes.unicode_minus'] = False

CLIENT_ID = "8Bx6RLqBs8X32IhrACqU"
CLIENT_SECRET = "6iw3Sfw8WM"

# ── 분석 대상 동네 (행정동 + 별칭 + 강남권 포함) ─────────────
AREAS = [
    # 기존 생활밀착 상권
    "화곡", "당산", "노량진", "청량리", "왕십리", "대학로",
    "을지로", "신길", "중계", "연희동",
    # 강남권
    "가로수길", "압구정로데오", "청담", "강남역", "역삼", "코엑스",
    "서초", "방배", "사당", "잠실",
    # 트렌디 상권 (기성)
    "성수", "홍대", "이태원", "망원", "연남", "합정",
    "익선동", "경리단길", "해방촌", "서촌", "북촌",
    # 신흥 후보
    "문래", "영등포", "도화동", "흑석", "상도",
    "봉천", "면목", "장안", "답십리", "미아",
    "신정", "화양", "군자", "건대", "뚝섬",
]

YEARS = ["2022", "2023", "2024", "2025"]
KEYWORD = "카페"  # 상권 활성도 프록시


def search_blog(query: str) -> int:
    encoded = urllib.parse.quote(query)
    url = f"https://openapi.naver.com/v1/search/blog?query={encoded}&display=1&sort=sim"
    req = urllib.request.Request(url)
    req.add_header("X-Naver-Client-Id", CLIENT_ID)
    req.add_header("X-Naver-Client-Secret", CLIENT_SECRET)
    try:
        res = json.loads(urllib.request.urlopen(req).read().decode("utf-8"))
        return res.get("total", 0)
    except Exception:
        return 0


# ── 연도별 블로그 포스팅 수 수집 ─────────────────────────────
records = []
total_calls = len(AREAS) * len(YEARS)
done = 0

print(f"총 {total_calls}회 API 호출 시작...\n")
for area in AREAS:
    row = {"area": area}
    for year in YEARS:
        count = search_blog(f"{area} {KEYWORD} {year}")
        row[year] = count
        done += 1
        print(f"  [{done:3d}/{total_calls}] {area} {year}: {count:,}")
        time.sleep(0.12)  # API 호출 제한 방지
    records.append(row)

df = pd.DataFrame(records)

# ── 증가율 계산 ───────────────────────────────────────────────
df["growth_22_25"] = ((df["2025"] - df["2022"]) / (df["2022"] + 1) * 100).round(1)
df["growth_23_25"] = ((df["2025"] - df["2023"]) / (df["2023"] + 1) * 100).round(1)
df["total_vol"]    = df[YEARS].sum(axis=1)

df = df.sort_values("growth_22_25", ascending=False).reset_index(drop=True)

output_path = "data/블로그트렌드_상권.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\n저장 완료: {output_path}")
print(df[["area","2022","2023","2024","2025","growth_22_25","total_vol"]].to_string(index=False))

# ══════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(22, 9))
fig.suptitle("서울 동네별 블로그 포스팅 증가율 — 신흥 상권 탐지", fontsize=14, fontweight="bold")

# ── 패널1: 증가율 Top20 (신흥 후보) ──────────────────────────
ax = axes[0]
top20 = df.nlargest(20, "growth_22_25").sort_values("growth_22_25")
colors = ["#d62728" if v >= 100 else "#ff7f0e" if v >= 50 else "#1f77b4"
          for v in top20["growth_22_25"]]
bars = ax.barh(top20["area"], top20["growth_22_25"], color=colors, alpha=0.85)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("블로그 포스팅 증가율 2022→2025 (%)", fontsize=10)
ax.set_title("신흥 상권 후보\n(빨강≥100%, 주황≥50%)", fontsize=11)
ax.grid(True, axis="x", alpha=0.3)

# ── 패널2: 연도별 추이 (상위 10개) ───────────────────────────
ax = axes[1]
top10 = df.nlargest(10, "growth_22_25")
x = np.arange(len(YEARS))
for _, row in top10.iterrows():
    vals = [row[y] for y in YEARS]
    ax.plot(x, vals, marker="o", label=row["area"], linewidth=1.8)
ax.set_xticks(x)
ax.set_xticklabels(YEARS)
ax.set_ylabel("블로그 포스팅 수", fontsize=10)
ax.set_title("증가율 상위 10개 동네\n연도별 포스팅 추이", fontsize=11)
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)

# ── 패널3: 누적 볼륨 vs 최근 증가율 산점도 ────────────────────
ax = axes[2]
scatter = ax.scatter(
    df["total_vol"], df["growth_22_25"],
    c=df["growth_22_25"], cmap="RdYlGn", s=80, alpha=0.8,
    vmin=df["growth_22_25"].min(), vmax=df["growth_22_25"].max()
)
for _, row in df.iterrows():
    ax.annotate(row["area"], (row["total_vol"], row["growth_22_25"]),
                fontsize=7.5, xytext=(4, 3), textcoords="offset points")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("누적 포스팅 볼륨 (2022~2025 합산)", fontsize=10)
ax.set_ylabel("증가율 2022→2025 (%)", fontsize=10)
ax.set_title("볼륨 vs 성장률\n(우상단=신흥, 좌상단=급부상)", fontsize=11)
plt.colorbar(scatter, ax=ax, label="증가율(%)")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_img = "data/blog_trend_상권.png"
plt.savefig(out_img, dpi=150, bbox_inches="tight")
print(f"\n시각화 저장 완료: {out_img}")
plt.show()
