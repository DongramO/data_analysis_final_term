# -*- coding: utf-8 -*-
"""
행정동 용도별 업종 매출 구성 상세 시각화
- 6패널: 업종 스택바 / 서울평균 편차 / 레이더 / 요일 패턴 / 시간대 패턴 / 핵심지표 분포
출력: image/area_type_detail.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA  = ROOT / "data" / "main_data"
OUT   = ROOT / "image" / "area_type_detail.png"

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 색상 설정 ──────────────────────────────────────────────────────────────
TYPE_ORDER  = ["오피스형", "혼합형", "주거형", "상권형"]
TYPE_COLORS = {"오피스형": "#E05C5C", "혼합형": "#F5A623",
               "주거형": "#4A90D9",  "상권형": "#27AE60"}
TYPE_N      = {"오피스형": 30, "혼합형": 66, "주거형": 309, "상권형": 11}

IND_COLORS  = ["#3498DB", "#E74C3C", "#9B59B6", "#1ABC9C", "#95A5A6"]
IND_LABELS  = ["FB카페", "FB식사", "주류유흥", "라이프스타일", "기타(비트렌드)"]

# ── 데이터 로드 ──────────────────────────────────────────────────────────
master = pd.read_csv(DATA / "트렌드매출_마스터.csv", encoding="utf-8-sig")
area   = pd.read_csv(DATA / "area_type_profile.csv",  encoding="utf-8-sig")
raw_q  = pd.read_csv(DATA / "분기별_원본피처.csv", encoding="utf-8-sig")

# 2024년 연간합계(raw_features는 분기별 → 연도별로 집계)
raw24 = raw_q[raw_q["연도"] == 2024].copy()
# 행정동별 합계
grp_cols = [c for c in raw24.columns if "매출금액" in c or "매출_금액" in c]
raw24_agg = raw24.groupby("행정동")[grp_cols].sum().reset_index()

# 구역유형 컬럼명 자동 탐색
area_dong_col = "행정동"
area_type_col = next((c for c in area.columns if "구역유형" in c and "코드" not in c), None)
area_weekend_col = next((c for c in area.columns if "주말집중도" in c and "2030" not in c), None)

# 마스터 2024
m24 = master[master["연도"] == 2024].copy()

# 병합
df = m24.merge(area[[area_dong_col, area_type_col, area_weekend_col]], on="행정동", how="left")
df = df.merge(raw24_agg, on="행정동", how="left")
df = df.dropna(subset=[area_type_col])

# 업종 비중 컬럼 계산 (raw_q에서)
total_col = next((c for c in df.columns if c == "총_매출금액" or c.startswith("총_매출금")), "총_매출")
if total_col not in df.columns:
    total_col = "총_매출"

cafe_col  = next((c for c in df.columns if "FB카페" in c and "금액" in c), None)
meal_col  = next((c for c in df.columns if "FB식사" in c and "금액" in c), None)
alc_col   = next((c for c in df.columns if "주류유흥" in c and "금액" in c), None)
life_col  = next((c for c in df.columns if "라이프스타일" in c and "금액" in c), None)
etc_col   = next((c for c in df.columns if "기타" in c and "금액" in c), None)

# 총매출 기준
denom = df[total_col].replace(0, np.nan)

ind_ratios = {}
for label, col in zip(IND_LABELS[:4], [cafe_col, meal_col, alc_col, life_col]):
    if col:
        ind_ratios[label] = df[col] / denom * 100
    else:
        ind_ratios[label] = pd.Series(np.nan, index=df.index)

# 기타 = 100 - (나머지 4개)
trend_sum = sum(v for v in ind_ratios.values())
ind_ratios["기타(비트렌드)"] = 100 - trend_sum

# 요일별 비중 컬럼
day_map = {"월": "월요일", "화": "화요일", "수": "수요일", "목": "목요일",
           "금": "금요일", "토": "토요일", "일": "일요일"}
day_cols = {}
for short, full in day_map.items():
    c = next((col for col in df.columns if full in col and "금액" in col), None)
    day_cols[short] = c

# 시간대 비중 컬럼
time_labels = ["00~06", "06~11", "11~14", "14~17", "17~21", "21~24"]
time_cols = {}
for t in time_labels:
    c = next((col for col in df.columns if t in col and "금액" in col), None)
    time_cols[t] = c

# 구역유형 그룹별 통계
def group_stat(series_dict, agg="median"):
    result = {}
    for t in TYPE_ORDER:
        mask = df[area_type_col] == t
        row = {}
        for k, s in series_dict.items():
            if s is not None and not isinstance(s, float):
                vals = s[mask].dropna()
            else:
                vals = pd.Series([np.nan])
            row[k] = vals.median() if agg == "median" else vals.mean()
        result[t] = row
    return pd.DataFrame(result).T  # index=TYPE_ORDER, columns=keys


# ── Figure 구성 ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor("#F8F9FA")
gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38,
              left=0.07, right=0.97, top=0.91, bottom=0.08)

ax1 = fig.add_subplot(gs[0, 0])   # 업종 구성 스택바
ax2 = fig.add_subplot(gs[0, 1])   # 서울 평균 편차 히트맵
ax3 = fig.add_subplot(gs[0, 2])   # 레이더
ax4 = fig.add_subplot(gs[1, 0])   # 요일 패턴
ax5 = fig.add_subplot(gs[1, 1])   # 시간대 패턴
ax6 = fig.add_subplot(gs[1, 2])   # 핵심지표 분포(violinplot)

for ax in [ax1, ax2, ax4, ax5, ax6]:
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("행정동 용도별 업종·소비 패턴 상세 비교", fontsize=18, fontweight="bold",
             y=0.96, color="#1A1A2E")

# ══════════════════════════════════════════════════════════════════════
# Panel 1: 100% 스택바 — 업종 구성
# ══════════════════════════════════════════════════════════════════════
ind_df = group_stat(ind_ratios, agg="median")
# 100% normalize
ind_norm = ind_df.div(ind_df.sum(axis=1), axis=0) * 100

bottom = np.zeros(len(TYPE_ORDER))
x = np.arange(len(TYPE_ORDER))
for i, (label, color) in enumerate(zip(IND_LABELS, IND_COLORS)):
    vals = ind_norm[label].values
    bars = ax1.bar(x, vals, bottom=bottom, color=color, label=label,
                   width=0.65, alpha=0.88, edgecolor="white", linewidth=1.2)
    # 5% 이상만 텍스트
    for xi, (v, b) in enumerate(zip(vals, bottom)):
        if v > 4.5:
            ax1.text(xi, b + v/2, f"{v:.1f}%", ha="center", va="center",
                     fontsize=8.5, fontweight="bold", color="white")
    bottom += vals

ax1.set_xticks(x)
ax1.set_xticklabels([f"{t}\n(n={TYPE_N[t]})" for t in TYPE_ORDER], fontsize=9.5)
ax1.set_ylabel("매출 구성 비중 (%)", fontsize=10)
ax1.set_ylim(0, 105)
ax1.set_title("① 업종 구성 비교 (100% 기준, 중앙값)", fontsize=12, fontweight="bold", pad=10)
ax1.legend(loc="upper right", fontsize=7.5, framealpha=0.8, ncol=1)
ax1.set_yticks([0, 25, 50, 75, 100])

# 오피스형 FB식사 강조 화살표
meal_vals = ind_norm["FB식사"].values
office_meal = meal_vals[0]
resident_meal = meal_vals[2]
ax1.annotate(f"FB식사 {office_meal:.1f}%\n(주거형 {resident_meal:.1f}% 대비 +{office_meal-resident_meal:.1f}%p)",
             xy=(0, ind_norm.loc["오피스형", "FB카페"] + office_meal/2),
             xytext=(-0.4, 55), fontsize=7.5, color="#C0392B",
             arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.2))

# ══════════════════════════════════════════════════════════════════════
# Panel 2: 서울 전체 평균 대비 편차 히트맵
# ══════════════════════════════════════════════════════════════════════
metrics = {
    "업종점수": "업종점수",
    "트렌드매출\n비중": "트렌드매출_비중",
    "금요\n효과": "금요효과",
    "주말\n비중": "주말비중",
    "야간\n비중": "야간비중",
    "2030\n비중": "2030비중",
}
metric_cols = list(metrics.values())

seoul_avg = df[metric_cols].median()

heat_data = []
for t in TYPE_ORDER:
    mask = df[area_type_col] == t
    sub = df[mask][metric_cols].median()
    heat_data.append((sub - seoul_avg) / seoul_avg * 100)  # % deviation

heat_df = pd.DataFrame(heat_data, index=TYPE_ORDER, columns=list(metrics.keys()))

vmax = max(abs(heat_df.values.min()), abs(heat_df.values.max()))
im = ax2.imshow(heat_df.values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

ax2.set_xticks(range(len(metrics)))
ax2.set_xticklabels(list(metrics.keys()), fontsize=9.5)
ax2.set_yticks(range(len(TYPE_ORDER)))
ax2.set_yticklabels([f"{t} (n={TYPE_N[t]})" for t in TYPE_ORDER], fontsize=9.5)
ax2.set_title("② 서울 전체 중앙값 대비 편차 (%)", fontsize=12, fontweight="bold", pad=10)

for i in range(len(TYPE_ORDER)):
    for j in range(len(metrics)):
        v = heat_df.iloc[i, j]
        text_color = "white" if abs(v) > vmax * 0.55 else "#333333"
        sign = "+" if v >= 0 else ""
        ax2.text(j, i, f"{sign}{v:.1f}%", ha="center", va="center",
                 fontsize=9, fontweight="bold", color=text_color)

cbar = plt.colorbar(im, ax=ax2, shrink=0.7, pad=0.02)
cbar.ax.tick_params(labelsize=8)
cbar.set_label("편차 (%)", fontsize=8)

ax2.spines[:].set_visible(False)

# ══════════════════════════════════════════════════════════════════════
# Panel 3: 레이더 차트 (Spider chart)
# ══════════════════════════════════════════════════════════════════════
radar_metrics = ["업종점수", "트렌드매출_비중", "금요효과", "주말비중", "야간비중", "2030비중"]
radar_labels  = ["업종점수", "트렌드매출\n비중", "금요효과\n(금÷목)", "주말비중", "야간비중", "2030비중"]
N_radar = len(radar_metrics)
angles = np.linspace(0, 2 * np.pi, N_radar, endpoint=False).tolist()
angles += angles[:1]

# Min-max 정규화 (0~1)
radar_vals = {}
metric_min = df[radar_metrics].quantile(0.05)
metric_max = df[radar_metrics].quantile(0.95)

for t in TYPE_ORDER:
    mask = df[area_type_col] == t
    vals = df[mask][radar_metrics].median()
    norm = (vals - metric_min) / (metric_max - metric_min).replace(0, 1)
    norm = norm.clip(0, 1)
    radar_vals[t] = norm.tolist() + [norm.tolist()[0]]

ax3.remove()
ax3 = fig.add_subplot(gs[0, 2], polar=True)
ax3.set_facecolor("white")

for t in TYPE_ORDER:
    color = TYPE_COLORS[t]
    ax3.plot(angles, radar_vals[t], color=color, linewidth=2.2, label=t)
    ax3.fill(angles, radar_vals[t], color=color, alpha=0.10)

ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(radar_labels, fontsize=9, color="#333333")
ax3.set_yticks([0.25, 0.5, 0.75, 1.0])
ax3.set_yticklabels(["25%", "50%", "75%", "최대"], fontsize=7, color="#888888")
ax3.set_title("③ 유형별 지표 레이더 (5~95분위 정규화)", fontsize=12,
              fontweight="bold", pad=18, color="#1A1A2E")
ax3.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9,
           framealpha=0.85, edgecolor="#CCCCCC")
ax3.grid(color="#DDDDDD", linestyle="--", alpha=0.7)

# ══════════════════════════════════════════════════════════════════════
# Panel 4: 요일별 매출 비중 패턴
# ══════════════════════════════════════════════════════════════════════
day_order = ["월", "화", "수", "목", "금", "토", "일"]
day_plot_ok = all(day_cols[d] is not None for d in day_order)

ax4.set_title("④ 요일별 매출 비중 패턴 (유형별 중앙값)", fontsize=12,
              fontweight="bold", pad=10)
ax4.set_ylabel("일별 매출 비중 (%)", fontsize=10)

if day_plot_ok:
    # 총매출 기준 비중
    for d in day_order:
        df[f"_day_{d}"] = df[day_cols[d]] / denom * 100

    for t in TYPE_ORDER:
        mask = df[area_type_col] == t
        vals = [df[mask][f"_day_{d}"].median() for d in day_order]
        ax4.plot(day_order, vals, marker="o", linewidth=2.2,
                 color=TYPE_COLORS[t], label=t, markersize=6)
        # 레이블: 토·일에만 표기
        for xi, d in enumerate(day_order):
            if d in ["토", "일"]:
                ax4.text(xi, vals[xi] + 0.3, f"{vals[xi]:.1f}%",
                         ha="center", fontsize=7.5, color=TYPE_COLORS[t])

    # 균등 기준선 (1/7 = 14.3%)
    ax4.axhline(100/7, color="#AAAAAA", linestyle="--", linewidth=1.2, alpha=0.7)
    ax4.text(6.05, 100/7, "균등\n14.3%", fontsize=7.5, color="#888888", va="center")
    ax4.axvspan(4.5, 6.5, color="#FFF9E6", alpha=0.5, zorder=0)
    ax4.text(5.5, ax4.get_ylim()[0] if ax4.get_ylim()[0] else 0,
             "주말", ha="center", fontsize=8, color="#888888", va="bottom")

else:
    ax4.text(0.5, 0.5, "요일별 데이터 없음", ha="center", va="center",
             transform=ax4.transAxes, fontsize=11, color="#888888")

ax4.legend(fontsize=8.5, loc="upper left", framealpha=0.8)

# ══════════════════════════════════════════════════════════════════════
# Panel 5: 시간대별 매출 비중 패턴
# ══════════════════════════════════════════════════════════════════════
ax5.set_title("⑤ 시간대별 매출 비중 패턴 (유형별 중앙값)", fontsize=12,
              fontweight="bold", pad=10)
ax5.set_ylabel("시간대 매출 비중 (%)", fontsize=10)

time_ok = all(time_cols[t] is not None for t in time_labels)

if time_ok:
    for t_label in time_labels:
        df[f"_time_{t_label}"] = df[time_cols[t_label]] / denom * 100

    x_pos = range(len(time_labels))
    for t in TYPE_ORDER:
        mask = df[area_type_col] == t
        vals = [df[mask][f"_time_{tl}"].median() for tl in time_labels]
        ax5.plot(x_pos, vals, marker="s", linewidth=2.2,
                 color=TYPE_COLORS[t], label=t, markersize=6)

    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(time_labels, fontsize=9)

    # 점심(11~14) · 저녁(17~21) 구간 강조
    ax5.axvspan(1.5, 2.5, color="#FFF0F0", alpha=0.5, zorder=0)
    ax5.axvspan(3.5, 4.5, color="#F0FFF0", alpha=0.5, zorder=0)
    ax5.text(2, ax5.get_ylim()[1] * 0.97 if ax5.get_ylim()[1] else 30,
             "점심피크", ha="center", fontsize=8, color="#C0392B")
    ax5.text(4, ax5.get_ylim()[1] * 0.97 if ax5.get_ylim()[1] else 30,
             "저녁피크", ha="center", fontsize=8, color="#27AE60")
else:
    ax5.text(0.5, 0.5, "시간대 데이터 없음", ha="center", va="center",
             transform=ax5.transAxes, fontsize=11, color="#888888")

ax5.legend(fontsize=8.5, loc="upper left", framealpha=0.8)

# ══════════════════════════════════════════════════════════════════════
# Panel 6: 핵심 지표 분포 비교 (Violin + Box)
# ══════════════════════════════════════════════════════════════════════
# 3개 지표를 나란히 violin으로
vio_metrics = ["트렌드매출_비중", "업종점수", "2030비중"]
vio_labels  = ["트렌드매출 비중", "업종점수", "2030 비중"]

ax6.set_title("⑥ 핵심 지표 분포 비교 (전체 행정동)", fontsize=12,
              fontweight="bold", pad=10)
ax6.set_facecolor("white")
ax6.spines[["top", "right"]].set_visible(False)

n_vio = len(vio_metrics)
n_types = len(TYPE_ORDER)
group_width = 1.0
gap = 0.3
positions = []
for gi, m in enumerate(vio_metrics):
    base = gi * (n_types * 0.22 + gap)
    for ti, t in enumerate(TYPE_ORDER):
        positions.append((gi, ti, base + ti * 0.22))

# Violin 그리기
for gi, m in enumerate(vio_metrics):
    base = gi * (n_types * 0.22 + gap)
    vio_data = []
    vio_pos  = []
    for ti, t in enumerate(TYPE_ORDER):
        mask = df[area_type_col] == t
        vals = df[mask][m].dropna().values * 100 if df[m].max() <= 1.0 else df[mask][m].dropna().values
        vio_data.append(vals)
        vio_pos.append(base + ti * 0.22)

    parts = ax6.violinplot(vio_data, positions=vio_pos, widths=0.18,
                           showmedians=True, showextrema=False)
    for pc, t in zip(parts["bodies"], TYPE_ORDER):
        pc.set_facecolor(TYPE_COLORS[t])
        pc.set_alpha(0.55)
    parts["cmedians"].set_color("#333333")
    parts["cmedians"].set_linewidth(1.5)

    # 그룹 레이블
    ax6.text(base + 0.33, -0.03, vio_labels[gi],
             ha="center", va="top", transform=ax6.get_xaxis_transform(),
             fontsize=9, fontweight="bold", color="#333333")
    if gi < n_vio - 1:
        ax6.axvline(base + n_types * 0.22, color="#DDDDDD", linewidth=1)

ax6.set_ylabel("(%)", fontsize=10)
ax6.set_xticks([])

# 범례
legend_patches = [mpatches.Patch(facecolor=TYPE_COLORS[t], alpha=0.7, label=t)
                  for t in TYPE_ORDER]
ax6.legend(handles=legend_patches, fontsize=8.5, loc="upper right", framealpha=0.85)

# 서울 평균 선 (각 지표별)
for gi, m in enumerate(vio_metrics):
    base = gi * (n_types * 0.22 + gap)
    avg_val = df[m].median() * 100 if df[m].max() <= 1.0 else df[m].median()
    span_start = base - 0.05
    span_end   = base + n_types * 0.22 - 0.05
    ax6.hlines(avg_val, span_start, span_end,
               color="#888888", linestyle="--", linewidth=1.2, alpha=0.8)
    ax6.text(span_end + 0.01, avg_val, f"서울\n{avg_val:.1f}%",
             fontsize=6.5, color="#888888", va="center")

# ── 하단 주석 ──────────────────────────────────────────────────────────────
fig.text(0.5, 0.025,
         "데이터: 서울시 상권분석서비스 추정매출 2024년 / 생활인구 2024.7~9 / "
         "중앙값 기준 | 레이더 정규화: 423개 동 5~95분위 스케일",
         ha="center", fontsize=8, color="#777777")

plt.savefig(str(OUT), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"[완료] {OUT}")
