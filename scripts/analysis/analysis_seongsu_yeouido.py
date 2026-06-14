# -*- coding: utf-8 -*-
"""
성수2가1동 vs 여의동(더현대) — 연령대별 이동·소비 패턴 비교
출력: image/seongsu_yeouido_compare.png
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import pearsonr
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT      = Path(__file__).resolve().parent
RAW_PATH  = ROOT / 'data/main_data/분기별_원본피처.csv'
MIG_PATH  = ROOT / 'data/main_data/분기별_주말유입인구.csv'
CMAP_PATH = ROOT / 'data/main_data/코드매핑_유입인구_매출.csv'
TREND_MIG_PATH = ROOT / 'data/main_data/living_migration_trend_quarterly.csv'
IMG_DIR   = ROOT / 'image'

TARGETS = ['성수2가1동', '여의동']
DONG_COLOR = {'성수2가1동': '#E05C5C', '여의동': '#3A5CA9'}
YEARS = [2020, 2021, 2022, 2023, 2024]

AGE_LABELS = ['10대', '20대', '30대', '40대', '50대', '60이상']
AGE_COLORS = ['#95A5A6', '#E05C5C', '#F5A623', '#27AE60', '#3A5CA9', '#8E44AD']

IND_COLS  = ['FB카페_매출금액', 'FB식사_매출금액', '주류유흥_매출금액', '라이프스타일_매출금액', '기타_매출금액']
IND_NAMES = ['FB카페', 'FB식사', '주류·유흥', '라이프스타일', '기타']
IND_COLORS = ['#5B9BD5', '#ED7D31', '#A5A5A5', '#FFC000', '#D0D0D0']

DAY_COLS  = ['월요일_매출_금액','화요일_매출_금액','수요일_매출_금액',
             '목요일_매출_금액','금요일_매출_금액','토요일_매출_금액','일요일_매출_금액']
DAY_NAMES = ['월','화','수','목','금','토','일']

# ── 데이터 로드 ────────────────────────────────────────────────────────────
raw     = pd.read_csv(RAW_PATH,       encoding='utf-8-sig')
mig_raw = pd.read_csv(MIG_PATH,       encoding='utf-8-sig')   # 연령대 구성 비율용
cmap    = pd.read_csv(CMAP_PATH,      encoding='utf-8-sig')
tmig    = pd.read_csv(TREND_MIG_PATH, encoding='utf-8-sig')   # 트렌드시간 이동 (금17+토+일)
tmig['이동인구'] = pd.to_numeric(tmig['이동인구'], errors='coerce')

# tmig 컬럼: 행정동, 연도, 분기, 나이, 이동인구
# 나이 코드 → 연령대 레이블 매핑
AGE_MAP = {10:'10대', 20:'20대', 30:'30대', 40:'40대', 50:'50대',
           60:'60이상', 70:'60이상', 80:'60이상'}
tmig['연령대'] = tmig['나이'].map(AGE_MAP)
tmig_grp = (tmig.groupby(['행정동','연도','분기','연령대'])['이동인구']
            .sum().reset_index())
tmig_grp['분기순번'] = (tmig_grp['연도'] - 2020) * 4 + tmig_grp['분기']

# 연령대 구성 비율용 — migration_quarterly (합계 기준)
mig_raw = mig_raw.merge(cmap[['migration_코드','행정동명']],
                         left_on='도착_행정동코드', right_on='migration_코드', how='left')
mig_raw.rename(columns={'행정동명': '행정동'}, inplace=True)

ALL_QTR = list(range(1, 21))

def qtr_label(q):
    y = 2020 + (q - 1) // 4
    qq = (q - 1) % 4 + 1
    return f"{str(y)[2:]}Q{qq}"

QTR_LABELS = [qtr_label(q) for q in ALL_QTR]

# 트렌드시간 이동 — 분기별 연령대
def get_age_quarterly(dong):
    sub = tmig_grp[tmig_grp['행정동'] == dong]
    rows = {}
    for label in AGE_LABELS:
        s = (sub[sub['연령대'] == label]
             .groupby('분기순번')['이동인구'].sum()
             .reindex(ALL_QTR))
        rows[label] = s
    return pd.DataFrame(rows)

# 트렌드시간 이동 — 연도별 연령대
def get_age_yearly(dong):
    sub = tmig_grp[tmig_grp['행정동'] == dong]
    rows = {}
    for label in AGE_LABELS:
        s = (sub[sub['연령대'] == label]
             .groupby('연도')['이동인구'].sum()
             .reindex(YEARS))
        rows[label] = s
    return pd.DataFrame(rows)

def get_age_ratio_yearly(dong):
    df = get_age_yearly(dong)
    total = df.sum(axis=1)
    return df.div(total, axis=0)

# ── 그림 ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 2, figsize=(18, 26))
fig.suptitle('성수2가1동 vs 여의동(더현대) — 연령대별 이동 · 소비 패턴 비교\n'
             '(이동: 금요일 17시 이후 + 토·일 / 매출: 트렌드시간 금저녁+토+일 기준)',
             fontsize=13, fontweight='bold')

# ═══════════════════════════════════════════════════════════════
# Row 0: 연령대별 이동인구 지수 (분기별, 2020Q1=100)
# ═══════════════════════════════════════════════════════════════
for pi, dong in enumerate(TARGETS):
    ax = axes[0][pi]
    age_df = get_age_quarterly(dong)   # index=1~20

    for label, c in zip(AGE_LABELS, AGE_COLORS):
        s = age_df[label]
        base = s.iloc[0]
        if pd.isna(base) or base == 0:
            continue
        idx = s / base * 100
        lw = 2.5 if label == '20대' else 1.2
        ms = 5   if label == '20대' else 0
        zorder = 5 if label == '20대' else 3
        ax.plot(ALL_QTR, idx, 'o-' if label == '20대' else '-',
                color=c, lw=lw, markersize=ms, label=label, zorder=zorder)

    ax.axhline(100, color='gray', lw=0.8, ls=':')
    # 연도 구분 수직선
    for yr_start in [5, 9, 13, 17]:   # 2021Q1=5, 2022Q1=9, ...
        ax.axvline(yr_start - 0.5, color='#CCCCCC', lw=0.8, ls='-')
    if dong == '여의동':
        ax.axvline(5, color='#8E44AD', lw=2, ls='--', alpha=0.8,
                   label='더현대 개장(2021Q1)')

    # x축: 연도Q1 위치에만 레이블
    tick_pos    = [1, 5, 9, 13, 17]
    tick_labels = ['2020Q1','2021Q1','2022Q1','2023Q1','2024Q1']
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_xlim(0.5, 20.5)
    ax.set_ylabel('이동인구 지수 (2020Q1=100)')
    ax.set_title(f'{dong}  연령대별 이동인구 지수 (분기별)', fontsize=11, fontweight='bold',
                 color=DONG_COLOR[dong])
    ax.legend(fontsize=8.5, ncol=2, loc='upper left')
    ax.set_facecolor('#F8F8F8')
    ax.grid(alpha=0.2)

# ═══════════════════════════════════════════════════════════════
# Row 1: 연령대 구성 비율 변화 (누적 면적 or 그룹바)
# ═══════════════════════════════════════════════════════════════
for pi, dong in enumerate(TARGETS):
    ax = axes[1][pi]
    ratio_df = get_age_ratio_yearly(dong) * 100   # %

    bottom = np.zeros(len(YEARS))
    x = np.arange(len(YEARS))
    for label, c in zip(AGE_LABELS, AGE_COLORS):
        vals = ratio_df[label].values
        bars = ax.bar(x, vals, bottom=bottom, color=c, label=label, width=0.6, alpha=0.9)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 8:
                ax.text(i, b + v/2, f'{v:.0f}%',
                        ha='center', va='center', fontsize=8,
                        fontweight='bold', color='white')
        bottom += vals

    # 20대 비율 꺾은선 오버레이 (오른쪽 축)
    ax2 = ax.twinx()
    ax2.plot(x, ratio_df['20대'].values, 'D--', color='#E05C5C',
             lw=2, markersize=6, label='20대 비율(%)', zorder=6)
    ax2.set_ylabel('20대 비율 (%)', color='#E05C5C', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='#E05C5C')
    _max20 = ratio_df['20대'].max()
    ax2.set_ylim(0, _max20 * 2.5 if pd.notna(_max20) and _max20 > 0 else 0.5)

    if dong == '여의동':
        ax.axvline(1, color='#8E44AD', lw=1.8, ls='--', alpha=0.7)  # 2021=index 1

    ax.set_xticks(x)
    ax.set_xticklabels(YEARS)
    ax.set_ylabel('연령대 구성 비율 (%)')
    ax.set_title(f'{dong}  연령대 구성 비율 변화', fontsize=11, fontweight='bold',
                 color=DONG_COLOR[dong])
    ax.legend(fontsize=8, ncol=3, loc='upper left')
    ax.set_facecolor('#F8F8F8')

# ═══════════════════════════════════════════════════════════════
# Row 2: 업종 구성 변화 (left) + 요일별 매출 비중 변화 (right)
# ═══════════════════════════════════════════════════════════════

# Panel [2,0]: 두 지역 업종 구성 비교 (2020 vs 2022 vs 2024)
ax = axes[2][0]
compare_years = [2020, 2022, 2024]
n_years = len(compare_years)
n_dongs = len(TARGETS)
group_w = 0.35
bar_w   = group_w / n_years

x_base = np.arange(n_dongs)

for yi, year in enumerate(compare_years):
    for di, dong in enumerate(TARGETS):
        sub = raw[(raw['행정동']==dong) & (raw['연도']==year)]
        total = sub['총_매출금액'].mean()
        bottom = 0
        x_pos = di + (yi - 1) * group_w

        for col, name, c in zip(IND_COLS, IND_NAMES, IND_COLORS):
            v = sub[col].mean() / total if total > 0 else 0
            ax.bar(x_pos, v*100, bottom=bottom*100, color=c,
                   width=bar_w*0.9, alpha=0.85 if year==2024 else 0.5,
                   edgecolor='white', linewidth=0.4)
            bottom += v

# 연도 레이블
for yi, year in enumerate(compare_years):
    for di in range(n_dongs):
        x_pos = di + (yi - 1) * group_w
        ax.text(x_pos, 102, str(year)[2:]+'년',
                ha='center', fontsize=7.5, color='#555555')

ax.set_xticks(x_base)
ax.set_xticklabels(TARGETS, fontsize=11, fontweight='bold')
for tick, dong in zip(ax.get_xticklabels(), TARGETS):
    tick.set_color(DONG_COLOR[dong])

# 범례 (업종)
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=c, label=n) for c, n in zip(IND_COLORS, IND_NAMES)]
ax.legend(handles=patches, fontsize=8.5, loc='upper right', ncol=2)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_ylabel('업종 구성 비중 (%)')
ax.set_title('업종 구성 비교  (좌=2020, 중=2022, 우=2024)', fontsize=11)
ax.set_facecolor('#F8F8F8')
ax.set_ylim(0, 115)

# Panel [2,1]: 요일별 매출 비중 비교 (2020 vs 2024)
ax = axes[2][1]
x = np.arange(7)
width = 0.2
year_alpha = {2020: 0.45, 2024: 0.9}
year_ls    = {2020: '--', 2024: '-'}

for di, dong in enumerate(TARGETS):
    for yi, year in enumerate([2020, 2024]):
        sub = raw[(raw['행정동']==dong) & (raw['연도']==year)]
        day_sums = sub[DAY_COLS].sum()
        total    = day_sums.sum()
        ratios   = day_sums / total * 100
        color    = DONG_COLOR[dong]
        offset   = (di * 2 + yi - 1.5) * width
        label    = f'{dong} {year}년'
        ax.bar(x + offset, ratios, width=width*0.9, color=color,
               alpha=year_alpha[year], label=label, edgecolor='white')

ax.axhline(100/7, color='gray', lw=0.8, ls=':', label=f'균등 ({100/7:.1f}%)')
ax.set_xticks(x)
ax.set_xticklabels(DAY_NAMES, fontsize=10)
ax.set_ylabel('매출 비중 (%)')
ax.set_title('요일별 매출 비중  (진한 막대=2024, 연한 막대=2020)', fontsize=11)
ax.legend(fontsize=8.5, ncol=2)
ax.set_facecolor('#F8F8F8')
ax.grid(axis='y', alpha=0.2)

# ═══════════════════════════════════════════════════════════════
# Row 3: 분기별 트렌드 매출 변화
# 트렌드_대표_매출 = 금요일 × 야간비중 + 토요일 + 일요일
# ═══════════════════════════════════════════════════════════════

# 트렌드_대표_매출 계산
raw['야간비중'] = (
    raw['시간대_17~21_매출_금액'] + raw['시간대_21~24_매출_금액']
) / raw['총_매출금액'].replace(0, np.nan)
raw['트렌드_대표_매출'] = (
    raw['금요일_매출_금액'] * raw['야간비중']
    + raw['토요일_매출_금액']
    + raw['일요일_매출_금액']
)
raw['분기순번'] = (raw['연도'] - 2020) * 4 + raw['분기']

# Panel [3,0]: 두 지역 트렌드매출 지수 비교 (2020Q1=100)
ax = axes[3][0]
for dong in TARGETS:
    sub = raw[raw['행정동'] == dong].groupby('분기순번')['트렌드_대표_매출'].sum().reindex(ALL_QTR)
    base = sub.iloc[0]
    idx  = sub / base * 100
    color = DONG_COLOR[dong]
    ax.plot(ALL_QTR, idx, 'o-', color=color, lw=2.2, markersize=5, label=dong)

ax.axhline(100, color='gray', lw=0.8, ls=':')
for yr_start in [5, 9, 13, 17]:
    ax.axvline(yr_start - 0.5, color='#CCCCCC', lw=0.8, ls='-')
ax.axvline(5, color='#8E44AD', lw=1.8, ls='--', alpha=0.7, label='더현대 개장(2021Q1)')

ax.set_xticks([1, 5, 9, 13, 17])
ax.set_xticklabels(['2020Q1','2021Q1','2022Q1','2023Q1','2024Q1'], fontsize=9)
ax.set_xlim(0.5, 20.5)
ax.set_ylabel('트렌드매출 지수 (2020Q1=100)')
ax.set_title('트렌드매출 지수 비교 — 분기별\n(금저녁+토+일 기준, 2020Q1=100)', fontsize=11)
ax.legend(fontsize=9)
ax.set_facecolor('#F8F8F8')
ax.grid(alpha=0.2)

# Panel [3,1]: 두 지역 트렌드매출 YoY 변화율 (분기별)
ax = axes[3][1]
for dong in TARGETS:
    sub = (raw[raw['행정동'] == dong]
           .groupby(['연도','분기'])['트렌드_대표_매출'].sum()
           .reset_index()
           .sort_values(['연도','분기']))
    sub['YoY'] = sub.groupby('분기')['트렌드_대표_매출'].pct_change() * 100
    sub['분기순번'] = (sub['연도'] - 2020) * 4 + sub['분기']
    sub = sub[sub['연도'] >= 2021]

    color = DONG_COLOR[dong]
    ax.plot(sub['분기순번'], sub['YoY'], 'o-', color=color,
            lw=2.2, markersize=5, label=dong)
    ax.fill_between(sub['분기순번'], 0, sub['YoY'],
                    alpha=0.08, color=color)

ax.axhline(0, color='black', lw=0.8, ls=':')
for yr_start in [5, 9, 13, 17]:
    ax.axvline(yr_start - 0.5, color='#CCCCCC', lw=0.8, ls='-')
ax.axvline(5, color='#8E44AD', lw=1.8, ls='--', alpha=0.7, label='더현대 개장(2021Q1)')

ax.set_xticks([5, 9, 13, 17])
ax.set_xticklabels(['2021Q1','2022Q1','2023Q1','2024Q1'], fontsize=9)
ax.set_xlim(4.5, 20.5)
ax.set_ylabel('트렌드매출 YoY 변화율 (%)')
ax.set_title('트렌드매출 YoY 변화율 — 분기별\n(금저녁+토+일 기준, 전년동기 대비)', fontsize=11)
ax.legend(fontsize=9)
ax.set_facecolor('#F8F8F8')
ax.grid(alpha=0.2)

plt.tight_layout()
out = IMG_DIR / 'seongsu_yeouido_compare.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[저장] {out.name}")
