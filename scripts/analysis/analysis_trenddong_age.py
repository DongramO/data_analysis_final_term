# -*- coding: utf-8 -*-
"""
대표 트렌드 상권 연도별 세대 유입 비중(주말 이동인구 기준) + 주말 매출 지수
- 세대 비중: migration_quarterly (금밤+토+일 필터 적용된 이동인구, 연령대별)
- 주말 매출: raw_features_quarterly (토+일 매출 합산, 2020=100 지수)
- 성수동: 4개 행정동 개별 표시
출력: image/trenddong_age_분석.png
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT     = Path(__file__).resolve().parent
RF_PATH  = ROOT / 'data/main_data/분기별_원본피처.csv'
MIG_PATH = ROOT / 'data/main_data/분기별_주말유입인구.csv'
MAP_PATH = ROOT / 'data/main_data/코드매핑_유입인구_매출.csv'
IMG_DIR  = ROOT / 'image'

# ── 대표 상권 코드 매핑 (성수동 개별 분리) ───────────────
DONG_MAP = {
    '홍대(서교동)':    [11440660],
    '연남동':          [11440710],
    '합정·망원':       [11440680, 11440690, 11440700],
    '이태원':          [11170650, 11170660],
    '한남동':          [11170685],
    '용산해방촌':      [11170520],
    '서촌':            [11110515],
    '북촌(가회동)':    [11110600],
    '삼청동':          [11110540],
    '대학로(혜화동)':  [11110650],
    '화양동':          [11215710],
    '성수1가1동':      [11200650],
    '성수1가2동':      [11200660],
    '성수2가1동':      [11200670],
    '성수2가3동':      [11200690],
    '낙성대동':        [11620585],
    '방배동':          [11650610],
    '잠실':            [11710650, 11710710],
}

YEARS = [2020, 2021, 2022, 2023, 2024]

# ══ 1. 주말 매출 지수 (raw_features, 토+일) ══════════════
rf = pd.read_csv(RF_PATH, encoding='utf-8-sig')
rf.columns = [c.strip() for c in rf.columns]
rf = rf.rename(columns={
    '행정동_코드':      'code',
    '연도':             'yr',
    '토요일_매출_금액': 'sat',
    '일요일_매출_금액': 'sun',
})
rf['weekend'] = rf['sat'] + rf['sun']
wk_agg = rf.groupby(['code','yr'])['weekend'].sum().reset_index()

def get_weekend_sales(codes):
    sub = wk_agg[wk_agg['code'].isin(codes)].groupby('yr')['weekend'].sum().reindex(YEARS)
    base = sub[2020]
    return (sub / base * 100).values if base and base > 0 else np.full(len(YEARS), np.nan)

# ══ 2. 주말 세대 유입 비중 (migration_quarterly, 이미 주말 필터) ══
_map = pd.read_csv(MAP_PATH, encoding='utf-8-sig')
_map.columns = ['mig_code','raw_code','gu','dong_map']
_map['raw_code'] = _map['raw_code'].astype('Int64')
code_mig2raw = _map.set_index('mig_code')['raw_code'].to_dict()

_mig = pd.read_csv(MIG_PATH, encoding='utf-8-sig')
_mig.columns = ['mig_code','yr','q','yrq','age','HE','WE','EE','mig_total']
_mig['raw_code'] = _mig['mig_code'].map(code_mig2raw)
_mig = _mig.dropna(subset=['raw_code'])
_mig['raw_code'] = _mig['raw_code'].astype(int)

_mig['gen'] = _mig['age'].apply(
    lambda a: '20' if a == 20 else ('30' if a == 30 else ('40p' if a >= 40 else 'other')))
mig_agg = _mig.groupby(['raw_code','yr','gen'])['mig_total'].sum().reset_index()
mig_tot = _mig.groupby(['raw_code','yr'])['mig_total'].sum().rename('tot').reset_index()
mig_agg = mig_agg.merge(mig_tot, on=['raw_code','yr'])
mig_agg['share'] = mig_agg['mig_total'] / mig_agg['tot'] * 100

def get_age_shares(codes):
    sub = mig_agg[mig_agg['raw_code'].isin(codes)]
    # 동이 여러 개면 이동인구 합산 후 비중 재계산
    num = sub.groupby(['yr','gen'])['mig_total'].sum().unstack(fill_value=0)
    tot = sub.groupby('yr')['mig_total'].sum()
    # 중복 합산 방지 (tot는 gen별 합 × 중복 있으므로 직접 계산)
    tot2 = num.sum(axis=1)
    r20  = (num.get('20',  0) / tot2 * 100).reindex(YEARS)
    r30  = (num.get('30',  0) / tot2 * 100).reindex(YEARS)
    r40p = (num.get('40p', 0) / tot2 * 100).reindex(YEARS)
    return r20.values, r30.values, r40p.values

# ══ 3. 시각화 ════════════════════════════════════════════
N     = len(DONG_MAP)
NCOLS = 3
NROWS = (N + NCOLS - 1) // NCOLS   # 6

fig, axes = plt.subplots(NROWS, NCOLS, figsize=(21, NROWS * 4.8))
fig.suptitle(
    '대표 트렌드 상권  │  세대별 주말 유입 비중(좌축, 이동인구 기준)  +  주말(토·일) 매출 지수(우축, 2020=100)',
    fontsize=12, fontweight='bold', y=1.005)

C20  = '#E05C5C'
C30  = '#F5A623'
C40P = '#3A5CA9'
CBAR = '#AACFE4'

for idx, (name, codes) in enumerate(DONG_MAP.items()):
    row, col = divmod(idx, NCOLS)
    ax = axes[row][col]

    r20, r30, r40p = get_age_shares(codes)
    w_idx = get_weekend_sales(codes)

    if np.all(np.isnan(r20)):
        ax.set_title(f'{name}\n(이동 데이터 없음)', fontsize=10)
        ax.axis('off')
        continue

    # ── 우측: 주말 매출 지수 바 ──────────────────────────
    ax2 = ax.twinx()
    ax2.bar(YEARS, w_idx, width=0.6, color=CBAR, alpha=0.40, zorder=1)
    ax2.set_ylim(0, max(np.nanmax(w_idx) * 1.55, 220))
    ax2.set_ylabel('주말 매출 지수\n(2020=100)', fontsize=7, color='#4477AA')
    ax2.tick_params(axis='y', labelsize=7, colors='#4477AA')
    ax2.spines['right'].set_edgecolor(CBAR)
    for yr, v in zip(YEARS, w_idx):
        if not np.isnan(v):
            ax2.text(yr, v + 2, f'{v:.0f}', ha='center', fontsize=6.5, color='#4477AA')
    ax2.set_zorder(1)

    # ── 좌측: 세대 유입 비중 라인 ────────────────────────
    ax.set_zorder(2)
    ax.patch.set_visible(False)

    ax.plot(YEARS, r20,  color=C20,  lw=2.5, marker='o', markersize=6, label='20대', zorder=3)
    ax.plot(YEARS, r30,  color=C30,  lw=2.5, marker='s', markersize=6, label='30대', zorder=3)
    ax.plot(YEARS, r40p, color=C40P, lw=2.5, marker='^', markersize=6, label='40+50+60', zorder=3)

    for yr, v2, v3, v4 in zip(YEARS, r20, r30, r40p):
        if not any(np.isnan([v2, v3, v4])):
            ax.text(yr, v2 + 0.5, f'{v2:.1f}%', ha='center', fontsize=6.5, color=C20,  zorder=4)
            ax.text(yr, v3 - 1.6, f'{v3:.1f}%', ha='center', fontsize=6.5, color=C30,  zorder=4)
            ax.text(yr, v4 + 0.5, f'{v4:.1f}%', ha='center', fontsize=6.5, color=C40P, zorder=4)

    d20  = r20[-1]  - r20[0]  if not np.isnan(r20[-1])  else np.nan
    d30  = r30[-1]  - r30[0]  if not np.isnan(r30[-1])  else np.nan
    d40p = r40p[-1] - r40p[0] if not np.isnan(r40p[-1]) else np.nan
    dsal = w_idx[-1] - 100    if not np.isnan(w_idx[-1]) else np.nan
    ax.set_title(
        f'{name}\n'
        f'20대: {d20:+.1f}%p  30대: {d30:+.1f}%p  40+: {d40p:+.1f}%p  │  주말매출: {dsal:+.0f}',
        fontsize=9, pad=5)

    ax.set_xticks(YEARS)
    ax.set_xticklabels(["'20","'21","'22","'23","'24"], fontsize=9)
    ax.set_ylabel('주말 유입 비중 (%)', fontsize=8)
    valid = np.concatenate([r20[~np.isnan(r20)], r30[~np.isnan(r30)], r40p[~np.isnan(r40p)]])
    if len(valid):
        ax.set_ylim(max(0, valid.min() - 6), min(100, valid.max() + 6))
    ax.grid(alpha=0.18, zorder=0)

    if idx == 0:
        handles = [
            Line2D([0],[0], color=C20,  lw=2, marker='o', markersize=5, label='20대 (이동인구)'),
            Line2D([0],[0], color=C30,  lw=2, marker='s', markersize=5, label='30대 (이동인구)'),
            Line2D([0],[0], color=C40P, lw=2, marker='^', markersize=5, label='40+50+60 (이동인구)'),
            mpatches.Patch(facecolor=CBAR, alpha=0.6, label='주말 매출 지수 (우축)'),
        ]
        ax.legend(handles=handles, fontsize=7.5, loc='upper right')

# 빈 칸 숨기기
for idx in range(N, NROWS * NCOLS):
    row, col = divmod(idx, NCOLS)
    axes[row][col].axis('off')

fig.tight_layout(h_pad=2.5, w_pad=1.5)
out = IMG_DIR / 'trenddong_age_분석.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[저장] {out.name}")
