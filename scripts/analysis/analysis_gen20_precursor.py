# -*- coding: utf-8 -*-
"""
20대 소비 변화가 상권 활성화 전조임을 검증 — 4가지 기준
기준: 분析_수치_근거.md §0-3 레퍼런스 4개 / §1-3 Tier1 2개

선별 기준 (미발견 전조 행정동):
  Gate: 주말집중도 >= 0.80 (오피스형 제거)
  c1:  최근 3년(2022~2024) 20대 절대 매출 slope > 0
  c2:  2024 트렌드업종 비중 >= 0.40
  c3:  2023~2024 QoQ_20대 양수 분기 >= 4/8
  c4:  20대비중(2024) < 0.35 (레퍼런스 미달 = 아직 미발견)

출력: image/gen20_precursor_01~05_*.png
"""

import sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT     = Path(__file__).resolve().parent
RAW_PATH = ROOT / 'data/main_data/분기별_원본피처.csv'
AREA_PATH= ROOT / 'data/main_data/area_type_profile.csv'
IMG_DIR  = ROOT / 'image'

# ── 그룹 정의 ─────────────────────────────────────────────────────────────
REF   = ['방배2동', '청운효자동', '합정동', '성수2가1동']
TIER1 = ['숭인2동', '신당5동']

GROUP_COLOR = {
    '레퍼런스(4)': '#E05C5C',
    'Tier1(2)':   '#F5A623',
    '신규 선별':  '#27AE60',
    '서울 전체':  '#AAAAAA',
}

SCREEN = dict(gate_weekend=0.80, trend_industry_min=0.40, qoq_pos_min=4, age20_ratio_max=0.35)

# ── 데이터 로드 ────────────────────────────────────────────────────────────
raw  = pd.read_csv(RAW_PATH,  encoding='utf-8-sig')
area = pd.read_csv(AREA_PATH, encoding='utf-8-sig')

# ── 기본 지표 계산 ─────────────────────────────────────────────────────────
EPS = np.nan  # 0으로 나누기 방지

raw['야간비중'] = (
    raw['시간대_17~21_매출_금액'] + raw['시간대_21~24_매출_금액']
) / raw['총_매출금액'].replace(0, np.nan)

raw['트렌드시간_매출'] = (
    raw['토요일_매출_금액'] + raw['일요일_매출_금액']
    + raw['금요일_매출_금액'] * raw['야간비중']
)
raw['트렌드시간_비중'] = raw['트렌드시간_매출'] / raw['총_매출금액'].replace(0, np.nan)
raw['트렌드시간_20대_추정'] = raw['연령대_20_매출_금액'] * raw['트렌드시간_비중']

raw['트렌드업종_매출'] = (
    raw['FB카페_매출금액'] + raw['FB식사_매출금액']
    + raw['주류유흥_매출금액'] + raw['라이프스타일_매출금액']
)
raw['트렌드업종_비중']     = raw['트렌드업종_매출'] / raw['총_매출금액'].replace(0, np.nan)
raw['20대비중']           = raw['연령대_20_매출_금액'] / raw['총_매출금액'].replace(0, np.nan)
raw['트렌드업종_20대비율'] = raw['연령대_20_매출_금액'] / raw['트렌드업종_매출'].replace(0, np.nan)

raw = raw.merge(area[['행정동','주말집중도','구역유형']], on='행정동', how='left')

# ── QoQ (전년동기 대비) ───────────────────────────────────────────────────
raw = raw.sort_values(['행정동','연도','분기']).reset_index(drop=True)
raw['20대_전년동기'] = raw.groupby(['행정동','분기'])['연령대_20_매출_금액'].shift(1)
raw['QoQ_20대'] = raw['연령대_20_매출_금액'] / raw['20대_전년동기'].replace(0, np.nan) - 1

# 분기 순번 (x축)
raw['분기순번'] = (raw['연도'] - 2020) * 4 + raw['분기']
raw['연도분기'] = raw['연도'].astype(str) + 'Q' + raw['분기'].astype(str)

# ── 연간 집계 ─────────────────────────────────────────────────────────────
annual = raw.groupby(['행정동','연도']).agg(
    트렌드업종_비중    =('트렌드업종_비중','mean'),
    트렌드시간_비중    =('트렌드시간_비중','mean'),
    이십대비중        =('20대비중','mean'),
    이십대매출        =('연령대_20_매출_금액','sum'),
    트렌드시간_20대   =('트렌드시간_20대_추정','sum'),
    트렌드업종_20대비율=('트렌드업종_20대비율','mean'),
    야간비중          =('야간비중','mean'),
    주말집중도        =('주말집중도','first'),
    구역유형          =('구역유형','first'),
).reset_index()

# ── slope 계산 ────────────────────────────────────────────────────────────
def s(vals, yrs):
    yrs = np.array(yrs)
    if len(vals) < 2 or np.isnan(vals).all():
        return np.nan
    x = yrs - yrs[0]
    return np.polyfit(x, np.array(vals), 1)[0]

slopes = annual.groupby('행정동').apply(lambda g: pd.Series({
    '이십대매출_slope_recent': s(
        g[g['연도']>=2022]['이십대매출'].values,
        g[g['연도']>=2022]['연도'].values),
    '이십대비중_slope_recent': s(
        g[g['연도']>=2022]['이십대비중'].values,
        g[g['연도']>=2022]['연도'].values),
    '트렌드업종_20대비율_slope': s(g['트렌드업종_20대비율'].values, g['연도'].values),
    '주말집중도': g['주말집중도'].iloc[0],
    '구역유형':  g['구역유형'].iloc[0],
    '트렌드업종_24': g[g['연도']==2024]['트렌드업종_비중'].mean(),
    '이십대비중_24': g[g['연도']==2024]['이십대비중'].mean(),
    '이십대비중_20': g[g['연도']==2020]['이십대비중'].mean(),
    '이십대매출_24': g[g['연도']==2024]['이십대매출'].sum(),
    '이십대매출_20': g[g['연도']==2020]['이십대매출'].sum(),
})).reset_index()

slopes['이십대비중_delta'] = slopes['이십대비중_24'] - slopes['이십대비중_20']

qoq_count = (
    raw[(raw['연도'].isin([2023,2024])) & (raw['QoQ_20대'] > 0)]
    .groupby('행정동').size().rename('QoQ양수분기수')
)
slopes = slopes.merge(qoq_count, on='행정동', how='left').fillna({'QoQ양수분기수': 0})
slopes['QoQ양수분기수'] = slopes['QoQ양수분기수'].astype(int)

# ── 후보 선별 ─────────────────────────────────────────────────────────────
gate    = slopes['주말집중도'] >= SCREEN['gate_weekend']
exclude = slopes['행정동'].isin(REF + TIER1)
c1 = slopes['이십대매출_slope_recent'] > 0
c2 = slopes['트렌드업종_24'] >= SCREEN['trend_industry_min']
c3 = slopes['QoQ양수분기수'] >= SCREEN['qoq_pos_min']
c4 = slopes['이십대비중_24'] < SCREEN['age20_ratio_max']

candidates_df = slopes[gate & ~exclude & c1 & c2 & c3 & c4].copy()
candidates_df = candidates_df.sort_values('이십대비중_slope_recent', ascending=False)
CANDIDATES = candidates_df['행정동'].head(10).tolist()

print(f"[선별 결과] {len(CANDIDATES)}개 행정동")
for d in CANDIDATES:
    r = candidates_df[candidates_df['행정동']==d].iloc[0]
    print(f"  {d}: 트렌드업종={r['트렌드업종_24']:.3f}, "
          f"20대비중={r['이십대비중_24']:.3f}({r['이십대비중_delta']:+.3f}), "
          f"20대매출slope_recent={r['이십대매출_slope_recent']/1e8:.1f}억/년, "
          f"QoQ양수={int(r['QoQ양수분기수'])}/8, {r['구역유형']}")

# ── 그룹 라벨 ─────────────────────────────────────────────────────────────
def label_dong(d):
    if d in REF:        return '레퍼런스(4)'
    if d in TIER1:      return 'Tier1(2)'
    if d in CANDIDATES: return '신규 선별'
    return '서울 전체'

raw['그룹'] = raw['행정동'].map(label_dong)
annual['그룹'] = annual['행정동'].map(label_dong)

GROUPS_DONGS = {
    '서울 전체':  raw['행정동'].unique().tolist(),
    '신규 선별':  CANDIDATES,
    'Tier1(2)':  TIER1,
    '레퍼런스(4)': REF,
}
GROUPS_ORDER = list(GROUPS_DONGS.keys())

YEARS = [2020, 2021, 2022, 2023, 2024]

# 분기 순번 ↔ 레이블 매핑
QS = raw[['분기순번','연도분기']].drop_duplicates().sort_values('분기순번')

# ═══════════════════════════════════════════════════════════════════════════
# 그림 1 — 기준① 트렌드 시간대 업종 성향
# ═══════════════════════════════════════════════════════════════════════════
IND_COLS  = ['FB카페_매출금액','FB식사_매출금액','주류유흥_매출금액','라이프스타일_매출금액','기타_매출금액']
IND_NAMES = ['FB카페','FB식사','주류·유흥','라이프스타일','기타']
IND_COLORS = ['#5B9BD5','#ED7D31','#A5A5A5','#FFC000','#E0E0E0']

fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
fig1.suptitle('기준①  트렌드 시간대 업종 성향 비교 (2024 기준)', fontsize=14, fontweight='bold')

# 왼쪽: 업종 구성 100% 스택바
ax = axes1[0]
group_ratios = {}
for g in GROUPS_ORDER:
    dongs = GROUPS_DONGS[g]
    sub = raw[(raw['행정동'].isin(dongs)) & (raw['연도']==2024)]
    totals = [sub[c].mean() for c in IND_COLS]
    total  = sum(totals)
    group_ratios[g] = [v/total for v in totals]

x = np.arange(len(GROUPS_ORDER))
bottom = np.zeros(len(GROUPS_ORDER))
for idx, (name, color) in enumerate(zip(IND_NAMES, IND_COLORS)):
    vals = [group_ratios[g][idx] for g in GROUPS_ORDER]
    ax.bar(x, vals, bottom=bottom, color=color, label=name, width=0.5)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 0.04:
            ax.text(i, b + v/2, f'{v*100:.1f}%', ha='center', va='center',
                    fontsize=8.5, fontweight='bold', color='white')
    bottom += np.array(vals)

ax.set_xticks(x)
ax.set_xticklabels(GROUPS_ORDER, fontsize=10)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylim(0, 1.05)
ax.set_ylabel('매출 비중')
ax.set_title('업종 구성 100% 스택바', fontsize=11)
ax.legend(loc='upper right', fontsize=8.5)
ax.set_facecolor('#F8F8F8')

# 오른쪽: 3개 트렌드시간 지표 비교
ax2 = axes1[1]
metrics = ['야간비중', '트렌드시간_비중', '트렌드업종_비중']
metric_labels = ['야간비중\n(17~24시)', '트렌드시간 비중\n(금야간+토+일)', '트렌드업종 비중']
g_vals = {}
for g, dongs in GROUPS_DONGS.items():
    sub = raw[(raw['행정동'].isin(dongs)) & (raw['연도']==2024)]
    g_vals[g] = [sub['야간비중'].mean(), sub['트렌드시간_비중'].mean(), sub['트렌드업종_비중'].mean()]

x2 = np.arange(3)
w  = 0.18
for gi, g in enumerate(GROUPS_ORDER):
    offset = (gi - 1.5) * w
    bars = ax2.bar(x2 + offset, g_vals[g], w, color=GROUP_COLOR.get(g,'#999'), label=g, alpha=0.85)
    for bar, val in zip(bars, g_vals[g]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=7)

ref_line = g_vals['레퍼런스(4)'][2]
ax2.axhline(ref_line, color='#E05C5C', linestyle='--', linewidth=1, alpha=0.7)
ax2.text(2.55, ref_line + 0.003, f'레퍼런스 {ref_line:.3f}', color='#E05C5C', fontsize=8)

ax2.set_xticks(x2)
ax2.set_xticklabels(metric_labels, fontsize=9)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax2.set_ylabel('비중')
ax2.set_title('트렌드 시간대 활성도 지표', fontsize=11)
ax2.legend(fontsize=8.5)
ax2.set_facecolor('#F8F8F8')

plt.tight_layout()
out1 = IMG_DIR / 'gen20_precursor_01_industry.png'
fig1.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f"[저장] {out1.name}")


# ═══════════════════════════════════════════════════════════════════════════
# 그림 2 — 기준② 트렌드 시간대 20대 소비 절대량 · 비율
# ═══════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('기준②  트렌드 시간대 20대 소비 절대량 · 비율 시계열\n(추정: 20대매출 × 트렌드시간비중, 교차 데이터 없어 근사)',
              fontsize=12, fontweight='bold')

for g, dongs in GROUPS_DONGS.items():
    sub = annual[annual['행정동'].isin(dongs)]
    med_abs   = sub.groupby('연도')['트렌드시간_20대'].median() / 1e8
    med_ratio = sub.groupby('연도')['이십대비중'].median() * 100
    color = GROUP_COLOR.get(g, '#999')
    lw = 2.5 if g != '서울 전체' else 1.5
    ls = '-' if g != '서울 전체' else '--'
    axes2[0].plot(YEARS, med_abs.reindex(YEARS), marker='o', color=color, lw=lw, ls=ls, label=g)
    axes2[1].plot(YEARS, med_ratio.reindex(YEARS), marker='o', color=color, lw=lw, ls=ls, label=g)

for ax in axes2:
    ax.axvspan(2021.5, 2022.5, alpha=0.07, color='orange')
    ax.text(2022, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1] else 1,
            '성수 전환', ha='center', fontsize=8, color='#CC7700', va='top')
    ax.set_xticks(YEARS)
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('#F8F8F8')
    ax.legend(fontsize=9)

axes2[0].set_ylabel('추정 트렌드시간 20대 매출 (억원)')
axes2[0].set_title('트렌드 시간대 20대 소비 절대량 (그룹 중앙값)', fontsize=10)
axes2[1].set_ylabel('20대 매출 비중 (%)')
axes2[1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))
axes2[1].set_title('전체 매출 중 20대 비중 시계열', fontsize=10)

plt.tight_layout()
out2 = IMG_DIR / 'gen20_precursor_02_trend_time.png'
fig2.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"[저장] {out2.name}")


# ═══════════════════════════════════════════════════════════════════════════
# 그림 3 — 기준③ 트렌드 업종 내 20대 비율 시계열
# ═══════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
fig3.suptitle('기준③  트렌드 업종 내 20대 비율 시계열\n(근사: 20대전체매출 / 트렌드업종매출, 업종×연령 교차 없어 추정)',
              fontsize=12, fontweight='bold')

for g, dongs in GROUPS_DONGS.items():
    sub = annual[annual['행정동'].isin(dongs)]
    med = sub.groupby('연도')['트렌드업종_20대비율'].median() * 100
    med20 = med.reindex([2020]).values[0] if not pd.isna(med.reindex([2020]).values[0]) else 0
    delta = (med - med20).reindex(YEARS)
    color = GROUP_COLOR.get(g, '#999')
    lw = 2.5 if g != '서울 전체' else 1.5
    ls = '-' if g != '서울 전체' else '--'
    axes3[0].plot(YEARS, med.reindex(YEARS), marker='o', color=color, lw=lw, ls=ls, label=g)
    axes3[1].plot(YEARS, delta, marker='s', color=color, lw=lw, ls=ls, label=g)

for ax in axes3:
    ax.axvspan(2021.5, 2022.5, alpha=0.07, color='orange')
    ax.set_xticks(YEARS)
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('#F8F8F8')
    ax.legend(fontsize=9)

axes3[0].set_ylabel('트렌드업종 내 20대 비율 (%)')
axes3[0].set_title('그룹별 트렌드업종 20대 비율 (중앙값)', fontsize=10)
axes3[1].axhline(0, color='black', lw=0.8, ls=':')
axes3[1].set_ylabel('2020년 대비 변화 (%p)')
axes3[1].set_title('2020년 기준 트렌드업종 20대 비율 변화폭', fontsize=10)

plt.tight_layout()
out3 = IMG_DIR / 'gen20_precursor_03_industry_share.png'
fig3.savefig(out3, dpi=150, bbox_inches='tight')
plt.close(fig3)
print(f"[저장] {out3.name}")


# ═══════════════════════════════════════════════════════════════════════════
# 그림 4 — 기준④ QoQ 20대 소비 변화율 (분기별)
# ═══════════════════════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(16, 6))
fig4.suptitle('기준④  20대 소비 QoQ 변화율 (전년동기 대비, 분기 중앙값)',
              fontsize=14, fontweight='bold')

qoq_raw = raw[raw['연도'] >= 2021].sort_values('분기순번')
qs_range = sorted(qoq_raw['분기순번'].unique())

for g, dongs in GROUPS_DONGS.items():
    sub = qoq_raw[qoq_raw['행정동'].isin(dongs)]
    med = sub.groupby('분기순번')['QoQ_20대'].median() * 100
    color = GROUP_COLOR.get(g, '#999')
    lw = 2.5 if g != '서울 전체' else 1.5
    ls = '-' if g != '서울 전체' else '--'
    ax4.plot(qs_range, med.reindex(qs_range), marker='o', ms=4,
             color=color, lw=lw, ls=ls, label=g)

ax4.axhline(0, color='black', lw=1.0)
qs_map = QS.set_index('분기순번')['연도분기']
ax4.set_xticks(qs_range)
ax4.set_xticklabels([qs_map.get(v,'') for v in qs_range], fontsize=8, rotation=45)
ax4.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f%%'))
ax4.set_ylabel('QoQ 변화율 (%)')
ax4.set_ylim(-80, 150)
ax4.grid(axis='y', alpha=0.3)
ax4.set_facecolor('#F8F8F8')
ax4.legend(fontsize=10)
ax4.fill_between(qs_range, 0, 150, alpha=0.04, color='green')

plt.tight_layout()
out4 = IMG_DIR / 'gen20_precursor_04_qoq.png'
fig4.savefig(out4, dpi=150, bbox_inches='tight')
plt.close(fig4)
print(f"[저장] {out4.name}")


# ═══════════════════════════════════════════════════════════════════════════
# 그림 5 — 종합: 선별 결과 버블차트 + 요약 테이블
# ═══════════════════════════════════════════════════════════════════════════
# 레퍼런스·Tier1 지표도 slopes에서 추출
ref_rows = slopes[slopes['행정동'].isin(REF + TIER1)].copy()
ref_rows['그룹'] = ref_rows['행정동'].map(label_dong)
cand_rows = candidates_df.head(10).copy()
cand_rows['그룹'] = '신규 선별'

all_comp = pd.concat([ref_rows, cand_rows], ignore_index=True)

fig5, axes5 = plt.subplots(1, 2, figsize=(18, 7))
fig5.suptitle('선별 결과 비교 — 레퍼런스 / Tier1 / 신규 선별 후보', fontsize=14, fontweight='bold')

# 왼쪽: 버블차트 (트렌드업종비중 vs 20대비중 변화, 버블=QoQ양수분기)
ax5l = axes5[0]
for _, row in all_comp.iterrows():
    color = GROUP_COLOR.get(row['그룹'], '#999')
    sz = max((int(row['QoQ양수분기수']) + 1) * 35, 60)
    ax5l.scatter(row['트렌드업종_24'], row['이십대비중_delta'],
                 s=sz, color=color, alpha=0.78, edgecolors='white', lw=0.8, zorder=3)
    ax5l.annotate(row['행정동'],
                  (row['트렌드업종_24'], row['이십대비중_delta']),
                  fontsize=7.5, ha='center', va='bottom',
                  xytext=(0, 5), textcoords='offset points', zorder=4)

ax5l.axhline(0, color='gray', lw=0.8, ls=':')
ax5l.axvline(SCREEN['trend_industry_min'], color='orange', lw=1.2, ls='--', alpha=0.7,
             label=f'트렌드업종 기준 {SCREEN["trend_industry_min"]}')
ax5l.axhline(slopes[slopes['행정동'].isin(REF)]['이십대비중_delta'].mean(),
             color='#E05C5C', lw=1, ls=':', alpha=0.7, label='레퍼런스 20대비중Δ 평균')
ax5l.set_xlabel('2024 트렌드업종 비중')
ax5l.set_ylabel('20대비중 변화 (2024−2020)')
ax5l.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax5l.set_title('트렌드업종 비중 vs 20대비중 상승폭\n(버블 크기 = 2023~2024 QoQ 양수 분기 수)', fontsize=10)
ax5l.set_facecolor('#F8F8F8')
legend_patches = [mpatches.Patch(color=v, label=k) for k, v in GROUP_COLOR.items()]
ax5l.legend(handles=legend_patches + [
    mpatches.Patch(color='orange', alpha=0.5, label=f'트렌드업종 기준'),
], fontsize=8)

# 오른쪽: 요약 테이블
ax5r = axes5[1]
ax5r.axis('off')

tbl_data = []
for _, row in all_comp.iterrows():
    tbl_data.append([
        row['행정동'],
        row['그룹'],
        row['구역유형'],
        f"{row['트렌드업종_24']:.3f}",
        f"{row['이십대비중_24']:.3f}",
        f"{row['이십대비중_delta']:+.3f}",
        f"{int(row['QoQ양수분기수'])}/8",
    ])

headers = ['행정동','그룹','구역유형','트렌드업종\n비중(24)','20대\n비중(24)','20대비중\nΔ','QoQ양수']
tbl = ax5r.table(cellText=tbl_data, colLabels=headers,
                  cellLoc='center', loc='center', bbox=[0,0,1,1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)

for (ri, ci), cell in tbl.get_celld().items():
    if ri == 0:
        cell.set_facecolor('#3A5CA9')
        cell.set_text_props(color='white', fontweight='bold')
    else:
        dong = tbl_data[ri-1][0]
        if dong in REF:
            cell.set_facecolor('#FDECEA')
        elif dong in TIER1:
            cell.set_facecolor('#FFF4E5')
        elif ri % 2 == 0:
            cell.set_facecolor('#F4F9F4')

ax5r.set_title('행정동별 선별 지표 요약', fontsize=10, pad=10)

plt.tight_layout()
out5 = IMG_DIR / 'gen20_precursor_05_candidates.png'
fig5.savefig(out5, dpi=150, bbox_inches='tight')
plt.close(fig5)
print(f"[저장] {out5.name}")

print("\n[완료] 5개 이미지 생성")
print(f"  신규 선별 후보: {CANDIDATES}")
