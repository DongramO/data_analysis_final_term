# -*- coding: utf-8 -*-
"""
QoQ 선행-후행 분석: 20대 생활이동 vs 소비비중
  - 선정 지역(레퍼런스 / Phase35 / Phase41 후보) 기준
  - 연도 구분 없이 분기 연속 시계열 (2020Q1=1 ~ 2024Q4=20)
  - 교차상관(CCF) lag -2 ~ +2 분기

출력:
  image/qoq_01_group_timeseries.png  — 그룹 평균 시계열 (이동 vs 소비)
  image/qoq_02_ccf_by_group.png      — 그룹별 교차상관 함수
  image/qoq_03_individual_grid.png   — 개별 지역 시계열 패널
  image/qoq_04_scatter_leadlag.png   — 이동(t) vs 소비(t+1) 산점도
"""

import os
import sys
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import pearsonr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('image', exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 지역 정의
# ─────────────────────────────────────────────────────────────
# ── 새 레퍼런스 4개 (2026-05-26 기준 재선정) ─────────────────
# 기준: S1(업종점수≥0.50) + S2(2030비중≥0.35) 전환
#        Gate: 주말집중도≥0.80 | 트렌드매출_비중≥0.30 | 대학가 제외
REF_NEW = ['방배2동', '청운효자동', '합정동', '성수2가1동']

# 비교 대상: Phase35 차세대 후보 / Phase41 이동선행 후보
PHASE35_TOP = [
    '청파동', '다산동', '후암동', '연희동',
    '망원2동', '보문동', '논현2동', '성산2동', '망원1동',
]
PHASE41_2Q = [
    '이촌2동', '신당5동', '황학동', '북아현동', '숭인2동',
    '숭인1동', '가양2동', '오륜동', '창신3동', '보문동',
]

GROUP_META = {
    'REF_신규':  {'dongs': REF_NEW,      'color': '#1565C0', 'ls': '-',  'label': '레퍼런스 신규(4)'},
    'Phase35':   {'dongs': PHASE35_TOP,  'color': '#E65100', 'ls': '--', 'label': 'Phase35 차세대(9)'},
    'Phase41':   {'dongs': PHASE41_2Q,   'color': '#7B1FA2', 'ls': ':',  'label': 'Phase41 이동선행(10)'},
}

# ─────────────────────────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────────────────────────
print('[1/5] 데이터 로드...')

code_map = pd.read_csv('data/main_data/코드매핑_유입인구_매출.csv', encoding='utf-8-sig')
mig_to_raw = dict(zip(code_map['migration_코드'], code_map['raw_코드'].astype(int)))
raw_to_name = dict(zip(code_map['raw_코드'].astype(int), code_map['행정동명']))

mig_raw = pd.read_csv('data/main_data/분기별_주말유입인구.csv', encoding='utf-8-sig')
rfq     = pd.read_csv('data/main_data/분기별_원본피처.csv', encoding='utf-8-sig')

# ─────────────────────────────────────────────────────────────
# 20대 이동비율 계산 (분기 단위)
# ─────────────────────────────────────────────────────────────
print('[2/5] 패널 구축...')

mig20  = mig_raw[mig_raw['연령대'] == 20].groupby(['도착_행정동코드', '연도', '분기'])['합계_이동인구'].sum().reset_index()
mig_all = mig_raw.groupby(['도착_행정동코드', '연도', '분기'])['합계_이동인구'].sum().reset_index()
mq = mig20.merge(mig_all, on=['도착_행정동코드', '연도', '분기'], suffixes=('_20', '_전체'))
mq['이동비율'] = mq['합계_이동인구_20'] / mq['합계_이동인구_전체'].replace(0, np.nan)
mq['raw_코드'] = mq['도착_행정동코드'].map(mig_to_raw)
mq['행정동'] = mq['raw_코드'].map(raw_to_name)
mq = mq.dropna(subset=['행정동', '이동비율'])

# 연속 YQ 인덱스 (2020Q1=1 ~ 2024Q4=20)
mq['YQ'] = (mq['연도'] - 2020) * 4 + mq['분기']
mq_panel = mq[['행정동', '연도', '분기', 'YQ', '이동비율']].copy()

# ─────────────────────────────────────────────────────────────
# 20대 소비비중 계산 (분기 단위)
# ─────────────────────────────────────────────────────────────
rfq['소비비중'] = rfq['연령대_20_매출_금액'] / rfq['총_매출금액'].replace(0, np.nan)
rfq['YQ'] = (rfq['연도'] - 2020) * 4 + rfq['분기']
sq_panel = rfq[['행정동', '연도', '분기', 'YQ', '소비비중']].copy()

# ─────────────────────────────────────────────────────────────
# 패널 조인 & QoQ 변화량
# ─────────────────────────────────────────────────────────────
panel = mq_panel.merge(sq_panel, on=['행정동', '연도', '분기', 'YQ'], how='inner')
panel = panel.sort_values(['행정동', 'YQ']).reset_index(drop=True)

panel['Δ이동'] = panel.groupby('행정동')['이동비율'].diff()
panel['Δ소비'] = panel.groupby('행정동')['소비비중'].diff()

# 그룹 레이블 부여
all_selected = REF_NEW + PHASE35_TOP + PHASE41_2Q

group_map = {}
for d in REF_NEW:     group_map[d] = 'REF_신규'
for d in PHASE35_TOP: group_map[d] = 'Phase35'
for d in PHASE41_2Q:  group_map[d] = 'Phase41'

panel['그룹'] = panel['행정동'].map(group_map)
panel_sel = panel[panel['행정동'].isin(all_selected)].copy()

# 서울 전체 평균 (기준선)
seoul_avg = panel.groupby('YQ')[['이동비율', '소비비중']].mean().reset_index()
seoul_avg['Δ이동'] = seoul_avg['이동비율'].diff()
seoul_avg['Δ소비'] = seoul_avg['소비비중'].diff()

print(f'  패널 완성: {len(panel):,}행 / 선정 지역 {panel_sel["행정동"].nunique()}개')

# ─────────────────────────────────────────────────────────────
# 교차상관 함수 계산
# ─────────────────────────────────────────────────────────────
print('[3/5] 교차상관 계산...')

LAGS = [-2, -1, 0, 1, 2]

def calc_ccf(sub_df, lags=LAGS):
    """
    lag > 0: 이동(t) → 소비(t+lag)  — 이동 선행
    lag < 0: 이동(t) → 소비(t+lag)  — 소비 선행 (이동 후행)
    """
    results = {}
    for lag in lags:
        pairs = []
        for _, grp in sub_df.groupby('행정동'):
            grp = grp.sort_values('YQ').reset_index(drop=True)
            x = grp['Δ이동'].values
            if lag == 0:
                y = grp['Δ소비'].values
            elif lag > 0:
                y = np.concatenate([grp['Δ소비'].values[lag:], [np.nan] * lag])
            else:
                y = np.concatenate([[np.nan] * abs(lag), grp['Δ소비'].values[:lag]])
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 5:
                pairs.append((x[mask], y[mask]))
        if pairs:
            xs = np.concatenate([p[0] for p in pairs])
            ys = np.concatenate([p[1] for p in pairs])
            valid = np.isfinite(xs) & np.isfinite(ys)
            if valid.sum() >= 10:
                r, p = pearsonr(xs[valid], ys[valid])
                results[lag] = {'r': r, 'p': p, 'n': int(valid.sum())}
            else:
                results[lag] = {'r': 0.0, 'p': 1.0, 'n': 0}
        else:
            results[lag] = {'r': 0.0, 'p': 1.0, 'n': 0}
    return results

ccf_results = {}
for gname, gmeta in GROUP_META.items():
    sub = panel_sel[panel_sel['그룹'] == gname]
    ccf_results[gname] = calc_ccf(sub)

# 서울 전체
ccf_results['서울전체'] = calc_ccf(panel[['행정동', 'YQ', 'Δ이동', 'Δ소비']].copy())

# 수준값 상관도 계산 (CCF level)
def calc_level_ccf(sub_df, lags=LAGS):
    results = {}
    for lag in lags:
        pairs = []
        for _, grp in sub_df.groupby('행정동'):
            grp = grp.sort_values('YQ').reset_index(drop=True)
            x = grp['이동비율'].values
            if lag == 0:
                y = grp['소비비중'].values
            elif lag > 0:
                y = np.concatenate([grp['소비비중'].values[lag:], [np.nan] * lag])
            else:
                y = np.concatenate([[np.nan] * abs(lag), grp['소비비중'].values[:lag]])
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() >= 5:
                pairs.append((x[mask], y[mask]))
        if pairs:
            xs = np.concatenate([p[0] for p in pairs])
            ys = np.concatenate([p[1] for p in pairs])
            valid = np.isfinite(xs) & np.isfinite(ys)
            if valid.sum() >= 10:
                r, p = pearsonr(xs[valid], ys[valid])
                results[lag] = {'r': r, 'p': p, 'n': int(valid.sum())}
            else:
                results[lag] = {'r': 0.0, 'p': 1.0, 'n': 0}
        else:
            results[lag] = {'r': 0.0, 'p': 1.0, 'n': 0}
    return results

level_ccf = {}
for gname, gmeta in GROUP_META.items():
    sub = panel_sel[panel_sel['그룹'] == gname]
    level_ccf[gname] = calc_level_ccf(sub)

# ─────────────────────────────────────────────────────────────
# 그룹 평균 시계열
# ─────────────────────────────────────────────────────────────
print('[4/5] 시각화...')

group_avg = {}
for gname, gmeta in GROUP_META.items():
    sub = panel_sel[panel_sel['그룹'] == gname]
    avg = sub.groupby('YQ')[['이동비율', '소비비중']].mean().reset_index()
    avg['Δ이동'] = avg['이동비율'].diff()
    avg['Δ소비'] = avg['소비비중'].diff()
    group_avg[gname] = avg

# YQ 레이블 (연 경계만 표시)
yq_labels_all = [f'{2020 + (q-1)//4}Q{(q-1)%4+1}' for q in range(1, 21)]
yq_ticks_major = [q for q in range(1, 21) if (q - 1) % 4 == 0]  # Q1만
yq_labels_major = [f'{2020 + (q-1)//4}' for q in yq_ticks_major]

# ── Fig 1: 그룹 평균 시계열 ──────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)
fig.suptitle('그룹별 20대 이동비율 vs 소비비중 — QoQ 연속 시계열\n(2020Q1 ~ 2024Q4, 분기 연속) | 레퍼런스: 신규 4개 기준', fontsize=14, fontweight='bold')

YQS = np.arange(1, 21)

for ax, (gname, gmeta) in zip(axes, GROUP_META.items()):
    avg = group_avg[gname]
    seoul = seoul_avg.copy()

    ax2 = ax.twinx()

    # 서울 전체 기준선 (배경)
    ax.fill_between(seoul['YQ'], seoul['이동비율'], alpha=0.08, color='gray')
    ax.plot(seoul['YQ'], seoul['이동비율'], color='gray', lw=1, ls=':', label='서울 전체 이동')
    ax2.plot(seoul['YQ'], seoul['소비비중'], color='gray', lw=1, ls='-.', label='서울 전체 소비')

    # 그룹 평균
    ax.plot(avg['YQ'], avg['이동비율'], color=gmeta['color'], lw=2.2, ls='-',
            marker='o', ms=4, label=f'{gmeta["label"]} 이동비율')
    ax2.plot(avg['YQ'], avg['소비비중'], color=gmeta['color'], lw=2.2, ls='--',
             marker='s', ms=4, label=f'{gmeta["label"]} 소비비중')

    # 연 구분선
    for yq in yq_ticks_major[1:]:
        ax.axvline(yq - 0.5, color='#ccc', lw=0.8, ls='--')

    # 상관계수 표시
    r0 = ccf_results[gname].get(0, {}).get('r', 0)
    r1 = ccf_results[gname].get(1, {}).get('r', 0)
    p1 = ccf_results[gname].get(1, {}).get('p', 1)
    sig1 = '***' if p1 < 0.001 else ('**' if p1 < 0.01 else ('*' if p1 < 0.05 else 'n.s.'))
    ax.text(0.01, 0.95,
            f'CCF lag0={r0:+.3f}  lag+1={r1:+.3f}{sig1}  (변화량 QoQ)',
            transform=ax.transAxes, fontsize=8.5, va='top',
            bbox=dict(fc='white', ec='#ddd', alpha=0.85, boxstyle='round,pad=0.3'))

    ax.set_ylabel('20대 이동비율', color=gmeta['color'], fontsize=9)
    ax2.set_ylabel('20대 소비비중', color=gmeta['color'], fontsize=9, rotation=-90, labelpad=15)
    ax.tick_params(axis='y', labelcolor=gmeta['color'])
    ax2.tick_params(axis='y', labelcolor=gmeta['color'])

    # 범례
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7.5, ncol=2)
    ax.set_title(gmeta['label'], fontsize=11, fontweight='bold', pad=6)
    ax.grid(axis='y', alpha=0.25)
    ax.spines[['top']].set_visible(False)

axes[-1].set_xticks(range(1, 21))
axes[-1].set_xticklabels(yq_labels_all, rotation=45, ha='right', fontsize=7.5)
for yq in yq_ticks_major:
    axes[-1].get_xticklabels()[yq - 1].set_fontweight('bold')
    axes[-1].get_xticklabels()[yq - 1].set_fontsize(9)

plt.tight_layout()
plt.savefig('image/qoq_01_group_timeseries.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('  → image/qoq_01_group_timeseries.png')

# ── Fig 2: 교차상관 (CCF) ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
fig.suptitle('교차상관 함수 (CCF): 20대 이동비율(t) → 소비비중(t+lag)\n'
             'lag > 0: 이동 선행 | lag < 0: 소비 선행 | QoQ 변화량 기준 | 레퍼런스: 신규 4개',
             fontsize=13, fontweight='bold')

GROUPS_VIZ = list(GROUP_META.keys()) + ['서울전체']
GROUP_COLORS = {
    'REF_신규': '#1565C0',
    'Phase35': '#E65100', 'Phase41': '#7B1FA2', '서울전체': '#555555'
}
GROUP_LABELS = {
    'REF_신규': '레퍼런스 신규(4)',
    'Phase35': 'Phase35 차세대(9)',
    'Phase41': 'Phase41 이동선행(10)',
    '서울전체': '서울 전체 423개',
}

for ax, gname in zip(axes.flat, GROUPS_VIZ):
    ccf = ccf_results[gname]
    rs  = [ccf[l]['r'] for l in LAGS]
    ps  = [ccf[l]['p'] for l in LAGS]
    ns  = [ccf[l]['n'] for l in LAGS]

    bar_colors = []
    for r, p in zip(rs, ps):
        if p < 0.05:
            bar_colors.append('#1565C0' if r > 0 else '#C62828')
        else:
            bar_colors.append('#B0BEC5')

    bars = ax.bar(LAGS, rs, color=bar_colors, edgecolor='white', linewidth=0.5, width=0.7)
    ax.axhline(0, color='black', lw=0.9)

    for bar, r, p, n in zip(bars, rs, ps, ns):
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        ypos = r + 0.002 if r >= 0 else r - 0.008
        va = 'bottom' if r >= 0 else 'top'
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f'r={r:+.3f}\n{sig}' if sig else f'r={r:+.3f}',
                ha='center', va=va, fontsize=8, fontweight='bold' if sig else 'normal')
        ax.text(bar.get_x() + bar.get_width()/2, 0,
                f'n={n}', ha='center', va='bottom', fontsize=6.5, color='#777')

    ax.set_title(GROUP_LABELS.get(gname, gname), fontsize=11, fontweight='bold', pad=6,
                 color=GROUP_COLORS.get(gname, '#555'))
    ax.set_xlabel('lag (분기)', fontsize=9)
    ax.set_ylabel('Pearson r (Δ이동, Δ소비)', fontsize=9)
    ax.set_xticks(LAGS)
    ax.set_xticklabels([f'lag{l:+d}' for l in LAGS], fontsize=9)
    ax.set_ylim(-0.45, 0.45)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.axvline(0, color='#ffcc00', lw=1.2, ls='--', alpha=0.7, zorder=0)

# 마지막 빈 axes 숨기기 (2×3=6, 그룹5개)
for ax in axes.flat[len(GROUPS_VIZ):]:
    ax.set_visible(False)

fig.text(0.5, -0.02,
    '■ 파랑: r>0, p<0.05  ■ 빨강: r<0, p<0.05  □ 회색: n.s.  |  '
    'lag+1: 이동이 소비보다 1분기 선행',
    ha='center', fontsize=9, color='#555')

plt.tight_layout()
plt.savefig('image/qoq_02_ccf_by_group.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('  → image/qoq_02_ccf_by_group.png')

# ── Fig 3: 개별 지역 시계열 패널 ─────────────────────────────
INDIVIDUAL_DONGS = REF_NEW + PHASE35_TOP[:6] + PHASE41_2Q[:4]  # 4+6+4=14개
n_row, n_col = 3, 5
fig, axes = plt.subplots(n_row, n_col, figsize=(20, 12))
fig.suptitle('개별 지역 QoQ 시계열: 20대 이동비율(파랑) vs 소비비중(주황)\n'
             '— 연속 분기 (2020Q1~2024Q4) | 선은 수준값, 막대 배경은 QoQ 변화 방향',
             fontsize=13, fontweight='bold')

for i, (ax, dong) in enumerate(zip(axes.flat, INDIVIDUAL_DONGS)):
    sub = panel[panel['행정동'] == dong].sort_values('YQ')
    if sub.empty:
        ax.set_visible(False)
        continue

    grp = group_map.get(dong, '기타')
    color = GROUP_META.get(grp, {}).get('color', '#555')

    ax2 = ax.twinx()

    # 이동비율 (좌축, 파랑)
    ax.plot(sub['YQ'], sub['이동비율'] * 100, color='#1565C0', lw=1.8, marker='o', ms=3)
    ax.fill_between(sub['YQ'], sub['이동비율'] * 100,
                    alpha=0.12, color='#1565C0')

    # 소비비중 (우축, 주황)
    ax2.plot(sub['YQ'], sub['소비비중'] * 100, color='#E65100', lw=1.8,
             ls='--', marker='s', ms=3)

    # 연 구분
    for yq in yq_ticks_major[1:]:
        ax.axvline(yq - 0.5, color='#ddd', lw=0.6)

    # 개별 CCF lag+1 계산
    x = sub['Δ이동'].values
    y_lag1 = np.concatenate([sub['Δ소비'].values[1:], [np.nan]])
    mask = np.isfinite(x) & np.isfinite(y_lag1)
    if mask.sum() >= 5:
        r1, p1 = pearsonr(x[mask], y_lag1[mask])
        sig = '***' if p1 < 0.001 else ('**' if p1 < 0.01 else ('*' if p1 < 0.05 else ''))
        lag1_str = f'lag+1 r={r1:+.2f}{sig}'
    else:
        lag1_str = 'lag+1 n/a'

    grp_short = {'REF_신규': 'REF', 'Phase35': 'P35', 'Phase41': 'P41'}.get(grp, grp)
    ax.set_title(f'{dong}\n[{grp_short}] {lag1_str}',
                 fontsize=8.5, fontweight='bold', color=color, pad=3)
    ax.set_xticks(yq_ticks_major)
    ax.set_xticklabels([str(2020 + (q-1)//4) for q in yq_ticks_major], fontsize=7)
    ax.tick_params(axis='y', labelsize=7, labelcolor='#1565C0')
    ax2.tick_params(axis='y', labelsize=7, labelcolor='#E65100')
    ax.set_ylabel('이동비율(%)', fontsize=6.5, color='#1565C0')
    ax2.set_ylabel('소비비중(%)', fontsize=6.5, color='#E65100', rotation=-90, labelpad=12)
    ax.grid(axis='y', alpha=0.2)
    ax.spines[['top']].set_visible(False)

# 남은 axes 숨기기
for ax in axes.flat[len(INDIVIDUAL_DONGS):]:
    ax.set_visible(False)

# 범례
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#1565C0', lw=2, marker='o', ms=4, label='20대 이동비율 (좌축, %)'),
    Line2D([0], [0], color='#E65100', lw=2, ls='--', marker='s', ms=4, label='20대 소비비중 (우축, %)'),
]
fig.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9,
           bbox_to_anchor=(0.98, 0.01))

plt.tight_layout()
plt.savefig('image/qoq_03_individual_grid.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('  → image/qoq_03_individual_grid.png')

# ── Fig 4: 이동(t) vs 소비(t+1) 산점도 ──────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
fig.suptitle('이동비율(t) → 소비비중(t+1분기) 산점도\n'
             '— 이동 선행 가설 검증: 기울기(β)가 양수이면 이동이 소비를 선도',
             fontsize=13, fontweight='bold')

for ax, (gname, gmeta) in zip(axes, GROUP_META.items()):
    sub = panel_sel[panel_sel['그룹'] == gname].sort_values(['행정동', 'YQ'])
    xs, ys = [], []
    for _, grp in sub.groupby('행정동'):
        grp = grp.reset_index(drop=True)
        for i in range(len(grp) - 1):
            x_val = grp.loc[i, '이동비율']
            y_val = grp.loc[i + 1, '소비비중']
            if np.isfinite(x_val) and np.isfinite(y_val):
                xs.append(x_val)
                ys.append(y_val)

    xs, ys = np.array(xs), np.array(ys)

    ax.scatter(xs * 100, ys * 100, alpha=0.35, s=14,
               color=gmeta['color'], edgecolors='none')

    # 추세선
    if len(xs) > 10:
        from scipy.stats import linregress
        slope, intercept, r, p, _ = linregress(xs * 100, ys * 100)
        xline = np.linspace((xs * 100).min(), (xs * 100).max(), 100)
        ax.plot(xline, slope * xline + intercept, color='black', lw=1.8, ls='--')
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
        ax.text(0.05, 0.92,
                f'β={slope:.3f}  r={r:+.3f}  {sig}\n(n={len(xs)})',
                transform=ax.transAxes, fontsize=9.5,
                bbox=dict(fc='white', ec='#ddd', alpha=0.9, boxstyle='round'))

    ax.set_title(gmeta['label'], fontsize=11, fontweight='bold', color=gmeta['color'])
    ax.set_xlabel('20대 이동비율(t) %', fontsize=9)
    ax.set_ylabel('20대 소비비중(t+1분기) %', fontsize=9)
    ax.grid(alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('image/qoq_04_scatter_leadlag.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('  → image/qoq_04_scatter_leadlag.png')

# ─────────────────────────────────────────────────────────────
# 요약 출력
# ─────────────────────────────────────────────────────────────
print('\n[5/5] CCF 요약 (QoQ 변화량 기준)')
print(f'{"그룹":<16} {"lag-2":>8} {"lag-1":>8} {"lag0":>8} {"lag+1":>8} {"lag+2":>8}')
print('-' * 62)
for gname in list(GROUP_META.keys()) + ['서울전체']:
    label = GROUP_LABELS.get(gname, gname)
    row = [ccf_results[gname].get(l, {}).get('r', 0) for l in LAGS]
    sig_row = []
    for l in LAGS:
        r = ccf_results[gname].get(l, {}).get('r', 0)
        p = ccf_results[gname].get(l, {}).get('p', 1)
        s = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        sig_row.append(f'{r:+.3f}{s}')
    print(f'{label:<16} ' + '  '.join(f'{v:>10}' for v in sig_row))

print('\n수준값 CCF 요약 (이동비율 수준 vs 소비비중 수준)')
print(f'{"그룹":<16} {"lag-2":>8} {"lag-1":>8} {"lag0":>8} {"lag+1":>8} {"lag+2":>8}')
print('-' * 62)
for gname in GROUP_META.keys():
    label = GROUP_META[gname]['label']
    sig_row = []
    for l in LAGS:
        r = level_ccf[gname].get(l, {}).get('r', 0)
        p = level_ccf[gname].get(l, {}).get('p', 1)
        s = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        sig_row.append(f'{r:+.3f}{s}')
    print(f'{label:<16} ' + '  '.join(f'{v:>10}' for v in sig_row))

# ── Fig 5: 수준값 CCF — 그룹별 lag 프로파일 ─────────────────
fig, ax = plt.subplots(figsize=(13, 6))
fig.suptitle('수준값 교차상관: 이동비율(t) vs 소비비중(t+lag)\n'
             '— lag 증가에 따라 r이 높아지면 "이동이 소비를 선행"함을 의미',
             fontsize=13, fontweight='bold')

x = np.arange(len(LAGS))
width = 0.18
group_keys = list(GROUP_META.keys())

for i, gname in enumerate(group_keys):
    rs = [level_ccf[gname].get(l, {}).get('r', 0) for l in LAGS]
    ps = [level_ccf[gname].get(l, {}).get('p', 1) for l in LAGS]
    color = GROUP_META[gname]['color']
    label = GROUP_META[gname]['label']
    offset = (i - len(group_keys)/2 + 0.5) * width
    bars = ax.bar(x + offset, rs, width, color=color, label=label,
                  alpha=0.85, edgecolor='white', linewidth=0.5)
    # 유의성 별표
    for j, (bar, r, p) in enumerate(zip(bars, rs, ps)):
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        if sig:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    sig, ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')

# 추세 화살표 설명
ax.annotate('→ lag 증가에 따라\n  r이 높아질수록\n  이동이 소비를 선행',
            xy=(4.4, 0.55), fontsize=8.5, color='#555',
            bbox=dict(fc='lightyellow', ec='#ccc', boxstyle='round'))

ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f'lag{l:+d}\n({["소비2Q선행","소비1Q선행","동시","이동1Q선행","이동2Q선행"][i]})'
                    for i, l in enumerate(LAGS)], fontsize=9)
ax.set_ylabel('Pearson r (수준값 기준)', fontsize=10)
ax.set_ylim(0, 0.75)
ax.grid(axis='y', alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig('image/qoq_05_level_ccf_profile.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('  → image/qoq_05_level_ccf_profile.png')

print('\n=== 완료 ===')
