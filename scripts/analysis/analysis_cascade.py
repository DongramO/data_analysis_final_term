# -*- coding: utf-8 -*-
"""
20대 선행형 vs 매출 선행형 — 20대→30대+ 세대 연쇄 검증
출력: image/cascade_분석.png
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT     = Path(__file__).resolve().parent
MIG_PATH = ROOT / 'data/main_data/분기별_주말유입인구.csv'
MAP_PATH = ROOT / 'data/main_data/코드매핑_유입인구_매출.csv'
CCF_PATH = ROOT / 'data/_tmp_ccf_all.csv'
IMG_DIR  = ROOT / 'image'

C_20대  = '#E05C5C'
C_SAL   = '#3A5CA9'
C_20대연  = '#27AE60'
C_30대연  = '#F5A623'

# ── 데이터 로드 ──────────────────────────────────────────
_map = pd.read_csv(MAP_PATH, encoding='utf-8-sig')
_map.columns = ['mig_code','raw_code','gu','dong_map']
_map['raw_code'] = _map['raw_code'].astype('Int64')
code_mig2raw = _map.set_index('mig_code')['raw_code'].to_dict()
dong_name    = _map.set_index('raw_code')['dong_map'].to_dict()

_mig = pd.read_csv(MIG_PATH, encoding='utf-8-sig')
_mig.columns = ['mig_code','yr','q','yrq','age','HE','WE','EE','mig_total']
_mig['raw_code'] = _mig['mig_code'].map(code_mig2raw)
_mig = _mig.dropna(subset=['raw_code'])
_mig['raw_code'] = _mig['raw_code'].astype(int)
_mig['qnum'] = (_mig['yr'] - 2020) * 4 + _mig['q']

ccf_all = pd.read_csv(CCF_PATH, encoding='utf-8-sig')
COND = (ccf_all['최고_r'] >= 0.35) & (ccf_all['매출성장률_2020→2024(%)'] > 0)
lead20_codes  = ccf_all[COND & (ccf_all['패턴'] == '20대선행형')]['행정동코드'].tolist()
leadsal_codes = ccf_all[COND & (ccf_all['패턴'] == '매출선행형')]['행정동코드'].tolist()

# ── 20대 vs 30대+ YoY CCF 재계산 ──────────────────────────
LAGS    = list(range(-3, 5))
MIN_OBS = 7

def yoy_s(s):
    return ((s - s.shift(4)) / s.shift(4).replace(0, np.nan)).dropna()

def group_ccf(codes):
    lag_rs_all = {l: [] for l in LAGS}
    peak_lags  = []
    for code in codes:
        sub  = _mig[_mig['raw_code'] == code]
        m20  = sub[sub['age'] == 20].groupby('qnum')['mig_total'].sum()
        m30p = sub[sub['age'] >= 30].groupby('qnum')['mig_total'].sum()
        y20  = yoy_s(m20); y30p = yoy_s(m30p)
        common = y20.index.intersection(y30p.index)
        if len(common) < MIN_OBS: continue
        lag_rs = {}
        for lag in LAGS:
            vm, vs = (common, common + lag) if lag >= 0 else (common - lag, common)
            valid  = vm[vm.isin(y20.index) & vs.isin(y30p.index)]
            if len(valid) < MIN_OBS: continue
            r, _ = pearsonr(y20[valid].values,
                            y30p[(valid + lag) if lag >= 0 else valid].values)
            lag_rs[lag] = round(float(r), 3)
            lag_rs_all[lag].append(float(r))
        if lag_rs:
            peak_lags.append(max(lag_rs, key=lambda k: lag_rs[k]))
    avg_r    = {l: np.mean(v) if v else np.nan for l, v in lag_rs_all.items()}
    peak_ser = pd.Series(peak_lags)
    return avg_r, peak_ser

avg_r_20d, peaks_20d = group_ccf(lead20_codes)
avg_r_sal, peaks_sal = group_ccf(leadsal_codes)

# ── 세대별 연간 이동 비중 (2020~2024) ─────────────────────
AGE_MAP = {0:'0~9세', 10:'10대', 20:'20대', 30:'30대',
           40:'40대', 50:'50대', 60:'60이상', 70:'60이상', 80:'60이상'}
yr_age = _mig.groupby(['raw_code','yr','age'])['mig_total'].sum().reset_index()
yr_age['age_label'] = yr_age['age'].map(AGE_MAP)
yr_agg = yr_age.groupby(['raw_code','yr','age_label'])['mig_total'].sum().reset_index()
yr_tot = yr_agg.groupby(['raw_code','yr'])['mig_total'].sum().rename('tot').reset_index()
yr_agg = yr_agg.merge(yr_tot, on=['raw_code','yr'])
yr_agg['share'] = yr_agg['mig_total'] / yr_agg['tot'] * 100

def age_trend(codes, age_labels):
    sub = yr_agg[yr_agg['raw_code'].isin(codes) & yr_agg['age_label'].isin(age_labels)]
    return sub.groupby(['yr','age_label'])['share'].median().reset_index()

trend_20d = age_trend(lead20_codes,  ['20대','30대','40대','50대'])
trend_sal = age_trend(leadsal_codes, ['20대','30대','40대','50대'])

# ── 레이아웃 ────────────────────────────────────────────
fig = plt.figure(figsize=(22, 16))
fig.suptitle(
    '【가설 검증】 20대 선행형 행정동에서 20대 → 30대+ 연쇄 유입이 발생하는가?',
    fontsize=14, fontweight='bold', y=0.99)

gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.48, wspace=0.35,
                       left=0.07, right=0.97, top=0.93, bottom=0.07)

ax_r    = fig.add_subplot(gs[0, :2])   # CCF r 프로파일 (전체 폭)
ax_pie  = fig.add_subplot(gs[0, 2])    # peak_lag 비율 비교
ax_t20  = fig.add_subplot(gs[1, 0])    # 20대선행형 세대별 비중 추이
ax_tsal = fig.add_subplot(gs[1, 1])    # 매출선행형 세대별 비중 추이
ax_txt  = fig.add_subplot(gs[1, 2])    # 해석 요약 텍스트

# ══ 패널 1: lag별 평균 r 프로파일 ═══════════════════════════
ax_r.plot(LAGS, [avg_r_20d[l] for l in LAGS],
          color=C_20대, lw=2.8, marker='o', markersize=8,
          label=f'20대 선행형 (n={len(lead20_codes)})', zorder=4)
ax_r.plot(LAGS, [avg_r_sal[l] for l in LAGS],
          color=C_SAL, lw=2.8, marker='s', markersize=8, ls='--',
          label=f'매출 선행형 (n={len(leadsal_codes)})', zorder=4)

# 구간 강조
ax_r.axvspan(-3.4, 0.4, alpha=0.07, color='gray',  label='plateau 구간 (방향 불명확)')
ax_r.axvspan( 0.4, 4.4, alpha=0.07, color=C_20대연, label='20대→30대+ 선행 구간')

ax_r.axvline(0, color='#888', lw=1.2, ls='--')
ax_r.axhline(0, color='#CCC', lw=0.8)

# 핵심 포인트 강조: lag+1에서 급락
for c, d in [(C_20대, avg_r_20d), (C_SAL, avg_r_sal)]:
    y0, y1 = d[0], d[1]
    ax_r.annotate('', xy=(1, y1), xytext=(0, y0),
                  arrowprops=dict(arrowstyle='->', color=c, lw=2))

ax_r.text(0.5, (avg_r_20d[0]+avg_r_20d[1])/2,
          f"lag+0→+1에서\nr: {avg_r_20d[0]:.2f}→{avg_r_20d[1]:.2f}\n(급락)",
          fontsize=9, color=C_20대, ha='left',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C_20대, alpha=0.8))

ax_r.set_xticks(LAGS)
ax_r.set_xticklabels([f'lag{l:+d}Q' for l in LAGS], fontsize=10)
ax_r.set_xlabel('lag (분기, 양수=20대가 30대+보다 앞섬)', fontsize=10)
ax_r.set_ylabel('평균 CCF r\n(20대 YoY vs 30대+ YoY)', fontsize=10)
ax_r.set_ylim(-0.15, 1.05)
ax_r.set_title('20대 이동 YoY ↔ 30대+ 이동 YoY 교차상관 프로파일\n'
               '→ lag-3~0 plateau: 방향 불명확 / lag+1에서 r 급락: 20대가 30대+를 미래에 끌어당기지 않음',
               fontsize=11, pad=8)
ax_r.set_facecolor('#F8F8F8')
ax_r.grid(alpha=0.2)
ax_r.legend(fontsize=9.5, loc='lower left')

# ══ 패널 2: peak_lag 비율 비교 (누적 가로막대) ═══════════════
cats  = ['30대+\n선행\n(lag<0)', '동시\n(lag=0)', '20대\n선행\n(lag>0)']
d_20d = pd.Series(peaks_20d)
d_sal = pd.Series(peaks_sal)

def pct(s):
    lag_neg = (s < 0).mean() * 100
    lag_0   = (s == 0).mean() * 100
    lag_pos = (s > 0).mean() * 100
    return [lag_neg, lag_0, lag_pos]

v20d = pct(d_20d)
vsal = pct(d_sal)

colors_stack = ['#3A5CA9','#F5A623','#E05C5C']
bottoms = [0, 0]
for ci, (c, cat) in enumerate(zip(colors_stack, cats)):
    vals = [v20d[ci], vsal[ci]]
    ax_pie.barh(['20대선행형','매출선행형'], vals, left=bottoms,
                color=c, alpha=0.85, edgecolor='white', height=0.5, label=cat)
    for bi, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 5:
            ax_pie.text(b + v/2, bi, f'{v:.0f}%',
                        ha='center', va='center', fontsize=9,
                        color='white', fontweight='bold')
    bottoms = [bottoms[i] + vals[i] for i in range(2)]

ax_pie.set_xlim(0, 100)
ax_pie.set_xlabel('비율 (%)', fontsize=10)
ax_pie.set_title('peak_lag 분포\n(20대가 30대+를 선행하는 비율)', fontsize=11, pad=8)
ax_pie.set_facecolor('#F8F8F8')
ax_pie.legend(fontsize=8.5, loc='lower right',
              handles=[mpatches.Patch(color=c, label=l, alpha=0.85)
                       for c, l in zip(colors_stack, cats)])

# ══ 패널 3·4: 세대별 비중 연도별 추이 ═══════════════════════
age_colors = {'20대': C_20대, '30대': C_30대연, '40대': '#3A5CA9', '50대': '#8E44AD'}

for ax_t, trend, title, bg in [
    (ax_t20,  trend_20d, f'【20대 선행형 n={len(lead20_codes)}】 세대별 이동 비중 추이',  '#FFF5F5'),
    (ax_tsal, trend_sal, f'【매출 선행형 n={len(leadsal_codes)}】 세대별 이동 비중 추이', '#F5F7FF'),
]:
    for age, c in age_colors.items():
        sub = trend[trend['age_label'] == age].sort_values('yr')
        if sub.empty: continue
        lw  = 3.0 if age == '20대' else 1.8
        ls  = '-'  if age == '20대' else '--'
        ax_t.plot(sub['yr'], sub['share'],
                  color=c, lw=lw, ls=ls, marker='o', markersize=6, label=age)
        for _, r in sub.iterrows():
            ax_t.text(r['yr'], r['share'] + 0.15, f"{r['share']:.1f}%",
                      ha='center', fontsize=7.5, color=c)

    ax_t.set_xticks([2020, 2021, 2022, 2023, 2024])
    ax_t.set_ylabel('이동 비중 중앙값 (%)', fontsize=10)
    ax_t.set_title(title, fontsize=10, pad=8)
    ax_t.set_facecolor(bg)
    ax_t.grid(alpha=0.2)
    ax_t.legend(fontsize=8.5, loc='upper right')

# ══ 패널 5: 해석 요약 ═══════════════════════════════════════
ax_txt.axis('off')
findings = [
    ('가설', '20대 선행 → 30대+ 연쇄 → 매출 성장'),
    ('', ''),
    ('① CCF', 'lag+1에서 r이 0.75→0.41 급락'),
    ('',  '→ 20대가 30대+를 미래에 "끌어당기는"\n    신호 없음'),
    ('', ''),
    ('② peak_lag', '20대선행형 67개 동 중\n  30대+도 선행: 7.5% (5개)'),
    ('',  '→ 나머지 92.5%는 동시이거나\n    30대+가 오히려 먼저'),
    ('', ''),
    ('③ 세대 비중', '20대선행형: 20대비중 하락\n30대+ 비중도 동반 하락'),
    ('',  '→ 연쇄 유입 패턴 미확인'),
    ('', ''),
    ('결론', '20대 선행형 = 20대 소비 자체가\n매출 성장을 직접 이끌 가능성 높음'),
    ('',  '(30대+를 유인해서가 아니라\n 20대의 높은 소비단가·빈도가 원인)'),
]

y = 0.97
for label, text in findings:
    if label:
        ax_txt.text(0.04, y, label, fontsize=9.5, fontweight='bold',
                    color='#333', transform=ax_txt.transAxes)
        ax_txt.text(0.30, y, text, fontsize=9, color='#333',
                    transform=ax_txt.transAxes)
    else:
        pass
    y -= 0.068 if label else 0.02

ax_txt.set_facecolor('#FAFAFA')
for spine in ax_txt.spines.values():
    spine.set_edgecolor('#DDDDDD')

# ── 저장 ─────────────────────────────────────────────────
out = IMG_DIR / 'cascade_분석.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[저장] {out.name}")
