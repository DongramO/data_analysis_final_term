# -*- coding: utf-8 -*-
"""
통합 대시보드 2종 생성
- dashboard_A_signals.png : 분석 근거 (레퍼런스 패턴 → 선행신호 → 생애주기)
- dashboard_B_candidates.png : 후보군 비교 (모든 검증 신호 한 화면)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 팔레트 ─────────────────────────────────────────────────────────────────
C_REF   = ['#1565C0','#457B9D','#2A9D8F','#E9C46A']   # 레퍼런스 4개 동
C_LEAD  = '#1565C0'   # 선행 지표
C_LAG   = '#6c757d'   # 후행 지표
C_CAND  = '#457B9D'   # 후보군

# ── 데이터 로드 ────────────────────────────────────────────────────────────
tsm = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig')
ml_tr = pd.read_csv('data/main_data/ml_dataset_quarterly_train.csv', encoding='utf-8-sig')
ml_pr = pd.read_csv('data/main_data/ml_dataset_quarterly_predict.csv', encoding='utf-8-sig')
ml = pd.concat([ml_tr, ml_pr], ignore_index=True)
blog = pd.read_csv('data/main_data/블로그트렌드_상권.csv', encoding='utf-8-sig')
naver = pd.read_csv('data/main_data/naver_trend_candidates.csv', encoding='utf-8-sig')
naver['period'] = pd.to_datetime(naver['period'])

# ── 레퍼런스 정의 ──────────────────────────────────────────────────────────
from report_config import REFERENCE_DONGS, load_v3, phase35_top15

REF_DONGS = REFERENCE_DONGS
REF_NAMES = REFERENCE_DONGS.copy()
REF_NAVER = ['방배', '서촌', '합정', '성수']

_v3d = load_v3()
CAND_DONGS = phase35_top15(_v3d)
CAND_SIMS = _v3d.set_index('행정동').loc[CAND_DONGS, '코사인유사도'].tolist()

# 블로그 매핑 (후보 행정동 → blog 상권명)
BLOG_MAP = {'역삼2동':'역삼','문래동':'문래','서초1동':'서초','장안1동':'장안'}
# 네이버 매핑 (후보 행정동 → naver group명)
NAVER_MAP = {'역삼2동':'역삼2동','문래동':'문래동','서초1동':'서초동','장안1동':'장안동',
             '아현동':'아현동','효창동':'효창동','교남동':'교남동','도곡1동':'도곡동',
             '양재2동':'양재동','풍납2동':'풍납동'}

# ── 보조 함수 ──────────────────────────────────────────────────────────────
def get_tsm_dong(dong):
    return tsm[tsm['행정동'] == dong].sort_values('연도')

def get_ml_dong(dong):
    return ml[ml.iloc[:,1] == dong].sort_values([ml.columns[2], ml.columns[3]])

def naver_recent_yoy(grp_name, n=6):
    sub = naver[naver['group'] == grp_name].sort_values('period')
    sub = sub.copy()
    sub['yoy'] = sub['ratio'].pct_change(12) * 100
    return sub.tail(n)['yoy'].mean()

def naver_norm_series(grp_name):
    sub = naver[naver['group'] == grp_name].sort_values('period').copy()
    base = sub[sub['period'].dt.year.isin([2020,2021])]['ratio'].mean()
    base = max(base, 1.0)
    sub['roll3'] = sub['ratio'].rolling(3, min_periods=1).mean()
    sub['norm'] = sub['roll3'] / base
    return sub


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD A — 분석 근거
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 15))
fig.patch.set_facecolor('#FAFAFA')
gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.50, wspace=0.38,
                       left=0.06, right=0.97, top=0.93, bottom=0.05)

fig.suptitle('서울 트렌드 상권 분석 — 근거 대시보드\n"2030이 먼저 이동하고, 주말형 상권화가 따라온다"',
             fontsize=15, fontweight='bold', y=0.98)

# ─────────────────────────────────────────────────────────────
# A1 (row0, col0-1): 레퍼런스 4개 동 분위 궤적
# ─────────────────────────────────────────────────────────────
ax_a1 = fig.add_subplot(gs[0, 0:2])

years = [2020,2021,2022,2023,2024]
for i, (dong, name) in enumerate(zip(REF_DONGS, REF_NAMES)):
    d = get_tsm_dong(dong)
    if len(d) == 0:
        continue
    y_vals = d.set_index('연도').reindex(years)['복합점수_분위'].values
    ax_a1.plot(years, y_vals, 'o-', color=C_REF[i], lw=2.5, ms=7, label=name, zorder=5)
    ax_a1.fill_between(years, y_vals, alpha=0.07, color=C_REF[i])
    ax_a1.annotate(f'{y_vals[-1]:.2f}', xy=(2024, y_vals[-1]),
                   xytext=(5,0), textcoords='offset points',
                   fontsize=8.5, color=C_REF[i], fontweight='bold', va='center')

# 2030이 이탈하기 시작하는 기준선 (0.70)
ax_a1.axhline(0.70, color='#888', lw=1, ls='--', alpha=0.5)
ax_a1.text(2020.05, 0.72, '상위 30% 기준선', fontsize=8, color='#888')

ax_a1.set_title('① 레퍼런스 상권 — 급부상 궤적 (복합점수 분위)', fontsize=11, fontweight='bold')
ax_a1.set_ylabel('복합점수 분위 (0~1)', fontsize=9)
ax_a1.set_xticks(years)
ax_a1.legend(fontsize=8.5, loc='upper left', ncol=2)
ax_a1.grid(alpha=0.3)
ax_a1.set_ylim(0, 1.05)
ax_a1.set_xlim(2019.7, 2024.5)

# ─────────────────────────────────────────────────────────────
# A2 (row0, col2): 레퍼런스 시그니처 — 선행 vs 후행
# ─────────────────────────────────────────────────────────────
ax_a2 = fig.add_subplot(gs[0, 2])

sig_feats = [
    ('평일_비중', +0.752, '선행'),
    ('2030_이동비율_변화', +0.535, '선행'),
    ('라이프스타일_비중', +0.299, '선행'),
    ('HE_비율', +0.235, '선행'),
    ('트렌드시간대_비중', -0.698, '후행'),
    ('방문형_강도', -0.690, '후행'),
    ('주말_비중', -0.671, '후행'),
]
feat_names = [f[0].replace('_비중','').replace('_변화','↑').replace('_','·') for f in sig_feats]
zvals = [f[1] for f in sig_feats]
types = [f[2] for f in sig_feats]
colors_sig = [C_LEAD if t=='선행' else C_LAG for t in types]

bars = ax_a2.barh(feat_names, zvals, color=colors_sig, alpha=0.85, height=0.7)
ax_a2.axvline(0, color='black', lw=1)
ax_a2.set_xlabel('Z-score (급부상 전 공통 패턴)', fontsize=8.5)
ax_a2.set_title('② 레퍼런스 시그니처\n(급부상 전 공통 특징)', fontsize=11, fontweight='bold')

# 선행/후행 레이블
from matplotlib.patches import Patch
leg = [Patch(color=C_LEAD, label='선행 신호 (+)'),
       Patch(color=C_LAG,  label='후행/아직 낮음 (−)')]
ax_a2.legend(handles=leg, fontsize=8, loc='lower right')
ax_a2.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, zvals):
    ax_a2.text(val + (0.02 if val>=0 else -0.02),
               bar.get_y()+bar.get_height()/2,
               f'{val:+.3f}', va='center', ha='left' if val>=0 else 'right', fontsize=8)

# ─────────────────────────────────────────────────────────────
# A3 (row1, col0-1): 선행 vs 후행 시계열 비교 (레퍼런스 4개 동)
# ─────────────────────────────────────────────────────────────
ax_a3a = fig.add_subplot(gs[1, 0])
ax_a3b = fig.add_subplot(gs[1, 1])

for i, dong in enumerate(REF_DONGS):
    d = get_ml_dong(dong)
    if len(d) == 0:
        continue
    # 연도별 평균
    ann = d.groupby(d.columns[2])[[d.columns[76], d.columns[83], d.columns[34]]].mean()
    ann.columns = ['2030이동비율','2030이동변화','트렌드시간대']
    yr = ann.index.values

    ax_a3a.plot(yr, ann['2030이동변화'].values, 'o-', color=C_REF[i], lw=2, ms=6,
                label=REF_NAMES[i])
    ax_a3b.plot(yr, ann['트렌드시간대'].values, 'o-', color=C_REF[i], lw=2, ms=6,
                label=REF_NAMES[i])

ax_a3a.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
ax_a3a.set_title('③ 선행 신호: 2030 이동비율 변화\n(↑급부상 전 2030이 먼저 유입)', fontsize=10, fontweight='bold',
                 color=C_LEAD)
ax_a3a.set_ylabel('전년대비 변화 (%p)', fontsize=9)
ax_a3a.legend(fontsize=7.5, ncol=2)
ax_a3a.grid(alpha=0.3)
ax_a3a.set_xticks([2020,2021,2022,2023,2024])

ax_a3b.set_title('④ 후행 신호: 트렌드시간대 비중\n(↑ 이후 주말형 상권으로 전환)', fontsize=10, fontweight='bold',
                 color=C_LAG)
ax_a3b.set_ylabel('트렌드시간대 비중', fontsize=9)
ax_a3b.legend(fontsize=7.5, ncol=2)
ax_a3b.grid(alpha=0.3)
ax_a3b.set_xticks([2020,2021,2022,2023,2024])

# ─────────────────────────────────────────────────────────────
# A4 (row1, col2): 성수 서브동별 2030 이동비율 — 생애주기 시각화
# ─────────────────────────────────────────────────────────────
ax_a4 = fig.add_subplot(gs[1, 2])

seongsu_dongs = ['성수1가1동','성수1가2동','성수2가1동','성수2가3동']
ss_colors = ['#1565C0','#457B9D','#2A9D8F','#F4A261']
ss_labels = ['성수1가1동\n(2022 고점)','성수1가2동\n(2023 고점)','성수2가1동\n(2024 상승)','성수2가3동\n(2023 고점)']

for i, dong in enumerate(seongsu_dongs):
    d = get_ml_dong(dong)
    if len(d) == 0:
        continue
    ann = d.groupby(d.columns[2])[d.columns[76]].mean()
    ax_a4.plot(ann.index.values, ann.values, 'o-',
               color=ss_colors[i], lw=2.2, ms=6, label=ss_labels[i])

ax_a4.set_title('⑤ 성수동 서브동별 2030 이동비율\n(생애주기: 유입 → 고점 → 희석)', fontsize=10, fontweight='bold')
ax_a4.set_ylabel('2030 이동비율 (%)', fontsize=9)
ax_a4.set_xticks([2020,2021,2022,2023,2024])
ax_a4.legend(fontsize=7.5)
ax_a4.grid(alpha=0.3)

# 화살표 주석 "2030 이탈 후 다음 지역으로"
ax_a4.annotate('성수1가1동\n2022 고점 후\n하락 중',
               xy=(2024, 30.4), xytext=(2022.5, 26),
               fontsize=7.5, color='#1565C0',
               arrowprops=dict(arrowstyle='->', color='#1565C0', lw=1.2))

# ─────────────────────────────────────────────────────────────
# A5 (row2, col0-2): 네이버 트렌드 — 레퍼런스 급등 시점 검증
# ─────────────────────────────────────────────────────────────
naver_ref_names = ['성수','여의','문래','신대방']
naver_ref_labels = ['성수\n(2022.07 급등)','여의(더현대)\n(2021.03 급등)','문래\n(미돌파)','신대방\n(미돌파)']
naver_colors = ['#1565C0','#457B9D','#2A9D8F','#E9C46A']
surge_marks = {'성수':'2022-07','여의':'2021-03'}

axes_a5 = [fig.add_subplot(gs[2, j]) for j in range(3)]

# 3개 subplot에 4개 시리즈 (성수/여의/문래 개별, 신대방은 겹쳐서)
for j, grp in enumerate(naver_ref_names[:3]):
    ax = axes_a5[j]
    sub = naver_norm_series(grp)
    col = naver_colors[j]

    ax.fill_between(sub['period'], sub['norm'], alpha=0.15, color=col)
    ax.plot(sub['period'], sub['norm'], color=col, lw=2.5, label=f'{grp} (3M롤링)')
    ax.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(1.8, color='#1565C0', lw=0.8, ls=':', alpha=0.6, label='급등 기준 1.8×')

    if grp in surge_marks:
        sm = pd.Timestamp(surge_marks[grp])
        ax.axvline(sm, color='#1565C0', lw=2, ls='--', alpha=0.9)
        ax.text(sm, sub['norm'].max()*0.92, sm.strftime('%y.%m'),
                fontsize=9, color='#1565C0', fontweight='bold', ha='left')

    ax.set_title(f'⑥ {naver_ref_labels[j]}', fontsize=9.5, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y.%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=40, ha='right', fontsize=7.5)
    ax.set_ylabel('2020-21 대비 배율', fontsize=8)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

plt.savefig('image/dashboard_A_signals.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print('저장: image/dashboard_A_signals.png')


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD B — 후보군 비교
# ══════════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(22, 18))
fig2.patch.set_facecolor('#FAFAFA')
gs2 = gridspec.GridSpec(3, 3, figure=fig2,
                        hspace=0.50, wspace=0.38,
                        left=0.06, right=0.97, top=0.94, bottom=0.04)

fig2.suptitle('차세대 트렌드 상권 후보군 비교 대시보드\n(패턴 유사도 · 블로그 · 네이버 · 임대료 · 분위 궤적)',
              fontsize=15, fontweight='bold', y=0.98)

# ─────────────────────────────────────────────────────────────
# B1 (row0, col0): 패턴 유사도 + 네이버 최근YoY 복합 막대
# ─────────────────────────────────────────────────────────────
ax_b1 = fig2.add_subplot(gs2[0, 0])

cands_short = ['양재2동','역삼2동','문래동','도곡1동','서초1동',
               '구로2동','풍납2동','장안1동','문정2동','교남동',
               '원효로1동','무악동','원효로2동','아현동','효창동'][::-1]
sims_rev = CAND_SIMS[::-1]
cand_colors_bar = [plt.cm.RdYlGn(s) for s in sims_rev]

bars_b1 = ax_b1.barh(cands_short, sims_rev, color=cand_colors_bar, alpha=0.85, height=0.7)
ax_b1.axvline(0.75, color='#888', lw=1, ls='--', alpha=0.6)
ax_b1.set_title('① 패턴 유사도\n(레퍼런스 급부상 전 패턴과의 일치도)', fontsize=10, fontweight='bold')
ax_b1.set_xlabel('코사인 유사도')
ax_b1.set_xlim(0.5, 1.0)
for bar, val in zip(bars_b1, sims_rev):
    ax_b1.text(val+0.002, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center', fontsize=7.5)
ax_b1.grid(axis='x', alpha=0.3)

# ─────────────────────────────────────────────────────────────
# B2 (row0, col1): 블로그 + 네이버 최근YoY 비교 (데이터 있는 것만)
# ─────────────────────────────────────────────────────────────
ax_b2 = fig2.add_subplot(gs2[0, 1])

# 블로그 성장률 (growth_20_25)
blog_data = {}
for dong, bname in BLOG_MAP.items():
    row = blog[blog['area'] == bname]
    if len(row) > 0:
        blog_data[dong] = float(row.iloc[0]['growth_20_25'])

# 네이버 최근 YoY
naver_yoy = {}
for dong, nname in NAVER_MAP.items():
    yoy = naver_recent_yoy(nname)
    if not np.isnan(yoy):
        naver_yoy[dong] = round(yoy, 1)

# 두 지표 모두 있는 것: 역삼2동, 문래동, 서초1동, 장안1동
both = sorted(set(blog_data) | set(naver_yoy))
x = np.arange(len(both))
w = 0.4

blog_vals = [blog_data.get(d, None) for d in both]
yoy_vals  = [naver_yoy.get(d, None) for d in both]

# 블로그: 2020→2025 성장률(%)
b_vals_plot = [v if v is not None else 0 for v in blog_vals]
y_vals_plot = [v if v is not None else np.nan for v in yoy_vals]

ax_b2_twin = ax_b2.twinx()
b1 = ax_b2.bar(x - w/2, b_vals_plot, width=w, color='#E9C46A', alpha=0.8, label='블로그 성장률 (2020→25, %)')
# filter out nan for naver
y_valid = [(xi, v) for xi, v in zip(x+w/2, y_vals_plot) if not np.isnan(v)]
if y_valid:
    xi_arr, yv_arr = zip(*y_valid)
    b2 = ax_b2_twin.bar(list(xi_arr), list(yv_arr), width=w, color='#457B9D', alpha=0.8, label='네이버 최근YoY (%)')

ax_b2.set_xticks(x)
ax_b2.set_xticklabels([d.replace('동','') for d in both], fontsize=8.5)
ax_b2.set_ylabel('블로그 성장률 (%)', fontsize=8.5, color='#E9C46A')
ax_b2_twin.set_ylabel('네이버 최근YoY (%)', fontsize=8.5, color='#457B9D')
ax_b2_twin.axhline(0, color='#457B9D', lw=0.5, ls='--', alpha=0.4)
ax_b2.set_title('② 블로그 성장률(2020→25)\nvs 네이버 최근 6개월 YoY', fontsize=10, fontweight='bold')
lines1, labels1 = ax_b2.get_legend_handles_labels()
lines2, labels2 = ax_b2_twin.get_legend_handles_labels()
ax_b2.legend(lines1+lines2, labels1+labels2, fontsize=7.5, loc='upper left')
ax_b2.grid(axis='y', alpha=0.3)

# ─────────────────────────────────────────────────────────────
# B3 (row0, col2): 네이버 최근YoY 전체 후보 순위
# ─────────────────────────────────────────────────────────────
ax_b3 = fig2.add_subplot(gs2[0, 2])

all_naver_yoy = []
for dong in cands_short:
    nname = NAVER_MAP.get(dong)
    if nname:
        yoy = naver_recent_yoy(nname)
        all_naver_yoy.append((dong, yoy if not np.isnan(yoy) else 0))
    else:
        all_naver_yoy.append((dong, np.nan))

all_naver_yoy_sorted = sorted([x for x in all_naver_yoy if not np.isnan(x[1])], key=lambda x: x[1])
names_yoy = [x[0] for x in all_naver_yoy_sorted]
vals_yoy  = [x[1] for x in all_naver_yoy_sorted]
cols_yoy  = ['#2ecc71' if v >= 0 else '#1565C0' for v in vals_yoy]

bars_b3 = ax_b3.barh(names_yoy, vals_yoy, color=cols_yoy, alpha=0.85, height=0.7)
ax_b3.axvline(0, color='black', lw=0.8)
ax_b3.set_title('③ 네이버 최근 6개월 YoY\n(전년 동월 대비 검색 변화율)', fontsize=10, fontweight='bold')
ax_b3.set_xlabel('전년 동월 대비 (%)')
for bar, val in zip(bars_b3, vals_yoy):
    ax_b3.text(val+(0.5 if val>=0 else -0.5), bar.get_y()+bar.get_height()/2,
               f'{val:+.1f}%', va='center', ha='left' if val>=0 else 'right', fontsize=7.5)
ax_b3.grid(axis='x', alpha=0.3)

# ─────────────────────────────────────────────────────────────
# B4 (row1, col0-1): 후보 상위 10개 + 레퍼런스 분위 궤적 비교
# ─────────────────────────────────────────────────────────────
ax_b4 = fig2.add_subplot(gs2[1, 0:2])

years = [2020,2021,2022,2023,2024]

# 레퍼런스 (얇은 배경선)
for i, dong in enumerate(REF_DONGS):
    d = get_tsm_dong(dong)
    if len(d) == 0:
        continue
    y_vals = d.set_index('연도').reindex(years)['복합점수_분위'].values
    ax_b4.plot(years, y_vals, '--', color=C_REF[i], lw=1.8, alpha=0.55,
               label=f'[REF] {REF_NAMES[i]}')

# 후보 상위 10개 (굵은 선)
cand_palette = plt.cm.tab10(np.linspace(0, 1, 10))
for j, dong in enumerate(CAND_DONGS[:10]):
    d = get_tsm_dong(dong)
    if len(d) == 0:
        continue
    y_vals = d.set_index('연도').reindex(years)['복합점수_분위'].values
    ax_b4.plot(years, y_vals, 'o-', color=cand_palette[j], lw=2, ms=5,
               label=f'{dong}')
    # 2024 값 레이블
    if not np.isnan(y_vals[-1]):
        ax_b4.annotate(f'{y_vals[-1]:.2f}', xy=(2024, y_vals[-1]),
                       xytext=(4, 0), textcoords='offset points',
                       fontsize=7, color=cand_palette[j], va='center')

ax_b4.axhline(0.70, color='#888', lw=1, ls='--', alpha=0.5)
ax_b4.text(2020.05, 0.72, '상위 30% 기준선', fontsize=8, color='#888')
ax_b4.set_title('④ 후보군 vs 레퍼런스 분위 궤적 비교\n(점선=레퍼런스 / 실선=후보군, 현재 저분위가 급등 전 조건)', fontsize=10, fontweight='bold')
ax_b4.set_ylabel('복합점수 분위', fontsize=9)
ax_b4.set_xticks(years)
ax_b4.legend(fontsize=7, ncol=4, loc='upper left', framealpha=0.8)
ax_b4.grid(alpha=0.3)
ax_b4.set_ylim(-0.05, 1.05)
ax_b4.set_xlim(2019.7, 2024.7)

# ─────────────────────────────────────────────────────────────
# B5 (row1, col2): 임대료·공실률 (데이터 있는 후보)
# ─────────────────────────────────────────────────────────────
ax_b5 = fig2.add_subplot(gs2[1, 2])

# 임대료 + 공실 데이터 직접 정의 (validate_scorecard 기반)
rental_data = {
    '역삼2동':  {'임대료변화': 17.0, '공실률': 12.1},
    '장안1동':  {'임대료변화': 14.1, '공실률': 8.1},
    '서초1동':  {'임대료변화': None, '공실률': None},
    '아현동':   {'임대료변화': None, '공실률': 5.7},
    '효창동':   {'임대료변화': -5.4, '공실률': 1.1},
    '양재2동':  {'임대료변화': 17.0, '공실률': 3.1},
}

r_dongs = [d for d, v in rental_data.items() if v['공실률'] is not None]
r_vacancy = [rental_data[d]['공실률'] for d in r_dongs]
r_rent_chg = [rental_data[d]['임대료변화'] if rental_data[d]['임대료변화'] else 0 for d in r_dongs]

x_r = np.arange(len(r_dongs))
w_r = 0.4
ax_b5_twin = ax_b5.twinx()

bars_r1 = ax_b5.bar(x_r - w_r/2, r_vacancy, width=w_r, color='#1565C0', alpha=0.75, label='공실률 (%) 2024')
bars_r2 = ax_b5_twin.bar(x_r + w_r/2, r_rent_chg, width=w_r,
                          color=['#2ecc71' if v>=0 else '#1565C0' for v in r_rent_chg],
                          alpha=0.75, label='임대료 변화율 (%)')

ax_b5.axhline(5, color='#1565C0', lw=0.8, ls='--', alpha=0.5)
ax_b5.set_xticks(x_r)
ax_b5.set_xticklabels([d.replace('동','') for d in r_dongs], fontsize=8.5)
ax_b5.set_ylabel('공실률 (%)', fontsize=8.5, color='#1565C0')
ax_b5_twin.set_ylabel('임대료 변화율 (%)', fontsize=8.5, color='green')
ax_b5.set_title('⑤ 임대료·공실률\n(낮을수록 상업화 초기 가능성)', fontsize=10, fontweight='bold')

lines_r = [bars_r1, bars_r2]
labels_r = ['공실률 (%) 2024', '임대료 변화율 (%)']
ax_b5.legend(lines_r, labels_r, fontsize=7.5)
ax_b5.grid(axis='y', alpha=0.3)

# ─────────────────────────────────────────────────────────────
# B6 (row2, full): 종합 스코어카드 히트맵
# ─────────────────────────────────────────────────────────────
ax_b6 = fig2.add_subplot(gs2[2, :])

# 스코어카드 데이터 구성
score_dongs = ['문래동','서초1동','장안1동','역삼2동','효창동','양재2동','아현동',
               '교남동','도곡1동','풍납2동','구로2동','문정2동','무악동','원효로1동','원효로2동']

signals = {
    '패턴유사도': [0.830,0.809,0.775,0.870,0.724,0.879,0.728,0.771,0.826,0.776,0.795,0.773,0.754,0.760,0.752],
    '상승여지\n(1-분위)': [1-0.040,1-0.265,1-0.043,1-0.161,1-0.108,1-0.158,1-0.095,1-0.066,1-0.038,1-0.076,1-0.005,1-0.163,1-0.062,1-0.002,1-0.017],
    '블로그성장\n(정규화)': [1.0, 0.59, 0.43, 0.34, None, None, None, None, None, None, None, None, None, None, None],
    '네이버YoY\n(정규화)': [naver_recent_yoy(NAVER_MAP.get(d,'')) for d in score_dongs],
    '공실률(역)\n낮을수록↑': [None,None,1-8.1/20,1-12.1/20,1-1.1/20,1-3.1/20,1-5.7/20,None,None,None,None,None,None,None,None],
}

# 네이버 YoY 정규화 (최솟값~최댓값 0~1)
yoys = signals['네이버YoY\n(정규화)']
valid_yoys = [y for y in yoys if not np.isnan(y)]
if valid_yoys:
    mn, mx = min(valid_yoys), max(valid_yoys)
    rng = mx - mn if mx != mn else 1
    signals['네이버YoY\n(정규화)'] = [(y-mn)/rng if not np.isnan(y) else None for y in yoys]

# 히트맵 배열
signal_keys = list(signals.keys())
mat = np.full((len(signal_keys), len(score_dongs)), np.nan)
for r, key in enumerate(signal_keys):
    for c, val in enumerate(signals[key]):
        if val is not None and not np.isnan(val):
            mat[r, c] = val

im = ax_b6.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

ax_b6.set_xticks(range(len(score_dongs)))
ax_b6.set_xticklabels(score_dongs, fontsize=8.5, rotation=35, ha='right')
ax_b6.set_yticks(range(len(signal_keys)))
ax_b6.set_yticklabels(signal_keys, fontsize=8.5)
ax_b6.set_title('⑥ 종합 스코어카드 히트맵 (초록=강한 신호 / 빨강=약한 신호 / 회색=데이터 없음)',
                fontsize=10, fontweight='bold')

for r in range(mat.shape[0]):
    for c in range(mat.shape[1]):
        if not np.isnan(mat[r, c]):
            ax_b6.text(c, r, f'{mat[r,c]:.2f}', ha='center', va='center',
                       fontsize=7.5, color='black' if 0.3 < mat[r,c] < 0.8 else 'white')
        else:
            ax_b6.text(c, r, '—', ha='center', va='center', fontsize=8, color='#aaa')

plt.colorbar(im, ax=ax_b6, orientation='vertical', fraction=0.01, pad=0.01, label='신호 강도 (0~1)')

plt.savefig('image/dashboard_B_candidates.png', dpi=150, bbox_inches='tight',
            facecolor=fig2.get_facecolor())
plt.close()
print('저장: image/dashboard_B_candidates.png')
print('완료')
