# -*- coding: utf-8 -*-
"""
급부상 상권 세대별 유입 패턴 딥다이브
출력: image/emerging_분석.png
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT      = Path(__file__).resolve().parent
MASTER    = ROOT / 'data/main_data/트렌드매출_마스터.csv'
MIG_PATH  = ROOT / 'data/main_data/분기별_주말유입인구.csv'
MAP_PATH  = ROOT / 'data/main_data/코드매핑_유입인구_매출.csv'
CCF_PATH  = ROOT / 'data/_tmp_ccf_all.csv'
IMG_DIR   = ROOT / 'image'

C_EMRG  = '#E05C5C'   # 급부상
C_LGND  = '#3A5CA9'   # 기성
C_NORM  = '#95A5A6'   # 일반
C_20대  = '#E05C5C'
C_30대  = '#F5A623'
C_40대  = '#3A5CA9'
C_50대  = '#27AE60'

# ── 1. 마스터 파일 로드 + 급부상/기성 분류 ─────────────────
master = pd.read_csv(MASTER, encoding='utf-8-sig')
master.columns = ['dong','code','yr','trend_sales','total_sales','trend_ratio','biz_score',
                  'ratio_2030','fri_effect','weekend_ratio','night_ratio',
                  'composite','composite_qtile','tier','tier_label']

p20 = master[master['yr']==2020][['code','dong','composite_qtile','total_sales']].copy()
p24 = master[master['yr']==2024][['code','composite_qtile','total_sales']].copy()
p20.columns = ['code','dong','qtile_2020','sales_2020']
p24.columns = ['code','qtile_2024','sales_2024']
pvt = p20.merge(p24, on='code')
pvt['momentum'] = pvt['qtile_2024'] - pvt['qtile_2020']
pvt['sales_growth'] = (pvt['sales_2024'] - pvt['sales_2020']) / pvt['sales_2020'] * 100

mom_thr = pvt['momentum'].quantile(0.85)
emrg_codes = pvt[(pvt['qtile_2020'] < 0.60) & (pvt['momentum'] >= mom_thr) & (pvt['sales_growth'] > 0)]['code'].tolist()
lgnd_codes = pvt[pvt['qtile_2020'] >= 0.80]['code'].tolist()
norm_codes = [c for c in pvt['code'].tolist() if c not in emrg_codes and c not in lgnd_codes]

print(f"급부상: {len(emrg_codes)}, 기성: {len(lgnd_codes)}, 일반: {len(norm_codes)}")

# ── 2. 복합점수 분위 추이 (연도별) ───────────────────────────
def qtile_trend(codes):
    sub = master[master['code'].isin(codes)]
    return sub.groupby('yr')['composite_qtile'].median().reset_index()

trend_emrg = qtile_trend(emrg_codes)
trend_lgnd = qtile_trend(lgnd_codes)
trend_norm = qtile_trend(norm_codes)

# ── 3. CCF 패턴 분포 ─────────────────────────────────────────
ccf_all = pd.read_csv(CCF_PATH, encoding='utf-8-sig')

def pattern_pct(codes):
    sub = ccf_all[ccf_all['행정동코드'].isin(codes)]
    cnt = sub['패턴'].value_counts()
    total = len(sub)
    if total == 0:
        return {'20대선행형': 0, '동시형': 0, '매출선행형': 0}
    return {k: cnt.get(k, 0) / total * 100 for k in ['20대선행형', '동시형', '매출선행형']}

pct_emrg = pattern_pct(emrg_codes)
pct_lgnd = pattern_pct(lgnd_codes)
pct_norm = pattern_pct(norm_codes)

# ── 4. 세대별 이동 비중 (급부상 그룹, 연도별) ────────────────
_map = pd.read_csv(MAP_PATH, encoding='utf-8-sig')
_map.columns = ['mig_code','raw_code','gu','dong_map']
_map['raw_code'] = _map['raw_code'].astype('Int64')
code_mig2raw = _map.set_index('mig_code')['raw_code'].to_dict()

_mig = pd.read_csv(MIG_PATH, encoding='utf-8-sig')
_mig.columns = ['mig_code','yr','q','yrq','age','HE','WE','EE','mig_total']
_mig['raw_code'] = _mig['mig_code'].map(code_mig2raw)
_mig = _mig.dropna(subset=['raw_code'])
_mig['raw_code'] = _mig['raw_code'].astype(int)

AGE_MAP = {10:'10대', 20:'20대', 30:'30대', 40:'40대', 50:'50대',
           60:'60이상', 70:'60이상', 80:'60이상'}
_mig['age_label'] = _mig['age'].map(AGE_MAP)

yr_age = _mig.groupby(['raw_code','yr','age_label'])['mig_total'].sum().reset_index()
yr_tot = yr_age.groupby(['raw_code','yr'])['mig_total'].sum().rename('tot').reset_index()
yr_age = yr_age.merge(yr_tot, on=['raw_code','yr'])
yr_age['share'] = yr_age['mig_total'] / yr_age['tot'] * 100

def age_share_trend(codes, age_labels):
    sub = yr_age[yr_age['raw_code'].isin(codes) & yr_age['age_label'].isin(age_labels)]
    return sub.groupby(['yr','age_label'])['share'].median().reset_index()

trend_age_emrg = age_share_trend(emrg_codes, ['20대','30대','40대','50대'])
trend_age_lgnd = age_share_trend(lgnd_codes, ['20대','30대','40대','50대'])

# ── 5. 급부상 초기(2020-2021) vs 후기(2023-2024) 세대 비중 ──
def period_share(codes, yrs, age_labels):
    sub = yr_age[yr_age['raw_code'].isin(codes) & yr_age['yr'].isin(yrs) & yr_age['age_label'].isin(age_labels)]
    return sub.groupby('age_label')['share'].median().reset_index()

AGE_ORDER = ['20대','30대','40대','50대']
early_emrg = period_share(emrg_codes, [2020,2021], AGE_ORDER)
late_emrg  = period_share(emrg_codes, [2023,2024], AGE_ORDER)
early_lgnd = period_share(lgnd_codes, [2020,2021], AGE_ORDER)
late_lgnd  = period_share(lgnd_codes, [2023,2024], AGE_ORDER)

# ── 6. 급부상 상권 Top20 리스트 ─────────────────────────────
top20 = pvt[pvt['code'].isin(emrg_codes)].sort_values('momentum', ascending=False).head(20)
top20 = top20.merge(_map[['raw_code','gu']].rename(columns={'raw_code':'code'}), on='code', how='left')
top20['gu'] = top20['gu'].fillna('')

# ── 레이아웃 ─────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 20))
fig.suptitle(
    '【급부상 상권 딥다이브】 2020-2024 신흥 부상 상권의 세대별 유입 패턴',
    fontsize=15, fontweight='bold', y=0.99)

gs = gridspec.GridSpec(3, 3, figure=fig,
                       hspace=0.50, wspace=0.35,
                       left=0.06, right=0.98, top=0.95, bottom=0.05)

ax1 = fig.add_subplot(gs[0, :2])  # 급부상 Top20 바 차트
ax2 = fig.add_subplot(gs[0, 2])   # CCF 패턴 비교
ax3 = fig.add_subplot(gs[1, 0])   # 복합점수 분위 추이
ax4 = fig.add_subplot(gs[1, 1])   # 세대별 비중 추이 - 급부상
ax5 = fig.add_subplot(gs[1, 2])   # 세대별 비중 추이 - 기성
ax6 = fig.add_subplot(gs[2, 0])   # 초기 vs 후기 세대 비중 변화 (급부상)
ax7 = fig.add_subplot(gs[2, 1])   # 초기 vs 후기 세대 비중 변화 (기성)
ax8 = fig.add_subplot(gs[2, 2])   # 해석 요약

# ══ 패널 1: 급부상 Top20 ═══════════════════════════════════
top20_plot = top20.sort_values('momentum')
y_pos = np.arange(len(top20_plot))
bars_q20 = ax1.barh(y_pos, top20_plot['qtile_2020'], height=0.6,
                    color='#BDC3C7', alpha=0.85, label='2020 분위')
bars_q24 = ax1.barh(y_pos, top20_plot['qtile_2024'], height=0.6,
                    left=top20_plot['qtile_2020'],
                    color=C_EMRG, alpha=0.80, label='모멘텀(2020→2024)')

# 화살표 대신 막대 끝에 값 표시
for i, (_, row) in enumerate(top20_plot.iterrows()):
    ax1.text(row['qtile_2024'] + 0.01, i,
             f"{row['qtile_2024']:.2f}", va='center', fontsize=7.5, color='#333')
    ax1.text(-0.02, i, f"↑{row['momentum']:.2f}",
             va='center', ha='right', fontsize=7, color=C_EMRG, fontweight='bold')

labels = []
for _, row in top20_plot.iterrows():
    gu = f"({row['gu']})" if row['gu'] else ''
    labels.append(f"{row['dong']} {gu}")
ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels, fontsize=8)
ax1.set_xlim(-0.10, 1.10)
ax1.axvline(0.6, color='#888', lw=1, ls='--', alpha=0.5)
ax1.axvline(0.8, color='#333', lw=1, ls='--', alpha=0.5)
ax1.text(0.61, -0.8, '분위 0.60', fontsize=7.5, color='#888')
ax1.text(0.81, -0.8, '분위 0.80', fontsize=7.5, color='#333')
ax1.set_xlabel('복합점수 분위', fontsize=10)
ax1.set_title(f'급부상 상권 Top20 (모멘텀 순, n={len(emrg_codes)}개 전체)\n회색=2020 출발 분위 / 빨강=2024까지 상승폭',
              fontsize=11, pad=8)
ax1.set_facecolor('#F8F8F8')
ax1.grid(axis='x', alpha=0.2)
ax1.legend(fontsize=9, loc='lower right')

# ══ 패널 2: CCF 패턴 분포 비교 ═══════════════════════════════
groups = ['급부상', '기성', '일반']
pcts   = [pct_emrg, pct_lgnd, pct_norm]
pat_keys = ['20대선행형', '동시형', '매출선행형']
pat_colors = [C_20대, '#F5A623', C_40대]

bottoms = [0] * 3
for ci, (pk, pc) in enumerate(zip(pat_keys, pat_colors)):
    vals = [p[pk] for p in pcts]
    ax2.barh(groups, vals, left=bottoms, color=pc, alpha=0.85,
             edgecolor='white', height=0.5, label=pk)
    for gi, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 6:
            ax2.text(b + v/2, gi, f'{v:.0f}%',
                     ha='center', va='center', fontsize=9,
                     color='white', fontweight='bold')
    bottoms = [bottoms[i] + vals[i] for i in range(3)]

ax2.set_xlim(0, 100)
ax2.set_xlabel('비율 (%)', fontsize=10)
ax2.set_title('CCF 패턴 분포 비교\n(급부상 vs 기성 vs 일반)', fontsize=11, pad=8)
ax2.set_facecolor('#F8F8F8')
ax2.legend(fontsize=8.5, loc='lower right',
           handles=[mpatches.Patch(color=c, label=l, alpha=0.85)
                    for c, l in zip(pat_colors, pat_keys)])

# ══ 패널 3: 복합점수 분위 추이 ════════════════════════════════
for trend, label, c, ls, lw in [
    (trend_emrg, f'급부상 (n={len(emrg_codes)})', C_EMRG, '-', 2.8),
    (trend_lgnd, f'기성 (n={len(lgnd_codes)})',    C_LGND, '--', 2.0),
    (trend_norm, f'일반 (n={len(norm_codes)})',    C_NORM, ':', 1.8),
]:
    ax3.plot(trend['yr'], trend['composite_qtile'],
             color=c, lw=lw, ls=ls, marker='o', markersize=6, label=label)
    for _, r in trend.iterrows():
        ax3.text(r['yr'], r['composite_qtile'] + 0.015, f"{r['composite_qtile']:.2f}",
                 ha='center', fontsize=7.5, color=c)

ax3.set_xticks([2020,2021,2022,2023,2024])
ax3.set_ylabel('복합점수 분위 (중앙값)', fontsize=10)
ax3.set_title('복합점수 분위 연도별 추이\n(급부상 vs 기성 vs 일반 비교)', fontsize=11, pad=8)
ax3.set_facecolor('#F8F8F8')
ax3.grid(alpha=0.2)
ax3.legend(fontsize=9)
ax3.set_ylim(0.0, 1.05)

# ══ 패널 4·5: 세대별 비중 추이 ══════════════════════════════
age_cfg = [('20대', C_20대, '-', 2.8), ('30대', C_30대, '--', 1.8),
           ('40대', C_40대, '-.', 1.8), ('50대', C_50대, ':', 1.8)]

for ax_t, trend_age, title, bg in [
    (ax4, trend_age_emrg, f'【급부상 n={len(emrg_codes)}】 세대별 이동 비중 추이', '#FFF5F5'),
    (ax5, trend_age_lgnd, f'【기성 n={len(lgnd_codes)}】 세대별 이동 비중 추이',    '#F5F7FF'),
]:
    for age, c, ls, lw in age_cfg:
        sub = trend_age[trend_age['age_label'] == age].sort_values('yr')
        if sub.empty: continue
        ax_t.plot(sub['yr'], sub['share'], color=c, lw=lw, ls=ls,
                  marker='o', markersize=6, label=age)
        for _, r in sub.iterrows():
            ax_t.text(r['yr'], r['share'] + 0.2, f"{r['share']:.1f}%",
                      ha='center', fontsize=7.5, color=c)
    ax_t.set_xticks([2020,2021,2022,2023,2024])
    ax_t.set_ylabel('이동 비중 중앙값 (%)', fontsize=10)
    ax_t.set_title(title, fontsize=10, pad=8)
    ax_t.set_facecolor(bg)
    ax_t.grid(alpha=0.2)
    ax_t.legend(fontsize=8.5, loc='upper right')

# ══ 패널 6·7: 초기(2020-21) vs 후기(2023-24) 세대 비중 변화 ═
for ax_b, early, late, grp_name, bg in [
    (ax6, early_emrg, late_emrg, '급부상', '#FFF5F5'),
    (ax7, early_lgnd, late_lgnd, '기성',   '#F5F7FF'),
]:
    x  = np.arange(len(AGE_ORDER))
    w  = 0.35
    v_e = [early[early['age_label']==a]['share'].values[0] if len(early[early['age_label']==a]) else 0 for a in AGE_ORDER]
    v_l = [late[late['age_label']==a]['share'].values[0]   if len(late[late['age_label']==a])  else 0 for a in AGE_ORDER]

    age_colors_bar = [C_20대, C_30대, C_40대, C_50대]
    ax_b.bar(x - w/2, v_e, width=w, color=age_colors_bar, alpha=0.55,
             edgecolor='white', linewidth=1.5)
    ax_b.bar(x + w/2, v_l, width=w, color=age_colors_bar, alpha=0.9,
             edgecolor='white', linewidth=1.5)

    for i, (ve, vl) in enumerate(zip(v_e, v_l)):
        diff = vl - ve
        sign = '+' if diff >= 0 else ''
        color = '#27AE60' if diff > 0 else '#E05C5C'
        ax_b.text(i, max(ve, vl) + 0.4, f'{sign}{diff:.1f}%p',
                  ha='center', fontsize=8.5, color=color, fontweight='bold')

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(AGE_ORDER, fontsize=10)
    ax_b.set_ylabel('이동 비중 중앙값 (%)', fontsize=10)
    ax_b.set_title(f'【{grp_name}】 세대별 비중: 초기(2020-21) vs 후기(2023-24)',
                   fontsize=10, pad=8,
                   color=C_EMRG if grp_name=='급부상' else C_LGND, fontweight='bold')
    ax_b.set_facecolor(bg)
    ax_b.grid(axis='y', alpha=0.2)
    legend_e = mpatches.Patch(facecolor='#999', alpha=0.55, label='2020-21 초기')
    legend_l = mpatches.Patch(facecolor='#999', alpha=0.9,  label='2023-24 후기')
    ax_b.legend(handles=[legend_e, legend_l], fontsize=8.5)

# ══ 패널 8: 해석 요약 ═══════════════════════════════════════
ax8.axis('off')

# 실제 계산값 추출
emrg_p20d = pct_emrg.get('20대선행형', 0)
emrg_pdyn = pct_emrg.get('동시형', 0)
emrg_psal = pct_emrg.get('매출선행형', 0)
lgnd_p20d = pct_lgnd.get('20대선행형', 0)

# 초기→후기 20대 비중 변화
v20_e = early_emrg[early_emrg['age_label']=='20대']['share'].values
v20_l = late_emrg[late_emrg['age_label']=='20대']['share'].values
delta20_emrg = (v20_l[0] - v20_e[0]) if len(v20_e) and len(v20_l) else 0

v30_e = early_emrg[early_emrg['age_label']=='30대']['share'].values
v30_l = late_emrg[late_emrg['age_label']=='30대']['share'].values
delta30_emrg = (v30_l[0] - v30_e[0]) if len(v30_e) and len(v30_l) else 0

findings = [
    ('대상',   f'급부상 상권 {len(emrg_codes)}개'),
    ('기준',   '2020 분위<0.60 → 모멘텀 상위 15%'),
    ('',       ''),
    ('① CCF', f'20대선행형 {emrg_p20d:.0f}% / 동시형 {emrg_pdyn:.0f}%'),
    ('',       f'매출선행형 {emrg_psal:.0f}%'),
    ('',       f'→ 기성 상권 20대선행형 {lgnd_p20d:.0f}%와 비교'),
    ('',       ''),
    ('② 세대', f'20대 비중 변화: {delta20_emrg:+.1f}%p (초기→후기)'),
    ('',       f'30대 비중 변화: {delta30_emrg:+.1f}%p'),
    ('',       '→ 어떤 세대가 먼저 유입됐는지'),
    ('',       '   시각적으로 확인 가능'),
    ('',       ''),
    ('③ 추이', '급부상 상권의 복합점수 분위'),
    ('',       '기성 대비 얼마나 빠르게 수렴?'),
    ('',       ''),
    ('주의',   '인터넷 바이럴·SNS 효과는'),
    ('',       '이 데이터에 반영되지 않음'),
]

y = 0.98
for label, text in findings:
    if label:
        ax8.text(0.03, y, label, fontsize=9, fontweight='bold',
                 color='#333', transform=ax8.transAxes)
        ax8.text(0.28, y, text, fontsize=9, color='#333',
                 transform=ax8.transAxes)
    else:
        ax8.text(0.28, y, text, fontsize=8.5, color='#555',
                 transform=ax8.transAxes)
    y -= 0.062 if label else 0.055

ax8.set_facecolor('#FAFAFA')
for spine in ax8.spines.values():
    spine.set_edgecolor('#DDDDDD')

# ── 저장 ─────────────────────────────────────────────────────
out = IMG_DIR / 'emerging_분석.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[저장] {out.name}")
