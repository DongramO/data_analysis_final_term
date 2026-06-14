# -*- coding: utf-8 -*-
"""청파동 심층 분석 — 6패널 대시보드"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 ──────────────────────────────────────────────────
tsm = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig')
ml_tr = pd.read_csv('data/main_data/ml_dataset_quarterly_train.csv', encoding='utf-8-sig')
ml_pr = pd.read_csv('data/main_data/ml_dataset_quarterly_predict.csv', encoding='utf-8-sig')
ml = pd.concat([ml_tr, ml_pr], ignore_index=True)

rental = pd.read_csv('data/main_data/임대료_공실률.csv', encoding='utf-8-sig')

YEARS = [2020, 2021, 2022, 2023, 2024]
QMAP  = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3}

# 분기 순서 정렬용 숫자
def yq_sort(yq_str):
    parts = str(yq_str).split('Q') if 'Q' in str(yq_str) else str(yq_str).split()
    try:
        yr = int(parts[0])
        q  = int(parts[1])
        return yr * 10 + q
    except:
        return 0

def get_tsm(dong):
    return tsm[tsm['행정동'] == dong].sort_values('연도')

def get_ml_annual(dong):
    sub = ml[ml.iloc[:, 1] == dong].copy()
    sel = [1, 2, 6, 7, 8, 9, 18, 25, 30, 31, 76, 83]
    names = ['행정동', '연도_분기', 'FB카페', 'FB식사', '주류유흥', '라이프스타일',
             '주말비중', '야간비중', '2030소비', '금요효과', '2030이동비율', '2030이동비율_변화']
    sub = sub.iloc[:, sel].copy()
    sub.columns = names
    sub['sort_key'] = sub['연도_분기'].apply(yq_sort)
    sub = sub.sort_values('sort_key')
    sub['연도'] = sub['연도_분기'].apply(lambda x: int(str(x)[:4]))
    return sub

# ── 비교 대상 동 ──────────────────────────────────────────────────
CHEONGPA = '청파동'
REFS = {'성수2가1동': 2022, '합정동': 2021, '청운효자동': 2021}  # 전환 연도
NEIGHBORS = ['청파동', '후암동', '이태원1동', '이태원2동', '효창동']
REF_COLOR = '#E74C3C'
CAND_COLOR = '#2980B9'
NBRS_COLORS = ['#2980B9', '#E67E22', '#8E44AD', '#27AE60', '#E74C3C']

# ── 피겨 ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 20))
fig.patch.set_facecolor('#F8F9FA')
gs = gridspec.GridSpec(3, 3, hspace=0.52, wspace=0.38,
                       left=0.06, right=0.97, top=0.93, bottom=0.05)

fig.suptitle('청파동 심층 분석 — 차세대 트렌드 상권 가능성 진단',
             fontsize=15, fontweight='bold', y=0.96)

# ── 패널 1: 5년 트렌드 지표 시계열 ──────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
cp = get_tsm(CHEONGPA)

metrics = {
    '업종점수':      ('#E74C3C', '-',  'o'),
    '2030비중':     ('#3498DB', '-',  's'),
    '주말비중':      ('#2ECC71', '--', '^'),
    '야간비중':      ('#9B59B6', '--', 'D'),
    '트렌드매출_비중': ('#F39C12', ':',  'x'),
}
for label, (color, ls, mk) in metrics.items():
    ax1.plot(cp['연도'], cp[label], color=color, linestyle=ls,
             marker=mk, markersize=6, linewidth=2, label=label)

# 임계선
ax1.axhline(0.50, color='#E74C3C', linewidth=0.8, linestyle=':', alpha=0.6)
ax1.axhline(0.35, color='#3498DB', linewidth=0.8, linestyle=':', alpha=0.6)
ax1.text(2024.05, 0.505, 'S1 임계(0.50)', fontsize=7, color='#E74C3C', alpha=0.8)
ax1.text(2024.05, 0.355, 'S2 임계(0.35)', fontsize=7, color='#3498DB', alpha=0.8)

ax1.set_title('① 청파동 5개 지표 시계열 (2020~2024)', fontsize=10, fontweight='bold')
ax1.set_xlabel('연도', fontsize=8)
ax1.set_ylabel('비중', fontsize=8)
ax1.legend(fontsize=7, loc='upper left')
ax1.set_xlim(2019.5, 2025.0)
ax1.set_ylim(0, 0.65)
ax1.grid(True, alpha=0.3)

# ── 패널 2: 레퍼런스 대비 현재 상태 비교 (레이더/바 차트) ────────
ax2 = fig.add_subplot(gs[0, 1])

compare_dongs = ['성수2가1동\n(전환 전 2020)', '성수2가1동\n(전환 후 2022)',
                 '합정동\n(전환 전 2020)', '청파동\n(현재 2024)']
compare_vals = {}

s2g1_pre  = get_tsm('성수2가1동')
s2g1_post = get_tsm('성수2가1동')
hapjong   = get_tsm('합정동')

def row(df, yr):
    r = df[df['연도'] == yr]
    if len(r) == 0:
        return [0] * 5
    r = r.iloc[0]
    return [r['업종점수'], r['2030비중'], r['주말비중'], r['야간비중'], r['트렌드매출_비중']]

data_compare = np.array([
    row(s2g1_pre,  2020),
    row(s2g1_post, 2022),
    row(hapjong,   2020),
    row(cp,        2024),
])
labels_c = ['업종점수', '2030비중', '주말비중', '야간비중', '트렌드매출']
colors_c = ['#BDC3C7', '#E74C3C', '#BDC3C7', '#2980B9']
x = np.arange(len(labels_c))
w = 0.2

for i, (vals, color, lbl) in enumerate(zip(data_compare, colors_c, compare_dongs)):
    bars = ax2.bar(x + i * w, vals, width=w, color=color,
                   alpha=0.85, label=lbl.replace('\n', ' '))

ax2.axhline(0.50, color='#E74C3C', linestyle=':', linewidth=1, alpha=0.6)
ax2.axhline(0.35, color='#3498DB', linestyle=':', linewidth=1, alpha=0.6)
ax2.set_xticks(x + 1.5 * w)
ax2.set_xticklabels(labels_c, fontsize=8)
ax2.set_title('② 레퍼런스 전환 전/후 vs 청파동 현재', fontsize=10, fontweight='bold')
ax2.set_ylabel('비중', fontsize=8)
ax2.legend(fontsize=6.5, loc='upper right')
ax2.set_ylim(0, 0.75)
ax2.grid(True, alpha=0.3, axis='y')

# ── 패널 3: 업종 구성 변화 (카페 급증 강조) ──────────────────────
ax3 = fig.add_subplot(gs[0, 2])
cp_ml = get_ml_annual(CHEONGPA)
annual_ml = cp_ml.groupby('연도')[['FB카페', 'FB식사', '주류유흥', '라이프스타일']].mean()

bar_colors = ['#E74C3C', '#3498DB', '#9B59B6', '#F39C12']
bar_labels  = ['F&B카페', 'F&B식사', '주류/유흥', '라이프스타일']
bottom = np.zeros(len(annual_ml))
for col, color, lbl in zip(['FB카페', 'FB식사', '주류유흥', '라이프스타일'],
                             bar_colors, bar_labels):
    vals = annual_ml[col].values
    ax3.bar(annual_ml.index, vals, bottom=bottom, color=color, alpha=0.85, label=lbl)
    bottom += vals

# 카페 비중 숫자 표시
for yr, val in zip(annual_ml.index, annual_ml['FB카페'].values):
    ax3.text(yr, val / 2, f'{val:.1%}', ha='center', va='center',
             fontsize=8, color='white', fontweight='bold')

ax3.set_title('③ 업종 구성 변화 (카페 비중 주목)', fontsize=10, fontweight='bold')
ax3.set_xlabel('연도', fontsize=8)
ax3.set_ylabel('매출 비중', fontsize=8)
ax3.legend(fontsize=8, loc='upper left')
ax3.grid(True, alpha=0.3, axis='y')

# ── 패널 4: 용산역 임대료·공실률 (2020~2025) ─────────────────────
ax4 = fig.add_subplot(gs[1, 0])
yongsan = rental[rental['상권명'] == '용산역'].sort_values('연도')

color_rent  = '#E74C3C'
color_vac   = '#3498DB'
ax4_twin = ax4.twinx()

ax4.bar(yongsan['연도'], yongsan['임대료'], color=color_rent,
        alpha=0.7, label='임대료 (천원/㎡)', zorder=2)
ax4_twin.plot(yongsan['연도'], yongsan['공실률'], color=color_vac,
              marker='o', linewidth=2.5, label='공실률 (%)', zorder=3)

# YoY 변화율 표시
for i in range(1, len(yongsan)):
    prev = yongsan.iloc[i-1]['임대료']
    curr = yongsan.iloc[i]['임대료']
    if prev > 0:
        chg = (curr - prev) / prev * 100
        yr  = yongsan.iloc[i]['연도']
        color_chg = '#E74C3C' if chg > 0 else '#3498DB'
        ax4.text(yr, curr + 0.5, f'{chg:+.1f}%',
                 ha='center', va='bottom', fontsize=8,
                 color=color_chg, fontweight='bold')

ax4.set_title('④ 용산역 임대료·공실률 (청파1·2동)', fontsize=10, fontweight='bold')
ax4.set_xlabel('연도', fontsize=8)
ax4.set_ylabel('임대료 (천원/㎡)', fontsize=8, color=color_rent)
ax4_twin.set_ylabel('공실률 (%)', fontsize=8, color=color_vac)
ax4.grid(True, alpha=0.3, axis='y')
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

# ── 패널 5: 2030 이동비율 분기별 추이 ──────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
cp_q = get_ml_annual(CHEONGPA)
cp_q['yq_num'] = cp_q['sort_key']

ref_dongs_q = {}
ref_colors_q = {'성수2가1동': '#E74C3C', '합정동': '#2ECC71', '청운효자동': '#9B59B6'}
for d, col in ref_colors_q.items():
    sub = get_ml_annual(d)
    ref_dongs_q[d] = (sub, col)

# 청파동
ax5.plot(range(len(cp_q)), cp_q['2030이동비율'].values,
         color=CAND_COLOR, linewidth=2.5, marker='o', markersize=5,
         label='청파동', zorder=5)

# 레퍼런스들 (얇게)
for d, (sub, col) in ref_dongs_q.items():
    ax5.plot(range(len(sub)), sub['2030이동비율'].values,
             color=col, linewidth=1.5, linestyle='--', alpha=0.6,
             label=f'{d} (레퍼런스)')

# 연도 구분선
for i in [4, 8, 12, 16]:
    ax5.axvline(i, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
yr_labels = ['2020\nQ1', '2021\nQ1', '2022\nQ1', '2023\nQ1', '2024\nQ1']
ax5.set_xticks([0, 4, 8, 12, 16])
ax5.set_xticklabels(yr_labels, fontsize=8)

ax5.set_title('⑤ 2030 이동비율 — 청파동 vs 레퍼런스 분기 추이', fontsize=10, fontweight='bold')
ax5.set_ylabel('2030 이동비율 (%)', fontsize=8)
ax5.legend(fontsize=7.5, loc='lower right')
ax5.grid(True, alpha=0.3)

# ── 패널 6: 인근 동 비교 (2024 스냅샷) ──────────────────────────
ax6 = fig.add_subplot(gs[1, 2])

nbr_data = []
for dong in NEIGHBORS:
    sub = tsm[(tsm['행정동'] == dong) & (tsm['연도'] == 2024)]
    if len(sub) == 0:
        continue
    r = sub.iloc[0]
    nbr_data.append({
        '행정동': dong,
        '업종점수': r['업종점수'],
        '2030비중': r['2030비중'],
        '주말비중': r['주말비중'],
        '야간비중': r['야간비중'],
    })
nbr_df = pd.DataFrame(nbr_data)

x6 = np.arange(len(nbr_df))
w6 = 0.2
metrics6 = [('업종점수', '#E74C3C'), ('2030비중', '#3498DB'),
            ('주말비중', '#2ECC71'), ('야간비중', '#9B59B6')]
for i, (col, color) in enumerate(metrics6):
    ax6.bar(x6 + i * w6, nbr_df[col], width=w6, color=color, alpha=0.85, label=col)

ax6.axhline(0.50, color='#E74C3C', linestyle=':', linewidth=1, alpha=0.5)
ax6.axhline(0.35, color='#3498DB', linestyle=':', linewidth=1, alpha=0.5)
ax6.set_xticks(x6 + 1.5 * w6)
ax6.set_xticklabels(nbr_df['행정동'], fontsize=9, rotation=15)
ax6.set_title('⑥ 용산 권역 인근 동 비교 (2024)', fontsize=10, fontweight='bold')
ax6.set_ylabel('비중', fontsize=8)
ax6.legend(fontsize=8)
ax6.set_ylim(0, 0.75)
ax6.grid(True, alpha=0.3, axis='y')

# 청파동 강조
ax6.get_xticklabels()[0].set_color(CAND_COLOR)
ax6.get_xticklabels()[0].set_fontweight('bold')

# ── 패널 7: 업종 vs 2030이동 산점도 (용산·마포·서초 권역) ────────
ax7 = fig.add_subplot(gs[2, :2])

# 2024 전체 미전환 동 (후보군) — 용산·마포·서초·종로 구
tsm24 = tsm[tsm['연도'] == 2024].copy()
gu_filter = tsm24['행정동'].str.contains(
    '청파|후암|효창|남영|이촌|서빙고|한남|이태원|보광|주성|동부이촌|용문|원효|'
    '망원|합정|서교|동교|연남|성산|상암|'
    '청룡|낙성대|봉천|신림|서원|'
    '필동|묵정|광희|을지|다산|신당'
)

scatter_df = tsm24.copy()

# 전체 배경 점 (회색)
ax7.scatter(scatter_df['업종점수'], scatter_df['2030비중'],
            c='lightgray', s=15, alpha=0.4, zorder=1)

# 용산 권역
yongsan_df = tsm24[tsm24['행정동'].str.contains('청파|후암|효창|남영|이태원|한남|보광|주성|용문|원효', na=False)]
ax7.scatter(yongsan_df['업종점수'], yongsan_df['2030비중'],
            c='#E67E22', s=60, alpha=0.8, zorder=3, label='용산구')

# 마포 권역
mapo_df = tsm24[tsm24['행정동'].str.contains('망원|합정|서교|동교|연남|성산|상암', na=False)]
ax7.scatter(mapo_df['업종점수'], mapo_df['2030비중'],
            c='#27AE60', s=60, alpha=0.8, zorder=3, label='마포구')

# 2024 확정 트렌드 (S1+S2 충족)
confirmed = tsm24[(tsm24['업종점수'] >= 0.50) & (tsm24['2030비중'] >= 0.35)]
ax7.scatter(confirmed['업종점수'], confirmed['2030비중'],
            c='#C0392B', s=80, alpha=0.9, zorder=4, marker='*', label='확정 트렌드 (S1+S2)')

# 청파동 강조
cp24 = tsm24[tsm24['행정동'] == '청파동'].iloc[0]
ax7.scatter(cp24['업종점수'], cp24['2030비중'],
            c='#2980B9', s=250, zorder=6, marker='*', label='청파동')
ax7.annotate('청파동', (cp24['업종점수'], cp24['2030비중']),
             xytext=(8, 8), textcoords='offset points',
             fontsize=10, fontweight='bold', color='#2980B9')

# 임계선
ax7.axvline(0.50, color='#E74C3C', linewidth=1.2, linestyle='--', alpha=0.7, label='업종 임계(0.50)')
ax7.axhline(0.35, color='#3498DB', linewidth=1.2, linestyle='--', alpha=0.7, label='2030 임계(0.35)')

# 인근 동 라벨
for _, row_d in yongsan_df.iterrows():
    if row_d['행정동'] != '청파동':
        ax7.annotate(row_d['행정동'], (row_d['업종점수'], row_d['2030비중']),
                     xytext=(4, 4), textcoords='offset points',
                     fontsize=7.5, color='#E67E22', alpha=0.9)

ax7.set_title('⑦ 2024 업종점수 × 2030비중 — 용산·마포권 포지셔닝', fontsize=10, fontweight='bold')
ax7.set_xlabel('업종점수 (S1)', fontsize=9)
ax7.set_ylabel('2030비중 (S2)', fontsize=9)
ax7.legend(fontsize=8, loc='upper left')
ax7.grid(True, alpha=0.3)

# ── 패널 8: 분석 요약 텍스트 ──────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')

summary_text = [
    '【청파동 종합 진단】',
    '',
    '✅ 강점 (전환 진행 중)',
    '  • 업종점수 2022년 0.50 돌파',
    '  • 카페 비중 2020→2021 3배 급증',
    '    (0.04% → 10.5%)',
    '  • 2030이동비율 2024년 27~28%',
    '    (꾸준히 상승)',
    '  • 용산역 임대료',
    '    2023→2024: +29.5% 급등',
    '    2024→2025: +7.1% 추가 상승',
    '  • 이태원 인접 — 2030 이탈 수혜',
    '',
    '⚠️ 약점 (아직 전환 미완)',
    '  • 2030비중 0.319 (임계 0.35 미달)',
    '  • 주말비중 0.226 (정체/하락)',
    '  • 블로그·네이버 커버리지 없음',
    '',
    '🔮 전망',
    '  임대료 시장이 먼저 반응 중.',
    '  2030이동↑ + 카페업종↑ =',
    '  성수 전환 직전 패턴과 동일.',
    '  2030비중 0.35 돌파 시점이',
    '  트렌드 상권 공식 확정 신호.',
]

for i, line in enumerate(summary_text):
    weight = 'bold' if line.startswith('【') or line.startswith('✅') or \
             line.startswith('⚠️') or line.startswith('🔮') else 'normal'
    color  = '#2C3E50' if not line.startswith(' ') else '#555555'
    ax8.text(0.03, 0.97 - i * 0.040, line,
             transform=ax8.transAxes,
             fontsize=8.5, verticalalignment='top',
             fontweight=weight, color=color)

ax8.set_facecolor('#EBF5FB')
ax8.set_title('⑧ 종합 진단', fontsize=10, fontweight='bold')

plt.savefig('image/cheongpa_deep_dive.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print('저장 완료: image/cheongpa_deep_dive.png')
