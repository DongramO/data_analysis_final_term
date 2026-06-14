# -*- coding: utf-8 -*-
"""
두 패턴 비교: 20대 선행형 vs 매출 선행형
출력: image/pattern_compare.png
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

ROOT     = Path(__file__).resolve().parent
CCF_PATH = ROOT / 'data/_tmp_ccf_all.csv'
MIG_PATH = ROOT / 'data/main_data/분기별_주말유입인구.csv'
RAW_PATH = ROOT / 'data/main_data/분기별_원본피처.csv'
MAP_PATH = ROOT / 'data/main_data/코드매핑_유입인구_매출.csv'
IMG_DIR  = ROOT / 'image'

C_20대  = '#E05C5C'   # 빨강  — 20대 선행형
C_매출  = '#3A5CA9'   # 파랑  — 매출 선행형
C_동시  = '#F5A623'   # 주황  — 동시형
C_BG20  = '#FFF5F5'
C_BGSAL = '#F5F7FF'

# ── 데이터 로드 ──────────────────────────────────────────
ccf = pd.read_csv(CCF_PATH, encoding='utf-8-sig')

_mig = pd.read_csv(MIG_PATH, encoding='utf-8-sig')
_mig.columns = ['mig_code','yr','q','yrq','age','HE','WE','EE','mig_total']

_raw = pd.read_csv(RAW_PATH, encoding='utf-8-sig')
raw2 = _raw.iloc[:, [0,1,3,4,5,18,51,52,53,54,55,56]].copy()
raw2.columns = ['raw_code','dong','yr','q','yrq','sales',
                'age10','age20','age30','age40','age50','age60']

_map = pd.read_csv(MAP_PATH, encoding='utf-8-sig')
_map.columns = ['mig_code','raw_code','gu','dong_map']
_map['raw_code'] = _map['raw_code'].astype('Int64')
code_mig2raw = _map.set_index('mig_code')['raw_code'].to_dict()
dong_name    = _map.set_index('raw_code')['dong_map'].to_dict()

def qnum(yr, q_): return int((yr - 2020) * 4 + q_)
_mig['qnum']     = _mig.apply(lambda r: qnum(r['yr'], r['q']), axis=1)
raw2['qnum']     = raw2.apply(lambda r: qnum(r['yr'], r['q']), axis=1)
_mig['raw_code'] = _mig['mig_code'].map(code_mig2raw)
_mig = _mig.dropna(subset=['raw_code'])
_mig['raw_code'] = _mig['raw_code'].astype(int)

# ── YoY 계산 ─────────────────────────────────────────────
def calc_yoy(df, key, val, new_col):
    df = df.sort_values([key, 'qnum']).copy()
    prev = df[[key, 'qnum', val]].rename(columns={val: 'prev'})
    prev['qnum'] = prev['qnum'] + 4
    df = df.merge(prev, on=[key, 'qnum'], how='left')
    df[new_col] = (df[val] - df['prev']) / df['prev'].replace(0, np.nan)
    return df.drop(columns=['prev'])

mig20   = (_mig[_mig['age'] == 20]
           .groupby(['raw_code', 'qnum'])['mig_total'].sum().reset_index()
           .rename(columns={'mig_total': 'mig20'}))
sales_q = raw2.groupby(['raw_code', 'qnum'])['sales'].sum().reset_index()
mig20   = calc_yoy(mig20,   'raw_code', 'mig20',  'mig20_yoy')
sales_q = calc_yoy(sales_q, 'raw_code', 'sales',  'sales_yoy')

def get_yoy(codes):
    df = (mig20[mig20['raw_code'].isin(codes)][['raw_code','qnum','mig20_yoy']]
          .merge(sales_q[sales_q['raw_code'].isin(codes)][['raw_code','qnum','sales_yoy']],
                 on=['raw_code','qnum'], how='inner'))
    df['행정동']       = df['raw_code'].map(dong_name)
    df['yr']          = 2020 + (df['qnum'] - 1) // 4
    df['q_n']         = ((df['qnum'] - 1) % 4) + 1
    df['yrq']         = df['yr'].astype(str) + 'Q' + df['q_n'].astype(str)
    df['mig_pct']     = (df['mig20_yoy'] * 100).round(1)
    df['sales_pct']   = (df['sales_yoy'] * 100).round(1)
    return df.dropna(subset=['mig_pct', 'sales_pct']).sort_values(['행정동', 'yrq'])

# ── 패턴 선별 ─────────────────────────────────────────────
COND = (ccf['최고_r'] >= 0.35) & (ccf['매출성장률_2020→2024(%)'] > 0)
lead20  = ccf[COND & (ccf['패턴'] == '20대선행형')].sort_values('최고_r', ascending=False)
leadsal = ccf[COND & (ccf['패턴'] == '매출선행형')].sort_values('최고_r', ascending=False)

# 20대이동 변화율 추가
yr_mig20 = _mig[_mig['age'] == 20].groupby(['raw_code', 'yr'])['mig_total'].sum().reset_index()
s20_ = yr_mig20[yr_mig20['yr'] == 2020].set_index('raw_code')['mig_total']
s24_ = yr_mig20[yr_mig20['yr'] == 2024].set_index('raw_code')['mig_total']
mig_growth = ((s24_ - s20_) / s20_ * 100).rename('mig20_growth').reset_index()
ccf = ccf.merge(mig_growth, left_on='행정동코드', right_on='raw_code', how='left')

# ── 시계열용 데이터 ────────────────────────────────────────
TOP_YOY = 5  # 패널당 행정동 수

ss_codes  = [11200650, 11200660, 11200670, 11200690]  # 성수동 계열

# 극단 이상치(매출 성장률 200% 초과) 제외
lead20_clean = lead20[lead20['매출성장률_2020→2024(%)'] <= 200]
top20_codes  = lead20_clean.head(TOP_YOY)['행정동코드'].tolist()
topsal_codes = leadsal[~leadsal['행정동코드'].isin(ss_codes)].head(TOP_YOY)['행정동코드'].tolist()

yoy_20d  = get_yoy(top20_codes)
yoy_sal  = get_yoy(topsal_codes + ss_codes)

# ── 색상 팔레트 (행정동별 고정) ────────────────────────────
PALETTE = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#8c564b',
           '#17becf','#bcbd22','#e377c2','#7f7f7f']

def assign_colors(dongs):
    return {d: PALETTE[i % len(PALETTE)] for i, d in enumerate(dongs)}

colors_20d  = assign_colors(yoy_20d['행정동'].unique())
colors_sal  = assign_colors(yoy_sal['행정동'].unique())

# ══════════════════════════════════════════════════════════
# 레이아웃: 3행 × 2열
# ══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 20))
fig.suptitle('20대 이동 선행형 vs 매출 선행형 — 두 패턴 비교 (2020–2024)',
             fontsize=15, fontweight='bold', y=0.99)

gs = gridspec.GridSpec(3, 2, figure=fig,
                       hspace=0.50, wspace=0.32,
                       left=0.08, right=0.97, top=0.95, bottom=0.05)

ax_dist = fig.add_subplot(gs[0, 0])
ax_scat = fig.add_subplot(gs[0, 1])
ax_20d  = fig.add_subplot(gs[1, 0])
ax_sal  = fig.add_subplot(gs[1, 1])
ax_b20  = fig.add_subplot(gs[2, 0])
ax_bsal = fig.add_subplot(gs[2, 1])

# ══ 패널 1: 패턴 분포 막대 ══════════════════════════════
pat_order  = ['20대선행형', '동시형', '매출선행형']
pat_counts = ccf['패턴'].value_counts().reindex(pat_order)
pat_valid  = {p: len(ccf[COND & (ccf['패턴'] == p)]) for p in pat_order}
bar_colors = [C_20대, C_동시, C_매출]

bars1 = ax_dist.bar(pat_order, pat_counts.values, color=bar_colors,
                    edgecolor='white', linewidth=1.5, width=0.5)

for bar, p, cnt in zip(bars1, pat_order, pat_counts.values):
    # 막대 위 — 전체 수
    ax_dist.text(bar.get_x() + bar.get_width() / 2, cnt + 3,
                 f'{cnt}개 ({cnt/len(ccf)*100:.0f}%)',
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333')
    # 막대 중간 — 조건 부합 수 (흰색, 충분히 높은 막대만)
    if cnt > 30:
        ax_dist.text(bar.get_x() + bar.get_width() / 2, cnt / 2,
                     f'r≥0.35 기준\n{pat_valid[p]}개',
                     ha='center', va='center', fontsize=9, color='white', fontweight='bold')

ax_dist.set_ylim(0, pat_counts.max() * 1.18)
ax_dist.set_ylabel('행정동 수', fontsize=10)
ax_dist.set_title('전체 423개 행정동 패턴 분류', fontsize=12, pad=10)
ax_dist.set_facecolor('#F8F8F8')
ax_dist.grid(axis='y', alpha=0.25)
ax_dist.tick_params(axis='x', labelsize=10)

# ══ 패널 2: 산포도 ═══════════════════════════════════════
for pat, c in [('20대선행형', C_20대), ('동시형', C_동시), ('매출선행형', C_매출)]:
    sub = ccf[COND & (ccf['패턴'] == pat)]
    ax_scat.scatter(sub['mig20_growth'], sub['매출성장률_2020→2024(%)'],
                    c=c, alpha=0.65, s=50, label=pat,
                    edgecolors='white', linewidths=0.7, zorder=3)

# 성수동 강조
ss_df = ccf[ccf['행정동'].str.contains('성수', na=False)]
ax_scat.scatter(ss_df['mig20_growth'], ss_df['매출성장률_2020→2024(%)'],
                c='black', s=90, zorder=5, marker='D', label='성수동 계열')
for _, r in ss_df.iterrows():
    if pd.notna(r['mig20_growth']):
        ax_scat.annotate(r['행정동'],
                         (r['mig20_growth'], r['매출성장률_2020→2024(%)']),
                         xytext=(5, 4), textcoords='offset points',
                         fontsize=8, color='#333')

p96x = np.nanpercentile(ccf['mig20_growth'].dropna(), 96)
p96y = np.nanpercentile(ccf['매출성장률_2020→2024(%)'].dropna(), 96)
ax_scat.set_xlim(-110, min(p96x + 30, 350))
ax_scat.set_ylim(-20, min(p96y + 30, 250))
ax_scat.axhline(0, color='#CCC', lw=1, ls='--')
ax_scat.axvline(0, color='#CCC', lw=1, ls='--')
ax_scat.set_xlabel('20대 이동인구 변화율 2020→2024 (%)', fontsize=10)
ax_scat.set_ylabel('총매출 성장률 2020→2024 (%)', fontsize=10)
ax_scat.set_title('패턴별 분포 — 20대 이동변화 vs 매출 성장', fontsize=12, pad=10)
ax_scat.set_facecolor('#F8F8F8')
ax_scat.grid(alpha=0.15)
ax_scat.legend(fontsize=8.5, loc='upper right',
               handles=[mpatches.Patch(color=C_20대, label='20대 선행형'),
                        mpatches.Patch(color=C_동시,  label='동시형'),
                        mpatches.Patch(color=C_매출, label='매출 선행형'),
                        Line2D([0],[0], marker='D', color='w', markerfacecolor='black',
                               markersize=8, label='성수동 계열')])

# ══ YoY 공통 함수 ════════════════════════════════════════
def plot_yoy_panel(ax, yoy_df, color_map, title, title_color, bg_color,
                   special_dongs=None):
    """점선=20대이동 YoY, 실선=매출 YoY"""
    xl = []
    for dong, c in color_map.items():
        sub = yoy_df[yoy_df['행정동'] == dong].sort_values('yrq')
        if sub.empty:
            continue
        xl  = sub['yrq'].tolist()
        x   = np.arange(len(sub))
        lw_s = 3.0 if (special_dongs and dong in special_dongs) else 2.0
        ls_s = ':'  if (special_dongs and dong in special_dongs) else '-'

        ax.plot(x, sub['mig_pct'],   color=c, lw=1.6, ls='--',
                marker='o', markersize=3.5, alpha=0.85)
        ax.plot(x, sub['sales_pct'], color=c, lw=lw_s, ls=ls_s,
                marker='s', markersize=3.5, alpha=0.85, label=dong)

    ax.axhline(0, color='#888', lw=1.0)

    # x축: 연도 시작 분기만 표시
    if xl:
        year_ticks = [i for i, lbl in enumerate(xl) if lbl.endswith('Q1')]
        year_labels = [lbl[:4] for lbl in xl if lbl.endswith('Q1')]
        ax.set_xticks(year_ticks)
        ax.set_xticklabels(year_labels, fontsize=10)
        ax.set_xlim(-0.5, len(xl) - 0.5)

    # y축 클리핑 (이상치 제거)
    vals = yoy_df[['mig_pct','sales_pct']].stack().dropna()
    if len(vals):
        lo = max(np.percentile(vals, 3) - 10, -120)
        hi = min(np.percentile(vals, 97) + 10,  200)
        ax.set_ylim(lo, hi)

    ax.set_ylabel('YoY (%)', fontsize=10)
    ax.set_facecolor(bg_color)
    ax.grid(alpha=0.15)
    ax.set_title(title, fontsize=11, color=title_color, fontweight='bold', pad=10)

    # 범례 ① 행정동명 (우상단)
    dong_legend = ax.legend(fontsize=8, loc='upper right', ncol=2,
                             title='행정동', title_fontsize=8,
                             framealpha=0.9)
    ax.add_artist(dong_legend)

    # 범례 ② 선 스타일 (좌하단)
    line_handles = [
        Line2D([0],[0], color='gray', lw=1.6, ls='--',
               marker='o', markersize=4, label='20대 이동 YoY'),
        Line2D([0],[0], color='gray', lw=2.0, ls='-',
               marker='s', markersize=4, label='매출 YoY'),
    ]
    ax.legend(handles=line_handles, fontsize=8.5, loc='lower left', framealpha=0.9)

# ══ 패널 3: 20대 선행형 YoY ══════════════════════════════
plot_yoy_panel(
    ax_20d, yoy_20d, colors_20d,
    title='【20대 선행형】 Top5 — 점선(20대이동 YoY)이 실선(매출 YoY)보다 먼저 움직임',
    title_color=C_20대, bg_color=C_BG20
)

# ══ 패널 4: 매출 선행형 YoY (성수동 포함) ═══════════════
ss_names = [dong_name.get(c,'') for c in ss_codes if c in dong_name]
plot_yoy_panel(
    ax_sal, yoy_sal, colors_sal,
    title='【매출 선행형】 Top5 + 성수동(굵은 점선) — 실선(매출 YoY)이 먼저 움직임',
    title_color=C_매출, bg_color=C_BGSAL,
    special_dongs=ss_names
)

# ══ 패널 5·6: Top12 r값 막대 ════════════════════════════
for ax_b, df_b, title, bar_c in [
    (ax_b20,  lead20.head(12),  '20대 선행형 Top12 (CCF r 순)',  C_20대),
    (ax_bsal, leadsal.head(12), '매출 선행형 Top12 (CCF r 순)',  C_매출),
]:
    df_b = df_b.reset_index(drop=True)
    y_pos = np.arange(len(df_b))

    ax_b.barh(y_pos, df_b['최고_r'].values[::-1],
              color=bar_c, alpha=0.82, edgecolor='white', height=0.65)

    for i, (_, row) in enumerate(df_b[::-1].iterrows()):
        r_val = row['최고_r']
        # 행정동명: r값이 충분히 크면 막대 안 흰색, 아니면 막대 오른쪽 검정
        if r_val >= 0.55:
            ax_b.text(0.01, i, row['행정동'],
                      va='center', ha='left', fontsize=8.5,
                      color='white', fontweight='bold')
        else:
            ax_b.text(r_val + 0.02, i, row['행정동'],
                      va='center', ha='left', fontsize=8.5,
                      color=bar_c, fontweight='bold')
        # 부가 정보 — 막대 끝 오른쪽
        info_x = max(r_val, 0.55) + 0.17
        ax_b.text(info_x, i,
                  f"lag={int(row['최고_선행분기'])}Q  +{row['매출성장률_2020→2024(%)']:.0f}%",
                  va='center', ha='left', fontsize=7.5, color='#555')

    ax_b.axvline(0.35, color='#AAAAAA', lw=1.2, ls='--', label='기준선 r=0.35')
    ax_b.set_xlim(0, 1.55)
    ax_b.set_ylim(-0.5, len(df_b) - 0.5)
    ax_b.set_yticks([])
    ax_b.set_xlabel('CCF 최고 상관계수 (r)', fontsize=10)
    ax_b.set_title(title, fontsize=11, color=bar_c, fontweight='bold', pad=10)
    ax_b.set_facecolor('#F8F8F8')
    ax_b.grid(axis='x', alpha=0.2)
    ax_b.legend(fontsize=8.5, loc='lower right')

# ── 저장 ─────────────────────────────────────────────────
out = IMG_DIR / 'pattern_compare.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[저장] {out.name}")
