# -*- coding: utf-8 -*-
"""성수2가1동 업종 구성 패턴과 가장 유사한 행정동 비교"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from numpy.linalg import norm
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT     = Path(__file__).resolve().parent
RAW_PATH = ROOT / 'data/main_data/분기별_원본피처.csv'
IMG_DIR  = ROOT / 'image'

REF   = ['방배2동', '청운효자동', '합정동', '성수2가1동']
TIER1 = ['숭인2동', '신당5동']
CANDS = ['잠실6동','자양3동','신대방1동','용산2가동','동화동',
         '문정1동','등촌2동','개포1동','사당3동']

IND_COLS  = ['FB카페_매출금액','FB식사_매출금액','주류유흥_매출금액','라이프스타일_매출금액','기타_매출금액']
IND_NAMES = ['FB카페','FB식사','주류·유흥','라이프스타일','기타']
IND_COLORS= ['#5B9BD5','#ED7D31','#A5A5A5','#FFC000','#D0D0D0']

raw = pd.read_csv(RAW_PATH, encoding='utf-8-sig')

y24 = raw[raw['연도']==2024].groupby('행정동')[IND_COLS].mean().reset_index()
total = y24[IND_COLS].sum(axis=1)
ratio_cols = []
for c in IND_COLS:
    col = c.replace('_매출금액','_r')
    y24[col] = y24[c] / total
    ratio_cols.append(col)

ref_vec = y24[y24['행정동']=='성수2가1동'][ratio_cols].values[0]
vecs    = y24[ratio_cols].values
cos_sim = vecs @ ref_vec / (norm(vecs, axis=1) * norm(ref_vec) + 1e-9)
y24['유사도'] = cos_sim

# 성수2가1동 제외 TOP10
top10 = y24[y24['행정동'] != '성수2가1동'].nlargest(10, '유사도')

# 표시 대상: 성수2가1동 + TOP10
show_list = ['성수2가1동'] + top10['행정동'].tolist()
show_df = y24[y24['행정동'].isin(show_list)].copy()

# 순서: 성수2가1동 맨 앞, 나머지 유사도 내림차순
show_df['_order'] = show_df['행정동'].map(
    lambda d: -1 if d == '성수2가1동' else top10[top10['행정동']==d]['유사도'].values[0]
)
show_df = show_df.sort_values('_order', ascending=False).reset_index(drop=True)

# 그룹 태그
def tag(d):
    if d == '성수2가1동': return '레퍼런스(기준)'
    if d in REF:          return '레퍼런스'
    if d in TIER1:        return 'Tier1'
    if d in CANDS:        return '신규 선별'
    return '신규 발견'

show_df['그룹'] = show_df['행정동'].map(tag)

TAG_COLOR = {
    '레퍼런스(기준)': '#E05C5C',
    '레퍼런스':       '#E05C5C',
    'Tier1':          '#F5A623',
    '신규 선별':      '#27AE60',
    '신규 발견':      '#4A90D9',
}

# ── 그림 ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(20, 7),
                         gridspec_kw={'width_ratios': [1, 1.6]})
fig.suptitle('성수2가1동 업종 구성 패턴과 가장 유사한 행정동 (2024)',
             fontsize=14, fontweight='bold')

# ── Panel 1: 성수2가1동 단독 파이 + 순위 설명 ─────────────────────────
ax1 = axes[0]
ax1.axis('off')

# 테이블형 비교
col_headers = ['행정동', '그룹', '유사도'] + IND_NAMES
rows = []
for _, r in show_df.iterrows():
    sim = '' if r['행정동']=='성수2가1동' else f"{r['유사도']:.4f}"
    vals = [f"{r[c]*100:.1f}%" for c in ratio_cols]
    rows.append([r['행정동'], r['그룹'], sim] + vals)

tbl = ax1.table(cellText=rows, colLabels=col_headers,
                cellLoc='center', loc='center', bbox=[0,0,1,1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)

for (ri, ci), cell in tbl.get_celld().items():
    if ri == 0:
        cell.set_facecolor('#3A5CA9')
        cell.set_text_props(color='white', fontweight='bold')
        continue
    dong = rows[ri-1][0]
    grp  = rows[ri-1][1]
    base = TAG_COLOR.get(grp, '#FFFFFF')
    if dong == '성수2가1동':
        cell.set_facecolor('#FDECEA')
        cell.set_text_props(fontweight='bold')
    elif ri % 2 == 0:
        cell.set_facecolor('#F7F7F7')

    # 업종 열 최댓값 강조
    if ci >= 3:
        vals = [float(rows[ri-1][c].replace('%','')) for c in range(3, len(col_headers))]
        if float(rows[ri-1][ci].replace('%','')) == max(vals):
            cell.set_text_props(fontweight='bold', color='#1A5276')

ax1.set_title('행정동별 업종 비중 & 유사도', fontsize=10, pad=10)

# ── Panel 2: 개별 스택바 (성수2가1동 기준 정렬) ──────────────────────
ax2 = axes[1]
n = len(show_df)
bottom = np.zeros(n)
x = np.arange(n)

for col, name, color in zip(ratio_cols, IND_NAMES, IND_COLORS):
    vals = show_df[col].values
    ax2.bar(x, vals, bottom=bottom, color=color, label=name, width=0.7)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 0.08:
            ax2.text(i, b + v/2, f'{v*100:.0f}%',
                     ha='center', va='center', fontsize=8,
                     fontweight='bold', color='white')
    bottom += vals

# 성수2가1동 구분선
ax2.axvline(-0.5, color='#E05C5C', lw=2)
ax2.axvline(0.5,  color='#E05C5C', lw=2)

# x축 레이블
xlabels = []
for _, r in show_df.iterrows():
    sim_str = '' if r['행정동']=='성수2가1동' else f"\n(유사도 {r['유사도']:.4f})"
    xlabels.append(r['행정동'] + sim_str)

ax2.set_xticks(x)
ax2.set_xticklabels(xlabels, fontsize=8.5, rotation=0)

# x축 레이블 색상
for tick, (_, r) in zip(ax2.get_xticklabels(), show_df.iterrows()):
    tick.set_color(TAG_COLOR.get(r['그룹'], '#000000'))
    if r['행정동'] == '성수2가1동':
        tick.set_fontweight('bold')

# 성수2가1동 업종별 비중을 수평 참조선으로 표시 (누적 구간)
cum = 0
for col, color in zip(ratio_cols, IND_COLORS):
    val = ref_vec[ratio_cols.index(col)]
    cum_prev = cum
    cum += val
    ax2.axhline(cum, color=color, lw=1.0, ls=':', alpha=0.8, xmin=0.04)

ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax2.set_ylim(0, 1.05)
ax2.set_ylabel('업종 구성 비중')
ax2.set_title('업종 구성 스택바 비교\n(점선: 성수2가1동 누적 구간 기준선 / x축 색상 = 그룹)', fontsize=10)
ax2.legend(loc='upper right', fontsize=9, ncol=2)
ax2.set_facecolor('#F8F8F8')

# 범례 (그룹 색상)
import matplotlib.patches as mpatches
group_patches = [mpatches.Patch(color=v, label=k) for k, v in TAG_COLOR.items()
                 if k in show_df['그룹'].values]
ax2.legend(handles=ax2.get_legend_handles_labels()[0][:5] + group_patches,
           labels=IND_NAMES + list({k for k in TAG_COLOR if k in show_df['그룹'].values}),
           loc='upper right', fontsize=8.5, ncol=2)

plt.tight_layout()
out = IMG_DIR / 'seongsu_pattern_similar.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[저장] {out.name}")
