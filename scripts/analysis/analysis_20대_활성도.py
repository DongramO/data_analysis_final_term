# -*- coding: utf-8 -*-
"""
"20대의 매출 활성도 = 상권의 활성도" 가설 검증
데이터: 행정동_업종별_매출정리.csv (전체 요일 기준 20대 매출)

출력: image/20대_활성도_분석.png
Panel A (상단): 18개 대표 트렌드 상권 연도별 20대 매출 비중(좌) + 총매출 지수(우)
Panel B (하단 좌): 20대 비중(2024) × 총매출 성장률(2020→2024) 산점도 (423개 동)
Panel C (하단 우): 20대 비중 분위별 총매출 성장률 분포 비교
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT    = Path(__file__).resolve().parent
SRC     = ROOT / 'data/행정동_업종별_매출정리.csv'
IMG_DIR = ROOT / 'image'

YEARS = [2020, 2021, 2022, 2023, 2024]

# ── 대표 트렌드 상권 (행정동 이름 기준) ────────────────────
DONG_MAP = {
    '홍대(서교동)':    ['서교동'],
    '연남동':          ['연남동'],
    '합정·망원':       ['합정동', '망원1동', '망원2동'],
    '이태원':          ['이태원1동', '이태원2동'],
    '한남동':          ['한남동'],
    '용산해방촌':      ['용산2가동'],
    '서촌':            ['청운효자동'],
    '북촌(가회동)':    ['가회동'],
    '삼청동':          ['삼청동'],
    '대학로(혜화동)':  ['혜화동'],
    '화양동':          ['화양동'],
    '성수1가1동':      ['성수1가1동'],
    '성수1가2동':      ['성수1가2동'],
    '성수2가1동':      ['성수2가1동'],
    '성수2가3동':      ['성수2가3동'],
    '낙성대동':        ['낙성대동'],
    '방배동':          ['방배2동'],
    '잠실':            ['잠실6동', '잠실본동'],
}

# ══ 1. 데이터 로드 & 행정동×연도 집계 ═══════════════════════
raw = pd.read_csv(SRC, encoding='utf-8-sig')
raw.columns = [c.strip() for c in raw.columns]

agg = raw.groupby(['행정동', '연도']).agg(
    총매출=('당월_매출_금액', 'sum'),
    매출20=('연령대_20_매출_금액', 'sum'),
).reset_index()
agg['비중20'] = agg['매출20'] / agg['총매출'] * 100

# ══ 2. 패널 A용: 대표 상권 데이터 ════════════════════════════
def get_dong_series(dongs):
    sub = agg[agg['행정동'].isin(dongs)].groupby('연도')[['총매출','매출20']].sum()
    sub = sub.reindex(YEARS)
    비중 = (sub['매출20'] / sub['총매출'] * 100).values
    base = sub['총매출'][2020]
    지수 = (sub['총매출'] / base * 100).values if base > 0 else np.full(5, np.nan)
    return 비중, 지수

# ══ 3. 패널 B·C용: 423개 동 성장률 산출 ═════════════════════
wide = agg.pivot_table(index='행정동', columns='연도', values=['총매출','비중20'])
wide.columns = [f'{v}_{y}' for v,y in wide.columns]
wide = wide.reset_index()

wide = wide.dropna(subset=['총매출_2020','총매출_2024','비중20_2020','비중20_2024'])
wide = wide[wide['총매출_2020'] > 0]

wide['매출성장률'] = (wide['총매출_2024'] - wide['총매출_2020']) / wide['총매출_2020'] * 100
wide['비중20_변화'] = wide['비중20_2024'] - wide['비중20_2020']
# 이상치 제거 (성장률 5th~95th percentile)
p5, p95 = wide['매출성장률'].quantile([0.05, 0.95])
scatter_df = wide[(wide['매출성장률'] >= p5) & (wide['매출성장률'] <= p95)].copy()

# 20대 비중 분위 (2024 기준 4분위)
scatter_df['비중분위'] = pd.qcut(scatter_df['비중20_2024'], 4,
                                  labels=['하위25%','25~50%','50~75%','상위25%'])

# ══ 4. 시각화 ════════════════════════════════════════════════
N     = len(DONG_MAP)
NCOLS = 3
NROWS_A = (N + NCOLS - 1) // NCOLS   # 6

fig = plt.figure(figsize=(21, NROWS_A * 4.0 + 9))

# ── Panel A: 대표 상권 그리드 ──────────────────────────────
C20  = '#E05C5C'
CBAR = '#AACFE4'

gs_top = fig.add_gridspec(NROWS_A, NCOLS,
                           top=1.00, bottom=0.36,
                           hspace=0.55, wspace=0.35)

for idx, (name, dongs) in enumerate(DONG_MAP.items()):
    row, col = divmod(idx, NCOLS)
    ax = fig.add_subplot(gs_top[row, col])

    비중, 지수 = get_dong_series(dongs)

    # 우축: 총매출 지수 바
    ax2 = ax.twinx()
    ax2.bar(YEARS, 지수, width=0.6, color=CBAR, alpha=0.40, zorder=1)
    ax2.set_ylim(0, max(np.nanmax(지수) * 1.55, 220) if not np.all(np.isnan(지수)) else 300)
    ax2.set_ylabel('총매출 지수\n(2020=100)', fontsize=7, color='#4477AA')
    ax2.tick_params(axis='y', labelsize=7, colors='#4477AA')
    ax2.spines['right'].set_edgecolor(CBAR)
    for yr, v in zip(YEARS, 지수):
        if not np.isnan(v):
            ax2.text(yr, v + 2, f'{v:.0f}', ha='center', fontsize=6, color='#4477AA')
    ax2.set_zorder(1)

    # 좌축: 20대 매출 비중 라인
    ax.set_zorder(2)
    ax.patch.set_visible(False)
    ax.plot(YEARS, 비중, color=C20, lw=2.5, marker='o', markersize=6, zorder=3)
    for yr, v in zip(YEARS, 비중):
        if not np.isnan(v):
            ax.text(yr, v + 0.4, f'{v:.1f}%', ha='center', fontsize=6.5, color=C20, zorder=4)

    d비중 = 비중[-1] - 비중[0] if not np.isnan(비중[-1]) else np.nan
    d지수 = 지수[-1] - 100    if not np.isnan(지수[-1]) else np.nan
    ax.set_title(
        f'{name}\n20대 비중: {비중[0]:.1f}% → {비중[-1]:.1f}% ({d비중:+.1f}%p)  │  총매출 지수: {d지수:+.0f}',
        fontsize=8.5, pad=4)

    ax.set_xticks(YEARS)
    ax.set_xticklabels(["'20","'21","'22","'23","'24"], fontsize=9)
    ax.set_ylabel('20대 매출 비중 (%)', fontsize=8)
    valid = 비중[~np.isnan(비중)]
    if len(valid):
        ax.set_ylim(max(0, valid.min() - 4), min(60, valid.max() + 4))
    ax.grid(alpha=0.18, zorder=0)

    if idx == 0:
        handles = [
            Line2D([0],[0], color=C20,  lw=2, marker='o', markersize=5, label='20대 매출 비중 (전체 요일)'),
            mpatches.Patch(facecolor=CBAR, alpha=0.6, label='총매출 지수 (우축, 2020=100)'),
        ]
        ax.legend(handles=handles, fontsize=8, loc='upper left')

# 빈 칸 숨기기
for idx in range(N, NROWS_A * NCOLS):
    row, col = divmod(idx, NCOLS)
    fig.add_subplot(gs_top[row, col]).axis('off')

# ── Panel B + C: 하단 두 패널 ─────────────────────────────
gs_bot = fig.add_gridspec(1, 2, top=0.30, bottom=0.04, wspace=0.35)

# Panel B: 산점도 (20대 비중 vs 총매출 성장률)
ax_b = fig.add_subplot(gs_bot[0, 0])
colors_b = scatter_df['비중분위'].map({
    '하위25%': '#AAAAAA', '25~50%': '#F5A623', '50~75%': '#E05C5C', '상위25%': '#8B1A1A'
})
ax_b.scatter(scatter_df['비중20_2024'], scatter_df['매출성장률'],
             c=colors_b, alpha=0.55, s=28, zorder=3)

# 회귀선
x_val = scatter_df['비중20_2024'].values
y_val = scatter_df['매출성장률'].values
slope, intercept, r, p, _ = stats.linregress(x_val, y_val)
x_line = np.linspace(x_val.min(), x_val.max(), 100)
ax_b.plot(x_line, slope * x_line + intercept, color='#333333', lw=1.8, ls='--', zorder=4)
ax_b.text(0.97, 0.05, f'r = {r:.2f}  (p{"<0.001" if p<0.001 else f"={p:.3f}"})',
          transform=ax_b.transAxes, ha='right', fontsize=9,
          bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

# 대표 상권 라벨
rep_dongs_flat = {d: name for name, dlist in DONG_MAP.items() for d in dlist}
for _, row in scatter_df.iterrows():
    if row['행정동'] in rep_dongs_flat:
        ax_b.annotate(rep_dongs_flat[row['행정동']],
                      (row['비중20_2024'], row['매출성장률']),
                      fontsize=6.5, ha='left', va='bottom',
                      xytext=(3, 3), textcoords='offset points', color='#333333')

ax_b.axhline(0, color='gray', lw=0.8, ls=':')
ax_b.set_xlabel('20대 매출 비중, 2024 (%)', fontsize=10)
ax_b.set_ylabel('총매출 성장률, 2020→2024 (%)', fontsize=10)
ax_b.set_title('20대 매출 비중 × 총매출 성장률\n(423개 행정동, 2024 기준)', fontsize=11, fontweight='bold')
ax_b.grid(alpha=0.2)

patch_labels = ['하위 25%','25~50%','50~75%','상위 25%']
patch_colors = ['#AAAAAA','#F5A623','#E05C5C','#8B1A1A']
ax_b.legend(handles=[mpatches.Patch(color=c, label=l) for c,l in zip(patch_colors, patch_labels)],
            title='20대 비중 분위', fontsize=8, title_fontsize=8, loc='upper left')

# Panel C: 20대 비중 분위별 총매출 성장률 분포
ax_c = fig.add_subplot(gs_bot[0, 1])
quartile_order = ['하위25%','25~50%','50~75%','상위25%']
data_groups = [scatter_df[scatter_df['비중분위']==q]['매출성장률'].values for q in quartile_order]
bp = ax_c.boxplot(data_groups, labels=['하위\n25%','25~\n50%','50~\n75%','상위\n25%'],
                  patch_artist=True, medianprops=dict(color='black', lw=2),
                  flierprops=dict(marker='o', markersize=3, alpha=0.4))
for patch, color in zip(bp['boxes'], patch_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# 각 그룹 중앙값 표시
for i, grp in enumerate(data_groups):
    med = np.median(grp)
    ax_c.text(i+1, med + 1.5, f'{med:.1f}%', ha='center', fontsize=9, fontweight='bold')

# 그룹 간 유의차 표시 (첫 그룹 vs 마지막 그룹)
stat, pval = stats.mannwhitneyu(data_groups[0], data_groups[-1], alternative='two-sided')
sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'n.s.'))
y_top = max(data_groups[0].max(), data_groups[-1].max()) + 5
ax_c.annotate('', xy=(4, y_top), xytext=(1, y_top),
              arrowprops=dict(arrowstyle='-', color='black'))
ax_c.text(2.5, y_top + 1.5, f'p{("<0.001" if pval<0.001 else f"={pval:.3f}")} {sig}',
          ha='center', fontsize=9)

ax_c.axhline(0, color='gray', lw=0.8, ls=':')
ax_c.set_xlabel('20대 매출 비중 분위 (2024 기준)', fontsize=10)
ax_c.set_ylabel('총매출 성장률, 2020→2024 (%)', fontsize=10)
ax_c.set_title('20대 매출 비중 분위별\n총매출 성장률 분포', fontsize=11, fontweight='bold')
ax_c.grid(alpha=0.2, axis='y')

# ── 전체 제목 ──────────────────────────────────────────────
fig.text(0.5, 1.003,
         '"20대의 매출 활성도가 상권의 활성도"  │  20대 매출 비중 (전체 요일 기준, 행정동_업종별_매출정리.csv)',
         ha='center', va='bottom', fontsize=13, fontweight='bold')

out = IMG_DIR / '20대_활성도_분석.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'[저장] {out.name}')
