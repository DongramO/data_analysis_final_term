"""
4개 지표별 전년도 대비 변화율 Top 15 상권 (2021~2024)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 & 조인 ────────────────────────────────────────
rental  = pd.read_csv('data/main_data/임대료_공실률.csv', encoding='utf-8-sig')
dongmap = pd.read_csv('data/main_data/rental_vacancy_dongmap.csv', encoding='utf-8-sig')
master  = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig',
                      usecols=['행정동', '행정동_코드', '연도', '복합점수', '복합점수_분위'])
mig     = pd.read_csv('data/main_data/분기별_주말유입인구.csv', encoding='utf-8-sig')
mapping = pd.read_csv('data/main_data/코드매핑_유입인구_매출.csv', encoding='utf-8-sig')
mig_to_raw = dict(zip(mapping['migration_코드'], mapping['raw_코드']))

mig_annual = (mig.groupby(['도착_행정동코드', '연도'])['합계_이동인구']
              .sum().reset_index())
mig_annual['행정동_코드'] = mig_annual['도착_행정동코드'].map(mig_to_raw)
mig_annual = (mig_annual.dropna(subset=['행정동_코드']).copy()
              .assign(행정동_코드=lambda x: x['행정동_코드'].astype(int))
              .groupby(['행정동_코드', '연도'])['합계_이동인구'].sum().reset_index()
              .rename(columns={'합계_이동인구': '트렌드_유동인구'}))

base = (rental[rental['연도'] <= 2024]
        .merge(dongmap[['상권명', '대표_행정동']], on='상권명', how='left')
        .merge(master.rename(columns={'행정동': '대표_행정동'}),
               on=['대표_행정동', '연도'], how='inner')
        .merge(mig_annual, on=['행정동_코드', '연도'], how='left'))

base = base.dropna(subset=['복합점수', '임대료', '트렌드_유동인구']).sort_values(['상권명', '연도'])

# ── YoY 변화율 계산 ───────────────────────────────────────────
base['상권_분위변화']   = base.groupby('상권명')['복합점수_분위'].diff() * 100   # %p
base['유동인구_변화율'] = base.groupby('상권명')['트렌드_유동인구'].pct_change() * 100  # %
base['임대료_변화율']   = base.groupby('상권명')['임대료'].pct_change() * 100       # %
base['공실률_변화']     = base.groupby('상권명')['공실률'].diff()                   # %p

df = base.dropna(subset=['상권_분위변화', '유동인구_변화율', '임대료_변화율', '공실률_변화'])
yoy_years = sorted(df['연도'].unique())   # 2021~2024

# ── 지표 정의 ─────────────────────────────────────────────────
# (col, 제목, 단위, 색상 진하게, 색상 연하게, 파일명, 정렬방향, 방향 주석)
indicators = [
    ('상권_분위변화',   '상권 복합점수 분위 변화 Top 15',
     '%p', '#2166ac', '#b3cde3', 'yoy_top15_상권분위.png',     False, '▲ 값 클수록 상권 지수 상승'),
    ('유동인구_변화율', '트렌드 유동인구 YoY 변화율 Top 15 (금밤·토·일)',
     '%',  '#1a9850', '#b2e0c4', 'yoy_top15_유동인구.png',     False, '▲ 값 클수록 유동인구 증가'),
    ('임대료_변화율',   '임대료 YoY 변화율 Top 15',
     '%',  '#c0392b', '#f4a58a', 'yoy_top15_임대료.png',       False, '▲ 값 클수록 임대료 상승'),
    ('공실률_변화',     '공실률 변화 Top 15 (증가 = 악화)',
     '%p', '#7b3294', '#c994c7', 'yoy_top15_공실률증가.png',   False, '▲ 값 클수록 공실률 급증(악화 심함)'),
]

# ── 지표별 시각화 ─────────────────────────────────────────────
for col, title, unit, dark, light, fname, reverse, note in indicators:
    cmap = LinearSegmentedColormap.from_list('', [light, dark])

    fig, axes = plt.subplots(1, 4, figsize=(23, 7))
    fig.suptitle(f'{title}\n(서울 임대료 연계 64개 상권, {note})',
                 fontsize=14, fontweight='bold', y=1.02)

    for ax, yr in zip(axes, yoy_years):
        sub = df[df['연도'] == yr].copy()

        sub['_plot_val'] = sub[col]
        df_yr = sub.nlargest(15, '_plot_val').sort_values('_plot_val', ascending=True)
        plot_vals = df_yr['_plot_val'].values
        n = len(df_yr)

        bar_colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
        bars = ax.barh(df_yr['상권명'], plot_vals,
                       color=bar_colors, edgecolor='white', linewidth=0.4, height=0.7)

        max_v = plot_vals.max() if len(plot_vals) > 0 else 1
        offset = abs(max_v) * 0.03

        for bar, val in zip(bars, plot_vals):
            # 원래 값으로 레이블 표시 (공실률은 감소량이므로 부호 표시)
            sign = '+' if val >= 0 else ''
            label = f'{sign}{val:.1f}{unit}'
            ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                    label, va='center', ha='left', fontsize=9,
                    fontweight='bold', color='#1a1a4a')

        ax.set_title(f'{yr}년 (vs {yr-1}년)', fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel(f'변화량 ({unit})', fontsize=9)
        ax.set_xlim(0, max_v * 1.3)
        ax.tick_params(axis='y', labelsize=9.5)
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'image/{fname}', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'저장: image/{fname}')

# ── 텍스트 요약 ───────────────────────────────────────────────
print()
for col, title, unit, *_ in indicators:
    print(f'=== {title} ===')
    for yr in yoy_years:
        sub = df[df['연도'] == yr]
        top3 = sub.nlargest(3, col)['상권명'].tolist()
        vals = sub.nlargest(3, col)[col].values
        entries = '  /  '.join(f'{nm} ({v:+.1f}{unit})' for nm, v in zip(top3, vals))
        print(f'  {yr}: {entries}')
    print()
