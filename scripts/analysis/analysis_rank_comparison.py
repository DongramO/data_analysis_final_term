"""
유동인구 × 상권 × 임대료/공실률 랭크 종합 비교
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 데이터 로드 ────────────────────────────────────────────────
rental   = pd.read_csv('data/main_data/임대료_공실률.csv', encoding='utf-8-sig')
dongmap  = pd.read_csv('data/main_data/rental_vacancy_dongmap.csv', encoding='utf-8-sig')
master   = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig',
                       usecols=['행정동', '행정동_코드', '연도', '복합점수', '복합점수_분위'])
mig     = pd.read_csv('data/main_data/분기별_주말유입인구.csv', encoding='utf-8-sig')
mapping = pd.read_csv('data/main_data/코드매핑_유입인구_매출.csv', encoding='utf-8-sig')
mig_to_raw = dict(zip(mapping['migration_코드'], mapping['raw_코드']))

# ── 트렌드 유동인구 연간 합계 (금밤·토·일, 여가이동) ───────────
mig_annual = (mig.groupby(['도착_행정동코드', '연도'])['합계_이동인구']
              .sum().reset_index())
mig_annual['행정동_코드'] = mig_annual['도착_행정동코드'].map(mig_to_raw)
mig_annual = (mig_annual.dropna(subset=['행정동_코드']).copy()
              .assign(행정동_코드=lambda x: x['행정동_코드'].astype(int))
              .groupby(['행정동_코드', '연도'])['합계_이동인구'].sum().reset_index()
              .rename(columns={'합계_이동인구': '트렌드_유동인구'}))

# ── 조인 ──────────────────────────────────────────────────────
df = (rental[rental['연도'] <= 2024]
      .merge(dongmap[['상권명', '대표_행정동']], on='상권명', how='left')
      .merge(master.rename(columns={'행정동': '대표_행정동'}),
             on=['대표_행정동', '연도'], how='inner')
      .merge(mig_annual, on=['행정동_코드', '연도'], how='left'))

df = df.dropna(subset=['복합점수', '임대료', '트렌드_유동인구'])
years = sorted(df['연도'].unique())
n = df['상권명'].nunique()
print(f'분석 대상: {n}개 상권 × {len(years)}개 연도 ({len(df)}행)')

# ── 연도별 랭크 (1=최상위) ────────────────────────────────────
for yr in years:
    mask = df['연도'] == yr
    sub  = df.loc[mask]
    df.loc[mask, '상권_랭크']   = sub['복합점수'].rank(ascending=False, method='min').astype(int)
    df.loc[mask, '유동인구_랭크'] = sub['트렌드_유동인구'].rank(ascending=False, method='min').astype(int)
    df.loc[mask, '임대료_랭크']  = sub['임대료'].rank(ascending=False, method='min').astype(int)
    df.loc[mask, '공실률_랭크']  = sub['공실률'].rank(ascending=True,  method='min').astype(int)  # 낮을수록 좋음

# ── 상위 15 선정 (2024 상권랭크 기준) ─────────────────────────
top15 = (df[df['연도'] == 2024]
         .nsmallest(15, '상권_랭크')['상권명'].tolist())

# ─────────────────────────────────────────────────────────────
# 그림 1: Bump chart (Top 15 상권, 3개 지표)
# ─────────────────────────────────────────────────────────────
palette = [plt.cm.tab20(i / 20) for i in range(20)]
c_map   = {nm: palette[i] for i, nm in enumerate(top15)}

fig1, axes = plt.subplots(1, 3, figsize=(24, 9))
fig1.suptitle('상위 15개 상권 — 연도별 랭크 추이 비교 (2020~2024)\n(1위 = 최상위 / 유동인구: 트렌드 시간대 기준 금밤·토·일 여가이동)',
              fontsize=14, fontweight='bold', y=1.02)

df_top = df[df['상권명'].isin(top15)]
rank_panels = [
    ('상권_랭크',   '상권 복합점수 랭크'),
    ('유동인구_랭크', '유동인구 랭크'),
    ('임대료_랭크',  '임대료 랭크'),
]

for ax, (col, title) in zip(axes, rank_panels):
    for nm in top15:
        sub = df_top[df_top['상권명'] == nm].sort_values('연도')
        if len(sub) < 2:
            continue
        xv, yv = sub['연도'].values, sub[col].values
        ax.plot(xv, yv, '-o', color=c_map[nm],
                linewidth=2, markersize=7, alpha=0.85, zorder=3)
        # 마커 위 랭크 숫자
        for xi, yi in zip(xv, yv):
            ax.text(xi, yi - 0.5, str(int(yi)),
                    ha='center', va='bottom', fontsize=6.5,
                    color=c_map[nm], fontweight='bold')
        # 2024 끝점 상권명
        row24 = sub[sub['연도'] == 2024]
        if not row24.empty:
            ax.text(2024.12, row24[col].values[0], nm,
                    va='center', ha='left', fontsize=8, color=c_map[nm])

    ax.set_xlim(2019.7, 2027)
    ax.set_ylim(n + 1.5, 0.2)   # 1위가 위쪽
    ax.set_xticks(years)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_ylabel('랭크 (낮을수록 상위)', fontsize=9)
    ax.grid(linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('image/rank_comparison_bump.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: image/rank_comparison_bump.png')

# ─────────────────────────────────────────────────────────────
# 그림 2: 상권랭크 vs 임대료랭크 산점도 (2020 / 2024)
# ─────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))
fig2.suptitle('상권 복합점수 랭크 vs 임대료 랭크 비교 (2020 → 2024)\n'
              '버블 크기 = 트렌드 유동인구 규모(금밤·토·일)  /  색상 = 공실률(%)',
              fontsize=13, fontweight='bold', y=1.02)

for ax, yr in zip(axes2, [2020, 2024]):
    sub = df[df['연도'] == yr].copy()
    # 버블 크기: 유동인구 1위 → 가장 큰 원
    sub['bubble'] = (n + 1 - sub['유동인구_랭크']) / n * 350 + 40

    sc = ax.scatter(sub['상권_랭크'], sub['임대료_랭크'],
                    s=sub['bubble'], c=sub['공실률'],
                    cmap='RdYlGn_r', alpha=0.75,
                    edgecolors='white', linewidths=0.6,
                    vmin=sub['공실률'].quantile(0.05),
                    vmax=sub['공실률'].quantile(0.95))

    # 상권랭크 상위 12개 레이블
    for _, row in sub.nsmallest(12, '상권_랭크').iterrows():
        ax.annotate(row['상권명'],
                    (row['상권_랭크'], row['임대료_랭크']),
                    xytext=(4, 4), textcoords='offset points',
                    fontsize=7.5, color='#333')

    # 1:1 대각선
    ax.plot([1, n], [1, n], '--', color='#aaa', linewidth=1, alpha=0.6)
    ax.text(n * 0.62, n * 0.52, '상권랭크=임대료랭크\n(균형선)',
            fontsize=7.5, color='#999', ha='center')

    ax.set_xlim(0, n + 2)
    ax.set_ylim(0, n + 2)
    ax.set_xlabel('상권 복합점수 랭크 (1=최상위)', fontsize=10)
    ax.set_ylabel('임대료 랭크 (1=최고가)', fontsize=10)
    ax.set_title(f'{yr}년', fontsize=12, fontweight='bold', pad=8)
    ax.grid(linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.colorbar(sc, ax=ax, label='공실률 (%)', shrink=0.8)

plt.tight_layout()
plt.savefig('image/rank_comparison_scatter.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: image/rank_comparison_scatter.png')

# ─────────────────────────────────────────────────────────────
# 그림 3: 종합 랭크 히트맵 (2024, 상권랭크 오름차순)
# ─────────────────────────────────────────────────────────────
d24 = (df[df['연도'] == 2024]
       .set_index('상권명')
       .sort_values('상권_랭크'))

cols_hm   = ['상권_랭크', '유동인구_랭크', '임대료_랭크', '공실률_랭크']
labels_hm = ['상권\n복합점수', '유동인구', '임대료', '공실률\n(↑낮을수록 우수)']
data_hm   = d24[cols_hm].values.T  # (4, n_상권)

fig3, ax3 = plt.subplots(figsize=(20, 4.5))
im = ax3.imshow(data_hm, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=n)

ax3.set_xticks(range(len(d24)))
ax3.set_xticklabels(d24.index, rotation=55, ha='right', fontsize=8)
ax3.set_yticks(range(4))
ax3.set_yticklabels(labels_hm, fontsize=10)

# 셀 내 랭크 숫자
for j in range(len(d24)):
    for i in range(4):
        val = int(data_hm[i, j])
        txt_color = 'white' if val > n * 0.55 else 'black'
        ax3.text(j, i, str(val), ha='center', va='center',
                 fontsize=7.5, color=txt_color)

plt.colorbar(im, ax=ax3, label='랭크 (1=최상위 녹색 / 최하위 적색)',
             fraction=0.012, pad=0.02)
ax3.set_title('2024년 기준 상권별 4개 지표 랭크 종합 비교 (상권 복합점수 순 정렬)',
              fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('image/rank_comparison_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: image/rank_comparison_heatmap.png')

# ─────────────────────────────────────────────────────────────
# 그림 4: 연도별 랭크 간 스피어만 상관계수 추이
# ─────────────────────────────────────────────────────────────
corr_pairs = [
    ('상권_랭크', '유동인구_랭크', '상권 vs 유동인구', '#2166ac'),
    ('상권_랭크', '임대료_랭크',   '상권 vs 임대료',   '#d73027'),
    ('상권_랭크', '공실률_랭크',   '상권 vs 공실률',   '#4dac26'),
    ('유동인구_랭크', '임대료_랭크', '유동인구 vs 임대료', '#7b3294'),
]

corr_results = {label: [] for _, _, label, _ in corr_pairs}
for yr in years:
    sub = df[df['연도'] == yr]
    for c1, c2, label, _ in corr_pairs:
        r, _ = stats.spearmanr(sub[c1], sub[c2])
        corr_results[label].append(r)

fig4, ax4 = plt.subplots(figsize=(10, 5))
for _, _, label, color in corr_pairs:
    ax4.plot(years, corr_results[label], '-o', color=color,
             linewidth=2, markersize=8, label=label)
    for yr, r in zip(years, corr_results[label]):
        ax4.text(yr, r + 0.012, f'{r:.2f}', ha='center', va='bottom',
                 fontsize=8, color=color)

ax4.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
ax4.set_xticks(years)
ax4.set_xlabel('연도', fontsize=11)
ax4.set_ylabel('스피어만 상관계수 (ρ)', fontsize=11)
ax4.set_title('연도별 랭크 간 스피어만 상관계수 추이\n(상권·유동인구·임대료·공실률)',
              fontsize=13, fontweight='bold', pad=10)
ax4.legend(fontsize=9.5, loc='lower right')
ax4.set_ylim(-0.3, 1.0)
ax4.grid(linestyle='--', alpha=0.35)
ax4.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('image/rank_comparison_corr_trend.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: image/rank_comparison_corr_trend.png')

# ── 텍스트 요약 ───────────────────────────────────────────────
print('\n=== 2024년 랭크 간 스피어만 상관계수 ===')
sub24 = df[df['연도'] == 2024]
for c1, c2, label, _ in corr_pairs:
    r, p = stats.spearmanr(sub24[c1], sub24[c2])
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    print(f'  {label}: ρ={r:.3f}  p={p:.4f} {sig}')

print('\n=== 2020→2024 상권랭크 상승 폭 Top 5 ===')
p20 = df[df['연도'] == 2020].set_index('상권명')[['상권_랭크', '임대료_랭크', '유동인구_랭크']]
p24 = df[df['연도'] == 2024].set_index('상권명')[['상권_랭크', '임대료_랭크', '유동인구_랭크']]
chg = p20.join(p24, lsuffix='_2020', rsuffix='_2024', how='inner')
chg['상권_상승'] = chg['상권_랭크_2020'] - chg['상권_랭크_2024']   # 양수 = 순위 상승
for nm, row in chg.nlargest(5, '상권_상승').iterrows():
    print(f'  {nm}: 상권 {int(row.상권_랭크_2020)}→{int(row.상권_랭크_2024)} '
          f'(+{int(row.상권_상승)}계단)  '
          f'임대료 {int(row.임대료_랭크_2020)}→{int(row.임대료_랭크_2024)}')

print('\n=== 상권은 높은데 임대료는 낮은 곳 (2024, 상권-임대료 랭크 차 상위 5) ===')
gap = sub24.copy()
gap['갭'] = gap['임대료_랭크'] - gap['상권_랭크']   # 양수 = 임대료 대비 상권 우위
for _, row in gap.nlargest(5, '갭').iterrows():
    print(f'  {row["상권명"]}: 상권 {int(row["상권_랭크"])}위 / 임대료 {int(row["임대료_랭크"])}위 '
          f'(갭 {int(row["갭"])}계단)')

print('\n=== 임대료는 높은데 상권은 낮은 곳 (2024, 임대료 고평가 상위 5) ===')
for _, row in gap.nsmallest(5, '갭').iterrows():
    print(f'  {row["상권명"]}: 상권 {int(row["상권_랭크"])}위 / 임대료 {int(row["임대료_랭크"])}위 '
          f'(갭 {int(row["갭"])}계단)')
