"""
전년도 대비 변화율 기준 — 상권·유동인구·임대료·공실률 종합 비교
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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
# 상권 복합점수 분위: percentile-point 변화 (×100 → %p 단위)
base['상권_분위변화']   = base.groupby('상권명')['복합점수_분위'].diff() * 100
# 유동인구: % 변화
base['유동인구_변화율'] = base.groupby('상권명')['트렌드_유동인구'].pct_change() * 100
# 임대료: % 변화
base['임대료_변화율']   = base.groupby('상권명')['임대료'].pct_change() * 100
# 공실률: %p 변화 (이미 %이므로 차이만)
base['공실률_변화']     = base.groupby('상권명')['공실률'].diff()

df = base.dropna(subset=['상권_분위변화', '유동인구_변화율', '임대료_변화율', '공실률_변화'])
yoy_years = sorted(df['연도'].unique())   # 2021~2024
n_sk = df['상권명'].nunique()
print(f'YoY 분석 대상: {n_sk}개 상권 × {len(yoy_years)}개 연도 ({len(df)}행)')

# ─────────────────────────────────────────────────────────────
# 그림 1: 4개 지표 YoY 변화 분포 (연도별 박스플롯)
# ─────────────────────────────────────────────────────────────
indicators = [
    ('상권_분위변화',   '상권 복합점수 분위 변화 (%p)',  '#2166ac'),
    ('유동인구_변화율', '유동인구 YoY 변화율 (%)',        '#1a9850'),
    ('임대료_변화율',   '임대료 YoY 변화율 (%)',          '#d73027'),
    ('공실률_변화',     '공실률 변화 (%p)\n(+: 증가, -: 감소)', '#7b3294'),
]

fig1, axes = plt.subplots(2, 2, figsize=(16, 11))
fig1.suptitle('4개 지표 연도별 YoY 변화 분포 (2021~2024)\n(64개 상권, 전년도 대비  |  유동인구: 트렌드 시간대 기준 금밤·토·일 여가이동)',
              fontsize=14, fontweight='bold', y=1.01)

for ax, (col, label, color) in zip(axes.flatten(), indicators):
    data_by_yr = [df[df['연도'] == yr][col].dropna().values for yr in yoy_years]

    bp = ax.boxplot(data_by_yr, tick_labels=yoy_years, patch_artist=True,
                    medianprops=dict(color='black', linewidth=2),
                    flierprops=dict(marker='o', markersize=3, alpha=0.5,
                                    markerfacecolor=color, markeredgecolor='none'))
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.45)

    # 평균 마커 + 값
    for i, (yr, d) in enumerate(zip(yoy_years, data_by_yr)):
        m = np.mean(d)
        ax.plot(i + 1, m, 'D', color=color, markersize=7, zorder=5)
        ax.text(i + 1, m, f'  {m:.1f}', va='center', ha='left', fontsize=8.5,
                color=color, fontweight='bold')

    ax.axhline(0, color='gray', linewidth=0.9, linestyle='--', alpha=0.6)
    ax.set_title(label, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('연도 (vs 전년도)', fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('image/yoy_comparison_boxplot.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: image/yoy_comparison_boxplot.png')

# ─────────────────────────────────────────────────────────────
# 그림 2: 상권분위변화 vs 임대료변화율 산점도 (연도별 4패널)
# ─────────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(17, 14))
fig2.suptitle('상권 분위 변화(%p) vs 임대료 변화율(%) — 연도별\n'
              '색상 = 유동인구 변화율  /  점선 = 추세선',
              fontsize=13, fontweight='bold', y=1.01)

for ax, yr in zip(axes2.flatten(), yoy_years):
    sub = df[df['연도'] == yr].copy()

    vmin = sub['유동인구_변화율'].quantile(0.05)
    vmax = sub['유동인구_변화율'].quantile(0.95)

    sc = ax.scatter(sub['상권_분위변화'], sub['임대료_변화율'],
                    c=sub['유동인구_변화율'], cmap='RdYlGn',
                    vmin=vmin, vmax=vmax,
                    s=65, alpha=0.8, edgecolors='white', linewidths=0.5, zorder=3)

    # 추세선 + 상관계수
    x, y = sub['상권_분위변화'].values, sub['임대료_변화율'].values
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() > 3:
        slope, intercept, r, p, _ = stats.linregress(x[mask], y[mask])
        xline = np.linspace(x[mask].min(), x[mask].max(), 100)
        ax.plot(xline, slope * xline + intercept, '--', color='#333', linewidth=1.5, alpha=0.7)
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        ax.text(0.97, 0.97, f'r={r:.3f}  ({sig})',
                transform=ax.transAxes, ha='right', va='top', fontsize=9.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.75))

    # 상권분위 상위 5 + 하위 5 레이블
    for _, row in sub.nlargest(5, '상권_분위변화').iterrows():
        ax.annotate(row['상권명'], (row['상권_분위변화'], row['임대료_변화율']),
                    xytext=(3, 3), textcoords='offset points', fontsize=7.5, color='#1a3a6a')
    for _, row in sub.nsmallest(5, '상권_분위변화').iterrows():
        ax.annotate(row['상권명'], (row['상권_분위변화'], row['임대료_변화율']),
                    xytext=(3, -9), textcoords='offset points', fontsize=7.5, color='#6a1a1a')

    ax.axhline(0, color='#bbb', linewidth=0.7, linestyle='--')
    ax.axvline(0, color='#bbb', linewidth=0.7, linestyle='--')
    ax.set_xlabel('상권 분위 변화 (%p)', fontsize=9.5)
    ax.set_ylabel('임대료 변화율 (%)', fontsize=9.5)
    ax.set_title(f'{yr}년 (vs {yr-1}년)', fontsize=11, fontweight='bold', pad=8)
    ax.grid(linestyle='--', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.colorbar(sc, ax=ax, label='유동인구 변화율 (%)', shrink=0.8)

plt.tight_layout()
plt.savefig('image/yoy_comparison_scatter.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: image/yoy_comparison_scatter.png')

# ─────────────────────────────────────────────────────────────
# 그림 3: 연도별 지표 간 스피어만 상관계수 추이
# ─────────────────────────────────────────────────────────────
corr_pairs = [
    ('상권_분위변화', '임대료_변화율',   '상권↑ ↔ 임대료↑',     '#d73027'),
    ('상권_분위변화', '유동인구_변화율', '상권↑ ↔ 유동인구↑',   '#2166ac'),
    ('유동인구_변화율', '임대료_변화율', '유동인구↑ ↔ 임대료↑', '#1a9850'),
    ('상권_분위변화', '공실률_변화',     '상권↑ ↔ 공실률변화',   '#7b3294'),
]

corr_tbl = {label: [] for _, _, label, _ in corr_pairs}
pval_tbl = {label: [] for _, _, label, _ in corr_pairs}

for yr in yoy_years:
    sub = df[df['연도'] == yr]
    for c1, c2, label, _ in corr_pairs:
        r, p = stats.spearmanr(sub[c1], sub[c2])
        corr_tbl[label].append(r)
        pval_tbl[label].append(p)

fig3, ax3 = plt.subplots(figsize=(11, 6))
for _, _, label, color in corr_pairs:
    ax3.plot(yoy_years, corr_tbl[label], '-o', color=color,
             linewidth=2.2, markersize=9, label=label)
    for yr, r, p in zip(yoy_years, corr_tbl[label], pval_tbl[label]):
        sig = '*' if p < 0.05 else ''
        offset = 0.025 if r >= 0 else -0.04
        ax3.text(yr, r + offset, f'{r:.2f}{sig}',
                 ha='center', va='bottom', fontsize=8.5, color=color, fontweight='bold')

ax3.axhline(0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
ax3.set_xticks(yoy_years)
ax3.set_xticklabels([f'{yr}\n(vs {yr-1})' for yr in yoy_years], fontsize=10)
ax3.set_ylim(-0.65, 0.75)
ax3.set_ylabel('스피어만 상관계수 (ρ)', fontsize=11)
ax3.set_title('연도별 지표 간 YoY 변화율 스피어만 상관계수 추이\n(* p<0.05)',
              fontsize=13, fontweight='bold', pad=10)
ax3.legend(fontsize=10, loc='lower left', framealpha=0.9)
ax3.grid(linestyle='--', alpha=0.35)
ax3.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('image/yoy_comparison_corr.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: image/yoy_comparison_corr.png')

# ─────────────────────────────────────────────────────────────
# 그림 4: 상권별 4개 지표 평균 YoY 변화 히트맵 (2021~2024 평균)
# ─────────────────────────────────────────────────────────────
avg_yoy = (df.groupby('상권명')[['상권_분위변화', '유동인구_변화율', '임대료_변화율', '공실률_변화']]
           .mean()
           .sort_values('상권_분위변화', ascending=False))

# 각 지표를 절대 최댓값으로 정규화 → -1~+1 (공실률은 감소 방향이 좋으므로 부호 반전)
def norm_col(s):
    m = s.abs().max()
    return s / m if m > 0 else s

norm = pd.DataFrame({
    '상권분위변화':  norm_col(avg_yoy['상권_분위변화']),
    '유동인구변화율': norm_col(avg_yoy['유동인구_변화율']),
    '임대료변화율':   norm_col(avg_yoy['임대료_변화율']),
    '공실률변화':     norm_col(-avg_yoy['공실률_변화']),   # 감소 = 좋음 → 양수 방향
}, index=avg_yoy.index)

col_labels_hm = ['상권 분위\n변화(%p)', '유동인구\n변화율(%)', '임대료\n변화율(%)',
                 '공실률 변화(%p)\n감소↑ 좋음']
raw_values = avg_yoy[['상권_분위변화', '유동인구_변화율', '임대료_변화율', '공실률_변화']].values.T
hm_data    = norm.values.T  # (4, n)

fig4, ax4 = plt.subplots(figsize=(21, 4.5))
im = ax4.imshow(hm_data, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)

ax4.set_xticks(range(len(avg_yoy)))
ax4.set_xticklabels(avg_yoy.index, rotation=55, ha='right', fontsize=8)
ax4.set_yticks(range(4))
ax4.set_yticklabels(col_labels_hm, fontsize=10)

for j in range(len(avg_yoy)):
    for i in range(4):
        raw = raw_values[i, j]
        txt_color = 'black' if abs(hm_data[i, j]) < 0.6 else 'white'
        fmt = f'{raw:+.1f}' if i != 0 else f'{raw:+.1f}'
        ax4.text(j, i, fmt, ha='center', va='center', fontsize=7, color=txt_color)

plt.colorbar(im, ax=ax4, label='정규화 점수 (-1=최저 / +1=최고)', fraction=0.012, pad=0.02)
ax4.set_title('상권별 4개 지표 평균 YoY 변화 종합 비교 (2021~2024 평균, 상권 분위 상승 순)\n'
              '※ 공실률 행은 감소(개선) 방향이 녹색',
              fontsize=12, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig('image/yoy_comparison_heatmap.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: image/yoy_comparison_heatmap.png')

# ── 텍스트 요약 ───────────────────────────────────────────────
print('\n=== 연도별 4개 지표 평균 YoY 변화 ===')
for yr in yoy_years:
    sub = df[df['연도'] == yr]
    print(f'  {yr}: 상권분위 {sub["상권_분위변화"].mean():+.1f}%p  '
          f'유동인구 {sub["유동인구_변화율"].mean():+.1f}%  '
          f'임대료 {sub["임대료_변화율"].mean():+.1f}%  '
          f'공실률 {sub["공실률_변화"].mean():+.1f}%p')

print('\n=== 2021~2024 평균 상권분위 상승 Top 5 (및 타 지표) ===')
for nm, row in avg_yoy.head(5).iterrows():
    print(f'  {nm}: 상권 {row["상권_분위변화"]:+.1f}%p  '
          f'유동인구 {row["유동인구_변화율"]:+.1f}%  '
          f'임대료 {row["임대료_변화율"]:+.1f}%  '
          f'공실률 {row["공실률_변화"]:+.1f}%p')

print('\n=== 상권↑ 이지만 임대료 변화 가장 더딘 곳 (갭 확대, 저평가 가능성) ===')
avg_yoy['갭'] = avg_yoy['상권_분위변화'] - avg_yoy['임대료_변화율']
for nm, row in avg_yoy.nlargest(5, '갭').iterrows():
    print(f'  {nm}: 상권 {row["상권_분위변화"]:+.1f}%p  임대료 {row["임대료_변화율"]:+.1f}%  (갭 {row["갭"]:+.1f})')

print('\n=== 임대료↑ 이지만 상권 변화 더딘 곳 (임대료 선행 또는 과도한 상승) ===')
for nm, row in avg_yoy.nsmallest(5, '갭').iterrows():
    print(f'  {nm}: 상권 {row["상권_분위변화"]:+.1f}%p  임대료 {row["임대료_변화율"]:+.1f}%  (갭 {row["갭"]:+.1f})')
