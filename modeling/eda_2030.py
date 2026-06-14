"""
EDA — 2030세대 유동성 · 소비 성향과 상권 활성화의 관계
핵심 질문:
  1) 2030 이동비율이 높을 때 상권이 활성화되는가?
  2) 2030 소비 변화와 트렌드 지수 변화가 같은 방향으로 움직이는가?
  3) 어떤 행정동에서 2030 - 상권이 동행하고, 어디서는 무관한가?

출력 이미지 (modeling/image/):
  eda_2030_01_mobility_target.png    — 2030 이동비율 분포 × 타겟
  eda_2030_02_consumption_target.png — 2030 소비비중 분포 × 타겟
  eda_2030_03_sync_scatter.png       — 동편차 동기화 scatter (2030 vs 트렌드)
  eda_2030_04_yoy_sync.png           — YoY 변화 동행 분석
  eda_2030_05_lead_lag.png           — 2030 모멘텀 → 다음 분기 타겟 (선행 효과)
  eda_2030_06_dong_correlation.png   — 행정동별 2030-상권 동행 정도
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

IMG_DIR    = Path('modeling/image')
TRAIN_PATH = 'data/main_data/ml_dataset_quarterly_train.csv'

LABEL_ORDER  = ['상승', '유지', '하락']
LABEL_COLORS = {'상승': '#2166ac', '유지': '#999999', '하락': '#d6604d'}

train = pd.read_csv(TRAIN_PATH, encoding='utf-8-sig')
df = train.dropna(subset=['타겟_레이블']).copy()

print(f'분석 데이터: {len(df)}행 ({df["행정동"].nunique()}개 동)')

# ══════════════════════════════════════════════════════════════
# Chart 1: 2030 이동비율 분포 × 타겟 클래스
# 상승 클래스 행정동이 이동비율 자체가 높은가?
# ══════════════════════════════════════════════════════════════
MOB_FEATS = [
    ('2030_이동비율',    '2030\n이동비율\n(전체)'),
    ('20대_이동비율',    '20대\n이동비율'),
    ('30대_이동비율',    '30대\n이동비율'),
    ('2030_이동비율_동편차',  '2030 이동비율\n동편차'),
    ('HE_비율',          'HE 비율\n(가정→여가)'),
]

fig, axes = plt.subplots(1, len(MOB_FEATS), figsize=(20, 5))
fig.suptitle('2030 이동비율 분포 by 타겟 클래스\n'
             '(이동비율 = 금Fri저녁+토+일 트렌드 시간대 도착 비율, 동편차 = 현재 - 자기 평균)',
             fontsize=13, fontweight='bold')

for ax, (col, label) in zip(axes, MOB_FEATS):
    if col not in df.columns:
        ax.set_visible(False)
        continue
    data = [df[df['타겟_레이블'] == lbl][col].dropna() for lbl in LABEL_ORDER]
    parts = ax.violinplot(data, positions=[0, 1, 2], showmedians=True,
                          showextrema=False, widths=0.6)
    for pc, lbl in zip(parts['bodies'], LABEL_ORDER):
        pc.set_facecolor(LABEL_COLORS[lbl])
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)

    for i, lbl in enumerate(LABEL_ORDER):
        med = df[df['타겟_레이블'] == lbl][col].median()
        ax.text(i, ax.get_ylim()[1] * 0.02 if i == 0 else med,
                f'{med:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 클래스 간 t-test p-value (상승 vs 하락)
    g1 = df[df['타겟_레이블'] == '상승'][col].dropna()
    g2 = df[df['타겟_레이블'] == '하락'][col].dropna()
    if len(g1) > 5 and len(g2) > 5:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(g1, g2, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.set_xlabel(f'상승 vs 하락: {sig} (p={p:.3f})', fontsize=7)

    ax.axhline(0 if '동편차' in col else df[col].median(),
               color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(LABEL_ORDER, fontsize=9)
    ax.set_title(label, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

axes[0].set_ylabel('이동비율 / 동편차')
plt.tight_layout()
plt.savefig(IMG_DIR / 'eda_2030_01_mobility_target.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: eda_2030_01_mobility_target.png')

# ══════════════════════════════════════════════════════════════
# Chart 2: 2030 소비비중 분포 × 타겟 클래스
# ══════════════════════════════════════════════════════════════
CON_FEATS = [
    ('2030소비_비중',         '2030\n소비비중'),
    ('연령대20_비중',          '20대\n소비비중'),
    ('연령대30_비중',          '30대\n소비비중'),
    ('2030소비_비중_동편차',   '2030 소비\n동편차'),
    ('2030소비_비중_변화',     '2030 소비\nYoY 변화'),
]

fig, axes = plt.subplots(1, len(CON_FEATS), figsize=(20, 5))
fig.suptitle('2030 소비비중 분포 by 타겟 클래스\n'
             '(소비비중 = 전체 매출 중 20·30대 비중)',
             fontsize=13, fontweight='bold')

for ax, (col, label) in zip(axes, CON_FEATS):
    if col not in df.columns:
        ax.set_visible(False)
        continue
    data = [df[df['타겟_레이블'] == lbl][col].dropna() for lbl in LABEL_ORDER]
    parts = ax.violinplot(data, positions=[0, 1, 2], showmedians=True,
                          showextrema=False, widths=0.6)
    for pc, lbl in zip(parts['bodies'], LABEL_ORDER):
        pc.set_facecolor(LABEL_COLORS[lbl])
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)

    for i, lbl in enumerate(LABEL_ORDER):
        med = df[df['타겟_레이블'] == lbl][col].median()
        ax.text(i, med, f'{med:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    g1 = df[df['타겟_레이블'] == '상승'][col].dropna()
    g2 = df[df['타겟_레이블'] == '하락'][col].dropna()
    if len(g1) > 5 and len(g2) > 5:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(g1, g2, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.set_xlabel(f'상승 vs 하락: {sig} (p={p:.3f})', fontsize=7)

    ax.axhline(0 if '변화' in col or '동편차' in col else df[col].median(),
               color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(LABEL_ORDER, fontsize=9)
    ax.set_title(label, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)

axes[0].set_ylabel('소비비중')
plt.tight_layout()
plt.savefig(IMG_DIR / 'eda_2030_02_consumption_target.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: eda_2030_02_consumption_target.png')

# ══════════════════════════════════════════════════════════════
# Chart 3: 동편차 동기화 scatter
# 2030이 자기 평균보다 높을 때 트렌드도 자기 평균보다 높은가?
# ══════════════════════════════════════════════════════════════
pairs = [
    ('2030소비_비중_동편차',   '트렌드시간대_비중_동편차', '2030소비 동편차', '트렌드시간대 동편차'),
    ('2030_이동비율_동편차',  '트렌드시간대_비중_동편차', '2030이동 동편차', '트렌드시간대 동편차'),
    ('2030소비_비중_변화',    '트렌드시간대_비중_변화',   '2030소비 YoY변화', '트렌드시간대 YoY변화'),
    ('2030_이동비율_변화',   '트렌드시간대_비중_변화',   '2030이동 YoY변화', '트렌드시간대 YoY변화'),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('2030 지표 vs 트렌드 지표 동기화 분석\n'
             '(색상 = 타겟 클래스, 우측상단 = 같은 방향으로 움직이는 케이스)',
             fontsize=13, fontweight='bold')

for ax, (xcol, ycol, xlabel, ylabel) in zip(axes.flatten(), pairs):
    if xcol not in df.columns or ycol not in df.columns:
        ax.set_visible(False)
        continue

    sub = df[[xcol, ycol, '타겟_레이블']].dropna()
    for lbl in LABEL_ORDER:
        s = sub[sub['타겟_레이블'] == lbl]
        ax.scatter(s[xcol], s[ycol], c=LABEL_COLORS[lbl],
                   alpha=0.25, s=12, label=lbl, rasterized=True)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.7, alpha=0.4)
    ax.axvline(0, color='black', linestyle='--', linewidth=0.7, alpha=0.4)

    # 추세선
    valid = sub.dropna()
    if len(valid) > 10:
        m, b = np.polyfit(valid[xcol], valid[ycol], 1)
        x_line = np.linspace(valid[xcol].quantile(0.02), valid[xcol].quantile(0.98), 50)
        ax.plot(x_line, m * x_line + b, 'k-', linewidth=1.5, alpha=0.6)
        r, p = pearsonr(valid[xcol], valid[ycol])
        ax.text(0.97, 0.97, f'r = {r:.3f}\np = {p:.3f}',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # 사분면 색칠 (동행 = 우상단/좌하단)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.fill_between([0, max(xlim)], 0, max(ylim), alpha=0.04, color='blue')
    ax.fill_between([min(xlim), 0], min(ylim), 0, alpha=0.04, color='blue')

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f'{xlabel.split(" ")[0]} vs 트렌드 동기화', fontsize=10)
    ax.legend(fontsize=7, markerscale=2)
    ax.spines[['top', 'right']].set_visible(False)

    ax.text(0.97, 0.03, '동행 구간', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=7, color='#2166ac', style='italic')

plt.tight_layout()
plt.savefig(IMG_DIR / 'eda_2030_03_sync_scatter.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: eda_2030_03_sync_scatter.png')

# ══════════════════════════════════════════════════════════════
# Chart 4: YoY 변화 동행 — 2030이 오를 때 트렌드도 오르는가?
# 2030소비/이동 변화 방향 × 트렌드 변화 방향 교차 분석
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('2030 YoY 변화 방향 × 트렌드 YoY 변화 방향 교차 분석\n'
             '(같은 방향 = 동행, 다른 방향 = 역행)',
             fontsize=13, fontweight='bold')

for ax, (col2030, label2030) in zip(axes, [
    ('2030소비_비중_변화', '2030 소비비중 변화'),
    ('2030_이동비율_변화', '2030 이동비율 변화'),
]):
    if col2030 not in df.columns:
        ax.set_visible(False)
        continue

    sub = df[[col2030, '트렌드시간대_비중_변화', '타겟_레이블']].dropna()
    sub = sub.copy()
    sub['2030방향'] = np.sign(sub[col2030]).map({1: '2030↑', -1: '2030↓', 0: '2030→'})
    sub['트렌드방향'] = np.sign(sub['트렌드시간대_비중_변화']).map({1: '트렌드↑', -1: '트렌드↓', 0: '트렌드→'})
    sub['동행여부'] = np.where(
        (sub[col2030] > 0) & (sub['트렌드시간대_비중_변화'] > 0), '동반상승',
        np.where(
            (sub[col2030] < 0) & (sub['트렌드시간대_비중_변화'] < 0), '동반하락',
            np.where(
                (sub[col2030] > 0) & (sub['트렌드시간대_비중_변화'] < 0), '2030↑트렌드↓',
                np.where(
                    (sub[col2030] < 0) & (sub['트렌드시간대_비중_변화'] > 0), '2030↓트렌드↑',
                    '변화없음'
                )
            )
        )
    )

    sync_order = ['동반상승', '동반하락', '2030↑트렌드↓', '2030↓트렌드↑']
    sync_colors = ['#2166ac', '#d6604d', '#fddbc7', '#d1e5f0']

    # 동행 유형별 타겟 분포
    result = []
    for sync in sync_order:
        sg = sub[sub['동행여부'] == sync]['타겟_레이블']
        for lbl in LABEL_ORDER:
            result.append({'동행유형': sync, '타겟': lbl, '비율': (sg == lbl).mean() * 100, 'n': len(sg)})
    result = pd.DataFrame(result)

    x = np.arange(len(sync_order))
    width = 0.25
    for i, lbl in enumerate(LABEL_ORDER):
        r_vals = [result[(result['동행유형'] == s) & (result['타겟'] == lbl)]['비율'].values[0]
                  for s in sync_order]
        bars = ax.bar(x + i * width, r_vals, width, label=lbl,
                      color=LABEL_COLORS[lbl], edgecolor='white', alpha=0.85)
        for bar, r in zip(bars, r_vals):
            if r > 8:
                ax.text(bar.get_x() + bar.get_width()/2, r + 0.3,
                        f'{r:.0f}%', ha='center', va='bottom', fontsize=7)

    # n 표시
    for j, sync in enumerate(sync_order):
        n = sub[sub['동행여부'] == sync].shape[0]
        ax.text(j + width, -4, f'n={n}', ha='center', va='top', fontsize=7, color='gray')

    ax.set_xticks(x + width)
    ax.set_xticklabels(sync_order, fontsize=9)
    ax.set_ylabel('타겟 비율 (%)')
    ax.set_ylim(-6, 90)
    ax.set_title(f'{label2030}\nvs 트렌드시간대_비중 변화 동행 유형별 타겟 분포', fontsize=10)
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(IMG_DIR / 'eda_2030_04_yoy_sync.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: eda_2030_04_yoy_sync.png')

# ══════════════════════════════════════════════════════════════
# Chart 5: 2030 모멘텀 → 다음 분기 타겟 (선행 효과)
# "2030이 먼저 올라오면 상권도 나중에 올라오는가?"
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('2030 이동/소비 모멘텀 → 다음 분기 타겟 분포 (선행 효과 검증)\n'
             '(모멘텀: 양수=연속상승, 음수=연속하락, 0=방향 미결)',
             fontsize=13, fontweight='bold')

for ax, (mom_col, label) in zip(axes, [
    ('2030소비_비중_모멘텀', '2030 소비비중 모멘텀'),
    ('2030_이동비율_모멘텀', '2030 이동비율 모멘텀'),
]):
    if mom_col not in df.columns:
        ax.set_visible(False)
        continue

    mom_vals = sorted(df[mom_col].dropna().unique())
    mom_vals = [v for v in mom_vals if df[df[mom_col] == v].shape[0] >= 20]

    rise_rates, hold_rates, fall_rates, ns = [], [], [], []
    for mv in mom_vals:
        sub = df[df[mom_col] == mv]['타겟_레이블']
        ns.append(len(sub))
        rise_rates.append((sub == '상승').mean() * 100)
        hold_rates.append((sub == '유지').mean() * 100)
        fall_rates.append((sub == '하락').mean() * 100)

    x = np.arange(len(mom_vals))
    ax.bar(x, rise_rates, color='#2166ac', alpha=0.85, label='상승')
    ax.bar(x, hold_rates, bottom=rise_rates, color='#999999', alpha=0.85, label='유지')
    ax.bar(x, fall_rates,
           bottom=[r+h for r, h in zip(rise_rates, hold_rates)],
           color='#d6604d', alpha=0.85, label='하락')

    for xi, (r, n) in enumerate(zip(rise_rates, ns)):
        ax.text(xi, 101, f'n={n}', ha='center', va='bottom', fontsize=7, color='gray')
        if r > 5:
            ax.text(xi, r/2, f'{r:.0f}%', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in mom_vals], fontsize=10)
    ax.set_xlabel('모멘텀 값 (연속 방향 횟수)', fontsize=10)
    ax.set_ylabel('타겟 비율 (%)')
    ax.set_ylim(0, 110)
    ax.set_title(f'{label}\n→ 다음 분기 타겟 분포', fontsize=11)
    ax.legend(fontsize=9)
    ax.axvline(x[len(x)//2] - 0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(IMG_DIR / 'eda_2030_05_lead_lag.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: eda_2030_05_lead_lag.png')

# ══════════════════════════════════════════════════════════════
# Chart 6: 행정동별 2030-상권 동행 정도
# 행정동마다 2030과 트렌드 지수가 얼마나 같이 움직이는가?
# ══════════════════════════════════════════════════════════════

# 행정동별 상관계수 계산
def dong_corr(group, xcol, ycol):
    sub = group[[xcol, ycol]].dropna()
    if len(sub) < 4:
        return np.nan
    r, _ = spearmanr(sub[xcol], sub[ycol])
    return r

corr_소비 = (df.groupby('행정동')
             .apply(lambda g: dong_corr(g, '2030소비_비중_변화', '트렌드시간대_비중_변화'))
             .rename('r_소비'))
corr_이동 = (df.groupby('행정동')
             .apply(lambda g: dong_corr(g, '2030_이동비율_변화', '트렌드시간대_비중_변화'))
             .rename('r_이동'))

corr_df = pd.concat([corr_소비, corr_이동], axis=1).dropna()
corr_df['r_평균'] = (corr_df['r_소비'] + corr_df['r_이동']) / 2

print(f'\n행정동별 2030-트렌드 상관계수 요약:')
print(f'  2030소비 vs 트렌드: 중앙값 r = {corr_df["r_소비"].median():.3f}')
print(f'  2030이동 vs 트렌드: 중앙값 r = {corr_df["r_이동"].median():.3f}')
print(f'  r > 0.3 (동행): {(corr_df["r_평균"] > 0.3).sum()}개 동')
print(f'  r < -0.3 (역행): {(corr_df["r_평균"] < -0.3).sum()}개 동')

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('행정동별 2030 지표 - 트렌드 지표 동행 정도 (Spearman r)\n'
             '(r > 0: 2030 오르면 트렌드도 오름, r < 0: 역행)',
             fontsize=13, fontweight='bold')

# 1) 상관계수 분포 히스토그램
ax = axes[0]
ax.hist(corr_df['r_소비'].dropna(), bins=25, alpha=0.6, color='#2166ac',
        label='2030소비 vs 트렌드', edgecolor='white')
ax.hist(corr_df['r_이동'].dropna(), bins=25, alpha=0.6, color='#d6604d',
        label='2030이동 vs 트렌드', edgecolor='white')
ax.axvline(corr_df['r_소비'].median(), color='#2166ac', linestyle='--', linewidth=1.5,
           label=f'소비 중앙값 {corr_df["r_소비"].median():.2f}')
ax.axvline(corr_df['r_이동'].median(), color='#d6604d', linestyle='--', linewidth=1.5,
           label=f'이동 중앙값 {corr_df["r_이동"].median():.2f}')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Spearman 상관계수 r')
ax.set_ylabel('행정동 수')
ax.set_title('2030-트렌드 상관계수 분포\n(423개 행정동)', fontsize=11)
ax.legend(fontsize=8)
ax.spines[['top', 'right']].set_visible(False)

# 2) 소비 r vs 이동 r scatter
ax2 = axes[1]
sc = ax2.scatter(corr_df['r_소비'], corr_df['r_이동'],
                 c=corr_df['r_평균'], cmap='RdYlBu', alpha=0.6, s=30, vmin=-0.6, vmax=0.6)
ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax2.axvline(0, color='black', linestyle='--', linewidth=0.8)

# 강한 동행 행정동 레이블 (r_평균 > 0.5)
top_dong = corr_df[corr_df['r_평균'] > 0.45].index
for dong in top_dong[:8]:
    ax2.annotate(dong[:4], (corr_df.loc[dong, 'r_소비'], corr_df.loc[dong, 'r_이동']),
                 fontsize=6, ha='left', xytext=(3, 3), textcoords='offset points')

# 강한 역행 행정동
bot_dong = corr_df[corr_df['r_평균'] < -0.35].index
for dong in bot_dong[:5]:
    ax2.annotate(dong[:4], (corr_df.loc[dong, 'r_소비'], corr_df.loc[dong, 'r_이동']),
                 fontsize=6, ha='right', color='#d6604d', xytext=(-3, -3), textcoords='offset points')

plt.colorbar(sc, ax=ax2, label='평균 r')
ax2.set_xlabel('2030소비 vs 트렌드 (r)')
ax2.set_ylabel('2030이동 vs 트렌드 (r)')
ax2.set_title('소비-트렌드 r vs 이동-트렌드 r\n(우상단 = 두 채널 모두 동행)', fontsize=11)
ax2.text(0.98, 0.98, '강한 동행', transform=ax2.transAxes,
         ha='right', va='top', fontsize=8, color='#2166ac', fontweight='bold')
ax2.text(0.02, 0.02, '강한 역행', transform=ax2.transAxes,
         ha='left', va='bottom', fontsize=8, color='#d6604d', fontweight='bold')
ax2.spines[['top', 'right']].set_visible(False)

# 3) 동행 상위·하위 행정동 막대
ax3 = axes[2]
top15  = corr_df.nlargest(12, 'r_평균')
bot15  = corr_df.nsmallest(8, 'r_평균')
show   = pd.concat([top15, bot15]).sort_values('r_평균')

bar_colors = ['#d6604d' if r < 0 else '#2166ac' for r in show['r_평균']]
ax3.barh(show.index, show['r_평균'], color=bar_colors, edgecolor='white', height=0.7)
ax3.axvline(0, color='black', linewidth=0.8)
ax3.set_xlabel('평균 Spearman r')
ax3.set_title('동행 강도 상위/하위 행정동', fontsize=11)
ax3.tick_params(axis='y', labelsize=7)
ax3.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(IMG_DIR / 'eda_2030_06_dong_correlation.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: eda_2030_06_dong_correlation.png')

# ── 텍스트 요약 ──────────────────────────────────────────────
print('\n' + '='*60)
print('2030-상권 동행 정도 상위 행정동 (소비+이동 평균 r 기준)')
print('='*60)
print(corr_df.nlargest(10, 'r_평균')[['r_소비', 'r_이동', 'r_평균']].round(3).to_string())
print()
print('2030-상권 역행 행정동 (평균 r 최저)')
print(corr_df.nsmallest(5, 'r_평균')[['r_소비', 'r_이동', 'r_평균']].round(3).to_string())
