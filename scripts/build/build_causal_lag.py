# -*- coding: utf-8 -*-
"""
인과 시차(causal lag) 분석
  - 이동비율(t) × 20대소비비중(t+lag): lag = 0, 1, 2 분기
  - 이동비율(t) × 임대료변화(t+lag):  lag = 0, 1, 2 연간 (임대료는 연간 단위)

출력:
  report/data/causal_lag_analysis.csv
  report/images/S3_04_causal_lag.png
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr

sys.stdout.reconfigure(encoding='utf-8')
os.makedirs('report/data', exist_ok=True)
os.makedirs('report/images', exist_ok=True)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

REF_DONGS = [
    '성수1가2동','역삼1동','청운효자동','삼성1동','성수2가1동','합정동','구로3동','군자동',
    '방배2동','가양1동','대학동','사직동','상계2동','성수2가3동','신촌동','휘경1동'
]

# ────────────────────────────────────────────────────────────────
# 데이터 로드
# ────────────────────────────────────────────────────────────────
print('[1/4] 데이터 로드...')
mig_raw = pd.read_csv('data/main_data/분기별_주말유입인구.csv', encoding='utf-8-sig')
kt      = pd.read_excel('data/archive/KT_서울생활이동데이터_행정동코드_20210907.xlsx')
kt.columns = ['시도','시군구','행정동코드','행정동명','전체명']
kt_map = dict(zip(kt['행정동코드'], kt['행정동명']))

raw = pd.read_csv('data/archive/서울시_상권분석서비스(추정매출-행정동)_2020_2024.csv', encoding='utf-8-sig')
rental   = pd.read_csv('data/main_data/임대료_공실률.csv', encoding='utf-8-sig')
rmap     = pd.read_csv('data/main_data/rental_vacancy_dongmap.csv', encoding='utf-8-sig')

# ────────────────────────────────────────────────────────────────
# 분기별 패널: 이동비율 + 20대소비비중
# ────────────────────────────────────────────────────────────────
print('[2/4] 분기별 패널 구축...')

# 이동 분기별 20대비율
mig20  = mig_raw[mig_raw['연령대']==20].groupby(['도착_행정동코드','연도','분기'])['합계_이동인구'].sum().reset_index()
mall   = mig_raw.groupby(['도착_행정동코드','연도','분기'])['합계_이동인구'].sum().reset_index()
mq     = mig20.merge(mall, on=['도착_행정동코드','연도','분기'], suffixes=('_20','_전체'))
mq['이동비율_20대'] = mq['합계_이동인구_20'] / mq['합계_이동인구_전체'].replace(0, np.nan)
mq['행정동'] = mq['도착_행정동코드'].map(kt_map)
mq['YQ'] = (mq['연도'] - 2020) * 4 + mq['분기']
mq = mq.dropna(subset=['행정동','이동비율_20대'])

# 매출 분기별 20대비중
raw['연도']   = raw['기준_년분기_코드'] // 10
raw['분기']   = raw['기준_년분기_코드'] % 10
raw['YQ']    = (raw['연도'] - 2020) * 4 + raw['분기']
sq = raw.groupby(['행정동_코드_명','YQ','연도','분기'])[['당월_매출_금액','연령대_20_매출_금액']].sum().reset_index()
sq.columns = ['행정동','YQ','연도','분기','총매출','20대매출']
sq['소비비중_20대'] = sq['20대매출'] / sq['총매출'].replace(0, np.nan)

# 패널 조인
panel = mq[['행정동','YQ','연도','분기','이동비율_20대']].merge(
    sq[['행정동','YQ','소비비중_20대']], on=['행정동','YQ'], how='inner')
panel = panel.sort_values(['행정동','YQ']).reset_index(drop=True)

# QoQ 변화
panel['Δ이동'] = panel.groupby('행정동')['이동비율_20대'].diff()
panel['Δ소비'] = panel.groupby('행정동')['소비비중_20대'].diff()

# ────────────────────────────────────────────────────────────────
# 분기별 lag 상관계수 (전체 423개 동 × 시간점)
# ────────────────────────────────────────────────────────────────
print('[3/4] lag 상관계수 계산...')

lag_rows = []

# 이동비율 × 소비비중 lag 0/1/2 분기
for lag in [0, 1, 2]:
    # 수준값 상관
    pairs_lv = []
    # 변화량 상관
    pairs_ch = []
    for dong, grp in panel.groupby('행정동'):
        grp = grp.sort_values('YQ').reset_index(drop=True)
        x_lv = grp['이동비율_20대'].values
        x_ch = grp['Δ이동'].values
        if lag == 0:
            y_lv = grp['소비비중_20대'].values
            y_ch = grp['Δ소비'].values
        else:
            y_lv = np.roll(grp['소비비중_20대'].values, -lag)
            y_lv[-lag:] = np.nan
            y_ch = np.roll(grp['Δ소비'].values, -lag)
            y_ch[-lag:] = np.nan
        # 유효 쌍
        mask_lv = np.isfinite(x_lv) & np.isfinite(y_lv)
        mask_ch = np.isfinite(x_ch) & np.isfinite(y_ch)
        if mask_lv.sum() >= 5:
            pairs_lv.append((x_lv[mask_lv], y_lv[mask_lv]))
        if mask_ch.sum() >= 5:
            pairs_ch.append((x_ch[mask_ch], y_ch[mask_ch]))

    # 전체 풀링
    def pool_corr(pairs):
        if not pairs:
            return 0.0, 1.0, 0
        xs = np.concatenate([p[0] for p in pairs])
        ys = np.concatenate([p[1] for p in pairs])
        mask = np.isfinite(xs) & np.isfinite(ys)
        if mask.sum() < 3:
            return 0.0, 1.0, 0
        r, p = pearsonr(xs[mask], ys[mask])
        return round(r, 4), round(p, 5), int(mask.sum())

    r_lv, p_lv, n_lv = pool_corr(pairs_lv)
    r_ch, p_ch, n_ch = pool_corr(pairs_ch)

    lag_rows.append({
        '분석축': '이동비율→소비비중',
        '데이터_단위': '분기',
        'lag': lag,
        'lag_설명': f'이동비율(t) → 소비비중(t+{lag}분기)',
        'r_수준값': r_lv, 'p_수준값': p_lv, 'n_수준값': n_lv,
        'r_변화량': r_ch, 'p_변화량': p_ch, 'n_변화량': n_ch,
        '유의성_수준값': '***' if p_lv<0.001 else ('**' if p_lv<0.01 else ('*' if p_lv<0.05 else 'n.s.')),
        '유의성_변화량': '***' if p_ch<0.001 else ('**' if p_ch<0.01 else ('*' if p_ch<0.05 else 'n.s.')),
    })
    print(f'  이동→소비 lag{lag}: r_수준={r_lv:.3f}{("***" if p_lv<0.001 else "")}  r_변화={r_ch:.3f}{("***" if p_ch<0.001 else "")}')

# 임대료 연간 lag 0/1/2년 (이동비율 연평균 × 임대료 변화율)
sang2dong = dict(zip(rmap['상권명'], rmap['대표_행정동']))
rental_panel = rental[['상권명','연도','임대료']].copy()
rental_panel['행정동'] = rental_panel['상권명'].map(sang2dong)
rental_panel = rental_panel.dropna(subset=['행정동'])
rental_ann = rental_panel.groupby(['행정동','연도'])['임대료'].mean().reset_index()

mig_ann = mq.groupby(['행정동','연도'])['이동비율_20대'].mean().reset_index()
panel_ann = mig_ann.merge(rental_ann, on=['행정동','연도'], how='inner').sort_values(['행정동','연도'])
panel_ann['Δ이동_yr'] = panel_ann.groupby('행정동')['이동비율_20대'].diff()
panel_ann['Δ임대료']  = panel_ann.groupby('행정동')['임대료'].diff()
panel_ann['임대료변화율'] = panel_ann['Δ임대료'] / panel_ann.groupby('행정동')['임대료'].shift(1)

for lag in [0, 1, 2]:
    pairs_lv, pairs_ch = [], []
    for dong, grp in panel_ann.groupby('행정동'):
        grp = grp.sort_values('연도').reset_index(drop=True)
        x_lv = grp['이동비율_20대'].values
        x_ch = grp['Δ이동_yr'].values
        if lag == 0:
            y_lv = grp['임대료'].values
            y_ch = grp['임대료변화율'].values
        else:
            y_lv = np.roll(grp['임대료'].values, -lag); y_lv[-lag:] = np.nan
            y_ch = np.roll(grp['임대료변화율'].values, -lag); y_ch[-lag:] = np.nan
        mask_lv = np.isfinite(x_lv) & np.isfinite(y_lv)
        mask_ch = np.isfinite(x_ch) & np.isfinite(y_ch)
        if mask_lv.sum() >= 3: pairs_lv.append((x_lv[mask_lv], y_lv[mask_lv]))
        if mask_ch.sum() >= 3: pairs_ch.append((x_ch[mask_ch], y_ch[mask_ch]))

    r_lv, p_lv, n_lv = pool_corr(pairs_lv)
    r_ch, p_ch, n_ch = pool_corr(pairs_ch)
    lag_rows.append({
        '분석축': '이동비율→임대료',
        '데이터_단위': '연간',
        'lag': lag,
        'lag_설명': f'이동비율(t) → 임대료(t+{lag}년)',
        'r_수준값': r_lv, 'p_수준값': p_lv, 'n_수준값': n_lv,
        'r_변화량': r_ch, 'p_변화량': p_ch, 'n_변화량': n_ch,
        '유의성_수준값': '***' if p_lv<0.001 else ('**' if p_lv<0.01 else ('*' if p_lv<0.05 else 'n.s.')),
        '유의성_변화량': '***' if p_ch<0.001 else ('**' if p_ch<0.01 else ('*' if p_ch<0.05 else 'n.s.')),
    })
    print(f'  이동→임대료 lag{lag}yr: r_수준={r_lv:.3f}  r_변화={r_ch:.3f}')

df_lag = pd.DataFrame(lag_rows)
df_lag.to_csv('report/data/causal_lag_analysis.csv', encoding='utf-8-sig', index=False)
print('  → report/data/causal_lag_analysis.csv 저장')

# ────────────────────────────────────────────────────────────────
# 시각화
# ────────────────────────────────────────────────────────────────
print('[4/4] 시각화...')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('인과 시차 분석: 20대 이동비율 → 소비비중 → 임대료\n(서울시 423개 행정동 × 2020~2024 분기 패널, 전체 pooling)', fontsize=13, fontweight='bold')

# ① 이동→소비 수준값 lag0/1/2
ax = axes[0]
sub = df_lag[df_lag['분석축']=='이동비율→소비비중']
colors = ['#1565C0' if p<0.05 else '#90A4AE' for p in sub['p_수준값']]
bars = ax.bar([f'lag{l}분기' for l in sub['lag']], sub['r_수준값'], color=colors, edgecolor='white', linewidth=0.5)
for bar, r, sig in zip(bars, sub['r_수준값'], sub['유의성_수준값']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f'r={r:.3f}\n{sig}',
            ha='center', va='bottom', fontsize=9.5, fontweight='bold')
ax.axhline(0, color='black', lw=0.8)
ax.set_title('① 이동비율(t) → 소비비중(t+lag)\n[수준값 상관, 전체 동 pooling]', fontsize=10, fontweight='bold')
ax.set_ylabel('Pearson r')
ax.set_ylim(-0.05, max(sub['r_수준값'])*1.4 + 0.05)
ax.grid(axis='y', alpha=0.3)

# ② 이동→소비 변화량 lag0/1/2
ax = axes[1]
colors = ['#00838F' if p<0.05 else '#90A4AE' for p in sub['p_변화량']]
bars = ax.bar([f'lag{l}분기' for l in sub['lag']], sub['r_변화량'], color=colors, edgecolor='white', linewidth=0.5)
for bar, r, sig in zip(bars, sub['r_변화량'], sub['유의성_변화량']):
    ypos = bar.get_height() + 0.003 if bar.get_height() >= 0 else bar.get_height() - 0.015
    ax.text(bar.get_x()+bar.get_width()/2, ypos, f'r={r:.3f}\n{sig}',
            ha='center', va='bottom' if bar.get_height()>=0 else 'top', fontsize=9.5, fontweight='bold')
ax.axhline(0, color='black', lw=0.8)
ax.set_title('② 이동비율 변화(t) → 소비비중 변화(t+lag)\n[QoQ 변화량 상관]', fontsize=10, fontweight='bold')
ax.set_ylabel('Pearson r')
ax.grid(axis='y', alpha=0.3)

# ③ 이동→임대료 lag0/1/2년
ax = axes[2]
sub2 = df_lag[df_lag['분석축']=='이동비율→임대료']
x = np.arange(len(sub2))
w = 0.35
colors_lv = ['#1565C0' if p<0.05 else '#90A4AE' for p in sub2['p_수준값']]
colors_ch = ['#00838F' if p<0.05 else '#BDBDBD' for p in sub2['p_변화량']]
b1 = ax.bar(x-w/2, sub2['r_수준값'], w, color=colors_lv, label='수준값', edgecolor='white')
b2 = ax.bar(x+w/2, sub2['r_변화량'], w, color=colors_ch, label='변화량(YoY)', edgecolor='white')
for bar, r, sig in zip(b1, sub2['r_수준값'], sub2['유의성_수준값']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f'{r:.3f}\n{sig}',
            ha='center', va='bottom', fontsize=8)
for bar, r, sig in zip(b2, sub2['r_변화량'], sub2['유의성_변화량']):
    ypos = bar.get_height() + 0.003 if bar.get_height() >= 0 else bar.get_height() - 0.02
    ax.text(bar.get_x()+bar.get_width()/2, ypos, f'{r:.3f}\n{sig}',
            ha='center', va='bottom' if bar.get_height()>=0 else 'top', fontsize=8)
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f'lag{l}년' for l in sub2['lag']])
ax.set_title('③ 이동비율(t) → 임대료(t+lag)\n[68개 상권 한정, 연간 단위]', fontsize=10, fontweight='bold')
ax.set_ylabel('Pearson r')
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)

# 범례 설명
fig.text(0.5, -0.03,
    '■ 파랑/주황: p<0.05 유의  □ 회색: n.s. (p≥0.05) | '
    '*** p<0.001  ** p<0.01  * p<0.05  n.s. 유의하지 않음',
    ha='center', fontsize=9, color='#555')

plt.tight_layout()
out = 'report/images/S3_04_causal_lag.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'  → {out}')
print('\n=== causal_lag_analysis 완료 ===')
print(df_lag[['lag_설명','r_수준값','유의성_수준값','r_변화량','유의성_변화량']].to_string(index=False))
