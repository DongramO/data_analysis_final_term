# -*- coding: utf-8 -*-
"""
20대 유입 선행 → 30+ 유입 후행 → 상권 발달 패턴 탐지
- 핵심 지표: 20대 이동 YoY(t) vs 30+이동 YoY(t+lag) CCF
- 필터: CCF peak lag > 0 (20대 선행) + 매출 증가
출력: image/age_lead_pattern.png
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import pearsonr
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT      = Path(__file__).resolve().parent
RAW_PATH  = ROOT / 'data/main_data/분기별_원본피처.csv'
MIG_PATH  = ROOT / 'data/main_data/분기별_주말유입인구.csv'
CMAP_PATH = ROOT / 'data/main_data/코드매핑_유입인구_매출.csv'
IMG_DIR   = ROOT / 'image'

REF   = ['방배2동','청운효자동','합정동','성수2가1동']
TIER1 = ['숭인2동','신당5동']

def group_label(d):
    if d in REF:   return '레퍼런스'
    if d in TIER1: return 'Tier1'
    return '기타'

# ── 데이터 로드 ────────────────────────────────────────────────────────
raw  = pd.read_csv(RAW_PATH,  encoding='utf-8-sig')
mig  = pd.read_csv(MIG_PATH,  encoding='utf-8-sig')
cmap = pd.read_csv(CMAP_PATH, encoding='utf-8-sig')

mig = mig.merge(cmap[['migration_코드','행정동명']],
                left_on='도착_행정동코드', right_on='migration_코드', how='left')
mig.rename(columns={'행정동명':'행정동'}, inplace=True)

# 연령대별 분기 집계 (절대 이동인구)
age_piv = mig[mig['연령대']>=10].pivot_table(
    index=['행정동','연도','분기'],
    columns='연령대', values='합계_이동인구', aggfunc='sum'
).reset_index()
age_piv.columns.name = None
age_piv['20대'] = age_piv[20]
age_piv['30이상'] = age_piv[[c for c in age_piv.columns
                              if isinstance(c,(int,np.integer)) and c>=30]].sum(axis=1)
age_piv['전체']  = age_piv[[c for c in age_piv.columns
                              if isinstance(c,(int,np.integer)) and c>=10]].sum(axis=1)
age_piv['20대비율']  = age_piv['20대']  / age_piv['전체'].replace(0,np.nan)
age_piv['30이상비율'] = age_piv['30이상'] / age_piv['전체'].replace(0,np.nan)
age_piv['분기순번'] = (age_piv['연도']-2020)*4 + age_piv['분기']

# 매출 데이터 병합
raw_q = raw[['행정동','연도','분기','총_매출금액']].copy()
df = age_piv.merge(raw_q, on=['행정동','연도','분기'], how='left')

# YoY 변화율
df = df.sort_values(['행정동','연도','분기'])
df['20대_YoY']   = df.groupby(['행정동','분기'])['20대'].pct_change()   * 100
df['30이상_YoY'] = df.groupby(['행정동','분기'])['30이상'].pct_change() * 100
df['매출_YoY']   = df.groupby(['행정동','분기'])['총_매출금액'].pct_change() * 100

YEARS = [2020,2021,2022,2023,2024]

# ── 행정동별 CCF 계산: 20대 YoY(t) vs 30이상 YoY(t+lag) ───────────────
LAGS = range(-3, 5)
results = []

for dong, grp in df.groupby('행정동'):
    sub = grp[(grp['연도']>=2021)].sort_values('분기순번')
    s20  = sub['20대_YoY'].values
    s30  = sub['30이상_YoY'].values
    smkt = sub['매출_YoY'].values
    n    = len(s20)
    valid = ~(np.isnan(s20) | np.isnan(s30))
    if valid.sum() < 8:
        continue

    ccf_rs = {}
    for lag in LAGS:
        if lag >= 0:
            x = s20[:n-lag] if lag>0 else s20
            y = s30[lag:]   if lag>0 else s30
        else:
            x = s20[-lag:]
            y = s30[:n+lag]
        vm = ~(np.isnan(x)|np.isnan(y))
        if vm.sum() >= 5:
            r,_ = pearsonr(x[vm], y[vm])
            ccf_rs[lag] = r
        else:
            ccf_rs[lag] = np.nan

    peak_lag = max(ccf_rs, key=lambda k: ccf_rs[k] if not np.isnan(ccf_rs[k]) else -999)
    peak_r   = ccf_rs[peak_lag]

    # 동기간 매출 증가 (2022~2024 연평균 YoY)
    mkt_recent = sub[sub['연도']>=2022]['매출_YoY'].median()
    # 20대 이동 선행기(2020→2022) slope
    early = grp[grp['연도']<=2022].groupby('연도')['20대'].mean()
    if len(early) >= 2:
        early_slope = np.polyfit(range(len(early)), early.values, 1)[0]
    else:
        early_slope = np.nan
    # 30이상 이동 후행기(2022→2024) slope
    late = grp[grp['연도']>=2022].groupby('연도')['30이상'].mean()
    if len(late) >= 2:
        late_slope = np.polyfit(range(len(late)), late.values, 1)[0]
    else:
        late_slope = np.nan

    results.append({
        '행정동': dong,
        '그룹': group_label(dong),
        'peak_lag': peak_lag,
        'peak_r': peak_r,
        'r_lag0': ccf_rs.get(0, np.nan),
        'r_lag1': ccf_rs.get(1, np.nan),
        'r_lag2': ccf_rs.get(2, np.nan),
        '매출_YoY_중앙': mkt_recent,
        '20대_초기slope': early_slope,
        '30이상_후기slope': late_slope,
        **{f'r_lag{k}': ccf_rs.get(k,np.nan) for k in LAGS},
    })

res = pd.DataFrame(results)

# 선별 기준: 20대 선행 (peak_lag >= 1), 매출 증가 (매출_YoY > 0), 30이상 후기slope > 0
cond = (
    (res['peak_lag'] >= 1) &
    (res['매출_YoY_중앙'] > 0) &
    (res['30이상_후기slope'] > 0) &
    (res['20대_초기slope'] > 0)
)
top = res[cond].sort_values('peak_r', ascending=False).head(20)
print(f"패턴 충족 행정동: {len(res[cond])}개")
print(top[['행정동','그룹','peak_lag','peak_r','r_lag0','r_lag1','매출_YoY_중앙',
           '20대_초기slope','30이상_후기slope']].to_string(index=False))

# ── 시각화 ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18,14))
fig.suptitle('20대 유입 선행 → 타세대 유입 후행 → 상권 발달 패턴 탐지',
             fontsize=14, fontweight='bold')

# ── Panel 1: peak_lag 분포 (히스토그램) ─────────────────────────────
ax1 = axes[0][0]
lag_counts = res['peak_lag'].value_counts().sort_index()
colors_bar = ['#E05C5C' if l > 0 else ('#AAAAAA' if l == 0 else '#4A90D9')
              for l in lag_counts.index]
bars = ax1.bar(lag_counts.index.astype(str), lag_counts.values, color=colors_bar)
ax1.axvline(ax1.get_xticks()[list(lag_counts.index).index(0)],
            color='black', lw=0.8, ls=':', label='lag=0')
for b, v in zip(bars, lag_counts.values):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+1, str(v),
             ha='center', fontsize=9, fontweight='bold')
ax1.set_xlabel('CCF Peak Lag (20대→타세대 이동)')
ax1.set_ylabel('행정동 수')
ax1.set_title('서울 422동 — 20대 이동과 30+이동 사이\nCCF 최고점 lag 분포\n(빨강=20대 선행, 파랑=30+선행)', fontsize=10)
ax1.set_facecolor('#F8F8F8')

# ── Panel 2: 레퍼런스·Tier1 CCF 곡선 ────────────────────────────────
ax2 = axes[0][1]
DONG_COLOR = {'성수2가1동':'#E05C5C','합정동':'#C0392B',
              '방배2동':'#E8836B','청운효자동':'#F1948A',
              '숭인2동':'#F5A623','신당5동':'#F0C27F'}

lag_cols = [c for c in res.columns if str(c).startswith('r_lag') and
            c != 'r_lag0_dup' and 'slope' not in str(c)]
lag_map = {c: int(str(c).replace('r_lag','')) for c in lag_cols}
lag_list = sorted(lag_map.values())

for dong in REF + TIER1:
    row = res[res['행정동']==dong]
    if len(row) == 0: continue
    row = row.iloc[0]
    rs = [row[f'r_lag{l}'] for l in lag_list]
    color = DONG_COLOR.get(dong,'#AAAAAA')
    lw = 2.5 if dong=='성수2가1동' else 1.8
    ls = '-' if dong in REF else '--'
    ax2.plot(lag_list, rs, 'o'+ls, color=color, lw=lw, label=dong, markersize=5)

ax2.axvline(0, color='black', lw=0.8, ls=':')
ax2.axhline(0, color='black', lw=0.5, ls=':')
ax2.fill_between([0.5,4.5],-1,1, alpha=0.06, color='orange', label='20대 선행 구간')
ax2.set_xticks(lag_list)
ax2.set_xticklabels([f'lag{l:+d}' for l in lag_list], fontsize=9)
ax2.set_xlabel('← 30+선행  |  20대선행 →')
ax2.set_ylabel('r  (20대 이동 YoY vs 30+이동 YoY)')
ax2.set_title('레퍼런스·Tier1 CCF 곡선\n(lag>0 = 20대가 먼저 증가, 이후 30+증가)', fontsize=10)
ax2.legend(fontsize=8.5, loc='lower right')
ax2.set_facecolor('#F8F8F8')
ax2.grid(alpha=0.2)
ax2.set_ylim(-1,1)

# ── Panel 3: TOP 선별 행정동 이중축 시계열 ──────────────────────────
ax3 = axes[1][0]
top5 = top.head(6)['행정동'].tolist()
# 레퍼런스 포함 비교를 위해 성수2가1동 추가
show_list = ['성수2가1동'] + [d for d in top5 if d != '성수2가1동'][:5]

cmap_colors = plt.cm.tab10(np.linspace(0, 0.7, len(show_list)))

for i, dong in enumerate(show_list):
    sub = df[df['행정동']==dong].groupby('연도').agg(
        이십대=('20대','sum'), 삼십이상=('30이상','sum'), 전체=('전체','sum')
    ).reindex(YEARS)
    # 2020=100 지수
    idx20 = sub['이십대']  / sub['이십대'].iloc[0]  * 100
    idx30 = sub['삼십이상'] / sub['삼십이상'].iloc[0] * 100
    color = DONG_COLOR.get(dong, cmap_colors[i])
    lw = 2.5 if dong=='성수2가1동' else 1.8
    ax3.plot(YEARS, idx20, 'o-',  color=color, lw=lw, label=f'{dong} 20대')
    ax3.plot(YEARS, idx30, 's--', color=color, lw=lw*0.7, alpha=0.6, label=f'{dong} 30+')

ax3.axhline(100, color='gray', lw=0.8, ls=':')
ax3.set_xticks(YEARS)
ax3.set_ylabel('이동인구 지수 (2020=100)')
ax3.set_title('20대 선행 패턴 TOP5 + 성수2가1동\n(실선=20대, 점선=30+, 2020=100 지수)', fontsize=10)
ax3.legend(fontsize=7.5, loc='upper left', ncol=2)
ax3.set_facecolor('#F8F8F8')
ax3.grid(alpha=0.2)

# ── Panel 4: 선별 TOP20 요약 테이블 ─────────────────────────────────
ax4 = axes[1][1]
ax4.axis('off')

top20 = top[['행정동','그룹','peak_lag','peak_r','매출_YoY_중앙',
             '20대_초기slope','30이상_후기slope']].head(15).copy()
top20.insert(0, '순위', range(1, len(top20)+1))
top20['peak_r']         = top20['peak_r'].map(lambda x: f'{x:.3f}')
top20['매출_YoY_중앙']  = top20['매출_YoY_중앙'].map(lambda x: f'{x:.1f}%')
top20['20대_초기slope'] = top20['20대_초기slope'].map(lambda x: f'{x/1000:.1f}천')
top20['30이상_후기slope'] = top20['30이상_후기slope'].map(lambda x: f'{x/1000:.1f}천')

headers = ['순위','행정동','그룹','선행\nlag','r','매출YoY\n중앙',
           '20대초기\nslope','30+후기\nslope']
tbl = ax4.table(cellText=top20.values.tolist(), colLabels=headers,
                cellLoc='center', loc='center', bbox=[0,0,1,1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
for (ri,ci), cell in tbl.get_celld().items():
    if ri == 0:
        cell.set_facecolor('#3A5CA9')
        cell.set_text_props(color='white', fontweight='bold')
    else:
        dong = top20.iloc[ri-1]['행정동']
        if dong in REF:
            cell.set_facecolor('#FDECEA')
        elif dong in TIER1:
            cell.set_facecolor('#FFF4E5')
        elif ri % 2 == 0:
            cell.set_facecolor('#F7F7F7')
ax4.set_title('20대 선행 패턴 TOP15\n(peak_lag>0, 매출증가, 30+이동 증가)', fontsize=10, pad=10)

plt.tight_layout()
out = IMG_DIR / 'age_lead_pattern.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n[저장] {out.name}")
