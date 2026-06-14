"""
2030세대 트렌드 선도 지역 분석
: 2030 소비비중 / 20대 이동비율 YoY 변화 Top 15 + 선정 지역 다중 지표 비교
  ► 바 색상 = 상권 복합점수 분위 변화 (초록: 상권↑, 빨강: 상권↓)
  ► 바 끝 레이블 = 1차 지표 + [분위변화] + 임대료변화(해당 시)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 1. 매출 마스터 (2030 소비비중, 복합점수) ─────────────────────
master = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig',
                     usecols=['행정동', '행정동_코드', '연도', '2030비중', '복합점수_분위'])
master = master.sort_values(['행정동_코드', '연도'])
master['2030소비_변화']  = master.groupby('행정동_코드')['2030비중'].diff() * 100   # %p
master['분위변화']       = master.groupby('행정동_코드')['복합점수_분위'].diff() * 100  # %p

dong_name = (master[['행정동_코드', '행정동']]
             .drop_duplicates()
             .set_index('행정동_코드')['행정동'])

# ── 2. 20대 이동비율 YoY ──────────────────────────────────────
mig     = pd.read_csv('data/main_data/분기별_주말유입인구.csv', encoding='utf-8-sig')
mapping = pd.read_csv('data/main_data/코드매핑_유입인구_매출.csv', encoding='utf-8-sig')
mig_to_raw = dict(zip(mapping['migration_코드'], mapping['raw_코드']))

annual = mig.groupby(['도착_행정동코드', '연도', '연령대'])['합계_이동인구'].sum().reset_index()
total  = annual.groupby(['도착_행정동코드', '연도'])['합계_이동인구'].sum().rename('총_이동인구')
annual = annual.join(total, on=['도착_행정동코드', '연도'])
annual['비율'] = annual['합계_이동인구'] / annual['총_이동인구'] * 100

age20 = annual[annual['연령대'] == 20][['도착_행정동코드', '연도', '비율']].copy()
age20.columns = ['migration_코드', '연도', '20대_이동비율']
age20['행정동_코드'] = age20['migration_코드'].map(mig_to_raw)
age20 = age20.dropna(subset=['행정동_코드']).copy()
age20['행정동_코드'] = age20['행정동_코드'].astype(int)
# 여러 migration 코드가 같은 행정동_코드로 매핑되는 경우 비율 평균 처리
age20 = (age20.groupby(['행정동_코드', '연도'])['20대_이동비율'].mean()
         .reset_index())
age20 = age20.sort_values(['행정동_코드', '연도'])
age20['20대_이동비율_변화'] = age20.groupby('행정동_코드')['20대_이동비율'].diff()
age20['행정동'] = age20['행정동_코드'].map(dong_name)
age20 = age20.merge(master[['행정동_코드', '연도', '분위변화']],
                    on=['행정동_코드', '연도'], how='left')

# ── 3. 임대료/공실률 YoY (행정동 코드 기준) ───────────────────────
rental  = pd.read_csv('data/main_data/임대료_공실률.csv', encoding='utf-8-sig')
dongmap = pd.read_csv('data/main_data/rental_vacancy_dongmap.csv', encoding='utf-8-sig')
code_ref = (master[['행정동', '행정동_코드']]
            .drop_duplicates()
            .rename(columns={'행정동': '대표_행정동'}))

rental_base = (rental[rental['연도'] <= 2024]
               .merge(dongmap[['상권명', '대표_행정동']], on='상권명', how='left')
               .merge(code_ref, on='대표_행정동', how='inner')
               .sort_values(['행정동_코드', '연도']))
rental_base['임대료_변화'] = rental_base.groupby('행정동_코드')['임대료'].pct_change() * 100
rental_base['공실률_변화'] = rental_base.groupby('행정동_코드')['공실률'].diff()
rental_yoy = (rental_base[rental_base['연도'] >= 2021]
              [['행정동_코드', '연도', '임대료_변화', '공실률_변화']].dropna())
# 행정동_코드 기준으로 중복 있을 수 있음 (여러 상권 → 같은 대표 행정동) → 평균 처리
rental_yoy = rental_yoy.groupby(['행정동_코드', '연도']).mean().reset_index()

yoy_years = [2021, 2022, 2023, 2024]

print(f'소비 기준: {master["행정동_코드"].nunique()}개 동')
print(f'이동 기준: {age20["행정동_코드"].nunique()}개 동')
print(f'임대료 연계: {rental_yoy["행정동_코드"].nunique()}개 동')

# ── 공통: 색상 설정 ───────────────────────────────────────────
divnorm  = mcolors.TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30)
bar_cmap = cm.RdYlGn   # 분위변화: 초록=상권↑, 빨강=상권↓

# ─────────────────────────────────────────────────────────────
# 그림 1: 2030 소비비중 변화 Top 15 (연도별)
# ─────────────────────────────────────────────────────────────
consume_df = (master.dropna(subset=['2030소비_변화'])
              .merge(rental_yoy, on=['행정동_코드', '연도'], how='left'))

fig1, axes = plt.subplots(1, 4, figsize=(26, 9))
fig1.suptitle('2030 소비비중 YoY 변화 Top 15 행정동 (2021~2024)\n'
              '바 색상 = 상권 복합점수 분위 변화 (초록↑ / 빨강↓)  |  레이블 = [분위변화] / {임대료변화}',
              fontsize=13, fontweight='bold', y=1.02)

for ax, yr in zip(axes, yoy_years):
    sub = consume_df[consume_df['연도'] == yr]
    df_yr = sub.nlargest(15, '2030소비_변화').sort_values('2030소비_변화', ascending=True)

    bars = ax.barh(
        df_yr['행정동'], df_yr['2030소비_변화'],
        color=[bar_cmap(divnorm(v)) if not np.isnan(v) else '#cccccc'
               for v in df_yr['분위변화']],
        edgecolor='white', linewidth=0.4, height=0.7
    )
    max_v = df_yr['2030소비_변화'].max()
    offset = max_v * 0.025

    for bar, (_, row) in zip(bars, df_yr.iterrows()):
        label = f'+{row["2030소비_변화"]:.1f}%p'
        if not pd.isna(row['분위변화']):
            sign = '+' if row['분위변화'] >= 0 else ''
            label += f'  [분위{sign}{row["분위변화"]:.0f}%p]'
        if not pd.isna(row['임대료_변화']):
            sign = '+' if row['임대료_변화'] >= 0 else ''
            label += f'  임대료{sign}{row["임대료_변화"]:.1f}%'

        ax.text(max_v + offset, bar.get_y() + bar.get_height() / 2,
                label, va='center', ha='left', fontsize=7.5)

    ax.set_xlim(0, max_v * 1.05)
    ax.set_title(f'{yr}년 (vs {yr-1}년)', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('2030 소비비중 변화 (%p)', fontsize=9)
    ax.tick_params(axis='y', labelsize=9.5)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

# 컬러바 범례
sm = cm.ScalarMappable(cmap=bar_cmap, norm=divnorm)
sm.set_array([])
cbar = fig1.colorbar(sm, ax=axes.tolist(), shrink=0.6, pad=0.01)
cbar.set_label('상권 복합점수 분위 변화 (%p)', fontsize=9)

plt.tight_layout()
plt.savefig('image/yoy_top15_2030소비.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: image/yoy_top15_2030소비.png')

# ─────────────────────────────────────────────────────────────
# 그림 2: 20대 이동비율 변화 Top 15 (연도별)
# ─────────────────────────────────────────────────────────────
mig_df = (age20.dropna(subset=['20대_이동비율_변화', '행정동'])
          .merge(rental_yoy, on=['행정동_코드', '연도'], how='left'))

fig2, axes = plt.subplots(1, 4, figsize=(26, 9))
fig2.suptitle('20대 이동비율 YoY 변화 Top 15 행정동 (2021~2024)\n'
              '바 색상 = 상권 복합점수 분위 변화 (초록↑ / 빨강↓)  |  레이블 = [분위변화] / {임대료변화}',
              fontsize=13, fontweight='bold', y=1.02)

for ax, yr in zip(axes, yoy_years):
    sub = mig_df[mig_df['연도'] == yr]
    df_yr = sub.nlargest(15, '20대_이동비율_변화').sort_values('20대_이동비율_변화', ascending=True)

    bars = ax.barh(
        df_yr['행정동'], df_yr['20대_이동비율_변화'],
        color=[bar_cmap(divnorm(v)) if not np.isnan(v) else '#cccccc'
               for v in df_yr['분위변화']],
        edgecolor='white', linewidth=0.4, height=0.7
    )
    max_v = df_yr['20대_이동비율_변화'].max()
    offset = max_v * 0.025

    for bar, (_, row) in zip(bars, df_yr.iterrows()):
        label = f'+{row["20대_이동비율_변화"]:.2f}%p'
        if not pd.isna(row['분위변화']):
            sign = '+' if row['분위변화'] >= 0 else ''
            label += f'  [분위{sign}{row["분위변화"]:.0f}%p]'
        if not pd.isna(row['임대료_변화']):
            sign = '+' if row['임대료_변화'] >= 0 else ''
            label += f'  임대료{sign}{row["임대료_변화"]:.1f}%'

        ax.text(max_v + offset, bar.get_y() + bar.get_height() / 2,
                label, va='center', ha='left', fontsize=7.5)

    ax.set_xlim(0, max_v * 1.05)
    ax.set_title(f'{yr}년 (vs {yr-1}년)', fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('20대 이동비율 변화 (%p)', fontsize=9)
    ax.tick_params(axis='y', labelsize=9.5)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)

sm2 = cm.ScalarMappable(cmap=bar_cmap, norm=divnorm)
sm2.set_array([])
cbar2 = fig2.colorbar(sm2, ax=axes.tolist(), shrink=0.6, pad=0.01)
cbar2.set_label('상권 복합점수 분위 변화 (%p)', fontsize=9)

plt.tight_layout()
plt.savefig('image/yoy_top15_20대이동.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: image/yoy_top15_20대이동.png')

# ─────────────────────────────────────────────────────────────
# 그림 3: 2024년 Top 15 (2030소비 기준) 다중 지표 종합 히트맵
# ─────────────────────────────────────────────────────────────
top15_2024 = (consume_df[consume_df['연도'] == 2024]
              .nlargest(15, '2030소비_변화')[['행정동_코드', '행정동']])

# 이동비율 변화 join
mig_2024 = mig_df[mig_df['연도'] == 2024][['행정동_코드', '20대_이동비율_변화']]

hm_base = (top15_2024
           .merge(consume_df[consume_df['연도'] == 2024]
                  [['행정동_코드', '2030소비_변화', '분위변화', '임대료_변화', '공실률_변화']],
                  on='행정동_코드', how='left')
           .merge(mig_2024, on='행정동_코드', how='left')
           .set_index('행정동')
           .sort_values('2030소비_변화', ascending=False))

# 히트맵용 5개 지표 (공실률은 감소가 좋으므로 부호 반전)
hm_cols   = ['2030소비_변화', '20대_이동비율_변화', '분위변화', '임대료_변화', '공실률_변화']
hm_labels = ['2030 소비비중\n변화(%p)', '20대 이동비율\n변화(%p)',
             '상권 복합점수\n분위 변화(%p)', '임대료\n변화율(%)', '공실률\n변화(%p)']
hm_sign   = [1, 1, 1, 1, -1]   # 공실률은 감소가 좋음 → 색상 반전용

hm_data   = hm_base[hm_cols].values   # (15, 5)
n_areas   = len(hm_base)

# 지표별 색상 정규화 (행 단위: 15개 지역 내 상대값)
fig3, ax3 = plt.subplots(figsize=(13, 9))
ax3.set_facecolor('#f8f8f8')

for j, (col, label, sign) in enumerate(zip(hm_cols, hm_labels, hm_sign)):
    col_vals = hm_data[:, j] * sign   # 공실률은 방향 반전
    valid    = col_vals[~np.isnan(col_vals)]
    if len(valid) > 1:
        vmin, vmax = valid.min(), valid.max()
        norm_func = mcolors.TwoSlopeNorm(
            vmin=min(vmin, -0.01), vcenter=0, vmax=max(vmax, 0.01))
    else:
        norm_func = mcolors.Normalize(0, 1)

    for i, val in enumerate(col_vals):
        if np.isnan(val):
            fc = '#dddddd'
            txt = 'N/A'
            tc = '#999'
        else:
            fc  = cm.RdYlGn(norm_func(val))
            raw = hm_data[i, j]
            sign_str = '+' if raw >= 0 else ''
            fmt = '.1f' if abs(raw) < 10 else '.0f'
            txt = f'{sign_str}{raw:{fmt}}'
            # 어두운 배경 → 흰 글씨, 밝은 배경 → 검은 글씨
            brightness = 0.299*fc[0] + 0.587*fc[1] + 0.114*fc[2]
            tc = 'white' if brightness < 0.5 else 'black'

        rect = plt.Rectangle([j - 0.45, i - 0.45], 0.9, 0.9,
                              facecolor=fc, edgecolor='white', linewidth=1.0)
        ax3.add_patch(rect)
        ax3.text(j, i, txt, ha='center', va='center', fontsize=9.5,
                 fontweight='bold', color=tc)

ax3.set_xlim(-0.5, len(hm_cols) - 0.5)
ax3.set_ylim(-0.5, n_areas - 0.5)
ax3.set_xticks(range(len(hm_cols)))
ax3.set_xticklabels(hm_labels, fontsize=10, fontweight='bold')
ax3.set_yticks(range(n_areas))
ax3.set_yticklabels(hm_base.index, fontsize=10)
ax3.xaxis.set_ticks_position('top')
ax3.xaxis.set_label_position('top')
ax3.invert_yaxis()

ax3.set_title('2024년 2030 소비비중 상승 Top 15 — 다중 지표 종합 비교\n'
              '(색상: 초록↑좋음 / 빨강↓나쁨  |  공실률: 감소 방향이 초록  |  N/A: 임대료 데이터 없음)',
              fontsize=12, fontweight='bold', pad=40)
plt.tight_layout()
plt.savefig('image/yoy_top15_2030_종합비교.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('저장: image/yoy_top15_2030_종합비교.png')

# ── 텍스트 요약 ───────────────────────────────────────────────
print('\n=== 2030 소비비중 변화 Top 5 (연도별) ===')
for yr in yoy_years:
    sub = consume_df[consume_df['연도'] == yr].nlargest(5, '2030소비_변화')
    print(f'  {yr}:')
    for _, row in sub.iterrows():
        div = f'  분위{row["분위변화"]:+.0f}%p' if not np.isnan(row['분위변화']) else ''
        rent = f'  임대료{row["임대료_변화"]:+.1f}%' if not np.isnan(row['임대료_변화']) else ''
        print(f'    {row["행정동"]}: 소비비중 {row["2030소비_변화"]:+.2f}%p{div}{rent}')

print('\n=== 20대 이동비율 변화 Top 5 (연도별) ===')
for yr in yoy_years:
    sub = mig_df[mig_df['연도'] == yr].nlargest(5, '20대_이동비율_변화')
    print(f'  {yr}:')
    for _, row in sub.iterrows():
        div = f'  분위{row["분위변화"]:+.0f}%p' if not np.isnan(row['분위변화']) else ''
        rent = f'  임대료{row["임대료_변화"]:+.1f}%' if not np.isnan(row['임대료_변화']) else ''
        print(f'    {row["행정동"]}: 이동비율 {row["20대_이동비율_변화"]:+.2f}%p{div}{rent}')
