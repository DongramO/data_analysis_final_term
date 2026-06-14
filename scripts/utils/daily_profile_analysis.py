# -*- coding: utf-8 -*-
"""
일단위 생활인구 프로파일 분석
- 요일별 유동인구 패턴으로 상권·오피스·주거 구역 구분
- LOCAL_PEOPLE_DONG 파일 사용 (시간대×행정동×연령)

컬럼 실제 위치 (pandas가 날짜를 index로 읽은 후):
  index  = YYYYMMDD 날짜
  iloc[:,0] = 시간대 (0-23)
  iloc[:,1] = 행정동코드 (8자리, 예: 11110515)
  iloc[:,2] = 총생활인구수
  iloc[:,3] = 남자0-9
  iloc[:,6] = 남자20-24  ┐
  iloc[:,7] = 남자25-29  ├ 2030 남
  iloc[:,8] = 남자30-34  ┘
  iloc[:,20] = 여자20-24 ┐
  iloc[:,21] = 여자25-29 ├ 2030 여
  iloc[:,22] = 여자30-34 ┘
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import glob, os
import warnings; warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ── 1. 코드 → 행정동명 매핑 ──────────────────────────────────────
rf = pd.read_csv('data/main_data/분기별_원본피처.csv', encoding='utf-8-sig')
code2name = rf.iloc[:, [0, 1]].drop_duplicates()
code2name.columns = ['코드', '행정동']
code2name = dict(zip(code2name['코드'].astype(int), code2name['행정동']))

# ── 2. 생활인구 파일 로드 (2024 하반기 3개월) ─────────────────────
files = sorted(glob.glob('data/population/LOCAL_PEOPLE_DONG_2024*.csv'))[-6:-3]  # 07,08,09
print(f'로드 파일: {[os.path.basename(f) for f in files]}')

rows = []
for fpath in files:
    df = pd.read_csv(fpath, encoding='utf-8-sig')
    # index = 날짜(YYYYMMDD), 컬럼은 한 칸씩 밀려 있음
    dates   = df.index.astype(int)
    hours   = df.iloc[:, 0].values
    codes   = df.iloc[:, 1].values.astype(int)
    pop_tot = df.iloc[:, 2].values
    pop_2030= df.iloc[:, [6,7,8,20,21,22]].sum(axis=1).values

    tmp = pd.DataFrame({
        '날짜': dates,
        '시간': hours,
        '코드': codes,
        '총인구': pop_tot,
        '2030인구': pop_2030,
    })
    rows.append(tmp)

pop = pd.concat(rows, ignore_index=True)

# 요일 추출 (0=월, 6=일)
pop['날짜'] = pd.to_datetime(pop['날짜'].astype(str), format='%Y%m%d')
pop['요일'] = pop['날짜'].dt.dayofweek   # 0=월 ~ 6=일
pop['행정동'] = pop['코드'].map(code2name)
pop = pop.dropna(subset=['행정동'])

print(f'총 행: {len(pop):,}  |  커버 행정동: {pop["행정동"].nunique()}')

# ── 3. 요일별 평균 인구 집계 ─────────────────────────────────────
# 날짜×행정동 기준 일별 합산 후 요일별 평균 (시간 합산)
daily = (pop.groupby(['날짜','요일','행정동'])[['총인구','2030인구']]
           .sum().reset_index())

# 요일별 평균
by_dow = (daily.groupby(['행정동','요일'])[['총인구','2030인구']]
               .mean().reset_index())

# 정규화: 해당 동의 전체 주간 평균 = 1
dong_mean = by_dow.groupby('행정동')[['총인구','2030인구']].transform('mean')
by_dow['총인구_norm']   = by_dow['총인구']   / dong_mean['총인구']
by_dow['2030인구_norm'] = by_dow['2030인구'] / dong_mean['2030인구']

# ── 4. 상권 구분 지표 계산 ────────────────────────────────────────
# 주말집중도 = (토+일 평균) / (평일 평균)
wknd  = by_dow[by_dow['요일'].isin([5, 6])].groupby('행정동')['총인구'].mean()
wkday = by_dow[by_dow['요일'].isin([0,1,2,3,4])].groupby('행정동')['총인구'].mean()
wknd_2030  = by_dow[by_dow['요일'].isin([5,6])].groupby('행정동')['2030인구'].mean()
wkday_2030 = by_dow[by_dow['요일'].isin([0,1,2,3,4])].groupby('행정동')['2030인구'].mean()

metrics = pd.DataFrame({
    '주말집중도':       wknd / wkday,
    '2030주말집중도':   wknd_2030 / wkday_2030,
    '주말_총인구':      wknd,
    '평일_총인구':      wkday,
}).reset_index().rename(columns={'index':'행정동'})
metrics.columns = ['행정동','주말집중도','2030주말집중도','주말_총인구','평일_총인구']

# TSM 업종점수 병합
tsm = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig')
tsm24 = tsm[tsm['연도']==2024][['행정동','업종점수','2030비중']]
metrics = metrics.merge(tsm24, on='행정동', how='left')

# 구역 분류
def classify(row):
    wc = row['주말집중도']
    if pd.isna(wc):
        return '불명'
    if wc < 0.80:
        return '오피스형'
    elif wc < 0.95:
        return '혼합형'
    elif wc < 1.10:
        return '주거형'
    else:
        return '상권형'

metrics['구역유형'] = metrics.apply(classify, axis=1)
metrics.to_csv('data/main_data/area_type_profile.csv', encoding='utf-8-sig', index=False)

# ── 5. 시각화 ────────────────────────────────────────────────────
DOW_LABELS = ['월', '화', '수', '목', '금', '토', '일']

FOCUS = {
    '서초4동':   ('#C0392B', '--'),
    '역삼1동':   ('#E74C3C', '--'),
    '청파동':    ('#2980B9', '-'),
    '후암동':    ('#3498DB', '-'),
    '합정동':    ('#27AE60', '-'),
    '이태원1동': ('#9B59B6', '-'),
    '연남동':    ('#F39C12', '-'),
    '성수2가1동':('#16A085', '-'),
}

fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor('#F8F9FA')
gs = gridspec.GridSpec(3, 3, hspace=0.50, wspace=0.38,
                       left=0.06, right=0.97, top=0.93, bottom=0.05)
fig.suptitle('일단위 생활인구 요일 프로파일 — 상권·오피스·주거 구역 구분\n(2024년 7-9월 평균, 서울 생활인구 데이터)',
             fontsize=13, fontweight='bold', y=0.96)

# ── 패널 1: 총 인구 요일 프로파일 ───────────────────────────────
ax1 = fig.add_subplot(gs[0, :2])
for dong, (color, ls) in FOCUS.items():
    sub = by_dow[by_dow['행정동'] == dong].sort_values('요일')
    if len(sub) == 0:
        continue
    ax1.plot(sub['요일'], sub['총인구_norm'], color=color, linestyle=ls,
             linewidth=2.5, marker='o', markersize=7, label=dong)

ax1.axhline(1.0, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
ax1.axvspan(4.5, 6.5, alpha=0.06, color='green', label='주말')
ax1.set_xticks(range(7))
ax1.set_xticklabels(DOW_LABELS, fontsize=11)
ax1.set_title('① 총 생활인구 요일 프로파일 (정규화, 주간 평균=1.0)', fontsize=10, fontweight='bold')
ax1.set_ylabel('정규화된 생활인구', fontsize=9)
ax1.legend(fontsize=8.5, ncol=2, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.3, 6.3)

# ── 패널 2: 2030 인구 요일 프로파일 ─────────────────────────────
ax2 = fig.add_subplot(gs[1, :2])
for dong, (color, ls) in FOCUS.items():
    sub = by_dow[by_dow['행정동'] == dong].sort_values('요일')
    if len(sub) == 0:
        continue
    ax2.plot(sub['요일'], sub['2030인구_norm'], color=color, linestyle=ls,
             linewidth=2.5, marker='s', markersize=7, label=dong)

ax2.axhline(1.0, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
ax2.axvspan(4.5, 6.5, alpha=0.06, color='green')
ax2.set_xticks(range(7))
ax2.set_xticklabels(DOW_LABELS, fontsize=11)
ax2.set_title('② 2030 생활인구 요일 프로파일 (정규화)', fontsize=10, fontweight='bold')
ax2.set_ylabel('정규화된 2030 인구', fontsize=9)
ax2.legend(fontsize=8.5, ncol=2, loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-0.3, 6.3)

# ── 패널 3: 주말집중도 분포 (히스토그램) ─────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
metrics_clean = metrics.dropna(subset=['주말집중도'])

color_map = {'오피스형': '#E74C3C', '혼합형': '#F39C12',
             '주거형': '#3498DB', '상권형': '#27AE60', '불명': 'gray'}
for utype, color in color_map.items():
    sub = metrics_clean[metrics_clean['구역유형'] == utype]['주말집중도']
    if len(sub) == 0:
        continue
    ax3.hist(sub, bins=20, color=color, alpha=0.6, label=f'{utype} (n={len(sub)})', density=True)

ax3.axvline(1.0, color='black', linewidth=1.5, linestyle='--', label='주간평균=1')
ax3.set_title('③ 주말집중도 분포\n(주말평균/평일평균)', fontsize=10, fontweight='bold')
ax3.set_xlabel('주말집중도', fontsize=9)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# ── 패널 4: 주말집중도 vs 2030주말집중도 산점도 ───────────────────
ax4 = fig.add_subplot(gs[1, 2])

sc_data = metrics_clean.dropna(subset=['2030주말집중도'])
ax4.scatter(sc_data['주말집중도'], sc_data['2030주말집중도'],
            c=sc_data['주말집중도'], cmap='RdYlGn',
            s=30, alpha=0.4, zorder=1, vmin=0.6, vmax=1.4)

# 1:1 선
lims = [0.5, 1.7]
ax4.plot(lims, lims, 'k--', linewidth=1, alpha=0.4, label='총인구=2030 동일')

# 기준선
ax4.axvline(1.0, color='gray', linewidth=0.7, linestyle=':', alpha=0.6)
ax4.axhline(1.0, color='gray', linewidth=0.7, linestyle=':', alpha=0.6)

# 구역 라벨
ax4.text(0.7, 1.35, '2030만 주말형\n(여가형 2030)', fontsize=8, color='#27AE60', alpha=0.8)
ax4.text(0.65, 0.7, '2030도 평일형\n(직장인 2030)', fontsize=8, color='#E74C3C', alpha=0.8)

for dong, (color, ls) in FOCUS.items():
    r = sc_data[sc_data['행정동'] == dong]
    if len(r) == 0:
        continue
    r = r.iloc[0]
    ax4.scatter(r['주말집중도'], r['2030주말집중도'], c=color,
                s=150, zorder=5, edgecolors='white', linewidth=1.5)
    ax4.annotate(dong, (r['주말집중도'], r['2030주말집중도']),
                 xytext=(5, 3), textcoords='offset points',
                 fontsize=8.5, fontweight='bold', color=color)

ax4.set_title('④ 총인구 주말집중도 vs 2030 주말집중도\n(1.0 위 = 주말에 더 많이 옴)', fontsize=9, fontweight='bold')
ax4.set_xlabel('총인구 주말집중도', fontsize=9)
ax4.set_ylabel('2030 주말집중도', fontsize=9)
ax4.legend(fontsize=8)
ax4.set_xlim(0.55, 1.75)
ax4.set_ylim(0.55, 1.75)
ax4.grid(True, alpha=0.3)

# ── 패널 5: 구역 유형별 대표 프로파일 (평균 곡선) ────────────────
ax5 = fig.add_subplot(gs[2, :2])
type_colors = {'오피스형': '#E74C3C', '혼합형': '#F39C12',
               '주거형': '#3498DB', '상권형': '#27AE60'}

for utype, color in type_colors.items():
    dongs_t = metrics_clean[metrics_clean['구역유형'] == utype]['행정동'].tolist()
    sub = by_dow[by_dow['행정동'].isin(dongs_t)]
    avg = sub.groupby('요일')['총인구_norm'].mean()
    se  = sub.groupby('요일')['총인구_norm'].sem()
    ax5.plot(avg.index, avg.values, color=color, linewidth=2.5,
             marker='o', markersize=8, label=f'{utype} (n={len(dongs_t)})')
    ax5.fill_between(avg.index, avg - se, avg + se, alpha=0.15, color=color)

ax5.axhline(1.0, color='gray', linewidth=0.8, linestyle=':', alpha=0.5)
ax5.axvspan(4.5, 6.5, alpha=0.06, color='green')
ax5.set_xticks(range(7))
ax5.set_xticklabels(DOW_LABELS, fontsize=11)
ax5.set_title('⑤ 구역 유형별 평균 요일 프로파일 (±1 SE)', fontsize=10, fontweight='bold')
ax5.set_ylabel('정규화된 생활인구', fontsize=9)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_xlim(-0.3, 6.3)

# ── 패널 6: 주요 동 요약 바 차트 ─────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
focus_metrics = metrics[metrics['행정동'].isin(list(FOCUS.keys()))].copy()
focus_metrics = focus_metrics.sort_values('주말집중도')

colors_bar = [FOCUS[d][0] for d in focus_metrics['행정동']]
bars = ax6.barh(focus_metrics['행정동'], focus_metrics['주말집중도'],
                color=colors_bar, alpha=0.85)
ax6.axvline(1.0, color='black', linewidth=1.2, linestyle='--', label='평균=1.0')

# 2030 주말집중도 점으로 오버레이
ax6.scatter(focus_metrics['2030주말집중도'], focus_metrics['행정동'],
            marker='D', s=80, color='gold', edgecolors='black',
            linewidth=1, zorder=5, label='2030 주말집중도')

# 구분 라벨
for i, (_, row_d) in enumerate(focus_metrics.iterrows()):
    ax6.text(row_d['주말집중도'] + 0.01, i,
             f"{row_d['구역유형']}  ({row_d['주말집중도']:.2f})",
             va='center', fontsize=8)

ax6.set_title('⑥ 주요 행정동 주말집중도\n(◆ = 2030 주말집중도)', fontsize=10, fontweight='bold')
ax6.set_xlabel('주말집중도 (주말평균/평일평균)', fontsize=8)
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3, axis='x')
ax6.set_xlim(0.5, 1.8)

plt.savefig('image/daily_profile_analysis.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print('저장 완료: image/daily_profile_analysis.png')

# 구역 분류 요약 출력
out = ['=== 구역 유형 분포 ===']
out.append(metrics_clean['구역유형'].value_counts().to_string())
out.append('')
out.append('=== 주요 동 주말집중도 ===')
focus_out = metrics[metrics['행정동'].isin(list(FOCUS.keys()))][
    ['행정동','주말집중도','2030주말집중도','구역유형']].sort_values('주말집중도')
out.append(focus_out.round(3).to_string(index=False))
with open('data/temp_profile_summary.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
