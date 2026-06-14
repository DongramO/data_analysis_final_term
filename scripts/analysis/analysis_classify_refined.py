# -*- coding: utf-8 -*-
"""
상권 유형 정밀 분류 (6티어)
============================
3개 축으로 분류:
  Level      : 2024년 복합점수_분위 (현재 위치)
  Momentum   : 2020→2024 분위 변화량 (성장 속도)
  Consistency: 2020~2024 선형 추세 R² (일관성)

6티어 정의:
  ① 레전드     : Level ≥ 0.90 + Consistency R² ≥ 0.5 (꾸준히 최상위)
  ② 라이징스타  : Level ≥ 0.70 + Momentum 상위 20% (상위권에서 계속 오름)
  ③ 급성장 신흥 : Level 0.50~0.90 + Momentum 상위 10% (빠르게 치고 오르는 중)
  ④ 잠재 후보  : Level 0.40~0.70 + Momentum 양수 + 상위 30% (주목할 예비군)
  ⑤ 성숙·정체  : Level ≥ 0.70 + ①②③ 조건 미충족 (이미 강세, 모멘텀 무관하게 상위권 전부)
  ⑥ 쇠퇴 주의  : 2020 Level ≥ 0.70 + 2024 Level < 0.50 (뚜렷한 하락)
  ⑦ 일반       : 나머지
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats as sp_stats
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

data_dir  = 'data'
image_dir = 'image'

# ════════════════════════════════════════════════════════════
# 1. 마스터 파일 로드 + 행정동별 연도 추세 계산
# ════════════════════════════════════════════════════════════
master = pd.read_csv(os.path.join(data_dir, '트렌드매출_마스터.csv'), encoding='utf-8-sig')

# 행정동별 5년 분위 궤적
def calc_trajectory(grp):
    years = grp['연도'].values
    ranks = grp['복합점수_분위'].values
    if len(years) < 3:
        return pd.Series({'momentum': np.nan, 'r2': np.nan, 'slope': np.nan})
    slope, intercept, r, p, se = sp_stats.linregress(years, ranks)
    return pd.Series({
        'momentum':  ranks[-1] - ranks[0],   # 2020→2024 분위 변화량
        'slope':     slope,                   # 연간 평균 분위 변화
        'r2':        r**2,                    # 추세 일관성 (1=완전 선형)
        'rank_2020': ranks[0],
        'rank_2024': ranks[-1],
        'rank_std':  ranks.std(),             # 변동성
    })

traj = master.groupby('행정동').apply(calc_trajectory).reset_index()

# 업종점수·2030비중 2024년 값
latest = master[master['연도'] == 2024][['행정동','업종점수','2030비중','트렌드매출_비중','복합점수']].copy()
traj = traj.merge(latest, on='행정동', how='left')

# Momentum 분위 (전체 행정동 기준)
traj['momentum_pct'] = traj['momentum'].rank(pct=True)


# ════════════════════════════════════════════════════════════
# 2. 6티어 분류
# ════════════════════════════════════════════════════════════
def classify(row):
    l24  = row['rank_2024']      # 2024 분위
    l20  = row['rank_2020']      # 2020 분위
    mom  = row['momentum_pct']   # 모멘텀 분위
    r2   = row['r2']             # 일관성
    if pd.isna(l24) or pd.isna(l20):
        return '데이터 없음'

    # ① 레전드: 최상위 + 꾸준함
    if l24 >= 0.90 and r2 >= 0.50:
        return '① 레전드'
    # ② 라이징스타: 상위권 + 강한 모멘텀
    if l24 >= 0.70 and mom >= 0.80:
        return '② 라이징스타'
    # ③ 급성장 신흥: 중간~상위 + 극강 모멘텀
    if l24 >= 0.50 and mom >= 0.90 and l20 < 0.70:
        return '③ 급성장 신흥'
    # ④ 잠재 후보: 중간권 + 양의 모멘텀
    if 0.40 <= l24 < 0.70 and mom >= 0.60 and l20 < 0.55:
        return '④ 잠재 후보'
    # ⑥ 쇠퇴 주의
    if l20 >= 0.70 and l24 < 0.50:
        return '⑥ 쇠퇴 주의'
    # ⑤ 성숙·정체: 분위 ≥ 0.70 이면서 ①②③⑥ 어디에도 해당하지 않는 모든 경우
    if l24 >= 0.70:
        return '⑤ 성숙·정체'
    return '⑦ 일반'

traj['상권티어'] = traj.apply(classify, axis=1)

# 결과 출력
print("=== 상권 티어 분류 결과 ===")
print(traj['상권티어'].value_counts().to_string())

TIER_ORDER = ['① 레전드','② 라이징스타','③ 급성장 신흥','④ 잠재 후보','⑤ 성숙·정체','⑥ 쇠퇴 주의','⑦ 일반']
TIER_COLORS = {
    '① 레전드':    '#8b0000',
    '② 라이징스타': '#e15759',
    '③ 급성장 신흥':'#59a14f',
    '④ 잠재 후보':  '#76b7b2',
    '⑤ 성숙·정체': '#f28e2b',
    '⑥ 쇠퇴 주의': '#4e79a7',
    '⑦ 일반':      '#d3d3d3',
}

for tier in TIER_ORDER[:-1]:
    sub = traj[traj['상권티어'] == tier].sort_values('rank_2024', ascending=False)
    print(f"\n[{tier}] ({len(sub)}개)")
    cols = ['행정동','rank_2020','rank_2024','momentum','r2','업종점수','2030비중']
    print(sub[cols].head(12).round(3).to_string(index=False))


# ════════════════════════════════════════════════════════════
# 3. 마스터 파일에 티어 반영
# ════════════════════════════════════════════════════════════
tier_map = traj.set_index('행정동')['상권티어']
master['상권티어'] = master['행정동'].map(tier_map)
master.to_csv(os.path.join(data_dir, '트렌드매출_마스터.csv'), index=False, encoding='utf-8-sig')
print(f"\n마스터 파일 업데이트: data/트렌드매출_마스터.csv")


# ════════════════════════════════════════════════════════════
# 시각화 1. Level × Momentum 산점도 (티어별 색상)
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(20, 9))
fig.suptitle('상권 6티어 정밀 분류\n(Level=2024년 분위, Momentum=2020→2024 변화량)',
             fontsize=14, fontweight='bold')

ax = axes[0]
for tier in TIER_ORDER:
    sub = traj[traj['상권티어'] == tier]
    if tier == '⑦ 일반':
        ax.scatter(sub['rank_2024'], sub['momentum'],
                   color=TIER_COLORS[tier], alpha=0.3, s=20, label=f'{tier} ({len(sub)})', zorder=1)
    else:
        ax.scatter(sub['rank_2024'], sub['momentum'],
                   color=TIER_COLORS[tier], alpha=0.85, s=60,
                   edgecolors='white', linewidths=0.6,
                   label=f'{tier} ({len(sub)})', zorder=3)

# 주목 상권 레이블
for tier in ['① 레전드','② 라이징스타','③ 급성장 신흥','④ 잠재 후보']:
    sub = traj[traj['상권티어'] == tier].nlargest(5, 'rank_2024')
    for _, row in sub.iterrows():
        ax.annotate(row['행정동'], (row['rank_2024'], row['momentum']),
                    fontsize=7, xytext=(3,2), textcoords='offset points',
                    color=TIER_COLORS[tier])

ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax.axvline(0.70, color='gray', linewidth=0.5, linestyle=':', alpha=0.6)
ax.set_xlabel('2024년 복합점수 분위 (0=하위, 1=상위)', fontsize=11)
ax.set_ylabel('모멘텀 (2020→2024 분위 변화량)', fontsize=11)
ax.set_title('Level × Momentum 분포', fontsize=12)
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.25)
ax.spines[['top','right']].set_visible(False)

# 분면 텍스트
ax.text(0.88, ax.get_ylim()[1]*0.9, '레전드\n라이징스타', ha='center', fontsize=8,
        color='#8b0000', fontweight='bold')
ax.text(0.20, ax.get_ylim()[1]*0.9, '급성장\n신흥·잠재', ha='center', fontsize=8,
        color='#27ae60', fontweight='bold')


# ════════════════════════════════════════════════════════════
# 시각화 2. 티어별 2030비중 × 트렌드매출비중 비교 (사후 검증)
# ════════════════════════════════════════════════════════════
ax = axes[1]
for tier in TIER_ORDER:
    sub = traj[traj['상권티어'] == tier].dropna(subset=['트렌드매출_비중','2030비중'])
    if tier == '⑦ 일반':
        ax.scatter(sub['트렌드매출_비중']*100, sub['2030비중']*100,
                   color=TIER_COLORS[tier], alpha=0.25, s=20, label=f'{tier} ({len(sub)})', zorder=1)
    else:
        ax.scatter(sub['트렌드매출_비중']*100, sub['2030비중']*100,
                   color=TIER_COLORS[tier], alpha=0.85, s=65,
                   edgecolors='white', linewidths=0.6,
                   label=f'{tier} ({len(sub)})', zorder=3)

ax.set_xlabel('트렌드 대표 매출 비중 % (금저녁+토+일)', fontsize=11)
ax.set_ylabel('2030세대 매출 비중 % (사후 검증용)', fontsize=11)
ax.set_title('티어별 트렌드매출 비중 vs 2030비중\n(2030 배제로 정의 → 사후에 어떻게 분포?)', fontsize=11)
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.25)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'classify_refined_overview.png'), dpi=150, bbox_inches='tight')
print("저장: image/classify_refined_overview.png")
plt.show()


# ════════════════════════════════════════════════════════════
# 시각화 3. 핵심 티어 연도별 트렌드대표매출 추이
# ════════════════════════════════════════════════════════════
key_tiers = ['① 레전드','② 라이징스타','③ 급성장 신흥','④ 잠재 후보']

annual = (
    master[master['상권티어'].isin(key_tiers)]
    .groupby(['연도','상권티어'])
    .agg(트렌드매출=('트렌드_대표_매출','sum'),
         비중평균=('트렌드매출_비중','mean'),
         gen2030=('2030비중','mean'))
    .reset_index()
)

# 2020=100 인덱스
for t in key_tiers:
    base = annual.loc[(annual['연도']==2020) & (annual['상권티어']==t), '트렌드매출'].values
    if len(base) > 0 and base[0] > 0:
        annual.loc[annual['상권티어']==t, '매출_idx'] = (
            annual.loc[annual['상권티어']==t, '트렌드매출'] / base[0] * 100
        )

fig2, axes2 = plt.subplots(1, 3, figsize=(20, 7))
fig2.suptitle('핵심 4개 티어 연도별 추이 비교 (2020=100)', fontsize=14, fontweight='bold')

for ax, ycol, ylabel in [
    (axes2[0], '매출_idx',   '트렌드 대표 매출 (인덱스)'),
    (axes2[1], '비중평균',   '트렌드매출 비중 평균 (%)'),
    (axes2[2], 'gen2030',   '2030비중 평균 (%)'),
]:
    for t in key_tiers:
        sub = annual[annual['상권티어']==t].sort_values('연도')
        if ycol not in sub.columns or sub[ycol].isna().all():
            continue
        yvals = sub[ycol] * (100 if ycol in ['비중평균','gen2030'] else 1)
        ax.plot(sub['연도'].astype(str), yvals,
                color=TIER_COLORS[t], marker='o', linewidth=2.5, markersize=7, label=t)
        ax.annotate(f"{yvals.iloc[-1]:.1f}",
                    (str(int(sub['연도'].iloc[-1])), yvals.iloc[-1]),
                    xytext=(5,0), textcoords='offset points',
                    color=TIER_COLORS[t], fontsize=8.5, fontweight='bold')

    if ycol == '매출_idx':
        ax.axhline(100, color='gray', linewidth=0.8, linestyle='--')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(ylabel, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'classify_refined_timeseries.png'), dpi=150, bbox_inches='tight')
print("저장: image/classify_refined_timeseries.png")
plt.show()

# ════════════════════════════════════════════════════════════
# 4. 모델링 피처 요약 출력
# ════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║          차세대 상권 예측 모델링 피처 요약                    ║
╠══════════════════════════════════════════════════════════════╣
║ [즉시 사용 가능 — 트렌드매출_마스터.csv]                    ║
║  · 트렌드매출_비중 (현재 + 변화량)                           ║
║  · 복합점수_분위 (현재 + 변화량 = Momentum)                  ║
║  · 업종점수 (F&B/라이프스타일/주류 비중)                     ║
║  · 금요효과 / 주말비중 / 야간비중                            ║
║  · 2030비중 (종속변수 또는 검증용)                           ║
║  · 분위 선형 기울기 (slope) / 일관성 (R²)                    ║
║                                                              ║
║ [원본 데이터 추가 계산 필요]                                 ║
║  · 업종 신규 진입률 (카페/팝업 신규 비중 변화)               ║
║  · 점포 다양성 지수 (업종 수 / 헤르핀달 지수)               ║
║  · 매출 집중도 (상위 업종이 매출 독점하는지)                 ║
║                                                              ║
║ [외부 수집 필요 — 현재 미보유]                               ║
║  · 임대료 수준 및 변화율  ← 가장 중요한 선행지표             ║
║  · 공실률 변화                                               ║
║  · 지하철 승하차 인원 변화 (유동인구 proxy)                  ║
║  · 신규 사업자 등록 건수 (창업 활성도)                       ║
║  · SNS 해시태그 언급량 변화                                  ║
║                                                              ║
║ [예측 목표 변수 (Target)]                                    ║
║  · 2년 후 상권티어 상향 여부 (분류 문제)                     ║
║  · 2년 후 복합점수_분위 (회귀 문제)                          ║
║  · 2년 후 트렌드매출_비중 변화량                             ║
╚══════════════════════════════════════════════════════════════╝
""")

print("완료!")
