# -*- coding: utf-8 -*-
"""
트렌드 대표 매출 마스터 파일 생성 + 기존/신흥 상권 분류
==========================================================

[트렌드 대표 매출 정의]
  금요일_추정_저녁 = 금요일_매출_금액 × 야간비중
    (야간비중 = (시간대_17~21 + 시간대_21~24) / 전체 시간대)
  트렌드_대표_매출 = 금요일_추정_저녁 + 토요일_매출_금액 + 일요일_매출_금액

  ※ 요일×시간 교차 데이터 미제공으로 인한 추정값
     금요일 야간비중을 전체 시간대 야간비중으로 proxy 사용

[기존/신흥 분류 기준]
  연도별 복합점수(업종+시간패턴)를 각 연도 내 표준화 → 분위 산출
  - 기존 강세: 2020 AND 2024 모두 상위 30%
  - 신흥 성장: 2020 하위 70%, 2024 상위 30% (새로 진입)
  - 쇠퇴:     2020 상위 30%, 2024 하위 70% (이탈)
  - 일반:     나머지

[출력]
  data/트렌드매출_마스터.csv — 행정동 × 연도별 트렌드 지표 통합본
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, sys
sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

data_dir  = 'data'
image_dir = 'image'

# ── 업종 카테고리 ──────────────────────────────────────────
CATEG_MAP = {
    'F&B (카페/디저트)': ['커피-음료', '제과점'],
    'F&B (식사)':       ['한식음식점', '일식음식점', '양식음식점', '중식음식점',
                         '분식전문점', '패스트푸드점', '치킨전문점'],
    '주류/유흥':        ['호프-간이주점', '노래방'],
    '라이프스타일':     ['일반의류', '편의점', '애완동물', '화초', '인테리어'],
}
TREND_CATEGS = list(CATEG_MAP.keys())

DAY_COLS  = ['월요일_매출_금액','화요일_매출_금액','수요일_매출_금액',
             '목요일_매출_금액','금요일_매출_금액','토요일_매출_금액','일요일_매출_금액']
TIME_COLS = ['시간대_00~06_매출_금액','시간대_06~11_매출_금액','시간대_11~14_매출_금액',
             '시간대_14~17_매출_금액','시간대_17~21_매출_금액','시간대_21~24_매출_금액']

# ════════════════════════════════════════════════════════════
# 1. 데이터 로드 및 기초 집계
# ════════════════════════════════════════════════════════════
print("데이터 로드 중...")
df = pd.read_csv(
    os.path.join(data_dir, '서울시_상권분석서비스(추정매출-행정동)_2020_2024.csv'),
    encoding='utf-8-sig',
)
df['year'] = df['기준_년분기_코드'] // 10
df['category'] = df['서비스_업종_코드_명'].map(
    lambda x: next((c for c, items in CATEG_MAP.items() if x in items), '기타')
)

# 행정동 × 연도별 집계
grp_cols = ['행정동_코드', '행정동_코드_명', 'year']
agg_dict = {c: 'sum' for c in DAY_COLS + TIME_COLS}
agg_dict.update({
    '당월_매출_금액': 'sum',
    '연령대_20_매출_금액': 'sum',
    '연령대_30_매출_금액': 'sum',
})
base = df.groupby(grp_cols).agg(agg_dict).reset_index()

# 업종 비중 (트렌드 4종)
categ_agg = (
    df.groupby(grp_cols + ['category'])['당월_매출_금액']
    .sum().reset_index()
)
trend_categ = categ_agg[categ_agg['category'].isin(TREND_CATEGS)]
trend_categ_sum = trend_categ.groupby(grp_cols)['당월_매출_금액'].sum().reset_index()
trend_categ_sum.columns = grp_cols + ['트렌드업종_매출']
base = base.merge(trend_categ_sum, on=grp_cols, how='left')


# ════════════════════════════════════════════════════════════
# 2. 트렌드 대표 매출 계산
# ════════════════════════════════════════════════════════════
day_total  = base[DAY_COLS].sum(axis=1)
time_total = base[TIME_COLS].sum(axis=1)

# 야간비중 (전체 시간대 기준)
base['야간비중']  = (base['시간대_17~21_매출_금액'] + base['시간대_21~24_매출_금액']) / time_total

# 금요일 저녁 추정 = 금요일 매출 × 야간비중
base['금요저녁_추정'] = base['금요일_매출_금액'] * base['야간비중']

# 트렌드 대표 매출 = 금요저녁(추정) + 토 + 일
base['트렌드_대표_매출'] = (
    base['금요저녁_추정'] +
    base['토요일_매출_금액'] +
    base['일요일_매출_금액']
)

# 파생 지표
base['총_매출']      = base['당월_매출_금액']
base['트렌드매출_비중'] = base['트렌드_대표_매출'] / base['총_매출']
base['업종점수']     = base['트렌드업종_매출'] / base['총_매출']
base['2030비중']    = (base['연령대_20_매출_금액'] + base['연령대_30_매출_금액']) / base['총_매출']
base['금요효과']    = base['금요일_매출_금액'] / base['목요일_매출_금액']
base['주말비중']    = (base['토요일_매출_금액'] + base['일요일_매출_금액']) / day_total


# ════════════════════════════════════════════════════════════
# 3. 연도별 복합점수 (각 연도 내 표준화)
# ════════════════════════════════════════════════════════════
def zscore(s):
    return (s - s.mean()) / s.std()

records = []
for yr, gdf in base.groupby('year'):
    tmp = gdf.copy()
    tmp['temporal_score'] = zscore(tmp['금요효과']) + zscore(tmp['주말비중']) + zscore(tmp['야간비중'])
    tmp['복합점수']       = zscore(tmp['업종점수']) + zscore(tmp['temporal_score'])
    tmp['복합점수_분위']  = tmp['복합점수'].rank(pct=True)   # 0~1, 높을수록 상위
    records.append(tmp)

base = pd.concat(records, ignore_index=True)


# ════════════════════════════════════════════════════════════
# 4. 기존/신흥/쇠퇴/일반 분류 (2020 vs 2024 분위 비교)
# ════════════════════════════════════════════════════════════
rank_2020 = base[base['year']==2020].set_index('행정동_코드_명')['복합점수_분위']
rank_2024 = base[base['year']==2024].set_index('행정동_코드_명')['복합점수_분위']

TOP = 0.70   # 상위 30% 임계값

def classify(dong):
    r20 = rank_2020.get(dong, np.nan)
    r24 = rank_2024.get(dong, np.nan)
    if pd.isna(r20) or pd.isna(r24):
        return '데이터 없음'
    if r20 >= TOP and r24 >= TOP:
        return '기존 강세'
    elif r20 < TOP and r24 >= TOP:
        return '신흥 성장'
    elif r20 >= TOP and r24 < TOP:
        return '쇠퇴'
    else:
        return '일반'

all_dongs = base['행정동_코드_명'].unique()
classification = {d: classify(d) for d in all_dongs}
base['상권유형'] = base['행정동_코드_명'].map(classification)

print("\n=== 상권 유형 분류 결과 ===")
type_counts = base[base['year']==2024]['상권유형'].value_counts()
print(type_counts.to_string())

for t in ['기존 강세', '신흥 성장', '쇠퇴']:
    dongs = [d for d, v in classification.items() if v == t]
    top_n = (base[base['year']==2024]
             .set_index('행정동_코드_명')
             .loc[[d for d in dongs if d in base[base['year']==2024]['행정동_코드_명'].values]]
             .nlargest(10, '복합점수')
             .index.tolist())
    print(f"\n[{t}] 상위 10개: {top_n}")


# ════════════════════════════════════════════════════════════
# 5. 마스터 파일 저장
# ════════════════════════════════════════════════════════════
SAVE_COLS = [
    '행정동_코드', '행정동_코드_명', 'year',
    '트렌드_대표_매출', '총_매출', '트렌드매출_비중',
    '업종점수', '2030비중',
    '금요효과', '주말비중', '야간비중',
    '복합점수', '복합점수_분위', '상권유형',
]
master = base[SAVE_COLS].copy()
master.columns = [
    '행정동_코드', '행정동', '연도',
    '트렌드_대표_매출', '총_매출', '트렌드매출_비중',
    '업종점수', '2030비중',
    '금요효과', '주말비중', '야간비중',
    '복합점수', '복합점수_분위', '상권유형',
]
out_path = os.path.join(data_dir, '트렌드매출_마스터.csv')
master.to_csv(out_path, index=False, encoding='utf-8-sig')
print(f"\n마스터 파일 저장: {out_path}")
print(f"크기: {master.shape}")


# ════════════════════════════════════════════════════════════
# 시각화 1. 상권 유형별 트렌드 대표 매출 추이 (2020→2024)
# ════════════════════════════════════════════════════════════
TYPE_COLORS = {
    '기존 강세': '#e15759',
    '신흥 성장': '#59a14f',
    '쇠퇴':     '#4e79a7',
    '일반':     '#bab0ac',
}

annual_type = (
    base.groupby(['year', '상권유형'])
    .agg(트렌드매출=('트렌드_대표_매출','sum'),
         총매출=('총_매출','sum'),
         비중평균=('트렌드매출_비중','mean'),
         gen2030=('2030비중','mean'))
    .reset_index()
)
# 2020=100 인덱스
for t in ['기존 강세', '신흥 성장', '쇠퇴']:
    base_val = annual_type.loc[(annual_type['year']==2020) & (annual_type['상권유형']==t), '트렌드매출'].values
    if len(base_val) > 0:
        base_v = base_val[0]
        annual_type.loc[annual_type['상권유형']==t, '매출_idx'] = (
            annual_type.loc[annual_type['상권유형']==t, '트렌드매출'] / base_v * 100
        )

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('상권 유형별 트렌드 대표 매출 추이 비교 (2020=100)', fontsize=14, fontweight='bold')

# (a) 트렌드 대표 매출 인덱스
ax = axes[0]
for t in ['기존 강세', '신흥 성장', '쇠퇴']:
    sub = annual_type[annual_type['상권유형']==t].sort_values('year')
    if '매출_idx' in sub.columns:
        ax.plot(sub['year'].astype(str), sub['매출_idx'],
                color=TYPE_COLORS[t], marker='o', linewidth=2.5, markersize=7, label=t)
        last = sub.iloc[-1]
        ax.annotate(f"{last['매출_idx']:.0f}",
                    (str(int(last['year'])), last['매출_idx']),
                    xytext=(5,0), textcoords='offset points',
                    color=TYPE_COLORS[t], fontsize=9, fontweight='bold')
ax.axhline(100, color='gray', linewidth=0.8, linestyle='--')
ax.set_title('트렌드 대표 매출\n(금저녁추정+토+일)', fontsize=11)
ax.set_ylabel('인덱스 (2020=100)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.spines[['top','right']].set_visible(False)

# (b) 트렌드 매출 비중 평균 추이
ax = axes[1]
for t in ['기존 강세', '신흥 성장', '쇠퇴']:
    sub = annual_type[annual_type['상권유형']==t].sort_values('year')
    ax.plot(sub['year'].astype(str), sub['비중평균']*100,
            color=TYPE_COLORS[t], marker='o', linewidth=2.5, markersize=7, label=t)
ax.set_title('트렌드 대표 매출 비중 평균\n(트렌드매출/총매출)', fontsize=11)
ax.set_ylabel('비중 (%)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.spines[['top','right']].set_visible(False)

# (c) 2030비중 평균 추이
ax = axes[2]
for t in ['기존 강세', '신흥 성장', '쇠퇴']:
    sub = annual_type[annual_type['상권유형']==t].sort_values('year')
    ax.plot(sub['year'].astype(str), sub['gen2030']*100,
            color=TYPE_COLORS[t], marker='o', linewidth=2.5, markersize=7, label=t)
ax.set_title('2030비중 평균 추이\n(사후 검증용)', fontsize=11)
ax.set_ylabel('2030 매출 비중 (%)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'trend_type_timeseries.png'), dpi=150, bbox_inches='tight')
print("\n저장: image/trend_type_timeseries.png")
plt.show()


# ════════════════════════════════════════════════════════════
# 시각화 2. 기존 강세 Top 15 vs 신흥 성장 Top 15 비교
# ════════════════════════════════════════════════════════════
def get_top(t, n=15, sort_by='복합점수'):
    return (base[(base['year']==2024) & (base['상권유형']==t)]
            .nlargest(n, sort_by)['행정동_코드_명'].tolist())

top_existing = get_top('기존 강세')
top_emerging = get_top('신흥 성장')

# 2020→2024 트렌드매출_비중 변화
def trend_ratio_change(dongs):
    r20 = base[base['year']==2020].set_index('행정동_코드_명')['트렌드매출_비중']
    r24 = base[base['year']==2024].set_index('행정동_코드_명')['트렌드매출_비중']
    return {d: (r24.get(d, np.nan) - r20.get(d, np.nan)) * 100 for d in dongs}

chg_exist = pd.Series(trend_ratio_change(top_existing)).dropna().sort_values()
chg_emerg = pd.Series(trend_ratio_change(top_emerging)).dropna().sort_values()

fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))
fig2.suptitle('기존 강세 vs 신흥 성장: 트렌드 대표 매출 비중 변화 (2020→2024)',
              fontsize=13, fontweight='bold')

for ax, chg, t, color in [
    (axes2[0], chg_exist, '기존 강세', '#e15759'),
    (axes2[1], chg_emerg, '신흥 성장', '#59a14f'),
]:
    bars = ax.barh(chg.index, chg.values, color=color, alpha=0.85, edgecolor='white')
    for bar, v in zip(bars, chg.values):
        ax.text(v + (0.1 if v >= 0 else -0.1), bar.get_y() + bar.get_height()/2,
                f'{v:+.1f}pp', va='center', ha='left' if v >= 0 else 'right', fontsize=8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('트렌드 대표 매출 비중 변화 (pp)', fontsize=10)
    ax.set_title(f'[{t}] 상위 15개\n트렌드매출 비중 변화', fontsize=11)
    ax.grid(True, axis='x', alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'trend_type_compare.png'), dpi=150, bbox_inches='tight')
print("저장: image/trend_type_compare.png")
plt.show()


# ════════════════════════════════════════════════════════════
# 시각화 3. 상권 유형 분포: 복합점수_분위 2020 vs 2024 산점도
# ════════════════════════════════════════════════════════════
rank_df = pd.DataFrame({'2020분위': rank_2020, '2024분위': rank_2024}).dropna()
rank_df['유형'] = rank_df.index.map(classification)

fig3, ax = plt.subplots(figsize=(12, 10))
fig3.suptitle('행정동별 복합점수 분위 변화 (2020 → 2024)\n각 점 = 행정동, 대각선 위 = 상승, 아래 = 하락',
              fontsize=13, fontweight='bold')

for t, color in TYPE_COLORS.items():
    sub = rank_df[rank_df['유형'] == t]
    ax.scatter(sub['2020분위'], sub['2024분위'],
               color=color, alpha=0.6 if t == '일반' else 0.85,
               s=30 if t == '일반' else 65,
               label=f'{t} ({len(sub)}개)',
               edgecolors='white' if t != '일반' else 'none',
               linewidths=0.5, zorder=3 if t != '일반' else 1)

# 주목 상권 레이블 — 신흥 성장 상위 10
for dong in top_emerging[:10]:
    if dong in rank_df.index:
        row = rank_df.loc[dong]
        ax.annotate(dong, (row['2020분위'], row['2024분위']),
                    fontsize=7.5, xytext=(4,2), textcoords='offset points', color='#27ae60')
# 기존 강세 상위 5
for dong in top_existing[:5]:
    if dong in rank_df.index:
        row = rank_df.loc[dong]
        ax.annotate(dong, (row['2020분위'], row['2024분위']),
                    fontsize=7.5, xytext=(4,2), textcoords='offset points', color='#c0392b')

ax.plot([0,1], [0,1], color='gray', linewidth=1, linestyle='--', label='변화 없음')
ax.axhline(TOP, color='gray', linewidth=0.5, linestyle=':', alpha=0.7)
ax.axvline(TOP, color='gray', linewidth=0.5, linestyle=':', alpha=0.7)

# 4분면 레이블
ax.text(0.02, 0.97, '신흥 성장', fontsize=10, color='#27ae60', fontweight='bold',
        transform=ax.transAxes, va='top')
ax.text(0.72, 0.97, '기존 강세', fontsize=10, color='#c0392b', fontweight='bold',
        transform=ax.transAxes, va='top')
ax.text(0.72, 0.05, '쇠퇴', fontsize=10, color='#4e79a7', fontweight='bold',
        transform=ax.transAxes, va='bottom')

ax.set_xlabel('2020년 복합점수 분위 (0=하위, 1=상위)', fontsize=11)
ax.set_ylabel('2024년 복합점수 분위 (0=하위, 1=상위)', fontsize=11)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, alpha=0.25)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(image_dir, 'trend_type_scatter.png'), dpi=150, bbox_inches='tight')
print("저장: image/trend_type_scatter.png")
plt.show()

print("\n완료!")
