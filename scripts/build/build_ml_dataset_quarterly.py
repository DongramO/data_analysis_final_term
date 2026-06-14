"""
ML 데이터셋 빌드 - 분기 단위 YoY + 행정동 패널 피처 버전
- 행 단위: 행정동 x 연도 x 분기
- YoY 피처: 동일 분기 전년 대비 변화 (계절성 유지, 계절간 왜곡 제거)
- 패널 피처: 행정동 내 시계열 패턴 (lag1, 동편차, 모멘텀)
  - lag1: 1년 전 같은 분기 실제값
  - 동편차: 현재값 - 해당 행정동의 역대 expanding mean (leakage 없음)
  - 모멘텀: YoY 변화 방향이 몇 분기 연속 유지되고 있는지 (부호 포함)
- 타겟: 동일 분기 1년 후 복합점수_분위 변화 → 상승/유지/하락
"""
import pandas as pd
import numpy as np
from scipy.stats import zscore

OUT_DIR   = 'data/main_data'
THRESHOLD = 5.0

# ─────────────────────────────────────────────────────────────
# 1. raw_features_quarterly → 분기별 비중 + 복합점수
# ─────────────────────────────────────────────────────────────
rf = pd.read_csv('data/main_data/분기별_원본피처.csv', encoding='utf-8-sig')

T = '총_매출금액'
def ratio(df, col): return df[col] / df[T]

rf['FB카페_비중']       = ratio(rf, 'FB카페_매출금액')
rf['FB식사_비중']       = ratio(rf, 'FB식사_매출금액')
rf['주류유흥_비중']     = ratio(rf, '주류유흥_매출금액')
rf['라이프스타일_비중'] = ratio(rf, '라이프스타일_매출금액')
rf['기타업종_비중']     = ratio(rf, '기타_매출금액')

rf['월요일_비중'] = ratio(rf, '월요일_매출_금액')
rf['화요일_비중'] = ratio(rf, '화요일_매출_금액')
rf['수요일_비중'] = ratio(rf, '수요일_매출_금액')
rf['목요일_비중'] = ratio(rf, '목요일_매출_금액')
rf['금요일_비중'] = ratio(rf, '금요일_매출_금액')
rf['토요일_비중'] = ratio(rf, '토요일_매출_금액')
rf['일요일_비중'] = ratio(rf, '일요일_매출_금액')
rf['주말_비중']   = (rf['토요일_매출_금액'] + rf['일요일_매출_금액']) / rf[T]

rf['시간대_00~06_비중'] = ratio(rf, '시간대_00~06_매출_금액')
rf['시간대_06~11_비중'] = ratio(rf, '시간대_06~11_매출_금액')
rf['시간대_11~14_비중'] = ratio(rf, '시간대_11~14_매출_금액')
rf['시간대_14~17_비중'] = ratio(rf, '시간대_14~17_매출_금액')
rf['시간대_17~21_비중'] = ratio(rf, '시간대_17~21_매출_금액')
rf['시간대_21~24_비중'] = ratio(rf, '시간대_21~24_매출_금액')
rf['야간_비중']         = (rf['시간대_17~21_매출_금액'] + rf['시간대_21~24_매출_금액']) / rf[T]

rf['여성_비중']     = ratio(rf, '여성_매출_금액')
rf['연령대20_비중'] = ratio(rf, '연령대_20_매출_금액')
rf['연령대30_비중'] = ratio(rf, '연령대_30_매출_금액')
rf['연령대40_비중'] = ratio(rf, '연령대_40_매출_금액')
rf['2030소비_비중'] = (rf['연령대_20_매출_금액'] + rf['연령대_30_매출_금액']) / rf[T]
rf['금요효과']      = rf['금요일_매출_금액'] / rf['목요일_매출_금액'].replace(0, np.nan)

rf['평일_비중'] = (
    rf['월요일_매출_금액'] + rf['화요일_매출_금액'] +
    rf['수요일_매출_금액'] + rf['목요일_매출_금액']
) / rf[T]

야간비중 = (rf['시간대_17~21_매출_금액'] + rf['시간대_21~24_매출_금액']) / (
    rf[['시간대_00~06_매출_금액', '시간대_06~11_매출_금액', '시간대_11~14_매출_금액',
        '시간대_14~17_매출_금액', '시간대_17~21_매출_금액', '시간대_21~24_매출_금액']].sum(axis=1)
).replace(0, np.nan)

rf['트렌드시간대_비중'] = (
    rf['금요일_매출_금액'] * 야간비중 +
    rf['토요일_매출_금액'] + rf['일요일_매출_금액']
) / rf[T]

rf['방문형_강도'] = rf['트렌드시간대_비중'] / rf['평일_비중'].replace(0, np.nan)

# 분기별 복합점수 계산 (연도+분기 내 표준화)
업종_cols = ['FB카페_매출금액', 'FB식사_매출금액', '주류유흥_매출금액', '라이프스타일_매출금액']
rf['업종점수_raw'] = rf[업종_cols].sum(axis=1) / rf[T]

period_scores = []
for (yr, q), grp in rf.groupby(['연도', '분기']):
    idx = grp.index
    업종z = zscore(grp['업종점수_raw'].fillna(grp['업종점수_raw'].median()), nan_policy='omit')
    금요z = zscore(grp['금요효과'].fillna(grp['금요효과'].median()), nan_policy='omit')
    주말z = zscore(grp['주말_비중'].fillna(grp['주말_비중'].median()), nan_policy='omit')
    야간z = zscore(grp['야간_비중'].fillna(grp['야간_비중'].median()), nan_policy='omit')
    시간z = 금요z + 주말z + 야간z
    복합  = zscore(업종z, nan_policy='omit') + zscore(시간z, nan_policy='omit')
    분위  = pd.Series(복합, index=idx).rank(pct=True)
    period_scores.append(pd.DataFrame({'복합점수_분위': 분위}, index=idx))

rf = rf.join(pd.concat(period_scores))

def assign_tier(pct):
    if pct >= 0.90: return '① 상위권'
    elif pct >= 0.70: return '② 중상위'
    elif pct >= 0.30: return '③ 중위권'
    else: return '④ 하위권'

rf['분기_티어'] = rf['복합점수_분위'].map(assign_tier)

print(f'raw_features: {len(rf)}행 ({rf["행정동_코드"].nunique()}개 동, {rf["연도"].nunique()}년 × 4분기)')

# ─────────────────────────────────────────────────────────────
# 2. YoY 피처 - 동일 분기 전년 대비 변화
# ─────────────────────────────────────────────────────────────
yoy_ratio_cols = [
    'FB카페_비중', 'FB식사_비중', '주류유흥_비중', '라이프스타일_비중',
    '주말_비중', '야간_비중', '금요일_비중',
    '여성_비중', '연령대20_비중', '연령대30_비중', '2030소비_비중', '금요효과',
    '평일_비중', '트렌드시간대_비중', '방문형_강도',
]

# 동일 분기 기준 정렬 후 diff → 같은 분기 전년 대비
rf = rf.sort_values(['행정동_코드', '분기', '연도'])
for col in yoy_ratio_cols:
    rf[f'{col}_변화'] = rf.groupby(['행정동_코드', '분기'])[col].diff()

# 타겟: 동일 분기 1년 후 복합점수_분위
rf['분위_next']    = rf.groupby(['행정동_코드', '분기'])['복합점수_분위'].shift(-1)
rf['분위변화_next'] = (rf['분위_next'] - rf['복합점수_분위']) * 100

# ─────────────────────────────────────────────────────────────
# 2-b. 행정동 패널 피처 (행정동 내 시계열 패턴)
# ─────────────────────────────────────────────────────────────
# 핵심 지표에 대해 lag1, 동편차, 모멘텀 계산
# rf는 이미 [행정동_코드, 분기, 연도] 순 정렬 완료

# lag1, 동편차를 계산할 수준(level) 피처
PANEL_LEVEL_COLS = [
    '트렌드시간대_비중', '방문형_강도', '2030소비_비중',
    '주말_비중', '금요효과', '야간_비중', '평일_비중',
    'FB카페_비중', '주류유흥_비중',
]

# 모멘텀을 계산할 YoY 변화 피처
PANEL_YOY_COLS = [
    '트렌드시간대_비중_변화', '방문형_강도_변화', '2030소비_비중_변화',
    '주말_비중_변화', '금요효과_변화',
]

# lag1: 1년 전 같은 분기 실제값
for col in PANEL_LEVEL_COLS:
    rf[f'{col}_lag1'] = rf.groupby(['행정동_코드', '분기'])[col].transform(
        lambda x: x.shift(1)
    )

# 동편차: 현재값 - 해당 행정동의 역대 expanding mean (leakage 방지를 위해 shift(1))
# 해석: 자기 자신의 과거 평균 대비 지금이 얼마나 높고/낮은가
for col in PANEL_LEVEL_COLS:
    rf[f'{col}_동편차'] = rf.groupby(['행정동_코드', '분기'])[col].transform(
        lambda x: x - x.expanding().mean().shift(1)
    )

# 모멘텀: YoY 변화 방향이 몇 분기 연속 유지되는지 (부호 포함)
# 예: +2 = 2년 연속 상승, -3 = 3년 연속 하락
def signed_momentum(x):
    """연속 같은 방향 카운트 (부호: 상승=양수, 하락=음수)"""
    vals = x.values
    result = np.zeros(len(vals))
    count = 0
    prev_sign = 0
    for i, val in enumerate(vals):
        if np.isnan(val) or val == 0:
            result[i] = 0
        else:
            s = 1 if val > 0 else -1
            if s == prev_sign:
                count += s
            else:
                count = s
                prev_sign = s
            result[i] = count
    return result

for col in PANEL_YOY_COLS:
    base = col.replace('_변화', '')
    rf[f'{base}_모멘텀'] = rf.groupby(['행정동_코드', '분기'])[col].transform(
        signed_momentum
    )

panel_level_new = [f'{c}_lag1' for c in PANEL_LEVEL_COLS] + \
                  [f'{c}_동편차' for c in PANEL_LEVEL_COLS]
panel_mom_new   = [f'{c.replace("_변화","")}_모멘텀' for c in PANEL_YOY_COLS]

print(f'패널 피처 추가: lag1×{len(PANEL_LEVEL_COLS)} + 동편차×{len(PANEL_LEVEL_COLS)} + 모멘텀×{len(PANEL_YOY_COLS)} = {len(panel_level_new)+len(panel_mom_new)}개')

# ─────────────────────────────────────────────────────────────
# 3. migration → 분기별 20대/30대/HE/WE/EE
# ─────────────────────────────────────────────────────────────
mig     = pd.read_csv('data/main_data/분기별_주말유입인구.csv', encoding='utf-8-sig')
mapping = pd.read_csv('data/main_data/코드매핑_유입인구_매출.csv', encoding='utf-8-sig')
mig_to_raw = dict(zip(mapping['migration_코드'], mapping['raw_코드']))

total_mig = (mig.groupby(['도착_행정동코드', '연도', '분기'])
             [['HE_이동인구', 'WE_이동인구', 'EE_이동인구', '합계_이동인구']]
             .sum().reset_index()
             .rename(columns={'HE_이동인구': 'HE_전체', 'WE_이동인구': 'WE_전체',
                               'EE_이동인구': 'EE_전체', '합계_이동인구': '총_이동인구'}))

mig_age = mig.merge(total_mig, on=['도착_행정동코드', '연도', '분기'])
mig_age['이동비율'] = mig_age['합계_이동인구'] / mig_age['총_이동인구'] * 100

age20 = (mig_age[mig_age['연령대'] == 20]
         [['도착_행정동코드', '연도', '분기', '이동비율']]
         .rename(columns={'이동비율': '20대_이동비율'}))
age30 = (mig_age[mig_age['연령대'] == 30]
         [['도착_행정동코드', '연도', '분기', '이동비율']]
         .rename(columns={'이동비율': '30대_이동비율'}))

total_mig['HE_비율'] = total_mig['HE_전체'] / total_mig['총_이동인구'] * 100
total_mig['WE_비율'] = total_mig['WE_전체'] / total_mig['총_이동인구'] * 100
total_mig['EE_비율'] = total_mig['EE_전체'] / total_mig['총_이동인구'] * 100

mig_feat = (age20
            .merge(age30, on=['도착_행정동코드', '연도', '분기'], how='outer')
            .merge(total_mig[['도착_행정동코드', '연도', '분기',
                               'HE_비율', 'WE_비율', 'EE_비율', '총_이동인구']],
                   on=['도착_행정동코드', '연도', '분기'], how='outer'))

mig_feat['2030_이동비율'] = mig_feat['20대_이동비율'].fillna(0) + mig_feat['30대_이동비율'].fillna(0)
mig_feat.loc[mig_feat['20대_이동비율'].isna() & mig_feat['30대_이동비율'].isna(), '2030_이동비율'] = np.nan

mig_feat['행정동_코드'] = mig_feat['도착_행정동코드'].map(mig_to_raw)
mig_feat = mig_feat.dropna(subset=['행정동_코드']).copy()
mig_feat['행정동_코드'] = mig_feat['행정동_코드'].astype(int)
mig_feat = (mig_feat.groupby(['행정동_코드', '연도', '분기'])
            [['20대_이동비율', '30대_이동비율', '2030_이동비율',
              'HE_비율', 'WE_비율', 'EE_비율', '총_이동인구']]
            .mean().reset_index())

# YoY 변화 (동일 분기 전년 대비)
mig_feat = mig_feat.sort_values(['행정동_코드', '분기', '연도'])
for col in ['20대_이동비율', '30대_이동비율', '2030_이동비율', 'HE_비율', 'WE_비율', 'EE_비율']:
    mig_feat[f'{col}_변화'] = mig_feat.groupby(['행정동_코드', '분기'])[col].diff()

# ─────────────────────────────────────────────────────────────
# 3-b. migration 패널 피처
# ─────────────────────────────────────────────────────────────
MIG_PANEL_COLS = ['2030_이동비율', '20대_이동비율', '30대_이동비율']

for col in MIG_PANEL_COLS:
    mig_feat[f'{col}_lag1'] = mig_feat.groupby(['행정동_코드', '분기'])[col].transform(
        lambda x: x.shift(1)
    )
    mig_feat[f'{col}_동편차'] = mig_feat.groupby(['행정동_코드', '분기'])[col].transform(
        lambda x: x - x.expanding().mean().shift(1)
    )
    mig_feat[f'{col}_모멘텀'] = mig_feat.groupby(['행정동_코드', '분기'])[f'{col}_변화'].transform(
        signed_momentum
    )

mig_panel_new = (
    [f'{c}_lag1' for c in MIG_PANEL_COLS] +
    [f'{c}_동편차' for c in MIG_PANEL_COLS] +
    [f'{c}_모멘텀' for c in MIG_PANEL_COLS]
)

print(f'migration 패널 피처 추가: {len(mig_panel_new)}개')
print(f'migration: {len(mig_feat)}행 ({mig_feat["행정동_코드"].nunique()}개 동)')

# ─────────────────────────────────────────────────────────────
# 4. 병합
# ─────────────────────────────────────────────────────────────
ratio_cols = [
    'FB카페_비중', 'FB식사_비중', '주류유흥_비중', '라이프스타일_비중', '기타업종_비중',
    '월요일_비중', '화요일_비중', '수요일_비중', '목요일_비중',
    '금요일_비중', '토요일_비중', '일요일_비중', '주말_비중',
    '시간대_00~06_비중', '시간대_06~11_비중', '시간대_11~14_비중',
    '시간대_14~17_비중', '시간대_17~21_비중', '시간대_21~24_비중', '야간_비중',
    '여성_비중', '연령대20_비중', '연령대30_비중', '연령대40_비중', '2030소비_비중',
    '금요효과', '총_매출금액',
    '평일_비중', '트렌드시간대_비중', '방문형_강도',
] + [f'{c}_변화' for c in yoy_ratio_cols]

rf_cols = ['행정동_코드', '행정동', '연도', '분기', '상권티어',
           '복합점수_분위', '분기_티어', '분위_next', '분위변화_next'] + \
          ratio_cols + panel_level_new + panel_mom_new

df = (rf[rf_cols]
      .merge(mig_feat, on=['행정동_코드', '연도', '분기'], how='left'))

# ─────────────────────────────────────────────────────────────
# 5. 타겟 분류
# ─────────────────────────────────────────────────────────────
def classify(v):
    if pd.isna(v):       return np.nan
    elif v >  THRESHOLD: return 2
    elif v < -THRESHOLD: return 0
    else:                return 1

df['타겟']       = df['분위변화_next'].apply(classify)
df['타겟_레이블'] = df['타겟'].map({2: '상승', 1: '유지', 0: '하락'})

# ─────────────────────────────────────────────────────────────
# 6. 저장
# ─────────────────────────────────────────────────────────────
mig_base_cols = [
    '20대_이동비율', '30대_이동비율', '2030_이동비율',
    'HE_비율', 'WE_비율', 'EE_비율', '총_이동인구',
    '20대_이동비율_변화', '30대_이동비율_변화', '2030_이동비율_변화',
    'HE_비율_변화', 'WE_비율_변화', 'EE_비율_변화',
]

feature_cols = (
    ratio_cols +
    panel_level_new +
    panel_mom_new +
    mig_base_cols +
    mig_panel_new
)

id_cols     = ['행정동_코드', '행정동', '연도', '분기', '상권티어', '분기_티어']
target_cols = ['분위변화_next', '타겟', '타겟_레이블']
all_cols    = id_cols + feature_cols + target_cols

# feature_cols 중 df에 없는 컬럼 체크
missing = [c for c in all_cols if c not in df.columns]
if missing:
    print(f'[경고] 누락된 컬럼: {missing}')

train = df[all_cols].dropna(subset=['타겟'])
pred  = df[df['연도'] == 2024][all_cols].copy()

train.to_csv(f'{OUT_DIR}/ml_dataset_quarterly_train.csv', encoding='utf-8-sig', index=False)
pred.to_csv(f'{OUT_DIR}/ml_dataset_quarterly_predict.csv', encoding='utf-8-sig', index=False)

# ─────────────────────────────────────────────────────────────
# 7. 요약
# ─────────────────────────────────────────────────────────────
print(f'\n=== 분기별 학습 데이터셋 ===')
print(f'행: {len(train)}  |  총 피처: {len(feature_cols)}개')
print(f'  - raw 수준 피처:    {len(ratio_cols)}개')
print(f'  - 행정동 lag1:      {len([c for c in panel_level_new if "_lag1" in c])}개')
print(f'  - 행정동 동편차:    {len([c for c in panel_level_new if "_동편차" in c])}개')
print(f'  - 행정동 모멘텀:    {len(panel_mom_new)}개')
print(f'  - migration 기본:   {len(mig_base_cols)}개')
print(f'  - migration 패널:   {len(mig_panel_new)}개')
print(f'연도-분기 범위: {train["연도"].min()}Q{train["분기"].min()} ~ {train["연도"].max()}Q{train["분기"].max()}')
print()
print('타겟 분포:')
print(train['타겟_레이블'].value_counts().to_string())
print()
print('결측치 비율 (>5% 피처만):')
na = (train[feature_cols].isna().sum() / len(train) * 100).round(1)
na_over5 = na[na > 5]
print(na_over5.to_string() if len(na_over5) > 0 else '  없음')
print()
print(f'=== 예측셋 (2024년 → 2025 상승 후보) ===')
print(f'행: {len(pred)}')
print()
print(f'저장 완료 -> {OUT_DIR}/')
