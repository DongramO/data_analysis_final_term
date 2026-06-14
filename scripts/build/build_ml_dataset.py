"""
ML 학습용 데이터셋 생성 (raw feature 기반)
피처 출처:
  - 분기별_원본피처.csv : 요일/시간대/연령대/업종별 원본 매출
  - 분기별_주말유입인구.csv    : 20대/30대/2030 이동비율, HE/WE/EE 유형
  - 트렌드매출_마스터.csv     : 복합점수_분위 (타겟 & 현재 위치), 상권티어
타겟: t → t+1 복합점수_분위 변화  →  상승(>+5%p) / 유지 / 하락(<-5%p)
"""
import pandas as pd
import numpy as np
import os

OUT_DIR   = 'data/main_data'
THRESHOLD = 5.0   # 분류 임계값 (%p)

# ─────────────────────────────────────────────────────────────
# 1. raw_features_quarterly → 연간 집계 + 비중 피처
# ─────────────────────────────────────────────────────────────
rf = pd.read_csv('data/main_data/분기별_원본피처.csv', encoding='utf-8-sig')

# 연간 합계 (금액·건수 원본)
amount_cols = [
    'FB카페_매출금액', 'FB식사_매출금액', '주류유흥_매출금액',
    '라이프스타일_매출금액', '기타_매출금액', '총_매출금액', '총_매출건수',
    '월요일_매출_금액', '화요일_매출_금액', '수요일_매출_금액',
    '목요일_매출_금액', '금요일_매출_금액', '토요일_매출_금액', '일요일_매출_금액',
    '시간대_00~06_매출_금액', '시간대_06~11_매출_금액', '시간대_11~14_매출_금액',
    '시간대_14~17_매출_금액', '시간대_17~21_매출_금액', '시간대_21~24_매출_금액',
    '남성_매출_금액', '여성_매출_금액',
    '연령대_10_매출_금액', '연령대_20_매출_금액', '연령대_30_매출_금액',
    '연령대_40_매출_금액', '연령대_50_매출_금액', '연령대_60_이상_매출_금액',
    '주중_매출_금액', '주말_매출_금액',
]
rf_annual = (rf.groupby(['행정동_코드', '연도'])[amount_cols]
               .sum().reset_index())

# 비중 계산 (모두 총_매출금액 대비)
T = '총_매출금액'
def ratio(col): return rf_annual[col] / rf_annual[T]

rf_annual['FB카페_비중']       = ratio('FB카페_매출금액')
rf_annual['FB식사_비중']       = ratio('FB식사_매출금액')
rf_annual['주류유흥_비중']     = ratio('주류유흥_매출금액')
rf_annual['라이프스타일_비중'] = ratio('라이프스타일_매출금액')
rf_annual['기타업종_비중']     = ratio('기타_매출금액')

rf_annual['월요일_비중'] = ratio('월요일_매출_금액')
rf_annual['화요일_비중'] = ratio('화요일_매출_금액')
rf_annual['수요일_비중'] = ratio('수요일_매출_금액')
rf_annual['목요일_비중'] = ratio('목요일_매출_금액')
rf_annual['금요일_비중'] = ratio('금요일_매출_금액')
rf_annual['토요일_비중'] = ratio('토요일_매출_금액')
rf_annual['일요일_비중'] = ratio('일요일_매출_금액')
rf_annual['주말_비중']   = (rf_annual['토요일_매출_금액'] + rf_annual['일요일_매출_금액']) / rf_annual[T]

rf_annual['시간대_00~06_비중'] = ratio('시간대_00~06_매출_금액')
rf_annual['시간대_06~11_비중'] = ratio('시간대_06~11_매출_금액')
rf_annual['시간대_11~14_비중'] = ratio('시간대_11~14_매출_금액')
rf_annual['시간대_14~17_비중'] = ratio('시간대_14~17_매출_금액')
rf_annual['시간대_17~21_비중'] = ratio('시간대_17~21_매출_금액')
rf_annual['시간대_21~24_비중'] = ratio('시간대_21~24_매출_금액')
rf_annual['야간_비중']         = (rf_annual['시간대_17~21_매출_금액'] +
                                   rf_annual['시간대_21~24_매출_금액']) / rf_annual[T]

rf_annual['여성_비중']    = ratio('여성_매출_금액')
rf_annual['연령대20_비중'] = ratio('연령대_20_매출_금액')
rf_annual['연령대30_비중'] = ratio('연령대_30_매출_금액')
rf_annual['연령대40_비중'] = ratio('연령대_40_매출_금액')
rf_annual['2030소비_비중'] = (rf_annual['연령대_20_매출_금액'] +
                               rf_annual['연령대_30_매출_금액']) / rf_annual[T]

rf_annual['금요효과']     = (rf_annual['금요일_매출_금액'] /
                              rf_annual['목요일_매출_금액'].replace(0, np.nan))

# YoY 변화 (비중 기준)
rf_annual = rf_annual.sort_values(['행정동_코드', '연도'])
yoy_ratio_cols = [
    'FB카페_비중', 'FB식사_비중', '주류유흥_비중', '라이프스타일_비중',
    '주말_비중', '야간_비중', '금요일_비중',
    '여성_비중', '연령대20_비중', '연령대30_비중', '2030소비_비중', '금요효과',
]
for col in yoy_ratio_cols:
    rf_annual[f'{col}_변화'] = rf_annual.groupby('행정동_코드')[col].diff()

print(f'raw_features 연간: {len(rf_annual)}행 ({rf_annual["행정동_코드"].nunique()}개 동)')

# ─────────────────────────────────────────────────────────────
# 2. migration → 20대/30대/2030/HE/WE/EE 비율 + YoY
# ─────────────────────────────────────────────────────────────
mig     = pd.read_csv('data/main_data/분기별_주말유입인구.csv', encoding='utf-8-sig')
mapping = pd.read_csv('data/main_data/코드매핑_유입인구_매출.csv', encoding='utf-8-sig')
mig_to_raw = dict(zip(mapping['migration_코드'], mapping['raw_코드']))

# 연간 집계
mig_yr = mig.groupby(['도착_행정동코드', '연도', '연령대']).agg(
    HE=('HE_이동인구', 'sum'),
    WE=('WE_이동인구', 'sum'),
    EE=('EE_이동인구', 'sum'),
    합계=('합계_이동인구', 'sum')
).reset_index()

# 전체 합계 (연령 무관)
total_mig = mig_yr.groupby(['도착_행정동코드', '연도'])[['HE', 'WE', 'EE', '합계']].sum().reset_index()
total_mig.columns = ['도착_행정동코드', '연도', 'HE_전체', 'WE_전체', 'EE_전체', '총_이동인구']

# 연령대별 이동비율
age_mig = mig_yr.merge(total_mig, on=['도착_행정동코드', '연도'])
age_mig['이동비율'] = age_mig['합계'] / age_mig['총_이동인구'] * 100

# 20대, 30대, 기타 피벗
age20 = age_mig[age_mig['연령대'] == 20][['도착_행정동코드', '연도', '이동비율']].rename(columns={'이동비율': '20대_이동비율'})
age30 = age_mig[age_mig['연령대'] == 30][['도착_행정동코드', '연도', '이동비율']].rename(columns={'이동비율': '30대_이동비율'})

# HE/WE/EE 비율 (전체 이동 중 유형별 비중)
he_we_ee = total_mig.copy()
he_we_ee['HE_비율'] = he_we_ee['HE_전체'] / he_we_ee['총_이동인구'] * 100
he_we_ee['WE_비율'] = he_we_ee['WE_전체'] / he_we_ee['총_이동인구'] * 100
he_we_ee['EE_비율'] = he_we_ee['EE_전체'] / he_we_ee['총_이동인구'] * 100
he_we_ee['트렌드_유동인구'] = he_we_ee['총_이동인구']

mig_feat = (age20
            .merge(age30,    on=['도착_행정동코드', '연도'], how='outer')
            .merge(he_we_ee[['도착_행정동코드', '연도', 'HE_비율', 'WE_비율', 'EE_비율', '트렌드_유동인구']],
                   on=['도착_행정동코드', '연도'], how='outer'))

mig_feat['2030_이동비율'] = mig_feat['20대_이동비율'].fillna(0) + mig_feat['30대_이동비율'].fillna(0)
mig_feat.loc[mig_feat['20대_이동비율'].isna() & mig_feat['30대_이동비율'].isna(), '2030_이동비율'] = np.nan

# 코드 매핑 (7자리 → 8자리)
mig_feat['행정동_코드'] = mig_feat['도착_행정동코드'].map(mig_to_raw)
mig_feat = mig_feat.dropna(subset=['행정동_코드']).copy()
mig_feat['행정동_코드'] = mig_feat['행정동_코드'].astype(int)
mig_feat = (mig_feat.groupby(['행정동_코드', '연도'])
            [['20대_이동비율', '30대_이동비율', '2030_이동비율',
              'HE_비율', 'WE_비율', 'EE_비율', '트렌드_유동인구']]
            .mean().reset_index())

# YoY 변화
mig_feat = mig_feat.sort_values(['행정동_코드', '연도'])
for col in ['20대_이동비율', '30대_이동비율', '2030_이동비율', 'HE_비율', 'WE_비율', 'EE_비율']:
    mig_feat[f'{col}_변화'] = mig_feat.groupby('행정동_코드')[col].diff()

print(f'migration 피처: {len(mig_feat)}행 ({mig_feat["행정동_코드"].nunique()}개 동)')

# ─────────────────────────────────────────────────────────────
# 3. trend_sales_master → 타겟 + 상권티어
# ─────────────────────────────────────────────────────────────
master = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig',
                     usecols=['행정동', '행정동_코드', '연도', '복합점수_분위', '상권티어'])
master = master.sort_values(['행정동_코드', '연도'])

tier_map = {'① 레전드': 6, '② 라이징스타': 5, '③ 급성장 신흥': 4,
            '④ 잠재 후보': 3, '⑤ 성숙·정체': 2, '⑥ 쇠퇴 주의': 1, '⑦ 일반': 0}
master['상권티어_코드'] = master['상권티어'].map(tier_map).fillna(0).astype(int)

# 타겟: t+1 복합점수_분위
master['분위_next']     = master.groupby('행정동_코드')['복합점수_분위'].shift(-1)
master['분위변화_next'] = (master['분위_next'] - master['복합점수_분위']) * 100

# ─────────────────────────────────────────────────────────────
# 4. 전체 병합
# ─────────────────────────────────────────────────────────────
ratio_cols = [
    'FB카페_비중', 'FB식사_비중', '주류유흥_비중', '라이프스타일_비중', '기타업종_비중',
    '월요일_비중', '화요일_비중', '수요일_비중', '목요일_비중',
    '금요일_비중', '토요일_비중', '일요일_비중', '주말_비중',
    '시간대_00~06_비중', '시간대_06~11_비중', '시간대_11~14_비중',
    '시간대_14~17_비중', '시간대_17~21_비중', '시간대_21~24_비중', '야간_비중',
    '여성_비중', '연령대20_비중', '연령대30_비중', '연령대40_비중', '2030소비_비중',
    '금요효과', '총_매출금액',
] + [f'{c}_변화' for c in yoy_ratio_cols]

df = (master
      .merge(rf_annual[['행정동_코드', '연도'] + ratio_cols],
             on=['행정동_코드', '연도'], how='left')
      .merge(mig_feat,
             on=['행정동_코드', '연도'], how='left'))

# ─────────────────────────────────────────────────────────────
# 5. 타겟 클래스
# ─────────────────────────────────────────────────────────────
def classify(v):
    if pd.isna(v):  return np.nan
    elif v >  THRESHOLD: return 2   # 상승
    elif v < -THRESHOLD: return 0   # 하락
    else:                return 1   # 유지

df['타겟']       = df['분위변화_next'].apply(classify)
df['타겟_레이블'] = df['타겟'].map({2: '상승', 1: '유지', 0: '하락'})

# ─────────────────────────────────────────────────────────────
# 7. 저장
# ─────────────────────────────────────────────────────────────
feature_cols = (
    ratio_cols +
    ['20대_이동비율', '30대_이동비율', '2030_이동비율',
     'HE_비율', 'WE_비율', 'EE_비율', '트렌드_유동인구',
     '20대_이동비율_변화', '30대_이동비율_변화', '2030_이동비율_변화',
     'HE_비율_변화', 'WE_비율_변화', 'EE_비율_변화']
)

id_cols     = ['행정동_코드', '행정동', '연도', '상권티어']
target_cols = ['분위변화_next', '타겟', '타겟_레이블']
all_cols    = id_cols + feature_cols + target_cols

# 학습셋: 2020~2023 피처 (타겟 있는 행)
train = df[all_cols].dropna(subset=['타겟'])
# 예측셋: 2024년 (타겟 없음 → 2025 후보)
pred  = df[df['연도'] == 2024][all_cols].copy()

train.to_csv(f'{OUT_DIR}/ml_dataset_train.csv', encoding='utf-8-sig', index=False)
pred.to_csv(f'{OUT_DIR}/ml_dataset_predict.csv', encoding='utf-8-sig', index=False)

# ─────────────────────────────────────────────────────────────
# 8. 요약
# ─────────────────────────────────────────────────────────────
print(f'\n=== 학습 데이터셋 ===')
print(f'행: {len(train)}  |  피처: {len(feature_cols)}개  |  동 수: {train["행정동_코드"].nunique()}')
print()
print('타겟 분포:')
print(train['타겟_레이블'].value_counts().to_string())
print()
print('결측치 비율 (>5% 피처만):')
na = (train[feature_cols].isna().sum() / len(train) * 100).round(1)
print(na[na > 5].to_string())
print()
print(f'=== 예측셋 (2024년 피처 → 2025 상승 후보) ===')
print(f'행: {len(pred)}')
print()
print(f'저장 완료 → {OUT_DIR}/')
