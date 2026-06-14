"""
차세대 상권 재선정 — 신규 레퍼런스 4개 기준
1. Phase35 재산정: 4개 레퍼런스 궤적 벡터에 대한 코사인 유사도
2. Phase41 재산정: 20대 이동 선행·소비 미전환 2사분면 (주말집중도 ≥ 0.80)
3. Tier 분류 및 결합
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ── 신규 레퍼런스 4개 ──────────────────────────────────────────
REF_NEW = ['방배2동', '청운효자동', '합정동', '성수2가1동']

# ── 대학가 제외 목록 (레퍼런스 선정 기준과 동일) ─────────────
# "이미 자리잡은 학생 소비 상권 — 여가형 트렌드 상권과 성격 다름"
UNIV_EXCLUDE = ['신촌동', '대학동', '휘경1동', '상계2동', '회기동']

# ── 데이터 로드 ────────────────────────────────────────────────
sales  = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig')
area   = pd.read_csv('data/main_data/area_type_profile.csv', encoding='utf-8-sig')
quad   = pd.read_csv('data/main_data/quadrant_20_result.csv', encoding='utf-8-sig')

# ── 분석 유효 동: 주말집중도 ≥ 0.80 (오피스형 제외) ──────────
valid_area = area[area['주말집중도'] >= 0.80][['행정동', '주말집중도', '구역유형']].copy()
valid_dongs = set(valid_area['행정동'])

# ── 2024년 기준 지표 (서술용) ─────────────────────────────────
s24 = sales[sales['연도'] == 2024][
    ['행정동', '업종점수', '2030비중', '트렌드매출_비중', '복합점수_분위']
].copy()

# ── 이미 기준 충족(성숙 상권) 제외 목록 ─────────────────────
# 2020년에 이미 S1+S2+트렌드 전부 충족 → 전환 불필요한 기성 상권
s20 = sales[sales['연도'] == 2020][['행정동', '업종점수', '2030비중', '트렌드매출_비중']]
already_mature = set(
    s20[(s20['업종점수'] >= 0.50) &
        (s20['2030비중'] >= 0.35) &
        (s20['트렌드매출_비중'] >= 0.30)]['행정동']
)
print(f"2020년 이미 기준 충족 (성숙 상권): {len(already_mature)}개")

# ─────────────────────────────────────────────────────────────
# Phase35: 코사인 유사도 분석 (신규 4개 레퍼런스 궤적 기준)
# ─────────────────────────────────────────────────────────────
FEATS = ['업종점수', '트렌드매출_비중', '복합점수_분위']
YEARS = [2020, 2021, 2022, 2023, 2024]

# 연도×지표 피벗 → 행정동별 15차원 벡터
pivot_parts = []
for feat in FEATS:
    p = sales.pivot_table(index='행정동', columns='연도', values=feat)
    p.columns = [f'{feat}_{y}' for y in p.columns]
    pivot_parts.append(p)

feat_mat = pd.concat(pivot_parts, axis=1).dropna()

# Z-score 정규화 (지표 간 스케일 차이 제거)
scaler = StandardScaler()
feat_scaled = pd.DataFrame(
    scaler.fit_transform(feat_mat),
    index=feat_mat.index,
    columns=feat_mat.columns
)

# 레퍼런스 4개 평균 벡터 (템플릿)
ref_vecs = feat_scaled.loc[feat_scaled.index.isin(REF_NEW)]
template = ref_vecs.mean(axis=0).values.reshape(1, -1)

# 코사인 유사도 계산
all_vecs = feat_scaled.values
sims = cosine_similarity(all_vecs, template).flatten()
sim_series = pd.Series(sims, index=feat_scaled.index, name='코사인유사도')

# 후보군 필터링
exclude = set(REF_NEW) | already_mature | set(UNIV_EXCLUDE)
sim_candidates = sim_series[
    sim_series.index.isin(valid_dongs) &
    ~sim_series.index.isin(exclude)
].sort_values(ascending=False)

# 2024년 지표 결합
sim_df = sim_candidates.reset_index().rename(columns={'index': '행정동'})
sim_df = sim_df.merge(s24, on='행정동', how='left')
sim_df = sim_df.merge(valid_area, on='행정동', how='left')

# 상위 15개 선택
TOP_N = 15
phase35_top = sim_df.head(TOP_N).copy()
phase35_set = set(phase35_top['행정동'])

print(f"\nPhase35 코사인 유사도 TOP{TOP_N}:")
print(phase35_top[['행정동', '코사인유사도', '업종점수', '트렌드매출_비중', '2030비중',
                    '복합점수_분위', '구역유형']].to_string(index=False))

# ─────────────────────────────────────────────────────────────
# Phase41: 이동 선행·소비 미전환 2사분면 재산정
# ─────────────────────────────────────────────────────────────
# quadrant_20_result: 매출활성도점수 (소비) vs 이동성장세점수 (이동)
# 2사분면 = 매출활성도 낮음(-) × 이동성장세 높음(+)

phase41_raw = quad[quad['사분면'] == '2사분면_유망'].copy()
phase41_raw = phase41_raw.merge(valid_area, on='행정동', how='inner')  # gate: 주말집중도 ≥ 0.80
phase41_raw = phase41_raw[~phase41_raw['행정동'].isin(exclude)]        # 레퍼런스·성숙·대학가 제외
phase41_raw = phase41_raw.merge(s24, on='행정동', how='left')
phase41_raw = phase41_raw.sort_values('이동성장세점수', ascending=False).reset_index(drop=True)

phase41_set = set(phase41_raw['행정동'])

print(f"\nPhase41 이동 선행 2사분면 ({len(phase41_raw)}개):")
print(phase41_raw[['행정동', '이동성장세점수', '매출활성도점수', '2030비중',
                    '업종점수', '트렌드매출_비중', '구역유형']].to_string(index=False))

# ─────────────────────────────────────────────────────────────
# Tier 분류: 교집합 → Tier1, P35전용 → Tier2, P41전용 → Tier3
# ─────────────────────────────────────────────────────────────
tier1 = phase35_set & phase41_set
tier2 = phase35_set - phase41_set
tier3 = phase41_set - phase35_set

def assign_tier(dong):
    if dong in tier1:   return 'Tier1_교집합'
    elif dong in tier2: return 'Tier2_업종전환'
    else:               return 'Tier3_이동선행'

# 전체 후보 통합
all_cands = (
    pd.concat([
        phase35_top[['행정동', '코사인유사도', '업종점수', '트렌드매출_비중',
                     '2030비중', '복합점수_분위', '주말집중도', '구역유형']],
        phase41_raw[['행정동', '업종점수', '트렌드매출_비중', '2030비중',
                     '복합점수_분위', '주말집중도', '구역유형']]
    ], ignore_index=True)
    .drop_duplicates('행정동')
    .assign(tier=lambda d: d['행정동'].map(assign_tier))
    .sort_values(['tier', '복합점수_분위'], ascending=[True, False])
    .reset_index(drop=True)
)

# ─────────────────────────────────────────────────────────────
# 그룹별 평균 지표 요약
# ─────────────────────────────────────────────────────────────
ref_stats = s24[s24['행정동'].isin(REF_NEW)][
    ['업종점수', '트렌드매출_비중', '2030비중', '복합점수_분위']].mean()
seoul_stats = s24[['업종점수', '트렌드매출_비중', '2030비중', '복합점수_분위']].mean()

grp_summary = all_cands.groupby('tier')[
    ['업종점수', '트렌드매출_비중', '2030비중', '복합점수_분위']
].mean()

# ─────────────────────────────────────────────────────────────
# 저장: CSV
# ─────────────────────────────────────────────────────────────
all_cands.to_csv('data/main_data/trend_candidates_v3.csv', index=False, encoding='utf-8-sig')
print("\n→ data/main_data/trend_candidates_v3.csv 저장 완료")

# ─────────────────────────────────────────────────────────────
# 저장: 텍스트 보고서
# ─────────────────────────────────────────────────────────────
with open('_tmp_candidate_reselect.txt', 'w', encoding='utf-8') as f:
    SEP = '='*70

    f.write(SEP + '\n')
    f.write('# 차세대 상권 재선정 결과 — 신규 레퍼런스 4개 기준\n')
    f.write('# 레퍼런스: 방배2동, 청운효자동, 합정동, 성수2가1동\n')
    f.write(SEP + '\n\n')

    # Phase35
    f.write('─'*70 + '\n')
    f.write(f'## Phase35: 코사인 유사도 기반 TOP{TOP_N} (레퍼런스 궤적 유사도)\n')
    f.write('## 피처: 업종점수·트렌드매출_비중·복합점수_분위 × 5년 (2020~2024)\n')
    f.write('─'*70 + '\n')
    disp35 = phase35_top[['행정동', '코사인유사도', '업종점수', '트렌드매출_비중',
                           '2030비중', '복합점수_분위', '주말집중도', '구역유형']].copy()
    for c in ['코사인유사도', '업종점수', '트렌드매출_비중', '2030비중', '복합점수_분위', '주말집중도']:
        disp35[c] = disp35[c].map('{:.3f}'.format)
    f.write(disp35.to_string(index=False) + '\n\n')

    # Phase41
    f.write('─'*70 + '\n')
    f.write(f'## Phase41: 이동 선행·소비 미전환 2사분면 ({len(phase41_raw)}개)\n')
    f.write('## 기준: 매출활성도 낮음(-) × 이동성장세 높음(+) | 주말집중도 ≥ 0.80\n')
    f.write('─'*70 + '\n')
    disp41 = phase41_raw[['행정동', '이동성장세점수', '매출활성도점수', '업종점수',
                           '트렌드매출_비중', '2030비중', '복합점수_분위', '구역유형']].copy()
    for c in ['이동성장세점수', '매출활성도점수', '업종점수', '트렌드매출_비중',
              '2030비중', '복합점수_분위']:
        disp41[c] = disp41[c].map('{:.3f}'.format)
    f.write(disp41.to_string(index=False) + '\n\n')

    # Tier 분류
    f.write('─'*70 + '\n')
    f.write('## Tier 분류\n')
    f.write('─'*70 + '\n')
    f.write(f'Tier1 교집합  ({len(tier1)}개): {sorted(tier1)}\n')
    f.write(f'Tier2 업종전환({len(tier2)}개): {sorted(tier2)}\n')
    f.write(f'Tier3 이동선행({len(tier3)}개): {sorted(tier3)}\n\n')

    # 그룹 평균 비교
    f.write('─'*70 + '\n')
    f.write('## 그룹별 평균 지표 비교 (2024)\n')
    f.write('─'*70 + '\n')
    rows = [
        ('서울 전체(423)', seoul_stats['업종점수'], seoul_stats['트렌드매출_비중'],
         seoul_stats['2030비중'], seoul_stats['복합점수_분위']),
    ]
    for tier in ['Tier3_이동선행', 'Tier2_업종전환', 'Tier1_교집합']:
        if tier in grp_summary.index:
            r = grp_summary.loc[tier]
            rows.append((tier, r['업종점수'], r['트렌드매출_비중'],
                         r['2030비중'], r['복합점수_분위']))
    rows.append(('레퍼런스 신규(4)', ref_stats['업종점수'], ref_stats['트렌드매출_비중'],
                 ref_stats['2030비중'], ref_stats['복합점수_분위']))

    header = f"{'그룹':<18}  {'업종점수':>8}  {'트렌드매출비중':>12}  {'2030비중':>8}  {'복합점수분위':>10}\n"
    f.write(header)
    f.write('-'*60 + '\n')
    for row in rows:
        f.write(f'{row[0]:<18}  {row[1]:>8.3f}  {row[2]:>12.3f}  {row[3]:>8.3f}  {row[4]:>10.3f}\n')

print('\n→ _tmp_candidate_reselect.txt 저장 완료')
print(f'\nTier1 교집합  ({len(tier1)}개): {sorted(tier1)}')
print(f'Tier2 업종전환({len(tier2)}개): {sorted(tier2)}')
print(f'Tier3 이동선행({len(tier3)}개): {sorted(tier3)}')
