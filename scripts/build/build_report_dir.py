# -*- coding: utf-8 -*-
"""
최종 보고서 자료 정리 스크립트 (v6 구조 기반)
  - report/images/ : 보고서에 쓸 시각화 파일 (섹션별 번호 재부여)
  - report/data/   : 보고서에 쓸 데이터 파일 + final_candidates_tier.csv 생성
  - report/README.md : 파일 인덱스
"""

import os, shutil
import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

REPORT = 'report'
IMG_DST = f'{REPORT}/images'
DAT_DST = f'{REPORT}/data'
os.makedirs(IMG_DST, exist_ok=True)
os.makedirs(DAT_DST, exist_ok=True)

# ────────────────────────────────────────────────────────────────
# 1. 이미지 복사 (섹션별 번호 재부여)
# ────────────────────────────────────────────────────────────────
IMAGE_MAP = [
    # (원본 경로, 저장명, 섹션, 설명)
    # 섹션 2: 지표 정의·배경
    ('image/gen20/01_consumption_trend.png',        'S2_01_consumption_trend.png',         '§2 배경',     '20대 매출 트렌드 개요'),
    ('image/gen20/02_gen20_vs_30s.png',             'S2_02_gen20_vs_30s.png',              '§2 배경',     '20대 vs 30대 소비 비교'),
    ('image/gen20/03_migration.png',                'S2_03_migration_trend.png',           '§2 배경',     '20대 이동비율 추이'),
    ('image/gen20/04_weekend_profile.png',          'S2_04_weekend_profile.png',           '§2 배경',     '주말집중도 프로파일'),
    ('report/images/S2_05_area_type_industry.png',  'S2_05_area_type_industry.png',        '§2 배경',     '구역유형별 트렌드 업종 구성'),
    # 섹션 3: 레퍼런스
    ('image/gen20/06_reference_trajectory.png',     'S3_01_reference_trajectory.png',      '§3 레퍼런스',  '레퍼런스 4개 동 궤적'),
    ('image/causal_chain_analysis.png',             'S3_02_causal_chain.png',              '§3 레퍼런스',  '이동→소비→임대료 인과관계'),
    ('image/dashboard_A_signals.png',               'S3_03_signals_dashboard.png',         '§3 레퍼런스',  '4개 독립 신호 대시보드'),
    ('report/images/S3_04_causal_lag.png',          'S3_04_causal_lag.png',                '§3 레퍼런스',  '이동↔소비↔임대료 lag 상관계수'),
    ('report/images/S3_05_ref_sales_timeseries.png','S3_05_ref_sales_timeseries.png',      '§3 레퍼런스',  '레퍼런스 16개 동 매출 시계열'),
    ('report/images/S3_06_ref_migration_timeseries.png','S3_06_ref_migration_timeseries.png','§3 레퍼런스','레퍼런스 16개 동 이동비율 시계열'),
    ('report/images/S3_07_ref_trend_industry.png',  'S3_07_ref_trend_industry.png',        '§3 레퍼런스',  '레퍼런스 동 트렌드 업종 전체 vs 20대 매출'),
    ('report/images/S3_08_ref_trend_timesales.png', 'S3_08_ref_trend_timesales.png',       '§3 레퍼런스',  '레퍼런스 동 트렌드 시간대 매출 비중'),
    # 섹션 4.1: Phase 35
    ('image/gen20/05_candidates.png',               'S4a_01_phase35_candidates.png',       '§4.1 Phase35','Phase 35 후보 TOP15'),
    # 섹션 4.2: Phase 41
    ('image/gen20/07_quadrant_analysis.png',        'S4b_01_quadrant.png',                 '§4.2 Phase41','4분면 산점도'),
    ('image/gen20/08_quadrant_timeseries.png',      'S4b_02_timeseries.png',               '§4.2 Phase41','유망 후보 시계열'),
    ('image/gen20/09_map_top15.png',                'S4b_03_map_top15.png',                '§4.2 Phase41','서울 지도 TOP15'),
    # 섹션 5: 비교·통합
    ('image/gen20/10_phase_compare.png',            'S5_01_phase35_vs_41.png',             '§5 비교',     'Phase 35 vs Phase 41 비교'),
    ('image/gen20/11_compare_map.png',              'S5_02_compare_map.png',               '§5 비교',     '교집합 지도 비교'),
    ('image/gen20/12_method_compare.png',           'S5_03_method_compare.png',            '§5 비교',     '우리 방식 vs 팀원 방식 비교'),
    ('report/images/S5_04_ref_vs_candidates.png',   'S5_04_ref_vs_candidates.png',         '§5 비교',     '레퍼런스 vs 후보 비교'),
]

print('[1/3] 이미지 복사...')
img_rows = []
for src, dst_name, section, desc in IMAGE_MAP:
    dst = f'{IMG_DST}/{dst_name}'
    if os.path.exists(src):
        # 이미 report/images/ 에 직접 저장된 파일은 복사 생략
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
        status = '✓'
    else:
        status = '✗ 파일없음'
    img_rows.append((section, dst_name, desc, status))
    print(f'  {status}  {dst_name}')

# ────────────────────────────────────────────────────────────────
# 2. 데이터 파일 + final_candidates_tier.csv 생성
# ────────────────────────────────────────────────────────────────
print('\n[2/3] 데이터 파일 정리...')

# Phase 35 TOP15 (서초4동 제외)
from report_config import load_v3, phase35_top15, tier1_dongs

_v3 = load_v3()
df35_top = _v3[_v3['행정동'].isin(phase35_top15(_v3))].sort_values('코사인유사도', ascending=False).head(15).copy()
df35_top = df35_top.rename(columns={'코사인유사도': '유사도'})
df35_top.to_csv(f'{DAT_DST}/phase35_candidates_top15.csv', encoding='utf-8-sig', index=False)
p35_set = set(df35_top['행정동'])
df35 = df35_top  # 하위 호환
print(f'  ✓ phase35_candidates_top15.csv ({len(df35_top)}개 동)')

# Phase 41 전체
df41 = pd.read_csv('data/main_data/quadrant_20_result.csv', encoding='utf-8-sig')
df41.to_csv(f'{DAT_DST}/phase41_quadrant_full.csv', encoding='utf-8-sig', index=False)
p41_q2 = df41[df41['사분면'] == '2사분면_유망'].nlargest(15, '이동성장세점수')
p41_set = set(p41_q2['행정동'])
print(f'  ✓ phase41_quadrant_full.csv ({len(df41)}개 동, 2사분면 {len(p41_q2)}개)')

# 주말집중도
shutil.copy2('data/main_data/area_type_profile.csv', f'{DAT_DST}/area_type_profile.csv')
at = pd.read_csv('data/main_data/area_type_profile.csv', encoding='utf-8-sig')
at_map = dict(zip(at['행정동'], at['구역유형']))
conc_map = dict(zip(at['행정동'], at['주말집중도']))
print(f'  ✓ area_type_profile.csv ({len(at)}개 동)')

# 임대료
shutil.copy2('data/main_data/임대료_공실률.csv', f'{DAT_DST}/임대료_공실률.csv')
print(f'  ✓ 임대료_공실률.csv')

# ── Tier 분류 통합 CSV ──────────────────────────────────────────
both = p35_set & p41_set
tier1 = sorted(both)
tier2 = sorted(p35_set - p41_set)
tier3 = sorted(p41_set - p35_set)

rows = []
for dong in tier1 + tier2 + tier3:
    tier = 1 if dong in both else (2 if dong in p35_set else 3)
    r35 = df35[df35['행정동'] == dong]
    sim = r35['유사도'].values[0] if len(r35) and '유사도' in r35.columns else (r35['코사인유사도'].values[0] if len(r35) else np.nan)
    genre = r35['업종점수'].values[0] if len(r35) else np.nan
    gen20_c = r35['2030비중'].values[0] if len(r35) else np.nan
    r41 = df41[df41['행정동'] == dong]
    mob = r41['이동성장세점수'].values[0] if len(r41) else np.nan
    quad = r41['사분면'].values[0] if len(r41) else '필터제외'
    rows.append({
        '행정동': dong,
        'Tier': tier,
        'Tier_설명': ['Tier1_최고확신', 'Tier2_중기안정', 'Tier3_초기급등'][tier - 1],
        'Phase35_유사도': round(sim, 4) if not np.isnan(sim) else None,
        'Phase35_업종점수': round(genre, 4) if not np.isnan(genre) else None,
        'Phase35_20대비중': round(gen20_c, 4) if not np.isnan(gen20_c) else None,
        'Phase41_이동성장세': round(mob, 4) if not np.isnan(mob) else None,
        'Phase41_사분면': quad,
        '주말집중도': round(conc_map.get(dong, np.nan), 4) if dong in conc_map else None,
        '구역유형': at_map.get(dong, '미측정'),
    })

tier_df = pd.DataFrame(rows)
tier_df.to_csv(f'{DAT_DST}/final_candidates_tier.csv', encoding='utf-8-sig', index=False)
print(f'  ✓ final_candidates_tier.csv (Tier1={len(tier1)}개, Tier2={len(tier2)}개, Tier3={len(tier3)}개)')
print(f'    Tier1: {tier1}')

# ── 타 스크립트가 직접 생성하는 파일 존재 확인 ─────────────────
DATA_EXTRA = [
    ('causal_lag_analysis.csv',            '이동↔소비↔임대료 lag 상관계수 (build_causal_lag.py)'),
    ('reference_3_2_1.csv',               '레퍼런스 16개 동 기본정보 §3.2.1 (build_reference_tables.py)'),
    ('reference_3_2_2.csv',               '레퍼런스 16개 동 매출·이동 수치 §3.2.2 (build_reference_tables.py)'),
    ('reference_signal_detail.csv',        '레퍼런스 동 외부 신호 상세 (build_reference_tables.py)'),
    ('reference_sales_qoq.csv',            '레퍼런스 동 분기별 매출 시계열 (build_report_supplement.py)'),
    ('reference_migration_timeseries.csv', '레퍼런스 동 분기별 이동비율 시계열 (build_report_supplement.py)'),
    ('reference_trend_industry_sales.csv', '레퍼런스 동 트렌드 업종 전체·20대 매출 (build_ref_trend_sales.py)'),
    ('reference_trend_time_sales.csv',     '레퍼런스 동 트렌드 시간대 매출·비중 (build_ref_trend_sales.py)'),
]
for fname, desc in DATA_EXTRA:
    path = f'{DAT_DST}/{fname}'
    status = '✓' if os.path.exists(path) else '✗ 파일없음'
    print(f'  {status} {fname}')

# ────────────────────────────────────────────────────────────────
# 3. README.md 작성
# ────────────────────────────────────────────────────────────────
print('\n[3/3] README.md 작성...')

tier_table = '\n'.join(
    f'| {r["행정동"]} | Tier {r["Tier"]} | {r["Tier_설명"]} | '
    f'{r["Phase35_유사도"] or "-"} | '
    f'{r["Phase41_이동성장세"] or "-"} | '
    f'{r["주말집중도"] or "-"} | {r["구역유형"]} |'
    for _, r in tier_df.iterrows()
)

img_table = '\n'.join(
    f'| {sec} | `{name}` | {desc} | {st} |'
    for sec, name, desc, st in img_rows
)

data_extra_rows = '\n'.join(
    f'| `{fname}` | {desc.split(" (")[0]} | — |'
    for fname, desc in DATA_EXTRA
)

readme = f"""# 최종 보고서 자료 디렉토리

> 생성 기준: 최종_보고서_내용_정리_v6.md
> 분석 대상: 서울시 423개 행정동 (2020~2024 분기)

---

## 디렉토리 구조

```
report/
├── images/   시각화 파일 (섹션별 번호)
├── data/     데이터 파일
└── README.md 이 파일
```

---

## 핵심 후보 통합 (Tier 분류)

| 행정동 | Tier | 분류 | Phase35 유사도 | Phase41 이동성장세 | 주말집중도 | 구역유형 |
|--------|------|------|:--------------:|:-----------------:|:---------:|---------|
{tier_table}

> **Tier 1** (Phase35 ∩ Phase41): 두 방법론 교집합 — 최고 확신 후보
> **Tier 2** (Phase35 only): 코사인 유사도 기반 중기 안정형
> **Tier 3** (Phase41 only): 이동 모멘텀 기반 초기 급등형

---

## 이미지 파일 목록

| 섹션 | 파일명 | 설명 | 상태 |
|------|--------|------|------|
{img_table}

---

## 데이터 파일 목록

| 파일명 | 내용 | 행 수 |
|--------|------|-------|
| `final_candidates_tier.csv` | **통합 Tier 1/2/3 후보 목록** (Phase35+41+주말집중도 결합) | {len(tier_df)}개 동 |
| `phase35_candidates_top15.csv` | Phase 35 코사인 유사도 TOP15 (서초4동 제외) | 15개 동 |
| `phase41_quadrant_full.csv` | Phase 41 4분면 전체 결과 (avg22≥3% + OLS기울기>0 필터 적용) | 53개 동 |
| `area_type_profile.csv` | 주말집중도·구역유형 (오피스/혼합/주거/상권형) | 측정 동 |
| `임대료_공실률.csv` | 임대료·공실률 (68개 상권, 2020~2024) | — |
{data_extra_rows}

---

## 미완성 항목 (v6 📌 작업 목록)

- [x] `causal_lag_analysis.csv` — 이동·소비·임대료 lag 상관계수 (§1.2 가설 검증)
- [x] 레퍼런스 16개 동 정보 표 (§3.2.1~3.2.3) 완성
- [ ] 주말집중도 미측정 동 보강 (송파1동·서초2동·다산동 등 Phase35 후보 중 미측정)
- [ ] 외부 자료 검증 (`reference_external_validation.md`, §6.3)
"""

with open(f'{REPORT}/README.md', 'w', encoding='utf-8') as f:
    f.write(readme)
print('  ✓ README.md')

print('\n=== 완료 ===')
print(f'  images/: {len([r for r in img_rows if "✓" in r[3]])}개 확인, {len([r for r in img_rows if "✗" in r[3]])}개 누락')
print(f'  data/:   {5 + len(DATA_EXTRA)}개 파일')
print(f'  Tier1({len(tier1)}): {tier1}')
print(f'  Tier2({len(tier2)}): {tier2}')
print(f'  Tier3({len(tier3)}): {tier3}')
