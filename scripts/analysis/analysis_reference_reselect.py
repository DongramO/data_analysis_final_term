"""
레퍼런스 동 재선정 분석
1. 현재 16개 동 트렌드매출_비중 분포 + 문제 진단
2. 새 기준으로 전체 423개 동 재선정
   - S1_new: 트렌드매출_비중 전환 (2020 미충족 → 이후 충족)
   - S2: 2030비중 ≥ 0.35 전환 (동일)
   - Gate: 주말집중도 ≥ 0.90 (오피스형 제거)
"""
import pandas as pd
import numpy as np

# ── 데이터 로드 ───────────────────────────────────────────
sales   = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig')
area    = pd.read_csv('data/main_data/area_type_profile.csv', encoding='utf-8-sig')
ref_old = pd.read_csv('report/data/reference_3_2_1.csv', encoding='utf-8-sig')
ref_3_2_2 = pd.read_csv('report/data/reference_3_2_2.csv', encoding='utf-8-sig')

old_list = ref_old['행정동'].tolist()  # 기존 16개

# ── 1. 현재 16개 동 트렌드매출_비중 분포 ──────────────────
sales24 = sales[sales['연도'] == 2024].copy()

old_stats = sales24[sales24['행정동'].isin(old_list)][
    ['행정동', '트렌드매출_비중', '업종점수', '2030비중', '복합점수_분위']
].merge(ref_3_2_2[['행정동', '레퍼런스등급', '주말집중도', '구역유형']], on='행정동', how='left') \
 .merge(ref_old[['행정동', 'S3_블로그신호', 'S4_임대료신호']], on='행정동', how='left') \
 .sort_values('트렌드매출_비중', ascending=False)

# 서울 전체 분포
seoul_q = sales24['트렌드매출_비중'].quantile([0.25, 0.50, 0.75, 0.90])

# ── 2. 전체 연도별 트렌드매출_비중 pivot ──────────────────
pivot = sales.pivot_table(index='행정동', columns='연도', values='트렌드매출_비중')
pivot_ind = sales.pivot_table(index='행정동', columns='연도', values='업종점수')
pivot_2030 = sales.pivot_table(index='행정동', columns='연도', values='2030비중')

# ── 3. 새 임계값 탐색 ─────────────────────────────────────
# 현재 16개 동 트렌드매출_비중 분포로 임계값 참고
old_trend = old_stats['트렌드매출_비중']

# 기존 S1 임계값(0.50)에 해당하는 트렌드매출_비중 수준 확인
# → 현재 16개 동 중 트렌드매출_비중 분포의 하위값을 새 임계값 후보로 사용
threshold_candidates = [0.28, 0.30, 0.32, 0.33]

# ── 4. 새 기준 재선정 ─────────────────────────────────────
# Gate: 주말집중도 ≥ 0.90 (혼합형+주거형+상권형만)
area_gate = area[area['주말집중도'] >= 0.90][['행정동', '주말집중도', '구역유형']].copy()
valid_dongs = area_gate['행정동'].tolist()

# ── Gate 0.80 (오피스형만 제외, 혼합형·주거형·상권형 포함) ──
area_gate = area[area['주말집중도'] >= 0.80][['행정동', '주말집중도', '구역유형']].copy()
valid_dongs = area_gate['행정동'].tolist()

results = {}
for thr in threshold_candidates:
    s1 = (pivot >= thr).astype(int)
    s2 = (pivot_2030 >= 0.35).astype(int)

    both_2020 = (s1[2020] == 1) & (s2[2020] == 1)
    converted = []
    for dong in pivot.index:
        if dong not in valid_dongs:
            continue
        if both_2020.get(dong, False):
            continue
        for yr in [2021, 2022, 2023, 2024]:
            s1v = s1.loc[dong, yr] if yr in s1.columns and dong in s1.index else 0
            s2v = s2.loc[dong, yr] if yr in s2.columns and dong in s2.index else 0
            if s1v == 1 and s2v == 1:
                first_yr = yr
                break
        else:
            continue
        converted.append({'행정동': dong, '전환연도': first_yr,
                          '트렌드매출_2020': pivot.loc[dong, 2020],
                          '트렌드매출_2024': pivot.loc[dong, 2024],
                          '2030비중_2020': pivot_2030.loc[dong, 2020],
                          '2030비중_2024': pivot_2030.loc[dong, 2024],
                          '업종점수_2020': pivot_ind.loc[dong, 2020],
                          '업종점수_2024': pivot_ind.loc[dong, 2024]})
    results[thr] = pd.DataFrame(converted)

# ── 5. 임계값별 결과 수 확인 후 적정값 선택 ──────────────
# (너무 적으면 0.28, 너무 많으면 0.33으로 조정)
THR = 0.30
new_ref = results[THR].copy()
new_ref = new_ref.merge(area_gate, on='행정동', how='left')
new_ref = new_ref.merge(sales24[['행정동', '복합점수_분위']], on='행정동', how='left')
new_ref = new_ref.sort_values('전환연도').reset_index(drop=True)

# 기존 대비 교집합·신규·이탈
old_set = set(old_list)
new_set = set(new_ref['행정동'].tolist())

# ── 출력 ──────────────────────────────────────────────────
with open('_tmp_ref_reselect.txt', 'w', encoding='utf-8') as f:

    f.write('='*70 + '\n')
    f.write('# 현재 16개 레퍼런스 동 진단 (2024년 기준)\n')
    f.write('='*70 + '\n\n')

    f.write('서울 전체 트렌드매출_비중 분포:\n')
    f.write(f'  Q1={seoul_q[0.25]:.3f}  Q2={seoul_q[0.50]:.3f}  '
            f'Q3={seoul_q[0.75]:.3f}  P90={seoul_q[0.90]:.3f}\n\n')

    f.write('현재 16개 동 트렌드매출_비중 (내림차순):\n')
    display_old = old_stats[['행정동', '레퍼런스등급', '구역유형', '주말집중도',
                              '트렌드매출_비중', '업종점수', '2030비중',
                              'S3_블로그신호', 'S4_임대료신호']].copy()
    display_old['트렌드매출_비중'] = display_old['트렌드매출_비중'].map('{:.3f}'.format)
    display_old['업종점수'] = display_old['업종점수'].map('{:.3f}'.format)
    display_old['2030비중'] = display_old['2030비중'].map('{:.3f}'.format)
    display_old['주말집중도'] = display_old['주말집중도'].map('{:.3f}'.format)
    f.write(display_old.to_string(index=False) + '\n\n')

    f.write('오피스형(주말집중도<0.80) 포함 현황 (제거 대상):\n')
    office_in_ref = ref_3_2_2[ref_3_2_2['행정동'].isin(old_list) & (ref_3_2_2['주말집중도'] < 0.80)]
    for _, r in office_in_ref.iterrows():
        f.write(f'  ❌ {r["행정동"]} ({r["구역유형"]}, 주말집중도={r["주말집중도"]:.3f})\n')
    f.write(f'\n총 {len(office_in_ref)}개 동이 오피스형(주말집중도<0.80) — 레퍼런스 제거 대상\n\n')

    f.write('='*70 + '\n')
    f.write('# 임계값별 신규 레퍼런스 수 (주말집중도≥0.80 gate, 혼합형 포함)\n')
    f.write('='*70 + '\n\n')
    f.write(f'  {"트렌드매출_비중 임계값":>20}  {"전환 동 수":>8}  {"기존 16개 교집합":>12}\n')
    f.write('-'*50 + '\n')
    for thr, df_r in results.items():
        overlap = len(set(df_r['행정동']) & old_set)
        f.write(f'  {thr:>20.2f}  {len(df_r):>8}  {overlap:>12}\n')

    f.write('\n')
    f.write('='*70 + '\n')
    f.write(f'# 신규 레퍼런스 선정 결과 (트렌드매출_비중≥{THR}, 주말집중도≥0.80)\n')
    f.write('='*70 + '\n\n')

    f.write(f'총 {len(new_ref)}개 동 선정\n\n')

    display_new = new_ref[['행정동', '전환연도', '구역유형', '주말집중도',
                            '트렌드매출_2020', '트렌드매출_2024',
                            '2030비중_2020', '2030비중_2024',
                            '업종점수_2020', '업종점수_2024']].copy()
    for c in ['주말집중도', '트렌드매출_2020', '트렌드매출_2024',
              '2030비중_2020', '2030비중_2024', '업종점수_2020', '업종점수_2024']:
        display_new[c] = display_new[c].map(lambda x: f'{x:.3f}' if pd.notnull(x) else '-')
    f.write(display_new.to_string(index=False) + '\n\n')

    f.write('── 기존 16개 대비 비교 ──\n')
    f.write(f'  교집합 ({len(old_set & new_set)}개): {sorted(old_set & new_set)}\n')
    f.write(f'  기존→이탈 ({len(old_set - new_set)}개): {sorted(old_set - new_set)}\n')
    f.write(f'  신규진입 ({len(new_set - old_set)}개): {sorted(new_set - old_set)}\n')

    f.write('\n\n── 전환연도별 분포 ──\n')
    f.write(new_ref['전환연도'].value_counts().sort_index().to_string() + '\n')

    f.write('\n\n── 구역유형별 분포 ──\n')
    f.write(new_ref['구역유형'].value_counts().to_string() + '\n')

print('완료 → _tmp_ref_reselect.txt')
