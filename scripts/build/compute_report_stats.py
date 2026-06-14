# -*- coding: utf-8 -*-
"""
보고서 수치 계산 — Korean literals 없이 index 기반 접근
결과: report_stats_output.txt
"""
import pandas as pd
import numpy as np

# ── 로드 ────────────────────────────────────────────────────────
ms   = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig')
r321 = pd.read_csv('report/data/reference_3_2_1.csv',       encoding='utf-8-sig')
lag  = pd.read_csv('report/data/causal_lag_analysis.csv',   encoding='utf-8-sig')
tier = pd.read_csv('report/data/final_candidates_tier.csv', encoding='utf-8-sig')
p35  = pd.read_csv('report/data/phase35_candidates_top15.csv', encoding='utf-8-sig')
p41  = pd.read_csv('report/data/phase41_quadrant_full.csv', encoding='utf-8-sig')
tri  = pd.read_csv('report/data/reference_trend_industry_sales.csv', encoding='utf-8-sig')
trt  = pd.read_csv('report/data/reference_trend_time_sales.csv',     encoding='utf-8-sig')

# ── 컬럼 인덱스 ─────────────────────────────────────────────────
# trend_sales_master: [행정동, 행정동코드, 연도, 트렌드대표매출, 총매출,
#                      트렌드매출비중, 업종점수, 2030비중, 금요효과, 주말비중,
#                      야간비중, 복합점수, 복합점수_분위, 상권티어, ...]
C_DONG  = ms.columns[0]
C_YEAR  = ms.columns[2]
C_GENRE = ms.columns[6]   # 업종점수
C_GEN20 = ms.columns[7]   # 2030비중
C_FRI   = ms.columns[8]   # 금요효과
C_WKND  = ms.columns[9]   # 주말비중
C_NIGHT = ms.columns[10]  # 야간비중
C_RANK  = ms.columns[12]  # 복합점수_분위

# tier: [행정동, Tier, Tier설명, P35유사도, P35업종점수, P35_20대비중,
#        P41이동성장세, P41사분면, 주말집중도, 구역유형]
TC_DONG = tier.columns[0]
TC_TIER = tier.columns[1]

# r321: [번호, 행정동, 자치구, 특성명칭, 전환연도분기, 전환상태연도, S1, S2, S3, S4]
R_DONG = r321.columns[1]
R_S3   = r321.columns[8]   # 외부신호: 네이버/블로그
R_S4   = r321.columns[9]   # 임대료신호

# lag: [분석명칭, 분석단위, lag수, lag설명, r_수준값, p_수준값, n_수준값, r_변화량, p_변화량, n_변화량, sig_수준값, sig_변화량]
L_NAME = lag.columns[0]
L_UNIT = lag.columns[1]
L_LAG  = lag.columns[2]
L_DESC = lag.columns[3]
L_R1   = lag.columns[4]   # r 수준값
L_P1   = lag.columns[5]   # p 수준값
L_N1   = lag.columns[6]   # n 수준값
L_R2   = lag.columns[7]   # r 변화량
L_P2   = lag.columns[8]
L_S1   = lag.columns[10]  # sig 수준값
L_S2   = lag.columns[11]

# p35: [행정동, 유사도, 업종점수, 2030비중, ...]
P35_DONG = p35.columns[0]
P35_SIM  = p35.columns[1]

# p41
P41_DONG = p41.columns[0]
P41_QUAD = p41.columns[p41.columns.str.contains('사분면').argmax()]
P41_SCORE = p41.columns[p41.columns.str.contains('이동성장세점수').argmax()]

# tri (reference_trend_industry_sales)
TRI_DONG   = tri.columns[0]
TRI_YEAR   = tri.columns[1]
TRI_TOTAL  = tri.columns[2]   # 트렌드업종_전체매출
TRI_GEN20  = tri.columns[3]   # 트렌드업종_20대매출
TRI_SHARE  = tri.columns[4]   # 트렌드업종_20대비중

# trt (reference_trend_time_sales)
TRT_DONG    = trt.columns[0]
TRT_YEAR    = trt.columns[1]
TRT_TOTAL   = trt.columns[2]  # 총매출
TRT_TREND   = trt.columns[3]  # 트렌드시간_매출
TRT_SHARE   = trt.columns[4]  # 트렌드시간_매출비중

# ── 필터 ────────────────────────────────────────────────────────
ms24 = ms[ms[C_YEAR] == 2024].copy()
ms20 = ms[ms[C_YEAR] == 2020].copy()

r321['_grade'] = r321.apply(
    lambda x: 'conf' if (str(x[R_S3]).strip()=='O' or str(x[R_S4]).strip()=='O') else 'cand',
    axis=1
)
ref_all  = r321[R_DONG].tolist()
ref_conf = r321[r321['_grade']=='conf'][R_DONG].tolist()
ref_cand = r321[r321['_grade']=='cand'][R_DONG].tolist()

ref24  = ms24[ms24[C_DONG].isin(ref_all)]
conf24 = ms24[ms24[C_DONG].isin(ref_conf)]
cand24 = ms24[ms24[C_DONG].isin(ref_cand)]
ref20  = ms20[ms20[C_DONG].isin(ref_all)]

t1 = tier[tier[TC_TIER]==1][TC_DONG].tolist()
t2 = tier[tier[TC_TIER]==2][TC_DONG].tolist()
t3 = tier[tier[TC_TIER]==3][TC_DONG].tolist()
t1_24 = ms24[ms24[C_DONG].isin(t1)]
t2_24 = ms24[ms24[C_DONG].isin(t2)]
t3_24 = ms24[ms24[C_DONG].isin(t3)]

p41_q2 = p41[p41[P41_QUAD].str.contains('2사분면')]
p35_24 = ms24[ms24[C_DONG].isin(p35[P35_DONG])]
p41_24 = ms24[ms24[C_DONG].isin(p41_q2[P41_DONG])]

tri24 = tri[tri[TRI_YEAR]==2024]
trt24 = trt[trt[TRT_YEAR]==2024]

# ── 기준값 ──────────────────────────────────────────────────────
s_gen20 = ms24[C_GEN20].mean()
s_genre = ms24[C_GENRE].mean()
s_wknd  = ms24[C_WKND].mean()
s_fri   = ms24[C_FRI].mean()
s_night = ms24[C_NIGHT].mean()

# ── 출력 구성 ────────────────────────────────────────────────────
lines = []
def p(s=''):
    lines.append(s)

p('='*65)
p('A. 서울 전체 423개 동 기준값 (2024)')
p('='*65)
p(f'  20대 소비비중   평균 {s_gen20*100:.2f}%  중앙값 {ms24[C_GEN20].median()*100:.2f}%')
p(f'  업종점수        평균 {s_genre*100:.2f}%  중앙값 {ms24[C_GENRE].median()*100:.2f}%')
p(f'  주말비중        평균 {s_wknd*100:.2f}%   중앙값 {ms24[C_WKND].median()*100:.2f}%')
p(f'  금요효과        평균 {s_fri:.4f}    중앙값 {ms24[C_FRI].median():.4f}')
p(f'  야간비중        평균 {s_night*100:.2f}%')
p()

p('='*65)
p('B. 레퍼런스 16개 동 vs 서울 전체 (2024)')
p('='*65)
for label, df in [('전체 16개', ref24), ('확정 8개', conf24), ('후보 8개', cand24)]:
    p(f'  [{label}]')
    p(f'    20대 소비비중: {df[C_GEN20].mean()*100:.2f}%  (서울 {s_gen20*100:.2f}%  차이 {(df[C_GEN20].mean()-s_gen20)*100:+.2f}%p)')
    p(f'    업종점수:      {df[C_GENRE].mean()*100:.2f}%  (서울 {s_genre*100:.2f}%  차이 {(df[C_GENRE].mean()-s_genre)*100:+.2f}%p)')
    p(f'    주말비중:      {df[C_WKND].mean()*100:.2f}%  (서울 {s_wknd*100:.2f}%  차이 {(df[C_WKND].mean()-s_wknd)*100:+.2f}%p)')
    p(f'    금요효과:      {df[C_FRI].mean():.4f}   (서울 {s_fri:.4f}  차이 {(df[C_FRI].mean()-s_fri):+.4f})')
    p(f'    복합점수_분위: {df[C_RANK].mean():.3f}   (상위 {(1-df[C_RANK].mean())*100:.0f}%)')
    p()

p('='*65)
p('C. 레퍼런스 동 2020 → 2024 변화 (확정 8개)')
p('='*65)
conf20 = ms20[ms20[C_DONG].isin(ref_conf)]
for col, label in [(C_GEN20, '20대 소비비중'), (C_GENRE, '업종점수'), (C_WKND, '주말비중')]:
    v20 = conf20[col].mean() * 100
    v24 = conf24[col].mean() * 100
    p(f'  {label}: {v20:.2f}% → {v24:.2f}%  (Δ{v24-v20:+.2f}%p / 상대변화 {(v24-v20)/v20*100:+.1f}%)')
p()

p('='*65)
p('D. 인과관계 lag 분석 결과 (pooled Pearson r)')
p('='*65)
p(f'  {"분석":<42} {"r(수준값)":>10} {"r(변화량)":>10} {"N":>7}')
p(f'  {"-"*65}')
for _, row in lag.iterrows():
    desc = str(row[L_DESC])[:42]
    p(f'  {desc:<42} {row[L_R1]:>+9.4f} {row[L_R2]:>+9.4f} {int(row[L_N1]):>7}  {row[L_S1]}')
p()

p('='*65)
p('E. Phase 35 후보 TOP15 특성 (2024 기준)')
p('='*65)
best35_idx = p35[P35_SIM].idxmax()
best35_dong = p35.loc[best35_idx, P35_DONG]
best35_sim  = p35.loc[best35_idx, P35_SIM]
p(f'  유사도 평균: {p35[P35_SIM].mean():.4f}  최고: {best35_sim:.4f} ({best35_dong})')
p(f'  업종점수 평균: {p35_24[C_GENRE].mean()*100:.2f}%  (서울 대비 {(p35_24[C_GENRE].mean()-s_genre)*100:+.2f}%p)')
p(f'  20대비중 평균: {p35_24[C_GEN20].mean()*100:.2f}%  (서울 대비 {(p35_24[C_GEN20].mean()-s_gen20)*100:+.2f}%p)')
p(f'  주말비중 평균: {p35_24[C_WKND].mean()*100:.2f}%   복합점수_분위: {p35_24[C_RANK].mean():.3f}')
p()

p('='*65)
p('F. Phase 41 2사분면(유망) 후보 특성')
p('='*65)
best41_idx  = p41_q2[P41_SCORE].idxmax()
best41_dong = p41_q2.loc[best41_idx, P41_DONG]
best41_sc   = p41_q2.loc[best41_idx, P41_SCORE]
p(f'  이동성장세점수 평균: {p41_q2[P41_SCORE].mean():.4f}  최고: {best41_sc:.4f} ({best41_dong})')
p(f'  20대비중 평균: {p41_24[C_GEN20].mean()*100:.2f}%  (서울 대비 {(p41_24[C_GEN20].mean()-s_gen20)*100:+.2f}%p)')
p(f'  주말비중 평균: {p41_24[C_WKND].mean()*100:.2f}%   복합점수_분위: {p41_24[C_RANK].mean():.3f}')
p()

p('='*65)
p('G. Tier별 후보 특성 비교 (2024)')
p('='*65)
for label, df in [('Tier1 교집합 2개', t1_24), ('Tier2 P35only 13개', t2_24), ('Tier3 P41only 13개', t3_24)]:
    if len(df) == 0:
        continue
    p(f'  [{label}]')
    p(f'    20대비중: {df[C_GEN20].mean()*100:.2f}%  (서울 대비 {(df[C_GEN20].mean()-s_gen20)*100:+.2f}%p)')
    p(f'    업종점수: {df[C_GENRE].mean()*100:.2f}%  (서울 대비 {(df[C_GENRE].mean()-s_genre)*100:+.2f}%p)')
    p(f'    주말비중: {df[C_WKND].mean()*100:.2f}%   복합점수_분위: {df[C_RANK].mean():.3f}')
    p()

p('='*65)
p('H. 레퍼런스 트렌드 업종 내 20대 매출 비중 (2024)')
p('='*65)
p(f'  전체 16개 평균: {tri24[TRI_SHARE].mean()*100:.2f}%')
p(f'  확정 8개 평균:  {tri24[tri24[TRI_DONG].isin(ref_conf)][TRI_SHARE].mean()*100:.2f}%')
p(f'  후보 8개 평균:  {tri24[tri24[TRI_DONG].isin(ref_cand)][TRI_SHARE].mean()*100:.2f}%')
idx_max = tri24[TRI_SHARE].idxmax()
idx_min = tri24[TRI_SHARE].idxmin()
p(f'  최고: {tri24.loc[idx_max, TRI_DONG]}  {tri24.loc[idx_max, TRI_SHARE]*100:.1f}%')
p(f'  최저: {tri24.loc[idx_min, TRI_DONG]}  {tri24.loc[idx_min, TRI_SHARE]*100:.1f}%')
p()

p('='*65)
p('I. 레퍼런스 트렌드 시간대 매출비중 (2024)')
p('='*65)
p(f'  전체 16개 평균: {trt24[TRT_SHARE].mean()*100:.2f}%')
p(f'  확정 8개 평균:  {trt24[trt24[TRT_DONG].isin(ref_conf)][TRT_SHARE].mean()*100:.2f}%')
p(f'  후보 8개 평균:  {trt24[trt24[TRT_DONG].isin(ref_cand)][TRT_SHARE].mean()*100:.2f}%')
idx_max_t = trt24[TRT_SHARE].idxmax()
idx_min_t = trt24[TRT_SHARE].idxmin()
p(f'  최고: {trt24.loc[idx_max_t, TRT_DONG]}  {trt24.loc[idx_max_t, TRT_SHARE]*100:.1f}%')
p(f'  최저: {trt24.loc[idx_min_t, TRT_DONG]}  {trt24.loc[idx_min_t, TRT_SHARE]*100:.1f}%')
p()

p('='*65)
p('J. 개별 레퍼런스 동 핵심 수치 (2024)')
p('='*65)
p(f'  {"동명":<12} {"등급":<4} {"20대비중":>9} {"업종점수":>9} {"주말비중":>9} {"금요효과":>9} {"복합분위":>9}')
p(f'  {"-"*60}')
for _, row in r321.iterrows():
    dong = row[R_DONG]
    g    = '확정' if row['_grade']=='conf' else '후보'
    ms_r = ms24[ms24[C_DONG]==dong]
    if len(ms_r) == 0:
        continue
    r = ms_r.iloc[0]
    p(f'  {dong:<12} {g:<4} {r[C_GEN20]*100:>8.1f}% {r[C_GENRE]*100:>8.1f}% {r[C_WKND]*100:>8.1f}% {r[C_FRI]:>9.3f} {r[C_RANK]:>9.3f}')

with open('report_stats_output.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('saved: report_stats_output.txt')
