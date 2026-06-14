"""
가설 근거 수치 자료 — 정량 표 일괄 생성
가설: "20대가 먼저 이동하는 곳이 다음 트렌드 상권이 된다"

출력 표:
 T1. 행정동 구역유형 분류 현황 및 핵심 지표 비교
 T2. 구역유형별 소비 업종 비중 비교 (오피스 vs 상권 차이)
 T3. 전체 매출 대비 20대 매출 비율 연도별 추이
 T4. 레퍼런스 4개 동 핵심 지표 (절대값 + 비율, 2020→2024)
 T5. 구역유형별 연도별 매출 성장률 (YoY %)
 T6. 건당 매출(proxy 방문객 지출) 비교
 T7. 레퍼런스 동 20대 이동인구 연도별 변화
 T8. 20대 이동비율 × 트렌드매출_비중 상관관계
"""
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

plt_ready = False  # 시각화 없이 수치 표만 출력

# ── 기준 ────────────────────────────────────────────────────────
REF  = ['방배2동', '청운효자동', '합정동', '성수2가1동']
TIER1 = ['숭인2동', '신당5동']
YEARS = [2020, 2021, 2022, 2023, 2024]

# ── 데이터 로드 ──────────────────────────────────────────────────
sales  = pd.read_csv('data/main_data/트렌드매출_마스터.csv',      encoding='utf-8-sig')
raw_q  = pd.read_csv('data/main_data/분기별_원본피처.csv',  encoding='utf-8-sig')
area   = pd.read_csv('data/main_data/area_type_profile.csv',       encoding='utf-8-sig')
mig_q  = pd.read_csv('data/main_data/분기별_주말유입인구.csv',     encoding='utf-8-sig')
code_map = pd.read_csv('data/main_data/코드매핑_유입인구_매출.csv', encoding='utf-8-sig')

# ── 구역유형 매핑 (sales에 붙이기) ───────────────────────────────
area_slim = area[['행정동', '주말집중도', '구역유형']].copy()
sales = sales.merge(area_slim, on='행정동', how='left')
sales['구역유형'] = sales['구역유형'].fillna('미분류')

# ── raw_features 연도별 집계 ─────────────────────────────────────
raw_ann = (
    raw_q.groupby(['행정동', '연도'])
    .agg(
        총매출=('총_매출금액', 'sum'),
        총건수=('총_매출건수', 'sum'),
        매출_20대=('연령대_20_매출_금액', 'sum'),
        매출_30대=('연령대_30_매출_금액', 'sum'),
        매출_40대=('연령대_40_매출_금액', 'sum'),
        매출_FB카페=('FB카페_매출금액', 'sum'),
        매출_FB식사=('FB식사_매출금액', 'sum'),
        매출_주류유흥=('주류유흥_매출금액', 'sum'),
        매출_라이프=('라이프스타일_매출금액', 'sum'),
        매출_기타=('기타_매출금액', 'sum'),
    )
    .reset_index()
)
raw_ann['20대비중'] = raw_ann['매출_20대'] / raw_ann['총매출']
raw_ann['2030비중'] = (raw_ann['매출_20대'] + raw_ann['매출_30대']) / raw_ann['총매출']
raw_ann['건당매출'] = raw_ann['총매출'] / raw_ann['총건수']
raw_ann['FB카페비중'] = raw_ann['매출_FB카페'] / raw_ann['총매출']
raw_ann['FB식사비중'] = raw_ann['매출_FB식사'] / raw_ann['총매출']
raw_ann['주류유흥비중'] = raw_ann['매출_주류유흥'] / raw_ann['총매출']
raw_ann['라이프비중'] = raw_ann['매출_라이프'] / raw_ann['총매출']
raw_ann['기타비중'] = raw_ann['매출_기타'] / raw_ann['총매출']
raw_ann['업종점수'] = raw_ann['FB카페비중'] + raw_ann['FB식사비중'] + raw_ann['주류유흥비중'] + raw_ann['라이프비중']

raw_ann = raw_ann.merge(area_slim, on='행정동', how='left')
raw_ann['구역유형'] = raw_ann['구역유형'].fillna('미분류')

# ── 이동인구 처리 ────────────────────────────────────────────────
mig_20 = mig_q[mig_q['연령대'] == 20].copy()
mig_20_ann = (
    mig_20.groupby(['도착_행정동코드', '연도'])['합계_이동인구']
    .sum().reset_index()
    .rename(columns={'합계_이동인구': '20대_이동인구'})
)
mig_all_ann = (
    mig_q.groupby(['도착_행정동코드', '연도'])['합계_이동인구']
    .sum().reset_index()
    .rename(columns={'합계_이동인구': '총_이동인구'})
)
mig_ann = mig_20_ann.merge(mig_all_ann, on=['도착_행정동코드', '연도'])
mig_ann['20대_이동비율'] = mig_ann['20대_이동인구'] / mig_ann['총_이동인구']

# 코드 매핑으로 행정동명 연결
code_map_slim = code_map[['migration_코드', '행정동명']].drop_duplicates()
mig_ann = mig_ann.merge(
    code_map_slim, left_on='도착_행정동코드', right_on='migration_코드', how='left'
).rename(columns={'행정동명': '행정동'})

# ── 결과 파일 ────────────────────────────────────────────────────
lines = []
SEP = '=' * 72
DIV = '-' * 72

def add(text=''):
    lines.append(text)

def pct(v, d=1):
    return f'{v*100:.{d}f}%' if pd.notnull(v) else '-'

def num(v, unit='억', d=1):
    if pd.isnull(v): return '-'
    if unit == '억': return f'{v/1e8:,.{d}f}'
    return f'{v:,.{d}f}'

add(SEP)
add('# 가설 근거 수치 자료 — 서울 423개 행정동 × 2020~2024')
add('# 가설: "20대가 먼저 이동하는 곳이 다음 트렌드 상권이 된다"')
add('# 레퍼런스: 방배2동·청운효자동·합정동·성수2가1동')
add(SEP)

# ══════════════════════════════════════════════════════════════════
# T1. 행정동 구역유형 분류 현황 및 핵심 지표 비교 (2024)
# ══════════════════════════════════════════════════════════════════
add()
add('## T1. 행정동 구역유형 분류 현황 및 핵심 지표 비교 (2024)')
add('   분류 기준: 주말집중도 (주말평균생활인구 / 평일평균생활인구)')
add(DIV)

s24 = sales[sales['연도'] == 2024].copy()
r24 = raw_ann[raw_ann['연도'] == 2024].copy()

TYPE_ORDER = ['오피스형', '혼합형', '주거형', '상권형']
grp = s24.groupby('구역유형')[
    ['업종점수', '2030비중', '트렌드매출_비중', '금요효과', '주말집중도']
].mean()
cnt = s24.groupby('구역유형').size().rename('동 수')
grp = pd.concat([cnt, grp], axis=1).reindex(TYPE_ORDER)

header = f"{'구역유형':<8}  {'동수':>4}  {'주말집중도':>8}  {'업종점수':>8}  {'트렌드매출비중':>12}  {'2030비중':>8}  {'금요효과':>8}"
add(header)
add(DIV)
cutoffs = {'오피스형': '< 0.80', '혼합형': '0.80~0.95', '주거형': '0.95~1.10', '상권형': '> 1.10'}
for t in TYPE_ORDER:
    if t not in grp.index: continue
    r = grp.loc[t]
    add(f"{t:<8}  {int(r['동 수']):>4}  {r['주말집중도']:>8.3f}  {r['업종점수']:>8.3f}  {r['트렌드매출_비중']:>12.3f}  {r['2030비중']:>8.3f}  {r['금요효과']:>8.3f}  ({cutoffs[t]})")
add()

# 레퍼런스 & 서울 전체 추가
seoul_avg = s24[['업종점수', '2030비중', '트렌드매출_비중', '금요효과', '주말집중도']].mean()
ref_avg   = s24[s24['행정동'].isin(REF)][['업종점수', '2030비중', '트렌드매출_비중', '금요효과', '주말집중도']].mean()
add(f"{'서울 전체':<8}  {len(s24):>4}  {seoul_avg['주말집중도']:>8.3f}  {seoul_avg['업종점수']:>8.3f}  {seoul_avg['트렌드매출_비중']:>12.3f}  {seoul_avg['2030비중']:>8.3f}  {seoul_avg['금요효과']:>8.3f}  (기준)")
add(f"{'레퍼런스':<8}  {len(REF):>4}  {ref_avg['주말집중도']:>8.3f}  {ref_avg['업종점수']:>8.3f}  {ref_avg['트렌드매출_비중']:>12.3f}  {ref_avg['2030비중']:>8.3f}  {ref_avg['금요효과']:>8.3f}  (4개 동)")

# ══════════════════════════════════════════════════════════════════
# T2. 구역유형별 소비 업종 비중 비교
# ══════════════════════════════════════════════════════════════════
add()
add()
add('## T2. 구역유형별 소비 업종 비중 비교 (2024, 전체 매출 대비 %)')
add('   핵심: 오피스형은 FB식사(점심) 비중이 높아 업종점수가 과장됨')
add(DIV)

r24_grp = r24.groupby('구역유형')[
    ['FB카페비중', 'FB식사비중', '주류유흥비중', '라이프비중', '기타비중', '업종점수']
].mean().reindex(TYPE_ORDER)

header2 = f"{'구역유형':<8}  {'FB카페':>7}  {'FB식사':>7}  {'주류유흥':>7}  {'라이프':>7}  {'기타':>7}  {'업종점수합':>9}"
add(header2)
add(DIV)
for t in TYPE_ORDER:
    if t not in r24_grp.index: continue
    r = r24_grp.loc[t]
    add(f"{t:<8}  {r['FB카페비중']*100:>7.1f}%  {r['FB식사비중']*100:>7.1f}%  {r['주류유흥비중']*100:>7.1f}%  {r['라이프비중']*100:>7.1f}%  {r['기타비중']*100:>7.1f}%  {r['업종점수']*100:>9.1f}%")
seoul_r = r24[['FB카페비중', 'FB식사비중', '주류유흥비중', '라이프비중', '기타비중', '업종점수']].mean()
ref_r   = r24[r24['행정동'].isin(REF)][['FB카페비중', 'FB식사비중', '주류유흥비중', '라이프비중', '기타비중', '업종점수']].mean()
add(DIV)
add(f"{'서울전체':<8}  {seoul_r['FB카페비중']*100:>7.1f}%  {seoul_r['FB식사비중']*100:>7.1f}%  {seoul_r['주류유흥비중']*100:>7.1f}%  {seoul_r['라이프비중']*100:>7.1f}%  {seoul_r['기타비중']*100:>7.1f}%  {seoul_r['업종점수']*100:>9.1f}%")
add(f"{'레퍼런스':<8}  {ref_r['FB카페비중']*100:>7.1f}%  {ref_r['FB식사비중']*100:>7.1f}%  {ref_r['주류유흥비중']*100:>7.1f}%  {ref_r['라이프비중']*100:>7.1f}%  {ref_r['기타비중']*100:>7.1f}%  {ref_r['업종점수']*100:>9.1f}%")

# ══════════════════════════════════════════════════════════════════
# T3. 전체 매출 대비 20대 매출 비율 연도별 추이
# ══════════════════════════════════════════════════════════════════
add()
add()
add('## T3. 전체 매출 대비 20대 매출 비율(%) 연도별 추이 — 그룹 비교')
add(DIV)

groups = {
    '서울전체(423)': raw_ann['행정동'].unique().tolist(),
    '오피스형': area[area['구역유형']=='오피스형']['행정동'].tolist(),
    '혼합형':   area[area['구역유형']=='혼합형']['행정동'].tolist(),
    '주거형':   area[area['구역유형']=='주거형']['행정동'].tolist(),
    '상권형':   area[area['구역유형']=='상권형']['행정동'].tolist(),
    '레퍼런스(4)': REF,
    'Tier1후보(2)': TIER1,
}

header3 = f"{'그룹':<14}" + ''.join(f"  {y}년" for y in YEARS) + "  Δ2020→24"
add(header3)
add(DIV)
for gname, dongs in groups.items():
    row = []
    vals = {}
    for y in YEARS:
        sub = raw_ann[(raw_ann['연도']==y) & (raw_ann['행정동'].isin(dongs))]
        v = sub['20대비중'].mean() if len(sub) > 0 else np.nan
        vals[y] = v
        row.append(f"{v*100:>6.1f}%" if pd.notnull(v) else '     -')
    delta = vals[2024] - vals[2020] if pd.notnull(vals[2024]) and pd.notnull(vals[2020]) else np.nan
    d_str = f"{delta*100:+.1f}%p" if pd.notnull(delta) else '     -'
    add(f"{gname:<14}" + ''.join(row) + f"  {d_str}")

# ══════════════════════════════════════════════════════════════════
# T4. 레퍼런스 4개 동 핵심 지표 (절대값+비율, 2020→2024)
# ══════════════════════════════════════════════════════════════════
add()
add()
add('## T4. 레퍼런스 4개 동 핵심 지표 (절대값 포함, 2020→2024)')
add('   단위: 매출 금액 = 억원')
add(DIV)

for dong in REF:
    add(f'\n[ {dong} ]')
    header4 = f"{'연도':>4}  {'전체매출(억)':>10}  {'20대매출(억)':>10}  {'20대비중':>8}  {'업종점수':>8}  {'트렌드매출비중':>12}  {'건당매출(만)':>10}"
    add(header4)
    add('-'*70)
    sub = raw_ann[raw_ann['행정동'] == dong].sort_values('연도')
    s_sub = sales[(sales['행정동'] == dong)].sort_values('연도')
    for _, row in sub.iterrows():
        y = int(row['연도'])
        ind_row = s_sub[s_sub['연도']==y]
        ind = ind_row['업종점수'].values[0] if len(ind_row) else np.nan
        trnd = ind_row['트렌드매출_비중'].values[0] if len(ind_row) else np.nan
        add(f"{y:>4}  {row['총매출']/1e8:>10,.1f}  {row['매출_20대']/1e8:>10,.1f}  "
            f"{row['20대비중']*100:>8.1f}%  {ind*100:>8.1f}%  {trnd*100:>12.1f}%  "
            f"{row['건당매출']/1e4:>10,.0f}")
    # 변화율
    base = sub[sub['연도']==2020]
    last = sub[sub['연도']==2024]
    if len(base) and len(last):
        chg_total = (last['총매출'].values[0] - base['총매출'].values[0]) / base['총매출'].values[0]
        chg_20    = (last['매출_20대'].values[0] - base['매출_20대'].values[0]) / base['매출_20대'].values[0]
        add(f"   ▶ 전체 매출 변화: {chg_total*100:+.1f}%  │  20대 매출 변화: {chg_20*100:+.1f}%")

# ══════════════════════════════════════════════════════════════════
# T5. 구역유형별 연도별 전체 매출 성장률 (YoY %)
# ══════════════════════════════════════════════════════════════════
add()
add()
add('## T5. 구역유형별 연도별 총 매출 성장률 (YoY %)')
add('   기준: 각 그룹 내 동 평균 총_매출금액 기준 YoY 성장률')
add(DIV)

header5 = f"{'그룹':<14}  {'2021':>7}  {'2022':>7}  {'2023':>7}  {'2024':>7}  {'4년누적':>8}"
add(header5)
add(DIV)

groups5 = {
    '서울전체(423)': None,
    '오피스형': '오피스형',
    '혼합형':   '혼합형',
    '주거형':   '주거형',
    '상권형':   '상권형',
    '레퍼런스(4)': 'REF',
}
for gname, gtype in groups5.items():
    if gtype is None:
        sub_all = raw_ann.groupby('연도')['총매출'].mean()
    elif gtype == 'REF':
        sub_all = raw_ann[raw_ann['행정동'].isin(REF)].groupby('연도')['총매출'].mean()
    else:
        dongs_g = area[area['구역유형']==gtype]['행정동'].tolist()
        sub_all = raw_ann[raw_ann['행정동'].isin(dongs_g)].groupby('연도')['총매출'].mean()

    yoys = []
    cumul = 1.0
    for y in [2021, 2022, 2023, 2024]:
        prev = sub_all.get(y-1, np.nan)
        curr = sub_all.get(y, np.nan)
        if pd.notnull(prev) and prev > 0:
            yoy = (curr - prev) / prev
            yoys.append(f"{yoy*100:>+7.1f}%")
            cumul *= (1 + yoy)
        else:
            yoys.append(f"{'   -':>7}")
    cumul_str = f"{(cumul-1)*100:>+8.1f}%"
    add(f"{gname:<14}  " + '  '.join(yoys) + f"  {cumul_str}")

# ══════════════════════════════════════════════════════════════════
# T6. 건당 매출(방문객 지출 proxy) 비교
# ══════════════════════════════════════════════════════════════════
add()
add()
add('## T6. 건당 매출(방문객 평균 지출 proxy) 비교 (2024, 단위: 원)')
add('   건당매출 = 총_매출금액 / 총_매출건수  (소비 강도 측정)')
add(DIV)

header6 = f"{'그룹':<14}  {'건당매출(원)':>12}  {'서울대비':>8}"
add(header6)
add(DIV)
seoul_per = r24['건당매출'].mean()
groups6 = {
    '서울전체(423)': r24,
    '오피스형': r24[r24['구역유형']=='오피스형'],
    '혼합형':   r24[r24['구역유형']=='혼합형'],
    '주거형':   r24[r24['구역유형']=='주거형'],
    '상권형':   r24[r24['구역유형']=='상권형'],
    '레퍼런스(4)': r24[r24['행정동'].isin(REF)],
    'Tier1후보(2)': r24[r24['행정동'].isin(TIER1)],
}
for gname, df_g in groups6.items():
    v = df_g['건당매출'].mean()
    diff = (v - seoul_per) / seoul_per
    add(f"{gname:<14}  {v:>12,.0f}  {diff*100:>+8.1f}%")

add()
add('  개별 레퍼런스 동:')
for dong in REF:
    v = r24[r24['행정동']==dong]['건당매출'].values
    if len(v):
        diff = (v[0] - seoul_per) / seoul_per
        add(f"    {dong:<10}  {v[0]:>12,.0f}  ({diff*100:>+.1f}%)")

# ══════════════════════════════════════════════════════════════════
# T7. 레퍼런스 동 20대 이동인구 연도별 변화
# ══════════════════════════════════════════════════════════════════
add()
add()
add('## T7. 레퍼런스 동 20대 이동인구 연도별 변화')
add('   출처: KT 생활이동 데이터 (금·토·일 트렌드 시간대)')
add(DIV)

header7 = f"{'행정동':<12}" + ''.join(f"  {y}년" for y in YEARS) + "  Δ(%)"
add(header7)
add(DIV)
for dong in REF + TIER1:
    sub = mig_ann[mig_ann['행정동']==dong].sort_values('연도')
    row = []
    v_base = None
    v_last = None
    for y in YEARS:
        r = sub[sub['연도']==y]
        if len(r):
            v = r['20대_이동인구'].values[0]
            row.append(f"  {v/1e4:>6.1f}만")
            if y == 2020: v_base = v
            if y == 2024: v_last = v
        else:
            row.append("       -")
    if v_base and v_last:
        delta = (v_last - v_base) / v_base * 100
        row.append(f"  {delta:>+.1f}%")
    add(f"{dong:<12}" + ''.join(row))

add()
add('  단위: 만 명 (연간 합계 도착 이동인구)')

# ── 20대 이동비율 함께 ──────────────────────────────────────────
add()
add('  [20대 이동비율 = 20대이동인구 / 전체이동인구]')
add(f"  {'행정동':<12}" + ''.join(f"  {y}년" for y in YEARS))
add('  ' + '-'*60)
for dong in REF + TIER1:
    sub = mig_ann[mig_ann['행정동']==dong].sort_values('연도')
    row = []
    for y in YEARS:
        r = sub[sub['연도']==y]
        v = r['20대_이동비율'].values[0] if len(r) else np.nan
        row.append(f"  {v*100:>6.1f}%" if pd.notnull(v) else "       -")
    add(f"  {dong:<12}" + ''.join(row))

# ══════════════════════════════════════════════════════════════════
# T8. 20대 이동비율 × 트렌드매출_비중 상관관계
# ══════════════════════════════════════════════════════════════════
add()
add()
add('## T8. 20대 이동비율 × 트렌드매출_비중 상관관계 (pooled, 행정동 × 연도)')
add('   수준값(level) 기준 피어슨 r')
add(DIV)

# 패널 구성
panel = (
    sales[['행정동', '연도', '트렌드매출_비중', '업종점수', '2030비중', '복합점수_분위']]
    .merge(
        mig_ann.groupby(['행정동', '연도'])[['20대_이동비율']].mean().reset_index(),
        on=['행정동', '연도'], how='inner'
    )
    .dropna()
)

add(f"  전체 패널 N = {len(panel)}  (행정동 × 연도)")
add()
pairs = [
    ('20대이동비율', '트렌드매출_비중', '20대 이동비율 → 트렌드매출_비중'),
    ('20대이동비율', '업종점수',       '20대 이동비율 → 업종점수'),
    ('20대이동비율', '2030비중',       '20대 이동비율 → 2030비중(소비)'),
    ('20대이동비율', '복합점수_분위',   '20대 이동비율 → 복합점수_분위'),
]
panel = panel.rename(columns={'20대_이동비율': '20대이동비율'})
header8 = f"  {'분석 방향':<30}  {'r':>6}  {'p값':>8}  {'N':>6}  해석"
add(header8)
add('  ' + DIV)
for x_col, y_col, label in pairs:
    sub = panel[[x_col, y_col]].dropna()
    r, p = pearsonr(sub[x_col], sub[y_col])
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
    direction = '양의 상관' if r > 0.1 else ('음의 상관' if r < -0.1 else '미미')
    add(f"  {label:<30}  {r:>+6.3f}  {p:>8.3e}  {len(sub):>6}  {sig} {direction}")

# 구역유형별 상관
add()
add('  [구역유형별 상관]')
add(f"  {'구역유형':<10}  {'r':>6}  {'p값':>8}  {'N':>5}")
add('  ' + '-'*40)
panel_typed = panel.merge(area_slim, on='행정동', how='left')
for t in TYPE_ORDER:
    sub = panel_typed[panel_typed['구역유형']==t][['20대이동비율', '트렌드매출_비중']].dropna()
    if len(sub) > 10:
        r, p = pearsonr(sub['20대이동비율'], sub['트렌드매출_비중'])
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
        add(f"  {t:<10}  {r:>+6.3f}  {p:>8.3e}  {len(sub):>5}  {sig}")

# ── 레퍼런스 동 개별 상관 ────────────────────────────────────────
add()
add('  [레퍼런스 4개 동 개별 시계열 상관 (5년 × 연도)]')
add(f"  {'행정동':<12}  {'r(이동↔트렌드매출)':>18}  {'r(이동↔2030비중)':>16}")
add('  ' + '-'*50)
for dong in REF:
    sub = panel[panel['행정동']==dong].sort_values('연도')
    if len(sub) >= 4:
        r1, _ = pearsonr(sub['20대이동비율'], sub['트렌드매출_비중'])
        r2, _ = pearsonr(sub['20대이동비율'], sub['2030비중'])
        add(f"  {dong:<12}  {r1:>+18.3f}  {r2:>16.3f}")

# ══════════════════════════════════════════════════════════════════
# 요약: 핵심 수치 대조표
# ══════════════════════════════════════════════════════════════════
add()
add()
add(SEP)
add('## 핵심 수치 대조표 — 가설 근거 요약 (2024)')
add(SEP)
add(f"{'항목':<22}  {'오피스형':>8}  {'서울전체':>8}  {'혼합형':>8}  {'주거형':>8}  {'상권형':>8}  {'레퍼런스':>8}")
add(DIV)

metrics = {
    '20대 매출비중(%)': ('raw_ann', '20대비중', 2024),
    '업종점수(%)':      ('s24',     '업종점수', None),
    '트렌드매출비중(%)': ('s24',     '트렌드매출_비중', None),
    '금요효과':         ('s24',     '금요효과', None),
}

def get_grp_val(df, col, grp_type):
    if grp_type == '서울전체':
        return df[col].mean()
    elif grp_type == '레퍼런스':
        return df[df['행정동'].isin(REF)][col].mean()
    else:
        return df[df['구역유형']==grp_type][col].mean()

GRP_ORDER = ['오피스형', '서울전체', '혼합형', '주거형', '상권형', '레퍼런스']
for mname, (src, col, yr) in metrics.items():
    if src == 'raw_ann':
        df_use = r24
    else:
        df_use = s24
    vals = [get_grp_val(df_use, col, g) for g in GRP_ORDER]
    mult = 100 if col in ['20대비중', '업종점수', '트렌드매출_비중'] else 1
    fmt = f"{mname:<22}" + ''.join(
        f"  {v*mult:>8.2f}" if pd.notnull(v) else f"  {'   -':>8}" for v in vals
    )
    add(fmt)

add()
add(SEP)

# ── 저장 ─────────────────────────────────────────────────────────
output = '\n'.join(lines)
with open('_tmp_evidence_tables.txt', 'w', encoding='utf-8') as f:
    f.write(output)

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
print(output)
print('\n-> _tmp_evidence_tables.txt 저장 완료')
