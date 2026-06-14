"""
시계열 3차원 연관 분석: 20대 이동인구 → 매출 전환 → 임대료 반응
대상: 레퍼런스 4개 + Tier1 2개 + Tier2 대표 2개
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── 데이터 로드 ───────────────────────────────────────────────────
sales   = pd.read_csv('data/main_data/트렌드매출_마스터.csv', encoding='utf-8-sig')
mig_q   = pd.read_csv('data/main_data/분기별_주말유입인구.csv', encoding='utf-8-sig')
codemap = pd.read_csv('data/main_data/코드매핑_유입인구_매출.csv', encoding='utf-8-sig')
rental  = pd.read_csv('data/main_data/임대료_공실률.csv', encoding='utf-8-sig')

YEARS = [2020, 2021, 2022, 2023, 2024]

# ── 분석 대상 지역 ────────────────────────────────────────────────
REGIONS = {
    # 레퍼런스
    '성수2가1동': {'tier': '레퍼런스', '임대료상권': '뚝섬'},
    '방배2동':   {'tier': '레퍼런스', '임대료상권': '방배역/내방역'},
    '합정동':    {'tier': '레퍼런스', '임대료상권': '홍대/합정'},
    '청운효자동': {'tier': '레퍼런스', '임대료상권': '광화문'},
    # Tier1 유망
    '숭인2동':   {'tier': 'Tier1 유망', '임대료상권': '동대문'},
    '신당5동':   {'tier': 'Tier1 유망', '임대료상권': '동대문'},
    # Tier2 유망 (레퍼런스 궤적 유사, 대표 2개)
    '창4동':     {'tier': 'Tier2 유망', '임대료상권': None},
    '장위1동':   {'tier': 'Tier2 유망', '임대료상권': None},
}

# ── 이동인구: 행정동명 매핑 ────────────────────────────────────────
codemap['행정동명'] = codemap['행정동명'].str.strip()
code_dict = dict(zip(codemap['행정동명'], codemap['migration_코드']))

# 연도별 20대 이동인구 (합계_이동인구, 연령대==20)
mig20 = mig_q[mig_q['연령대'] == 20].copy()
mig_all = mig_q.copy()  # 전체 연령 합계용

# 연도별 집계 (분기 합산)
mig20_ann = mig20.groupby(['도착_행정동코드', '연도'])['합계_이동인구'].sum().reset_index()
mig20_ann.columns = ['코드', '연도', '20대이동']
mig_tot_ann = mig_all.groupby(['도착_행정동코드', '연도'])['합계_이동인구'].sum().reset_index()
mig_tot_ann.columns = ['코드', '연도', '전체이동']

mig_ann = mig20_ann.merge(mig_tot_ann, on=['코드', '연도'])
mig_ann['20대이동비율'] = mig_ann['20대이동'] / mig_ann['전체이동']

# ── 매출: 행정동 × 연도 ───────────────────────────────────────────
s_cols = ['행정동', '연도', '업종점수', '트렌드매출_비중', '2030비중', '총_매출']
sales_ann = sales[s_cols].copy()
sales_ann['20대매출_비중'] = sales_ann['2030비중']  # 2030비중이 20대+30대

# ── 임대료: 상권 × 연도 ───────────────────────────────────────────
rent_dict = {}
for _, row in rental.iterrows():
    key = (row['상권명'], int(row['연도']))
    rent_dict[key] = {'임대료': row['임대료'], '공실률': row['공실률']}

# ══════════════════════════════════════════════════════════════════
# 지역별 시계열 테이블 생성
# ══════════════════════════════════════════════════════════════════
SEP = '─' * 80

def get_mig_row(dong, year):
    code = code_dict.get(dong)
    if code is None:
        return np.nan, np.nan, np.nan
    row = mig_ann[(mig_ann['코드'] == code) & (mig_ann['연도'] == year)]
    if row.empty:
        return np.nan, np.nan, np.nan
    r = row.iloc[0]
    return r['20대이동'] / 10000, r['전체이동'] / 10000, r['20대이동비율'] * 100

def get_sales_row(dong, year):
    row = sales_ann[(sales_ann['행정동'] == dong) & (sales_ann['연도'] == year)]
    if row.empty:
        return np.nan, np.nan, np.nan
    r = row.iloc[0]
    return r['업종점수'] * 100, r['트렌드매출_비중'] * 100, r['2030비중'] * 100

def get_rent_row(sangkwon, year):
    if sangkwon is None:
        return np.nan, np.nan
    key = (sangkwon, year)
    if key not in rent_dict:
        return np.nan, np.nan
    d = rent_dict[key]
    return d['임대료'], d['공실률']

# ── 출력 ──────────────────────────────────────────────────────────
print("=" * 80)
print("시계열 3차원 연관 분석: 20대 이동 → 매출 전환 → 임대료 반응")
print("출처: KT 생활이동(금밤+토+일) / 서울시 상권분석서비스 / 한국부동산원")
print("=" * 80)

results = {}  # 나중에 비교 테이블용으로 저장

for dong, meta in REGIONS.items():
    tier = meta['tier']
    sang = meta['임대료상권']

    print(f"\n{SEP}")
    note = f" ※ 임대료: {sang} 상권 기준 (간접)" if sang in ['동대문'] else (f" ※ 임대료: {sang} 상권 기준" if sang else " ※ 임대료: 데이터 없음")
    print(f"▶ {dong}  [{tier}]{note}")
    print(SEP)

    hdr = (f"  {'연도':>4}  "
           f"{'20대이동':>8}  {'전체이동':>8}  {'20대비율':>7}  "
           f"{'업종점수':>7}  {'트렌드매출':>8}  {'2030비중':>7}  "
           f"{'임대료':>7}  {'공실률':>6}")
    print(hdr)
    print("  " + "-" * 76)

    rows_data = []
    for yr in YEARS:
        mig20_val, mig_tot_val, mig_ratio = get_mig_row(dong, yr)
        bs, tm, s30 = get_sales_row(dong, yr)
        rent_val, vac_val = get_rent_row(sang, yr)

        def fmt(v, dec=1, suffix=''):
            return f"{v:.{dec}f}{suffix}" if pd.notna(v) else "  -"

        print(f"  {yr:>4}  "
              f"{fmt(mig20_val, 1, '만'):>8}  "
              f"{fmt(mig_tot_val, 1, '만'):>8}  "
              f"{fmt(mig_ratio, 1, '%'):>7}  "
              f"{fmt(bs, 1, '%'):>7}  "
              f"{fmt(tm, 1, '%'):>8}  "
              f"{fmt(s30, 1, '%'):>7}  "
              f"{fmt(rent_val, 1):>7}  "
              f"{fmt(vac_val, 1, '%'):>6}")

        rows_data.append({
            '행정동': dong, '연도': yr, 'tier': tier,
            '20대이동(만)': mig20_val, '20대비율(%)': mig_ratio,
            '업종점수(%)': bs, '트렌드매출(%)': tm,
            '임대료': rent_val, '공실률(%)': vac_val
        })

    results[dong] = rows_data

    # 변화량 요약 (2020→2024)
    d0 = rows_data[0]
    d4 = rows_data[4]
    def delta(key, is_pct=False):
        v0, v4 = d0.get(key), d4.get(key)
        if pd.isna(v0) or pd.isna(v4): return "  -"
        if is_pct: return f"{v4-v0:+.1f}%p"
        return f"{(v4/v0-1)*100:+.1f}%" if v0 > 0 else "  -"

    print(f"\n  Δ(2020→2024)  "
          f"이동: {delta('20대이동(만)')}  "
          f"비율: {delta('20대비율(%)', True)}  "
          f"업종: {delta('업종점수(%)', True)}  "
          f"트렌드: {delta('트렌드매출(%)', True)}  "
          f"임대료: {delta('임대료')}")

# ══════════════════════════════════════════════════════════════════
# 비교 요약 테이블: 2024년 스냅샷 + 2020→2024 변화
# ══════════════════════════════════════════════════════════════════
print(f"\n\n{'=' * 80}")
print("비교 요약 — 2024 스냅샷 및 2020→2024 변화율")
print("=" * 80)

all_rows = []
for dong_rows in results.values():
    all_rows.extend(dong_rows)
df = pd.DataFrame(all_rows)

df24 = df[df['연도'] == 2024].copy()
df20 = df[df['연도'] == 2020].copy()

merged = df24.merge(df20[['행정동', '20대이동(만)', '20대비율(%)', '업종점수(%)', '트렌드매출(%)', '임대료']],
                    on='행정동', suffixes=('_24', '_20'))

merged['Δ이동'] = ((merged['20대이동(만)_24'] / merged['20대이동(만)_20'] - 1) * 100).round(1)
merged['Δ이동비율'] = (merged['20대비율(%)_24'] - merged['20대비율(%)_20']).round(1)
merged['Δ업종'] = (merged['업종점수(%)_24'] - merged['업종점수(%)_20']).round(1)
merged['Δ트렌드'] = (merged['트렌드매출(%)_24'] - merged['트렌드매출(%)_20']).round(1)
merged['Δ임대료'] = ((merged['임대료_24'] / merged['임대료_20'] - 1) * 100).round(1)

TIER_ORDER = ['레퍼런스', 'Tier1 유망', 'Tier2 유망']
merged['tier_order'] = merged['tier'].map({t: i for i, t in enumerate(TIER_ORDER)})
merged = merged.sort_values(['tier_order', 'Δ이동'], ascending=[True, False])

def fmtd(v, suffix='%'):
    if pd.isna(v): return "  -"
    return f"{v:+.1f}{suffix}"

print(f"\n  {'지역':<8} {'구분':^8} {'20대이동_24':>9} {'Δ이동':>7} {'Δ이동비율':>8} "
      f"{'업종_24':>7} {'Δ업종':>6} {'트렌드_24':>8} {'Δ트렌드':>7} "
      f"{'임대료_24':>8} {'Δ임대료':>7} {'공실률_24':>8}")
print("  " + "-" * 92)

for _, r in merged.iterrows():
    mig = f"{r['20대이동(만)_24']:.1f}만" if pd.notna(r['20대이동(만)_24']) else "  -"
    rent = f"{r['임대료_24']:.1f}" if pd.notna(r['임대료_24']) else "  -"
    vac  = f"{r['공실률(%)']:.1f}%" if pd.notna(r['공실률(%)']) else "  -"
    bs   = f"{r['업종점수(%)_24']:.1f}%" if pd.notna(r['업종점수(%)_24']) else "  -"
    tm   = f"{r['트렌드매출(%)_24']:.1f}%" if pd.notna(r['트렌드매출(%)_24']) else "  -"

    print(f"  {r['행정동']:<8} {r['tier']:^8} {mig:>9} {fmtd(r['Δ이동'], '%'):>7} "
          f"{fmtd(r['Δ이동비율'], '%p'):>8} "
          f"{bs:>7} {fmtd(r['Δ업종'], '%p'):>6} {tm:>8} {fmtd(r['Δ트렌드'], '%p'):>7} "
          f"{rent:>8} {fmtd(r['Δ임대료'], '%'):>7} {vac:>8}")

# ══════════════════════════════════════════════════════════════════
# 전환 단계 패턴 해석
# ══════════════════════════════════════════════════════════════════
print(f"\n\n{'=' * 80}")
print("전환 단계별 패턴 해석")
print("=" * 80)
print("""
[Stage 1 — 이동 선행, 소비 미전환]  Tier3/초기 Tier1
  · 20대 이동비율 증가  /  업종점수 낮음  /  임대료 낮음
  · 예: 성수2가1동 2020~2021년 / 숭인2동·신당5동 현재

[Stage 2 — 소비 전환 중]  Tier1~Tier2
  · 이동+비율 동시 상승  /  업종점수 급등  /  트렌드매출 증가
  · 임대료 아직 낮거나 완만한 상승 (아직 발견 전)
  · 예: 성수2가1동 2022년 / 숭인2동·신당5동 현재 진입 가능

[Stage 3 — 상권 성숙, 임대료 반응]  레퍼런스 완성 단계
  · 이동비율 안정 또는 전체 절대량 증가  /  업종점수 고위 유지
  · 임대료 급등  /  공실률 저위 (수요 > 공급)
  · 예: 성수2가1동 2022~2024년 (뚝섬 임대료 +51.1%)

[Stage 4 — 포화·과열]  홍대/합정 등 기성 상권
  · 임대료 정체·하락  /  공실률 상승  /  20대 다른 지역으로 이탈 시작
  · 예: 홍대/합정 (임대료 -6.7%, 공실률 상승)
""")
