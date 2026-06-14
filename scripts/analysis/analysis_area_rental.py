"""
구역유형 분류 기준 수치 + 임대료·공실률 연관 분석
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

# ── 데이터 로드 ──────────────────────────────────────────────────
area   = pd.read_csv('data/main_data/area_type_profile.csv', encoding='utf-8-sig')
rental = pd.read_csv('data/main_data/임대료_공실률.csv', encoding='utf-8-sig')
dongmap = pd.read_csv('data/main_data/rental_vacancy_dongmap.csv', encoding='utf-8-sig')

# ══════════════════════════════════════════════════════════════════
# PART 1: 구역유형 분류 기준 — 주말집중도 분포 수치
# ══════════════════════════════════════════════════════════════════
TYPE_ORDER = ['오피스형', '혼합형', '주거형', '상권형']
THRESHOLDS = {'오피스형': (0, 0.80), '혼합형': (0.80, 0.95),
              '주거형': (0.95, 1.10), '상권형': (1.10, 99)}

print("=" * 70)
print("PART 1. 구역유형 분류 기준 — 주말집중도 분포")
print("=" * 70)
print(f"\n{'구역유형':<8} {'임계값':^12} {'동 수':>5} {'최솟값':>7} {'최댓값':>7} {'평균':>7} {'중앙값':>7} {'표준편차':>8}")
print("-" * 65)
for t in TYPE_ORDER:
    sub = area[area['구역유형'] == t]['주말집중도']
    lo, hi = THRESHOLDS[t]
    thr = f"{lo:.2f}~{hi:.2f}" if hi < 99 else f"≥ {lo:.2f}"
    print(f"{t:<8} {thr:^12} {len(sub):>5} {sub.min():>7.3f} {sub.max():>7.3f} "
          f"{sub.mean():>7.3f} {sub.median():>7.3f} {sub.std():>8.3f}")

# 대표 동 예시 (최댓값·최솟값 근처)
print("\n── 구역유형별 대표 예시 ──")
for t in TYPE_ORDER:
    sub = area[area['구역유형'] == t].sort_values('주말집중도')
    low5  = sub.head(3)[['행정동','주말집중도']].values
    high5 = sub.tail(3)[['행정동','주말집중도']].values
    print(f"\n[{t}]")
    print(f"  하위 3: " + " / ".join(f"{r[0]}({r[1]:.3f})" for r in low5))
    print(f"  상위 3: " + " / ".join(f"{r[0]}({r[1]:.3f})" for r in high5))

# 레퍼런스 4개 주말집중도 확인
print("\n── 레퍼런스 4개 주말집중도 ──")
refs = ['방배2동', '청운효자동', '합정동', '성수2가1동']
ref_data = area[area['행정동'].isin(refs)][['행정동', '주말집중도', '구역유형']]
print(ref_data.to_string(index=False))

# ══════════════════════════════════════════════════════════════════
# PART 2: 임대료·공실률 × 구역유형 연관
# ══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("PART 2. 임대료·공실률 × 구역유형 연관")
print("=" * 70)

# 상권명 → 대표 행정동 → 구역유형 매핑
dongmap_dict = dict(zip(dongmap['상권명'], dongmap['대표_행정동']))
area_type_dict = dict(zip(area['행정동'], area['구역유형']))

rental = rental.copy()
rental['대표_행정동'] = rental['상권명'].map(dongmap_dict)
rental['구역유형'] = rental['대표_행정동'].map(area_type_dict)

# 매핑 현황
mapped = rental.dropna(subset=['구역유형'])
print(f"\n상권 데이터 매핑: 전체 {len(rental['상권명'].unique())}개 상권 중 "
      f"{len(mapped['상권명'].unique())}개 매핑 완료")
print(f"구역유형 분포: {mapped.groupby('구역유형')['상권명'].nunique().to_dict()}")

# 2024년 기준 구역유형별 임대료·공실률 평균
r24 = mapped[mapped['연도'] == 2024].copy()
print(f"\n── 2024년 기준 구역유형별 평균 임대료·공실률 (N=상권 수) ──")
grp = r24.groupby('구역유형').agg(
    N=('상권명', 'count'),
    임대료_평균=('임대료', 'mean'),
    임대료_중앙=('임대료', 'median'),
    공실률_평균=('공실률', 'mean'),
    공실률_중앙=('공실률', 'median'),
).round(1)
grp = grp.reindex([t for t in TYPE_ORDER if t in grp.index])
print(grp.to_string())

# ── 레퍼런스 지역 상권의 임대료·공실률 시계열 ──
print("\n── 레퍼런스 연관 상권 임대료·공실률 추이 ──")
REF_SANGKWON = {
    '뚝섬':        '성수2가1동(인근)',
    '방배역/내방역': '방배2동',
    '광화문':       '청운효자동',
    '서촌':         '청운효자동',
    '홍대/합정':    '합정동',
}
for sang, dong in REF_SANGKWON.items():
    sub = rental[rental['상권명'] == sang][['연도', '임대료', '공실률']].sort_values('연도')
    if sub.empty:
        continue
    print(f"\n[{sang} → {dong}]")
    print(f"  {'연도':>4}  {'임대료(천원/㎡)':>14}  {'공실률(%)':>9}")
    for _, row in sub.iterrows():
        yr = int(row['연도'])
        rent = f"{row['임대료']:.1f}" if pd.notna(row['임대료']) else "  -"
        vac  = f"{row['공실률']:.1f}" if pd.notna(row['공실률']) else "  -"
        print(f"  {yr:>4}  {rent:>14}  {vac:>9}")

# ── Tier1 후보(숭인2동·신당5동) 인근 상권 확인 ──
print("\n── Tier1 후보 인근 상권 (매핑 가능한 상권) ──")
tier1_adjacent = {
    '동대문':  '황학동(Tier3 인근)',  # 황학동 포함
    '왕십리':  '왕십리2동(근처)',
    '약수역':  '약수동(근처)',
}
for sang, note in tier1_adjacent.items():
    sub = rental[(rental['상권명'] == sang) & (rental['연도'] == 2024)][['임대료', '공실률']]
    if sub.empty:
        continue
    row = sub.iloc[0]
    print(f"  [{sang} / {note}] 임대료 {row['임대료']:.1f}, 공실률 {row['공실률']:.1f}%")

# ── 임대료 상승률 비교: 성수(뚝섬) vs 홍대합정 vs 명동 ──
print("\n── 임대료 변화율 비교 (2020→2024, 데이터 있는 상권) ──")
COMPARE = ['뚝섬', '홍대/합정', '방배역/내방역', '압구정', '이태원', '건대입구', '명동']
r_compare = rental[rental['상권명'].isin(COMPARE)].copy()
pivot_rent = r_compare.pivot_table(index='상권명', columns='연도', values='임대료')
pivot_vac  = r_compare.pivot_table(index='상권명', columns='연도', values='공실률')

print(f"\n{'상권':^12} {'임대료 2020':>11} {'임대료 2024':>11} {'Δ임대료':>8} {'공실률 2020':>11} {'공실률 2024':>11}")
print("-" * 62)
for sang in COMPARE:
    if sang not in pivot_rent.index:
        continue
    r20 = pivot_rent.loc[sang, 2020] if 2020 in pivot_rent.columns else np.nan
    r24_ = pivot_rent.loc[sang, 2024] if 2024 in pivot_rent.columns else np.nan
    v20 = pivot_vac.loc[sang, 2020] if 2020 in pivot_vac.columns else np.nan
    v24_ = pivot_vac.loc[sang, 2024] if 2024 in pivot_vac.columns else np.nan
    delta_r = f"+{(r24_/r20-1)*100:.1f}%" if pd.notna(r20) and pd.notna(r24_) and r20>0 else "  -"
    r20s  = f"{r20:.1f}" if pd.notna(r20) else "  -"
    r24s  = f"{r24_:.1f}" if pd.notna(r24_) else "  -"
    v20s  = f"{v20:.1f}%" if pd.notna(v20) else "  -"
    v24s  = f"{v24_:.1f}%" if pd.notna(v24_) else "  -"
    print(f"{sang:^12} {r20s:>11} {r24s:>11} {delta_r:>8} {v20s:>11} {v24s:>11}")
