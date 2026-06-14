# -*- coding: utf-8 -*-
"""
최종_보고서_내용_정리_v6.md 에 필요한 데이터·이미지를 기존 파일에서 찾아 report/ 에 조합한다.

- 수치 근거: report/분석_수치_근거.md, report/data/*.csv, data/main_data/*.csv
- 이미지: image/, image/gen20/, report/images/ → report/images/ (섹션 번호)
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "report"
IMG_DST = REPORT / "images"
DAT_DST = REPORT / "data"
EVIDENCE_MD = REPORT / "분석_수치_근거.md"
V6_MD = ROOT / "최종_보고서_내용_정리_v6.md"
MAPPING_MD = REPORT / "최종보고서_v6_데이터_조합.md"

IMG_DST.mkdir(parents=True, exist_ok=True)
DAT_DST.mkdir(parents=True, exist_ok=True)

# build_report_dir.py 와 동일 + QoQ·신규 레퍼런스 차트
IMAGE_MAP = [
    ("image/gen20/01_consumption_trend.png", "S2_01_consumption_trend.png", "§2", "20대 매출 트렌드"),
    ("image/gen20/02_gen20_vs_30s.png", "S2_02_gen20_vs_30s.png", "§2", "20대 vs 30대"),
    ("image/gen20/03_migration.png", "S2_03_migration_trend.png", "§2", "20대 이동비율"),
    ("image/gen20/04_weekend_profile.png", "S2_04_weekend_profile.png", "§2", "주말집중도"),
    ("report/images/S2_05_area_type_industry.png", "S2_05_area_type_industry.png", "§2", "구역유형별 업종"),
    ("image/gen20/06_reference_trajectory.png", "S3_01_reference_trajectory.png", "§3", "레퍼런스 궤적"),
    ("image/causal_chain_analysis.png", "S3_02_causal_chain.png", "§3", "인과관계"),
    ("image/dashboard_A_signals.png", "S3_03_signals_dashboard.png", "§3", "4신호 대시보드"),
    ("report/images/S3_04_causal_lag.png", "S3_04_causal_lag.png", "§3", "lag 상관"),
    ("report/images/S3_05_ref_sales_timeseries.png", "S3_05_ref_sales_timeseries.png", "§3", "매출 시계열"),
    ("report/images/S3_06_ref_migration_timeseries.png", "S3_06_ref_migration_timeseries.png", "§3", "이동 시계열"),
    ("report/images/S3_07_ref_trend_industry.png", "S3_07_ref_trend_industry.png", "§3", "트렌드 업종"),
    ("report/images/S3_08_ref_trend_timesales.png", "S3_08_ref_trend_timesales.png", "§3", "시간대 매출"),
    ("image/gen20/05_candidates.png", "S4a_01_phase35_candidates.png", "§4.1", "Phase35 TOP15"),
    ("image/gen20/07_quadrant_analysis.png", "S4b_01_quadrant.png", "§4.2", "4분면"),
    ("image/gen20/08_quadrant_timeseries.png", "S4b_02_timeseries.png", "§4.2", "후보 시계열"),
    ("image/gen20/09_map_top15.png", "S4b_03_map_top15.png", "§4.2", "지도 TOP15"),
    ("image/gen20/10_phase_compare.png", "S5_01_phase35_vs_41.png", "§5", "Phase35 vs 41"),
    ("image/gen20/11_compare_map.png", "S5_02_compare_map.png", "§5", "교집합 지도"),
    ("image/gen20/12_method_compare.png", "S5_03_method_compare.png", "§5", "방법 비교"),
    ("report/images/S5_04_ref_vs_candidates.png", "S5_04_ref_vs_candidates.png", "§5", "레퍼런스 vs 후보"),
    ("image/qoq_01_group_timeseries.png", "S3_09_qoq_group_timeseries.png", "§3", "QoQ 그룹 시계열"),
    ("image/qoq_02_ccf_by_group.png", "S3_10_qoq_ccf_by_group.png", "§3", "QoQ CCF"),
    ("image/qoq_05_level_ccf_profile.png", "S3_11_qoq_level_ccf.png", "§3", "QoQ 수준 CCF"),
]

# v6 섹션 → 데이터 파일 매핑
DATA_MAP = [
    ("§1.2 가설·lag", "causal_lag_analysis.csv", "report/data/causal_lag_analysis.csv", "이동↔소비↔임대료 lag"),
    ("§2.3 구역유형", "area_type_profile.csv", "data/main_data/area_type_profile.csv", "423동 주말집중도·구역유형"),
    ("§3.2.1 레퍼런스 기본", "reference_3_2_1.csv", "report/data/reference_3_2_1.csv", "레퍼런스 16개 S1~S4"),
    ("§3.2.2 레퍼런스 수치", "reference_3_2_2.csv", "report/data/reference_3_2_2.csv", "업종·2030·주말집중도"),
    ("§3 레퍼런스 시계열", "reference_sales_qoq.csv", "report/data/reference_sales_qoq.csv", "분기 매출"),
    ("§3", "reference_migration_timeseries.csv", "report/data/reference_migration_timeseries.csv", "분기 이동"),
    ("§3", "reference_trend_industry_sales.csv", "report/data/reference_trend_industry_sales.csv", "트렌드 업종 매출"),
    ("§3", "reference_trend_time_sales.csv", "report/data/reference_trend_time_sales.csv", "시간대 매출"),
    ("§3", "reference_signal_detail.csv", "report/data/reference_signal_detail.csv", "외부 신호"),
    ("§4.1 Phase35", "phase35_candidates_top15.csv", "data/main_data/trend_candidates_v3.csv", "코사인 TOP15 (v3 기준)"),
    ("§4.2 Phase41", "phase41_quadrant_full.csv", "data/main_data/quadrant_20_result.csv", "4분면 전체"),
    ("§5 Tier 통합", "final_candidates_with_tier.csv", "(본 스크립트 생성)", "trend_candidates_v3 기반"),
    ("§5 Tier (구버전)", "final_candidates_tier.csv", "report/data/final_candidates_tier.csv", "v2 레퍼런스·보문/성산2 교집합"),
    ("§6 임대료", "임대료_공실률.csv", "data/main_data/임대료_공실률.csv", "68 상권"),
    ("§2·분석 마스터", "트렌드매출_마스터.csv", "data/main_data/트렌드매출_마스터.csv", "매출 핵심 지표"),
    ("§2·이동", "분기별_주말유입인구.csv", "data/main_data/분기별_주말유입인구.csv", "KT 분기 이동"),
    ("§2·이동", "분기별_원본피처.csv", "data/main_data/분기별_원본피처.csv", "분기 feature"),
]

NEW_REF = ["방배2동", "청운효자동", "합정동", "성수2가1동"]


def copy_image(src_rel: str, dst_name: str) -> str:
    src = ROOT / src_rel
    dst = IMG_DST / dst_name
    if not src.is_file():
        return "없음"
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return "있음"


def build_tier_v3() -> pd.DataFrame:
    """분석_수치_근거 §1 기준: trend_candidates_v3.csv → final_candidates_with_tier.csv"""
    v3 = pd.read_csv(ROOT / "data/main_data/trend_candidates_v3.csv", encoding="utf-8-sig")
    q41 = pd.read_csv(ROOT / "data/main_data/quadrant_20_result.csv", encoding="utf-8-sig")
    q41_map = q41.set_index("행정동")

    rows = []
    for _, r in v3.iterrows():
        dong = r["행정동"]
        tier_raw = str(r.get("tier", ""))
        if "Tier1" in tier_raw:
            tier, tier_desc = 1, "Tier1_교집합"
        elif "Tier2" in tier_raw:
            tier, tier_desc = 2, "Tier2_업종전환"
        else:
            tier, tier_desc = 3, "Tier3_이동선행"

        mob = quad = None
        if dong in q41_map.index:
            mob = round(float(q41_map.loc[dong, "이동성장세점수"]), 4)
            quad = q41_map.loc[dong, "사분면"]

        rows.append(
            {
                "행정동": dong,
                "Tier": tier,
                "Tier_설명": tier_desc,
                "Phase35_유사도": round(float(r["코사인유사도"]), 4) if pd.notna(r.get("코사인유사도")) else None,
                "Phase35_업종점수": round(float(r["업종점수"]), 4),
                "Phase35_트렌드매출_비중": round(float(r["트렌드매출_비중"]), 4),
                "Phase35_2030비중": round(float(r["2030비중"]), 4),
                "Phase35_복합점수_분위": round(float(r["복합점수_분위"]), 4),
                "Phase41_이동성장세": mob,
                "Phase41_사분면": quad,
                "주말집중도": round(float(r["주말집중도"]), 4),
                "구역유형": r["구역유형"],
                "레퍼런스_신규4_여부": dong in NEW_REF,
            }
        )
    return pd.DataFrame(rows)


def build_phase35_top15_v3() -> pd.DataFrame:
    v3 = pd.read_csv(ROOT / "data/main_data/trend_candidates_v3.csv", encoding="utf-8-sig")
    t2 = v3[v3["tier"].str.contains("Tier2", na=False)].copy()
    t2 = t2.sort_values("코사인유사도", ascending=False).head(15)
    t2 = t2.rename(
        columns={
            "코사인유사도": "유사도",
            "트렌드매출_비중": "트렌드매출_비중",
        }
    )
    return t2


def df_to_md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    d = df.head(max_rows) if max_rows else df
    cols = list(d.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in d.iterrows():
        cells = [str(row[c]) if pd.notna(row[c]) else "" for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_mapping_md(img_status: list, tier_v3: pd.DataFrame) -> None:
    ref41 = tier_v3[tier_v3["레퍼런스_신규4_여부"]]
    tier1 = tier_v3[tier_v3["Tier"] == 1]["행정동"].tolist()

    data_rows = []
    for section, out_name, src, note in DATA_MAP:
        p = ROOT / src if not src.startswith("(") else DAT_DST / out_name
        exists = p.is_file() if not src.startswith("(") else (DAT_DST / out_name).is_file()
        data_rows.append(f"| {section} | `{out_name}` | `{src}` | {'✓' if exists else '✗'} | {note} |")

    img_table = "\n".join(
        f"| {sec} | `{name}` | {desc} | {st} | `{src}` |"
        for src, name, sec, desc, st in img_status
    )

    ref16_path = DAT_DST / "reference_3_2_1.csv"
    ref16_md = ""
    if ref16_path.is_file():
        ref16 = pd.read_csv(ref16_path, encoding="utf-8-sig")
        ref16_md = "### §3.2.1 레퍼런스 16개 (기존 기준, report/data)\n\n" + df_to_md_table(ref16) + "\n\n"

    ref4_md = "### §0 신규 레퍼런스 4개 (분석_수치_근거 §0-3, 현행 분석 기준)\n\n"
    ref4_md += "| 행정동 | 구역유형 | 전환연도 | 비고 |\n|--------|---------|:-------:|------|\n"
    ref4_md += "| 방배2동 | 혼합형 | 2021 | 주말집중도 0.937, 트렌드매출_비중 0.355 |\n"
    ref4_md += "| 청운효자동 | 주거형 | 2021 | 서촌권 |\n"
    ref4_md += "| 합정동 | 주거형 | 2021 | 홍대 인근, 성숙 상권 |\n"
    ref4_md += "| 성수2가1동 | 혼합형 | 2022 | 이동 선행→소비 전환 대표 사례 |\n\n"

    tier_md = df_to_md_table(
        tier_v3[
            [
                "행정동",
                "Tier",
                "Tier_설명",
                "Phase35_유사도",
                "Phase41_이동성장세",
                "주말집중도",
                "구역유형",
            ]
        ]
    )

    body = f"""# 최종 보고서 v6 — 데이터·이미지 조합 인덱스

> 자동 생성: `assemble_v6_report.py`  
> 기준 문서: `최종_보고서_내용_정리_v6.md`  
> 수치 근거: `report/분석_수치_근거.md`

---

## 1. 버전 정합성 (중요)

| 항목 | v6 원문 | 현행 분석 (2026-05-26~) |
|------|---------|-------------------------|
| 레퍼런스 | 16개 동 (S1·S2 전환) | **신규 4개** (주말집중도≥0.80·트렌드매출_비중≥0.30) |
| Tier1 교집합 | 보문동·성산2동 | **숭인2동·신당5동** (`trend_candidates_v3`) |
| Phase35 TOP | 청파동·후암동·연희동 (v2) | 숭인2동·창4동·신당5동 등 (`trend_candidates_v3`) |

보고서 본문(v6)은 16개 레퍼런스 전제로 작성되어 있으나, **수치·후보 통합은 `final_candidates_with_tier.csv`(v3)를 우선**한다.  
16개 표·차트는 `reference_3_2_*.csv` 및 S3_* 이미지로 그대로 활용 가능.

---

## 2. v6 섹션 → 데이터 파일

| v6 섹션 | 산출 파일 | 원본 경로 | 상태 | 설명 |
|---------|-----------|-----------|:----:|------|
{chr(10).join(data_rows)}

---

## 3. v6 섹션 → 이미지 (`report/images/`)

| 섹션 | 파일명 | 설명 | 상태 | 원본 |
|------|--------|------|:----:|------|
{img_table}

원본 탐색 경로: `image/`, `image/gen20/`, `report/images/`  
누락 시 해당 분석 스크립트 재실행 (`analysis_gen20.py`, `analysis_qoq_lead_lag.py`, `build_report_dir.py` 등).

---

## 4. 채워진 표 (CSV → Markdown)

{ref4_md}
{ref16_md}
### §5 Tier 통합 — 신규 기준 (`final_candidates_with_tier.csv`)

**Tier1 교집합**: {', '.join(tier1) if tier1 else '(없음)'}

{tier_md}

---

## 5. 아직 없는 v6 📌 항목

| 항목 | 상태 | 비고 |
|------|:----:|------|
| `reference_external_validation.md` | 미작성 | §6.3 웹 검색·외부 자료 |
| `reference_dongs_detail.md` | 미작성 | 동별 1페이지 정성 요약 |
| 423동 × 분기 통합 지표 단일 CSV | 부분 | `분기별_원본피처.csv` + `분기별_주말유입인구.csv` 조인 가능 |
| 2025 사후 검증 | 미확보 | 데이터 없음 |

---

## 6. 빠른 사용법

```bash
python assemble_v6_report.py   # report/data·images 갱신 + 본 파일 재생성
python build_report_dir.py     # 구버전(v2) Tier·이미지 (보문/성산2 교집합)
```

- **보고서 수치 인용**: `report/분석_수치_근거.md` → 섹션 번호(§)로 역추적  
- **표·csv**: `report/data/final_candidates_with_tier.csv`  
- **그림 삽입**: `report/images/S*.png`
"""
    MAPPING_MD.write_text(body, encoding="utf-8")


def patch_v6_tables() -> None:
    """v6 문서 §3.2 TBD 표를 report/data 기준으로 치환"""
    if not V6_MD.is_file():
        return
    text = V6_MD.read_text(encoding="utf-8")
    if "(TBD)" not in text:
        return

    ref1 = pd.read_csv(DAT_DST / "reference_3_2_1.csv", encoding="utf-8-sig")
    ref2 = pd.read_csv(DAT_DST / "reference_3_2_2.csv", encoding="utf-8-sig")

    t1 = df_to_md_table(ref1)
    t2_cols = [
        "행정동",
        "업종점수_2020",
        "업종점수_2024",
        "2030비중_2020",
        "2030비중_2024",
        "20대이동YoY피크(%)",
        "주말집중도",
        "구역유형",
    ]
    t2 = df_to_md_table(ref2[t2_cols])

    import re

    block_321 = re.compile(
        r"(#### 3\.2\.1 레퍼런스 16개 동 기본 정보\n\n)(>.*?\n\n)(\| #.*?\| 16 \| \(TBD\) \|.*?\| --- \|.*?\n)",
        re.DOTALL,
    )
    repl_321 = rf"\1> 아래 표는 `report/data/reference_3_2_1.csv` 와 동기화됨 (기존 16개 기준).\n\n{t1}\n"
    text, n1 = block_321.subn(repl_321, text, count=1)

    block_322 = re.compile(
        r"(#### 3\.2\.2 레퍼런스 동별 핵심 수치\n\n)(?:>.*?\n\n)?(\| 행정동 \|.*?\n\| --- \|.*?\n(?:\| [^\n]+\n)+)",
        re.DOTALL,
    )
    repl_322 = (
        rf"\1> `report/data/reference_3_2_2.csv` 기준. "
        rf"신규 레퍼런스 4개는 `report/최종보고서_v6_데이터_조합.md` §4 참고.\n\n{t2}\n"
    )
    text, n2 = block_322.subn(repl_322, text, count=1)

    note = (
        "\n\n> **데이터 조합 (자동)**: 상세 매핑·신규 Tier는 "
        "`report/최종보고서_v6_데이터_조합.md`, "
        "`report/data/final_candidates_with_tier.csv` 참고.\n"
    )
    if "최종보고서_v6_데이터_조합" not in text:
        text = text.replace(
            "> 💡 **사용 안내**:",
            "> 💡 **사용 안내**:" + note,
            1,
        )

    if n1 or n2:
        V6_MD.write_text(text, encoding="utf-8")
        print(f"  ✓ 최종_보고서_내용_정리_v6.md §3.2 표 갱신 (3.2.1={n1}, 3.2.2={n2})")
    else:
        print("  · v6 §3.2 표 패턴 미매칭 — 조합.md만 갱신")


def main() -> None:
    print("[1/4] 이미지 조합 → report/images/")
    img_status = []
    for src, dst_name, sec, desc in IMAGE_MAP:
        st = copy_image(src, dst_name)
        img_status.append((src, dst_name, sec, desc, st))
        print(f"  [{st}] {dst_name}")

    print("\n[2/4] 데이터 복사·생성")
    shutil.copy2(ROOT / "data/main_data/area_type_profile.csv", DAT_DST / "area_type_profile.csv")
    shutil.copy2(ROOT / "data/main_data/임대료_공실률.csv", DAT_DST / "임대료_공실률.csv")

    p35 = build_phase35_top15_v3()
    p35.to_csv(DAT_DST / "phase35_candidates_top15_v3.csv", encoding="utf-8-sig", index=False)
    print(f"  ✓ phase35_candidates_top15_v3.csv ({len(p35)}행)")

    shutil.copy2(ROOT / "data/main_data/quadrant_20_result.csv", DAT_DST / "phase41_quadrant_full.csv")

    tier_v3 = build_tier_v3()
    tier_v3.to_csv(DAT_DST / "final_candidates_with_tier.csv", encoding="utf-8-sig", index=False)
    print(
        f"  ✓ final_candidates_with_tier.csv "
        f"(Tier1={sum(tier_v3['Tier']==1)}, Tier2={sum(tier_v3['Tier']==2)}, Tier3={sum(tier_v3['Tier']==3)})"
    )

    print("\n[3/4] 조합 인덱스 Markdown")
    write_mapping_md(img_status, tier_v3)
    print(f"  ✓ {MAPPING_MD.relative_to(ROOT)}")

    print("\n[4/4] v6 본문 표 보강")
    patch_v6_tables()

    ok_img = sum(1 for *_, st in img_status if st == "있음")
    print(f"\n=== 완료: 이미지 {ok_img}/{len(img_status)}, Tier1={tier_v3.loc[tier_v3['Tier']==1,'행정동'].tolist()} ===")


if __name__ == "__main__":
    main()
