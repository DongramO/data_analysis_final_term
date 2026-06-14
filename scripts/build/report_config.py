# -*- coding: utf-8 -*-
"""
보고서·시각화 공통 기준 (2026-05-26 확정)
- 레퍼런스: 주말집중도≥0.80, 트렌드매출_비중≥0.30, S1·S2 전환 4개 동
- 후보: trend_candidates_v3.csv (Tier1/2/3)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

# 신규 레퍼런스 (모든 보고서 시각화 기준)
REFERENCE_DONGS = ["방배2동", "청운효자동", "합정동", "성수2가1동"]

# 구버전 16개 (비교·레거시 차트용, 기본 사용 금지)
REFERENCE_LEGACY_16 = [
    "성수1가2동",
    "역삼1동",
    "청운효자동",
    "삼성1동",
    "성수2가1동",
    "합정동",
    "구로3동",
    "군자동",
    "방배2동",
    "가양1동",
    "대학동",
    "사직동",
    "상계2동",
    "성수2가3동",
    "신촌동",
    "휘경1동",
]

UNIV_EXCLUDE = ["신촌동", "대학동", "휘경1동", "상계2동", "회기동"]

V3_PATH = ROOT / "data" / "main_data" / "trend_candidates_v3.csv"


def load_v3() -> pd.DataFrame:
    return pd.read_csv(V3_PATH, encoding="utf-8-sig")


def tier1_dongs(df: pd.DataFrame | None = None) -> list[str]:
    df = load_v3() if df is None else df
    return df[df["tier"].str.contains("Tier1", na=False)]["행정동"].tolist()


def tier2_dongs(df: pd.DataFrame | None = None) -> list[str]:
    df = load_v3() if df is None else df
    return df[df["tier"].str.contains("Tier2", na=False)]["행정동"].tolist()


def tier3_dongs(df: pd.DataFrame | None = None) -> list[str]:
    df = load_v3() if df is None else df
    return df[df["tier"].str.contains("Tier3", na=False)]["행정동"].tolist()


def phase35_top15(df: pd.DataFrame | None = None) -> list[str]:
    """코사인 유사도 상위 15 (Tier2 업종전환 중심 + Tier1 포함)."""
    df = load_v3() if df is None else df
    sub = df[df["tier"].str.contains("Tier2", na=False)].sort_values("코사인유사도", ascending=False)
    top = sub.head(15)["행정동"].tolist()
    for d in tier1_dongs(df):
        if d not in top:
            top = [d] + top[:14]
    return top[:15]


def compare_candidate_dongs(df: pd.DataFrame | None = None) -> list[str]:
    """레퍼런스 vs 후보 비교 차트용 (Tier1 + Tier2 상위)."""
    df = load_v3() if df is None else df
    t2 = df[df["tier"].str.contains("Tier2", na=False)].nlargest(6, "코사인유사도")["행정동"].tolist()
    return list(dict.fromkeys(tier1_dongs(df) + t2))


def phase41_top15() -> list[str]:
    q = pd.read_csv(ROOT / "data" / "main_data" / "quadrant_20_result.csv", encoding="utf-8-sig")
    return (
        q[q["사분면"] == "2사분면_유망"]
        .nlargest(15, "이동성장세점수")["행정동"]
        .tolist()
    )
