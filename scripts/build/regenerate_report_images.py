# -*- coding: utf-8 -*-
"""
보고서용 시각화 전체 재생성 (신규 레퍼런스 4개 + trend_candidates_v3 기준)

실행: python regenerate_report_images.py
"""

import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent

STEPS = [
    ("차세대 후보 v3 산출", "analysis_candidate_reselect.py"),
    ("gen20 기본 6종", "analysis_gen20.py"),
    ("4분면 분석", "analysis_quadrant_20.py"),
    ("QoQ 선행-후행", "analysis_qoq_lead_lag.py"),
    ("보고서 보조 차트", "build_report_supplement.py"),
    ("레퍼런스 트렌드 매출", "build_ref_trend_sales.py"),
    ("lag 분석", "build_causal_lag.py"),
    ("인과 사슬", "causal_chain_analysis.py"),
    ("Phase35 vs 41", "analysis_phase_compare.py"),
    ("지도 TOP15", "map_top15.py"),
    ("방법 비교", "analysis_method_compare.py"),
    ("신호 대시보드", "dashboard_viz.py"),
    ("report/images 복사", "assemble_v6_report.py"),
    ("HTML 보고서", "report/build_html_report.py"),
]


def main() -> None:
    print("=" * 60)
    print("보고서 시각화 일괄 재생성 (신규 기준)")
    print("=" * 60)
    failed = []
    for label, script in STEPS:
        path = ROOT / script
        if not path.is_file():
            print(f"  ✗ SKIP {script} (파일 없음)")
            failed.append(script)
            continue
        print(f"\n▶ {label}: {script}")
        r = subprocess.run([sys.executable, str(path)], cwd=str(ROOT))
        if r.returncode != 0:
            print(f"  ✗ 실패 (exit {r.returncode})")
            failed.append(script)
        else:
            print(f"  ✓ 완료")
    print("\n" + "=" * 60)
    if failed:
        print(f"실패 {len(failed)}개: {failed}")
        sys.exit(1)
    print("전체 완료 → report/images/ 및 image/gen20/ 확인")
    print("HTML: report/최종보고서.html (python report/build_html_report.py)")


if __name__ == "__main__":
    main()
