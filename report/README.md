# 최종 보고서 자료

## 보고서 본문 (여기서 시작)

**[`최종보고서.html`](최종보고서.html)** — **그림 포함 보기 (권장)**  
탐색기에서 더블클릭 → Chrome·Edge. `images/` 폴더 PNG 24장 연동.

**[`최종보고서.md`](최종보고서.md)** — 텍스트·편집용  
MD 미리보기에서는 로컬 그림이 안 보일 수 있음 → HTML 사용.

HTML 재생성: `python report/build_html_report.py`

## Word / PowerPoint

| 용도 | 파일 | 생성 |
|------|------|------|
| 인쇄·제출 (Word) | `통합보고서.docx` | `python build_docx_report.py` |
| **발표 (PPT)** | `최종보고서_발표.pptx` | `python build_pptx_report.py` |

**Word DOCX를 PPT에 복사하면** 표·그림·구분선이 깨지는 경우가 많습니다.  
→ PPT는 위 `.pptx`를 쓰거나, PPT에서 **삽입 → 그림**으로 `images/` PNG를 넣으세요.

수치·해석의 상세 근거: [`분석_수치_근거.md`](분석_수치_근거.md)

## 그림

`images/` — 섹션별 번호 (S2 배경, S3 레퍼런스, S4 후보, S5 비교)

## 참고

- 작업용 목차·📌 체크리스트: 프로젝트 루트 `최종_보고서_내용_정리_v6.md`
- 데이터·CSV 인덱스(개발용): `최종보고서_v6_데이터_조합.md`, `data/` (보고서 작성 시 필수 아님)
