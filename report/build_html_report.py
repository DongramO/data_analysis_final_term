# -*- coding: utf-8 -*-
"""최종보고서.md → 최종보고서.html (브라우저에서 그림 포함 표시)"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
MD = HERE / "최종보고서.md"
OUT = HERE / "최종보고서.html"


def main() -> None:
    md_text = MD.read_text(encoding="utf-8")
    # MD 미리보기: report/ 폴더 기준 ./images/
    md_text = md_text.replace("](images/", "](./images/")
    md_json = json.dumps(md_text, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>서울시 차세대 트렌드 상권 발굴 분석 보고서</title>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    body {{
      font-family: "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
      line-height: 1.65; color: #1a1a1a;
      max-width: 920px; margin: 0 auto; padding: 2rem 1.5rem 4rem;
      background: #fafafa;
    }}
    .notice {{
      background: #e8f4fc; border: 1px solid #7eb8da;
      padding: 1rem 1.25rem; border-radius: 8px; margin-bottom: 2rem; font-size: 0.95rem;
    }}
    #content {{
      background: #fff; padding: 2rem 2.5rem; border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,.08);
    }}
    #content img {{ max-width: 100%; height: auto; display: block; margin: 1.25rem auto;
      border: 1px solid #ddd; border-radius: 4px; }}
    #content table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.92rem; }}
    #content th, #content td {{ border: 1px solid #ccc; padding: 0.45rem 0.65rem; }}
    #content th {{ background: #f0f0f0; }}
    #content h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.4rem; }}
    #content h2 {{ margin-top: 2rem; color: #222; }}
    #content blockquote {{ background: #fff8e6; border-left: 4px solid #e6a800;
      padding: 0.5rem 1rem; margin: 1rem 0; }}
  </style>
</head>
<body>
  <div class="notice">
    <strong>그림이 보이는 버전입니다.</strong>
    탐색기에서 이 파일(<code>최종보고서.html</code>)을 더블클릭해 Chrome·Edge로 여세요.
    그림 파일은 같은 폴더의 <code>images/</code> 에 있습니다.
  </div>
  <article id="content"></article>
  <script>
    const md = {md_json};
    document.getElementById("content").innerHTML = marked.parse(md, {{ breaks: true }});
  </script>
</body>
</html>"""

    OUT.write_text(html, encoding="utf-8")
    print(f"✓ {OUT}")


if __name__ == "__main__":
    main()
