"""
PDF를 페이지별로 잘라서 저장. (JSON은 자르지 않고, eval_queries 만들 때 page_no만 사용해 relevant_pdf 이름을 정함.)

- 입력: 원본 PDF 디렉터리
- 출력: 페이지별 PDF 디렉터리 — 각 PDF는 {원본_stem}_1.pdf, _2.pdf, ... (1-based)

이후 build_eval_dataset_from_labels.py --for-split-pdfs 로 eval_queries.csv를 만들면
relevant_pdf가 위와 동일한 이름(stem_1.pdf)이 되어, page_no 컬럼 없이 평가 가능.

사용법 (프로젝트 루트에서):
  python test/split_pdf_and_label_by_page.py --pdf-dir test/test_dir_pdf/pdf --out-dir test/test_dir_pdf/pdf_pages
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def split_pdf_by_pages(pdf_path: Path, out_dir: Path) -> list[Path]:
    """
    PDF를 1페이지씩 잘라 out_dir에 {stem}_1.pdf, {stem}_2.pdf, ... 로 저장.
    Returns: 생성된 파일 경로 리스트.
    """
    import fitz
    out_dir.mkdir(parents=True, exist_ok=True)
    created = []
    doc = fitz.open(pdf_path)
    try:
        stem = pdf_path.stem
        for i in range(len(doc)):
            one = fitz.open()
            one.insert_pdf(doc, from_page=i, to_page=i)
            out_name = f"{stem}_{i + 1}.pdf"
            out_path = out_dir / out_name
            one.save(out_path)
            one.close()
            created.append(out_path)
    finally:
        doc.close()
    return created


def main():
    parser = argparse.ArgumentParser(
        description="PDF를 페이지별로 잘라 저장 (stem_1.pdf, stem_2.pdf, ...)"
    )
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "test" / "test_dir_pdf" / "pdf")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "test" / "test_dir_pdf" / "pdf_pages")
    args = parser.parse_args()

    pdf_files = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"PDF 없음: {args.pdf_dir}")
        return 1

    total = 0
    for p in pdf_files:
        created = split_pdf_by_pages(p, args.out_dir)
        total += len(created)
        print(f"  {p.name} → {len(created)}개 페이지")
    print(f"총 {total}개 페이지 PDF 저장: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
