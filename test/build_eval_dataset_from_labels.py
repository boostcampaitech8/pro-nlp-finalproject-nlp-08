"""
test_dir_pdf/label (QA3_*.json) 에서 평가용 데이터셋 생성.

- label: QA3_*.json, pdf: QA2_*.pdf. 파일명 끝 doc id(예: 000111)로 매칭.
- source_data_info[] 각 탭의 qa_data[].question → query, page_no는 자를 때만 사용(아래 --for-split-pdfs).

일반: CSV에 query, relevant_pdf, page_no, doc_name, answer.
--for-split-pdfs: 먼저 split_pdf_and_label_by_page.py 로 PDF를 페이지별로 자른 뒤,
  relevant_pdf=stem_1.pdf 형식으로 출력. CSV에는 page_no 컬럼 없음.

사용법:
  python test/build_eval_dataset_from_labels.py
  python test/split_pdf_and_label_by_page.py --pdf-dir test/test_dir_pdf/pdf --out-dir test/test_dir_pdf/pdf_pages
  python test/build_eval_dataset_from_labels.py --for-split-pdfs --out-csv test/test_dir_pdf/eval_queries.csv
  python test/eval_pdf_encoder_baselines.py --pdf-dir test/test_dir_pdf/pdf_pages --queries-csv test/test_dir_pdf/eval_queries.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def doc_id_from_qa_label_stem(stem: str) -> str | None:
    """QA3 라벨 파일 stem에서 doc id 추출. 예: QA3_240820_00_000111 -> 000111."""
    parts = stem.split("_")
    if len(parts) >= 1:
        return parts[-1]
    return None


def find_pdf_for_doc_id(pdf_dir: Path | None, doc_id: str) -> str | None:
    """pdf_dir에서 doc_id가 포함된 PDF 파일명(basename) 반환. 없으면 None."""
    if not pdf_dir or not pdf_dir.is_dir():
        return None
    for p in pdf_dir.glob("*.pdf"):
        if doc_id in p.name or p.stem.endswith(doc_id):
            return p.name
    return None


def get_pdf_page_count(pdf_path: Path) -> int | None:
    """PDF 쪽수 반환. 실패 시 None."""
    try:
        import fitz

        doc = fitz.open(pdf_path)
        try:
            return len(doc)
        finally:
            doc.close()
    except Exception:
        return None


def load_qa_label_rows(
    label_path: Path,
    pdf_dir: Path | None,
    verify_pages: bool = False,
    for_split_pdfs: bool = False,
) -> list[dict]:
    """
    QA3 형식 JSON에서 (query, relevant_pdf, page_no, doc_name, answer) 행 리스트 생성.
    - source_data_info[] 각 항목의 qa_data[].question, page_no 사용.
    - page_no가 있으면 해당 페이지가 PDF에 존재하는지 확인(verify_pages=True 시).
    """
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            raw = f.read()
        if not raw.strip():
            return []
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [건너뜀] JSON 파싱 실패: {label_path.name} — {e}")
        return []
    except OSError as e:
        print(f"  [건너뜀] 파일 읽기 실패: {label_path.name} — {e}")
        return []

    stem = label_path.stem
    doc_id = doc_id_from_qa_label_stem(stem)
    if not doc_id:
        return []

    pdf_basename = find_pdf_for_doc_id(pdf_dir, doc_id) if pdf_dir else None
    pdf_path = (pdf_dir / pdf_basename) if (pdf_dir and pdf_basename) else None
    if verify_pages and pdf_path and not pdf_path.is_file():
        pdf_path = None

    doc_name = ""
    try:
        doc_name = (data.get("raw_data_info") or {}).get("doc_name") or ""
    except Exception:
        pass

    source_list = data.get("source_data_info")
    if not isinstance(source_list, list):
        return []

    rows = []
    for tab in source_list:
        page_nos = tab.get("page_no")
        page_no = (
            int(page_nos[0]) if (page_nos and len(page_nos) > 0) else None
        )
        qa_list = tab.get("qa_data")
        if not isinstance(qa_list, list):
            continue
        for q in qa_list:
            question = (q.get("question") or "").strip()
            if not question:
                continue

            if for_split_pdfs:
                if page_no is None:
                    continue
                if pdf_basename:
                    stem = Path(pdf_basename).stem
                    relevant_pdf = f"{stem}_{page_no}.pdf"
                else:
                    relevant_pdf = f"QA2_00_{doc_id}_{page_no}.pdf"
                row = {
                    "query": question,
                    "relevant_pdf": relevant_pdf,
                    "doc_name": doc_name,
                    "answer": (q.get("answer") or "")[:500],
                }
            else:
                if not pdf_basename:
                    relevant_pdf = f"QA2_00_{doc_id}.pdf"
                else:
                    relevant_pdf = pdf_basename
                if (
                    verify_pages
                    and page_no is not None
                    and pdf_path is not None
                ):
                    total = get_pdf_page_count(pdf_path)
                    if total is not None and page_no > total:
                        continue
                row = {
                    "query": question,
                    "relevant_pdf": relevant_pdf,
                    "page_no": page_no if page_no is not None else "",
                    "doc_name": doc_name,
                    "answer": (q.get("answer") or "")[:500],
                }
            rows.append(row)
    return rows


def build_dataset(
    label_dir: Path,
    pdf_dir: Path | None,
    out_csv: Path,
    out_json: Path | None = None,
    verify_pages: bool = False,
    for_split_pdfs: bool = False,
) -> list[dict]:
    """label_dir의 QA3_*.json에서 qa_data.question + page_no 기반으로 CSV 생성."""
    label_files = sorted(label_dir.glob("QA3_*.json"))
    if not label_files:
        label_files = sorted(label_dir.glob("*.json"))

    all_rows: list[dict] = []
    for lp in label_files:
        all_rows.extend(
            load_qa_label_rows(
                lp,
                pdf_dir,
                verify_pages=verify_pages,
                for_split_pdfs=for_split_pdfs,
            )
        )

    if not all_rows:
        print(
            "생성된 행 없음. label의 QA3_*.json 및 source_data_info.qa_data, page_no를 확인하세요."
        )
        return all_rows

    fieldnames = (
        ["query", "relevant_pdf", "doc_name", "answer"]
        if for_split_pdfs
        else ["query", "relevant_pdf", "page_no", "doc_name", "answer"]
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"총 {len(all_rows)}개 쿼리 저장: {out_csv}")

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, ensure_ascii=False, indent=2)
        print(f"JSON 저장: {out_json}")

    return all_rows


def main():
    parser = argparse.ArgumentParser(
        description="test_dir_pdf 라벨(QA3)에서 qa_data.question + page_no로 평가 데이터셋 생성"
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=ROOT / "test" / "test_dir_pdf" / "label",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=ROOT / "test" / "test_dir_pdf" / "pdf",
        help="없으면 relevant_pdf는 QA2_00_<doc_id>.pdf 형식으로 출력",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "test" / "test_dir_pdf" / "eval_queries.csv",
    )
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument(
        "--verify-pages",
        action="store_true",
        help="PDF 실제 쪽수와 page_no가 맞는 행만 포함 (page_no > PDF쪽수 이면 제외)",
    )
    parser.add_argument(
        "--no-pdf-dir",
        action="store_true",
        help="pdf 디렉터리 없이 JSON만으로 생성 (relevant_pdf는 예상 이름)",
    )
    parser.add_argument(
        "--for-split-pdfs",
        action="store_true",
        help="페이지별 잘린 PDF용 CSV 생성. relevant_pdf=stem_1.pdf 형식, page_no 컬럼 없음. 먼저 split_pdf_and_label_by_page.py 로 PDF 자르기.",
    )
    args = parser.parse_args()

    pdf_dir = None if args.no_pdf_dir else args.pdf_dir
    out_json = args.out_json
    if out_json is None and args.out_csv:
        out_json = args.out_csv.with_suffix(".json")

    build_dataset(
        label_dir=args.label_dir,
        pdf_dir=pdf_dir,
        out_csv=args.out_csv,
        out_json=out_json,
        verify_pages=args.verify_pages,
        for_split_pdfs=args.for_split_pdfs,
    )


if __name__ == "__main__":
    main()
