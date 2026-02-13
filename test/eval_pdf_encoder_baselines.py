"""
PDF 인코더 비교 평가: 텍스트만 / OCR / 텍스트+OCR / 텍스트+VLM

동일 PDF·동일 쿼리로 네 방식의 검색 결과를 비교해 Recall@k, MRR을 출력합니다.
CSV는 query, relevant_pdf 만 사용 (page_no 없음. 페이지별 평가 시에는 PDF를 먼저 페이지별로 자른 뒤 해당 디렉터리 지정).

사용법 (프로젝트 루트에서):
  python test/eval_pdf_encoder_baselines.py --pdf-dir test/test_dir_pdf/pdf_pages --queries-csv test/test_dir_pdf/eval_queries.csv
  python test/eval_pdf_encoder_baselines.py --pdf-dir test/last_test --queries-csv test/test_dir_pdf/eval_queries.csv  # last_test 안의 PDF만 사용, 해당 쿼리만 자동 필터
"""

import argparse
import csv
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "test"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import torch
from omegaconf import OmegaConf

from app.services.pdf_processor import (
    pdf_to_combined_text,
    pdf_to_text_only,
    pdf_to_ocr_text,
    pdf_to_text_plus_ocr,
    _get_vlm_client,
)
from app.services.text_encoder import TextEncoder


def load_config(config_path: Path):
    raw = OmegaConf.load(config_path)
    cfg = raw.get("default", raw)
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def get_pdf_texts(pdf_paths: list[Path], method: str, cfg, vlm_client=None):
    """method: 'text_only' | 'ocr' | 'text_plus_ocr' | 'combined'"""
    texts = []
    for p in pdf_paths:
        if method == "text_only":
            t = pdf_to_text_only(str(p))
        elif method == "ocr":
            t = pdf_to_ocr_text(str(p))
        elif method == "text_plus_ocr":
            t = pdf_to_text_plus_ocr(str(p))
        else:
            t = pdf_to_combined_text(str(p), cfg, vlm_client=vlm_client)
        texts.append(t if t else "")
    return texts


def recall_at_k(rank_1based: int, k: int) -> float:
    return 1.0 if rank_1based and rank_1based <= k else 0.0


def mrr(rank_1based: int) -> float:
    return 1.0 / rank_1based if rank_1based else 0.0


def rank_of_relevant(
    query_emb: list[float],
    doc_embeddings: list[list[float]],
    relevant_idx: int,
) -> int | None:
    """
    정답 문서가 몇 위(1-based)인지 반환. 없으면 None.
    유사도: 쿼리·문서 임베딩의 내적 (q @ doc). jina 등 L2 정규화 임베딩이면 내적 = 코사인 유사도.
    """
    if relevant_idx < 0 or relevant_idx >= len(doc_embeddings):
        return None
    q = torch.tensor(query_emb, dtype=torch.float32)
    docs = torch.tensor(doc_embeddings, dtype=torch.float32)
    scores = (q.unsqueeze(0) @ docs.T).squeeze(0)
    order = torch.argsort(scores, descending=True)
    for r, idx in enumerate(order.tolist(), start=1):
        if idx == relevant_idx:
            return r
    return None


def resolve_relevant_idx(pdf_paths: list[Path], rel_basename: str) -> int:
    """relevant_pdf basename으로 문서 인덱스 반환. 정확한 파일명 또는 doc_id로 매칭."""
    path_to_idx = {p.name: i for i, p in enumerate(pdf_paths)}
    idx = path_to_idx.get(
        rel_basename, path_to_idx.get(Path(rel_basename).name, -1)
    )
    if idx >= 0:
        return idx
    stem = Path(rel_basename).stem
    parts = stem.split("_")
    if len(parts) >= 1:
        doc_id = parts[-1]
        for i, p in enumerate(pdf_paths):
            if doc_id in p.name or p.stem.endswith(doc_id):
                return i
    return -1


def try_build_queries_csv_from_labels(pdf_dir: Path) -> Path | None:
    """
    라벨 디렉터리에서 build_eval_dataset_from_labels.build_dataset을 호출해
    eval 쿼리 CSV를 생성합니다. 성공 시 생성된 CSV 경로, 실패 시 None.
    for_split_pdfs=True 이므로 원본 라벨·원본 PDF 디렉터리를 넘겨 stem_1.pdf 형식이 나오게 함.
    """
    test_dir_pdf = TEST_DIR / "test_dir_pdf"
    label_plain = test_dir_pdf / "label"
    pdf_origin = test_dir_pdf / "pdf_"
    if not label_plain.is_dir():
        return None
    build_script = TEST_DIR / "build_eval_dataset_from_labels.py"
    if not build_script.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        "build_eval_dataset_from_labels", build_script
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out_csv = test_dir_pdf / "eval_queries.csv"
    try:
        rows = mod.build_dataset(
            label_dir=label_plain,
            pdf_dir=pdf_origin if pdf_origin.is_dir() else None,
            out_csv=out_csv,
            out_json=None,
            verify_pages=False,
            for_split_pdfs=True,
        )
        if rows and out_csv.exists():
            print(
                f"쿼리 CSV 없음 → build_eval_dataset_from_labels로 생성: {out_csv} ({len(rows)}행)"
            )
            return out_csv
    except Exception as e:
        print(f"build_eval_dataset_from_labels 호출 실패: {e}")
    return None


def run_evaluation(
    pdf_dir: Path,
    queries_csv: Path,
    config_path: Path,
    top_k: int = 5,
):
    cfg = load_config(config_path)
    model_name = cfg.model_name.text
    text_encoder = TextEncoder(model_name=model_name)

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"PDF 없음: {pdf_dir}")
        return

    queries: list[tuple[str, str]] = []
    with open(queries_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("query") or "").strip()
            rel = (row.get("relevant_pdf") or "").strip()
            if q and rel:
                queries.append((q, rel))

    # pdf_dir에 있는 PDF만 참조하는 쿼리만 사용 (정확한 파일명 일치 우선, 없으면 doc_id 매칭)
    pdf_names = {p.name for p in pdf_paths}
    n_before = len(queries)
    queries_exact = [
        (q, rel)
        for q, rel in queries
        if rel in pdf_names or Path(rel).name in pdf_names
    ]
    if queries_exact:
        queries = queries_exact
    else:
        queries = [
            (q, rel)
            for q, rel in queries
            if resolve_relevant_idx(pdf_paths, rel) >= 0
        ]
    if n_before > len(queries):
        print(f"pdf_dir 기준 필터: 쿼리 {n_before}개 → {len(queries)}개 사용")
    if not queries:
        print(
            "쿼리 없음. CSV의 relevant_pdf가 pdf_dir 안의 파일과 매칭되는 행이 없습니다."
        )
        return

    vlm_client = (
        _get_vlm_client(cfg)
        if getattr(getattr(cfg, "vlm", None), "enabled", True)
        else None
    )

    results = {}
    for method_name, method_key in [
        ("텍스트만 (PyMuPDF)", "text_only"),
        ("OCR (easyocr)", "ocr"),
        ("텍스트+OCR (그림 포함)", "text_plus_ocr"),
        ("텍스트+VLM (제안)", "combined"),
    ]:
        t0 = time.perf_counter()
        texts = get_pdf_texts(pdf_paths, method_key, cfg, vlm_client)
        with torch.no_grad():
            doc_embs = text_encoder.model.encode(
                texts, task="retrieval.passage"
            ).tolist()
        query_embs = [text_encoder.emb_query(q) for q, _ in queries]
        time_sec = time.perf_counter() - t0

        recall1_sum = recall5_sum = mrr_sum = 0.0
        n = len(queries)
        for (q, rel), q_emb in zip(queries, query_embs):
            relevant_idx = resolve_relevant_idx(pdf_paths, rel)
            rank = rank_of_relevant(q_emb, doc_embs, relevant_idx)
            recall1_sum += recall_at_k(rank, 1)
            recall5_sum += recall_at_k(rank, top_k)
            mrr_sum += mrr(rank)
        results[method_name] = {
            "Recall@1": recall1_sum / n,
            f"Recall@{top_k}": recall5_sum / n,
            "MRR": mrr_sum / n,
            "time_sec": time_sec,
        }

    print("\n=== PDF 인코더 비교 평가 ===\n")
    print(f"PDF 디렉터리: {pdf_dir}")
    print(f"문서 수: {len(pdf_paths)}")
    print(f"쿼리 개수: {n}")
    print(f"Top-K: {top_k}\n")
    print(
        f"{'방식':<25} {'Recall@1':>10} {f'Recall@{top_k}':>12} {'MRR':>10} {'시간(초)':>10}"
    )
    print("-" * 72)
    for name, vals in results.items():
        print(
            f"{name:<25} {vals['Recall@1']:>10.3f} {vals[f'Recall@{top_k}']:>12.3f} {vals['MRR']:>10.3f} {vals['time_sec']:>10.2f}"
        )
    print()
    return results


def main():
    default_pdf_dir = ROOT / "test" / "test_dir_pdf" / "pdf_pages"
    parser = argparse.ArgumentParser(
        description="PDF 인코더 베이스라인 비교 (텍스트만 / OCR / 텍스트+OCR / 텍스트+VLM)"
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=default_pdf_dir,
        help="PDF 디렉터리 (기본: test/last_test)",
    )
    parser.add_argument(
        "--queries-csv", type=Path, help="query, relevant_pdf 컬럼 CSV"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config" / "config.yaml"
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Recall@k의 k (기본: 3)"
    )
    args = parser.parse_args()

    queries_csv = args.queries_csv
    if not queries_csv or not queries_csv.exists():
        auto_csv = try_build_queries_csv_from_labels(args.pdf_dir)
        if auto_csv is not None:
            queries_csv = auto_csv
        else:
            sample = args.pdf_dir / "eval_queries_sample.csv"
            if not sample.exists():
                sample.parent.mkdir(parents=True, exist_ok=True)
                with open(sample, "w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["query", "relevant_pdf"])
                    pdfs = list(args.pdf_dir.glob("*.pdf"))
                    if pdfs:
                        w.writerow(["이 문서의 내용을 요약하면", pdfs[0].name])
                print(f"라벨로 CSV 생성 실패 → 샘플 쿼리 CSV 생성: {sample}")
            queries_csv = sample
    else:
        queries_csv = Path(args.queries_csv)

    run_evaluation(
        pdf_dir=args.pdf_dir,
        queries_csv=Path(queries_csv),
        config_path=args.config,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
