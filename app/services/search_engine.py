import os
import pathlib
from typing import List, Dict, Tuple, Optional, Callable
from omegaconf import DictConfig

from app.services.file_manager import FileManager
from app.services.vector_store import VectorStore


def search(
    query: str,
    target_dir: str,
    target_extensions: Dict[str, List[str]],
    cfg: DictConfig,
    top_k: Optional[int] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    주어진 질의어와 경로, 확장자 목록을 바탕으로 검색을 수행합니다.
    Args:
        query (str): 검색 질의 문자열
        target_dir (str): 검색 대상 루트 경로
        target_extensions (Dict[str, List[str]]): 검색 대상 확장자 목록
            예: {"image': ['.png', '.jpg'], 'text': ['.txt', '.md']}
        cfg (DictConfig): 설정 객체
        progress_callback (Callable[[int, str], None], optional): 진행 상황 콜백 함수
        top_k (Optional[int]): 상위 k개 결과 반환. None이면 cfg.search.top_k 사용

    Returns:
        Dict[str, List[Tuple[str, float]]]: 타입별(예: 'image', 'text') 상위 k개 검색 결과
            예: {'image': [('/path/to/image1.jpg', 0.95), ('/path/to/image2.png', 0.93)],
                  'text': [('/path/to/doc1.txt', 0.89)]}
    """

    def update_progress(progress: int, message: str):
        if progress_callback:
            progress_callback(progress, message)

    # 파일 스캔 단계 (0-100%)
    update_progress(0, "파일 스캔 중")
    file_manager = FileManager(target_dir, target_extensions)
    file_paths = file_manager.scan_directory()
    update_progress(100, "파일 스캔 완료")

    # 모델 로드 및 벡터 스토어 초기화 단계 (0-100%)
    update_progress(0, "모델 로드 중")
    vector_store = VectorStore(
        target_dir, target_extensions, cfg, progress_callback=update_progress
    )
    update_progress(100, "모델 로드 완료")

    # 벡터 스토어 동기화 단계 (0-100%)
    update_progress(0, "벡터 스토어 동기화 중")
    vector_store.sync_vector_store(file_paths)
    update_progress(100, "벡터 스토어 동기화 완료")

    update_progress(0, "검색 실행 중")
    k = top_k if top_k is not None else cfg.search.top_k
    results = vector_store.search(
        query, top_k=k, progress_callback=update_progress
    )
    update_progress(100, "검색 완료")

    return results
