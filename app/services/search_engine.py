import os
import pathlib
from typing import List, Dict, Tuple, Optional
from omegaconf import DictConfig

from app.services.file_manager import FileManager
from app.services.vector_store import VectorStore


def search(
    query: str,
    target_dir: str,
    target_extensions: Dict[str, List[str]],
    cfg: DictConfig,
    top_k: Optional[int] = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    주어진 질의어와 경로, 확장자 목록을 바탕으로 검색을 수행합니다.
    Args:
        query (str): 검색 질의 문자열
        target_dir (str): 검색 대상 루트 경로
        target_extensions (Dict[str, List[str]]): 검색 대상 확장자 목록
            예: {"image': ['.png', '.jpg'], 'text': ['.txt', '.md']}
        cfg (DictConfig): 설정 객체
        top_k (Optional[int]): 상위 k개 결과 반환. None이면 cfg.search.top_k 사용

    Returns:
        Dict[str, List[Tuple[str, float]]]: 타입별(예: 'image', 'text') 상위 k개 검색 결과
            예: {'image': [('/path/to/image1.jpg', 0.95), ('/path/to/image2.png', 0.93)],
                  'text': [('/path/to/doc1.txt', 0.89)]}
    """
    actual_top_k: int = top_k if top_k is not None else cfg.search.top_k

    file_manager = FileManager(target_dir, target_extensions)
    vector_store = VectorStore(target_dir, target_extensions, cfg)

    file_paths = file_manager.scan_directory()
    vector_store.sync_vector_store(file_paths)

    results = vector_store.search(query, top_k=actual_top_k)

    return results
