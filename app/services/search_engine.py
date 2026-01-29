import os
import pathlib
from typing import List, Dict, Any

from app.services.file_manager import FileManager
from app.services.vector_store import VectorStore
from app.services.encoder import SiglipEncoder


def search(
    query: str, target_path: str, extensions: List[str], top_k: int, cache_path: str
) -> List[Dict[str, Any]]:
    """
    주어진 질의어와 경로, 확장자 목록을 바탕으로 검색을 수행합니다.
    Args:
        query (str): 검색 질의 문자열
        target_path (str): 검색 대상 루트 경로
        extensions (List[str]): 검색 대상 확장자 목록
        top_k (int): 상위 k개 결과 반환
        cache_path (str): 캐시 파일 경로
    Returns:
        List[Dict[str, Any]]: 검색 결과 목록
    """
    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    cache_full_path = os.path.join(ROOT_DIR, cache_path)
    vector_store = VectorStore(cache_full_path)
    encoder = SiglipEncoder()
    file_manager = FileManager(vector_store, target_path)

    need_embed_files = file_manager.sync_and_filter()
    print(f"Files need embedding: {need_embed_files}")
    if need_embed_files:
        embeded_vector_list = encoder.create_emb_list(need_embed_files)
        for file_path, embeded_vector in zip(
            need_embed_files, embeded_vector_list
        ):
            vector_store.create_cache(
                create_key=file_path,
                mtime=file_manager.get_file_mtime(file_path),
                create_embed=embeded_vector,
            )
    query_embed = encoder.create_emb_txt_query(query)
    results = vector_store.search(target_path, query_embed, top_k)

    vector_store.save_cache()

    return results
