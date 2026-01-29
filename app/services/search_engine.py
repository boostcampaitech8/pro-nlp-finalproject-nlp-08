import os
from typing import List, Dict, Any

from app.services.file_manager import FileManager
from app.services.vector_store import VectorStore
from app.services.encoder import SiglipEncoder


def search(
    query: str, root_path: str, extensions: List[str], top_k: int
) -> List[Dict[str, Any]]:
    """
    주어진 질의어와 경로, 확장자 목록을 바탕으로 검색을 수행합니다.
    Args:
        query (str): 검색 질의 문자열
        root_path (str): 검색 대상 루트 경로
        extensions (List[str]): 검색 대상 확장자 목록
        top_k (int): 상위 k개 결과 반환
    Returns:
        List[Dict[str, Any]]: 검색 결과 목록
    """
    vector_store = VectorStore()
    encoder = SiglipEncoder()
    file_manager = FileManager()

    all_files = file_manager.scan_directory(root_path, extensions)
    need_embed_files = file_manager.sync_and_filter(all_files, vector_store)
    if need_embed_files:
        embeded_vector_list = encoder.create_emb_list(need_embed_files)
    for file_path, embeded_vector in zip(
        need_embed_files, embeded_vector_list
    ):
        vector_store.create_cache(
            create_key=file_path,
            mtime=os.path.getmtime(file_path),
            create_embed=embeded_vector,
        )
    query_embed = encoder.create_emb_query([query])[0]
    results = vector_store.search(query_embed, top_k=top_k)

    return results
