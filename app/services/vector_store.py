# 캐시 조회 저장 삭제 업데이트
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from app.services.encoder import Encoder
import sqlite3
import os
import faiss
from collections import defaultdict
from omegaconf import DictConfig


class VectorStore:
    """
    pickle 기반 VectorStore 구현 및 조회, 삽입, 삭제 기능
    지정된 경로의 pickle 파일을 읽어 데이터베이스를 메모리에 로드

    Args:
        target_dir (str): 사용자가 지정한 디렉토리 경로
        target_extensions (Dict[str, List[str]]): 임베딩할 파일 확장자 목록
        cfg (DictConfig): 설정

    Returns:
        None
    """

    def __init__(
        self,
        target_dir: str,
        target_extensions: Dict[str, List[str]],
        cfg: DictConfig,
    ):
        ROOT_DIR = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        self.target_dir = target_dir
        self.target_extensions = target_extensions
        self.db_path = os.path.join(ROOT_DIR, cfg.db_path)
        self.sqlite_path = os.path.abspath(
            os.path.join(self.db_path, "metadata.db")
        )
        # target_extensions의 키를 target_types로 활용하여 필요한 encoder만 로딩
        target_types = set(target_extensions.keys())
        self.encoder = Encoder(cfg=cfg, target_types=target_types)
        self.batch_size = dict(cfg.batch_size)
        self.faiss_indices = {}

        os.makedirs(self.db_path, exist_ok=True)
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE,
                    mtime REAL,
                    extension TEXT,
                    type TEXT
                )
                """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_extension ON metadata(extension)"
            )

    def _load_faiss_index(
        self, type_: str, dim: Optional[int] = None
    ) -> Optional[faiss.IndexIDMap]:
        """
        인덱스를 메모리(캐시)에서 가져오거나 디스크에서 로드합니다.
        파일이 없으면 새 인덱스를 생성합니다.

        Args:
            type_ (str): 인덱스 유형
            dim (Optional[int]): 임베딩 벡터의 차원 수
        """
        faiss_index_path = self._get_faiss_index_path(type_)

        if type_ in self.faiss_indices:
            return self.faiss_indices[type_]

        if os.path.exists(faiss_index_path):
            index = faiss.read_index(faiss_index_path)
            self.faiss_indices[type_] = index
            return index

        if dim is not None:
            base_index = faiss.IndexFlatIP(dim)
            index = faiss.IndexIDMap(base_index)
            self.faiss_indices[type_] = index
            return index

        return None

    def _get_faiss_index_path(self, extension: str) -> str:
        """
        주어진 확장자에 대한 Faiss 인덱스 파일 경로를 반환합니다.
        Args:
            extension (str): 파일 확장자
        Returns:
            str: Faiss 인덱스 파일 경로
        """
        return os.path.join(self.db_path, f"index_{extension}.faiss")

    def _fetch_db_metadata(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Summary:
            주어진 디렉토리와 확장자 목록에 따라 데이터베이스에서 메타데이터를 가져옵니다.
        Args:
            target_dir (str): 사용자가 지정한 디렉토리 경로
            target_extensions (Dict[str, List[str]]): 임베딩할 파일 확장자 목록
        Returns:
            Dict[str, Dict[str, Dict[str, Any]]]: 파일 경로를 키로 하고 메타데이터를 값으로 하는 딕셔너리
            예: {'image': {'/path/to/image1.jpg': {'mtime': 1234567890.0, 'id': 1}, ...}, 'text': {...}}
        """
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()

            target_extensions_flat = [
                ext
                for sublist in self.target_extensions.values()
                for ext in sublist
            ]
            place_holders = ", ".join(["?"] * len(target_extensions_flat))
            cursor.execute(
                f"SELECT file_path, mtime, id, type FROM metadata WHERE file_path LIKE ? AND extension IN ({place_holders})",
                ([f"{self.target_dir}%"] + target_extensions_flat),
            )
            db_files = defaultdict(dict)
            for row in cursor.fetchall():
                path, mtime, id_, type_ = row
                db_files[type_][path] = {"mtime": mtime, "id": id_}

        return dict(db_files)

    def _remove_data(self, to_delete_ids: Dict[str, List[int]]) -> None:
        """
        Summary:
            주어진 ID 목록에 따라 데이터베이스와 Faiss 인덱스에서 데이터를 삭제합니다.
        Args:
            to_delete_ids (Dict[str, List[int]]): 삭제할 데이터의 ID 목록
        """
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            for type_, ids in to_delete_ids.items():
                if not ids:
                    continue
                placeholders = ", ".join(["?"] * len(ids))
                cursor.execute(
                    f"DELETE FROM metadata WHERE id IN ({placeholders})", ids
                )
                index = self._load_faiss_index(type_)
                faiss_index_path = self._get_faiss_index_path(type_)
                if index:
                    id_array = np.array(ids).astype("int64")
                    index.remove_ids(id_array)
                    faiss.write_index(index, faiss_index_path)
                    self.faiss_indices[type_] = index
            conn.commit()

    def _add_data(
        self,
        to_embed_paths: Dict[str, List[str]],
        embed_list: Dict[str, List[List[float]]],
    ) -> None:
        """
        Summary:
            주어진 파일 경로와 임베딩 목록에 따라 데이터베이스와 Faiss 인덱스에 데이터를 추가합니다.
        Args:
            to_embed_paths (Dict[str, List[str]]): 임베딩할 파일 경로 목록
            embed_list (Dict[str, List[List[float]]]): 파일 경로에 해당하는 임베딩 벡터 목록
        """
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()

            for type_, paths in to_embed_paths.items():
                embeddings = embed_list.get(type_, [])
                if not embeddings:
                    continue
                faiss_index_path = self._get_faiss_index_path(type_)
                dim = len(embeddings[0])
                index = self._load_faiss_index(type_, dim)
                new_ids = []

                for path in paths:
                    mtime = os.path.getmtime(path)
                    extension = os.path.splitext(path)[1].lower()
                    cursor.execute(
                        "INSERT INTO metadata (file_path, mtime, extension, type) VALUES (?, ?, ?, ?)",
                        (path, mtime, extension, type_),
                    )
                    new_ids.append(cursor.lastrowid)

                vectors_np = np.array(embeddings).astype("float32")

                faiss.normalize_L2(vectors_np)
                ids_np = np.array(new_ids).astype("int64")
                index.add_with_ids(vectors_np, ids_np)
                faiss.write_index(index, faiss_index_path)
                self.faiss_indices[type_] = index

            conn.commit()

    def sync_vector_store(
        self,
        local_files: Dict[str, List[str]],
    ) -> None:
        """
        Summary:
            주어진 디렉토리와 확장자 목록에 따라 파일들을 임베딩하고 벡터 스토어에 저장합니다.
        Args:
            local_files: (Dict[str, List[str]]): 임베딩할 파일들의 전체 경로 리스트
            예: {'image': ['/path/to/image1.jpg', '/path/to/image2.png'], 'text': ['/path/to/doc1.txt']}
        """
        db_files = self._fetch_db_metadata()
        to_delete_ids = defaultdict(
            list
        )  # db에서 삭제할 것: 로컬에서 삭제된 것, 수정된 것, faiss에서 삭제하기 위해 faiss index(id)를 담음
        to_embed_paths = defaultdict(
            list
        )  # 임베딩 해야할 것: 로컬에서 수정된 것, 새로 생긴 것. 임배딩 생성을 위해 path를 담음

        for type, files in db_files.items():
            for path, info in files.items():
                if path not in local_files.get(type, []):
                    to_delete_ids[type].append(info["id"])
                elif info["mtime"] != os.path.getmtime(path):
                    to_delete_ids[type].append(info["id"])
                    to_embed_paths[type].append(path)

        for type_, paths in local_files.items():
            to_embed_paths[type_].extend(
                list(set(paths) - set(db_files.get(type_, {}).keys()))
            )

        self._remove_data(to_delete_ids)
        embed_list = self.encoder.create_embedding_list(
            dict(to_embed_paths), self.batch_size
        )
        self._add_data(dict(to_embed_paths), embed_list)

    def search(
        self,
        query: str,
        top_k: int,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        코사인 유사도를 통해 쿼리와 유사한 절대경로를 topk개 반환

        Args:
            query (str): 자연어 쿼리
            topk (int): 반환할 유사도 상위 개수

        Returns:
            Dict[str, List[Tuple[str, float]]]: 유사도 상위 topk개의 절대경로와 유사도 점수
            예: {'image': [('/path/to/similar_image1.jpg', 0.95), ...], 'text': [('/path/to/similar_doc1.txt', 0.89), ...]}
        """
        results = defaultdict(list)
        db_files = self._fetch_db_metadata()
        allowed_indices = {}

        for type_, paths_dict in db_files.items():
            type_id_map = {}
            for path, info in paths_dict.items():
                row_id = info["id"]
                type_id_map[row_id] = path
            allowed_indices[type_] = type_id_map

        for type_, id_map in allowed_indices.items():
            query_embedding = self.encoder.create_query_embedding(query, type_)
            query_vector = (
                np.array(query_embedding).astype("float32").reshape(1, -1)
            )
            faiss.normalize_L2(query_vector)

            index = self._load_faiss_index(type_)
            if index is None or index.ntotal == 0:
                continue
            id_array = np.array(list(id_map.keys())).astype("int64")
            selector = faiss.IDSelectorBatch(id_array)
            params = faiss.SearchParameters(sel=selector)

            D, I = index.search(query_vector, top_k, params=params)
            for idx, score in zip(I[0], D[0]):
                if idx == -1:
                    continue
                results[type_].append((id_map[idx], float(score)))

        return dict(results)
