from typing import Dict, List

import torch
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from pathlib import Path

from app.services.base_encoder import BaseEncoder


class TextEncoder(BaseEncoder):
    _cache: Dict[str, AutoModel] = {}

    def __init__(
        self,
        model_name: str,
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name

        # 캐시 경로는 프로젝트 내부로 고정
        project_root = Path(__file__).resolve().parents[2]
        self.cache_dir = project_root / "model" / "hf_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = self._load_model()

    def _load_model(self) -> SentenceTransformer:
        """모델을 로컬 캐시 우선으로 로드하고, 없으면 다운로드합니다."""
        if self.model_name in self._cache:
            return self._cache[self.model_name]

        try:
            model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.cache_dir),
                device=self.device,
            )
        except Exception:
            model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.cache_dir),
                device=self.device,
            )


        model = model.to(self.device).eval()

        self._cache[self.model_name] = model
        return model

    def emb_file_list(
        self, file_paths: List[str], batch_size: int
    ) -> List[List[float]]:
        """텍스트 파일을 읽어 임베딩 벡터를 반환합니다.

        Args:
            file_paths: 텍스트 파일 절대 경로

        Returns:
            임베딩 벡터
        """

        embedding: List[List[float]] = []
        total_batches = (len(file_paths) + batch_size - 1) // batch_size
        for batch_paths in tqdm(
            self._chunks(file_paths, batch_size),
            total=total_batches,
            desc="Embedding texts",
        ):
            batch_texts: List[str] = []
            for text_path in batch_paths:
                with open(text_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    batch_texts.append(text)
            with torch.no_grad():
                embeddings = self.model.encode(
                    ['passage: '+ text for text in batch_texts]
                , normalize_embeddings=True)
            embedding.extend(embeddings.tolist())
        return embedding

    def emb_query(self, query: str) -> List[float]:
        """텍스트를 읽어 임베딩 벡터를 반환합니다.

        Args:
            query: 텍스트 문자열

        Returns:
            임베딩 벡터
        """
        with torch.no_grad():
            embeddings = self.model.encode(['query: '+ query], normalize_embeddings=True)
        return embeddings[0].squeeze().tolist()
