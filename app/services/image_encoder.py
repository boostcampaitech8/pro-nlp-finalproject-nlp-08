from typing import Dict, List, Tuple

import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from app.services.base_encoder import BaseEncoder


class ImageEncoder(BaseEncoder):
    _cache: Dict[str, Tuple[AutoModel, AutoProcessor]] = {}

    def __init__(
        self,
        model_name: str,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model, self.processor = self._load_model()

    def _load_model(self) -> Tuple[AutoModel, AutoProcessor]:
        """모델을 로컬 캐시 우선으로 로드하고, 없으면 다운로드합니다."""
        if self.model_name in self._cache:
            return self._cache[self.model_name]

        try:
            # 1) 로컬 캐시만으로 로드 시도
            model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
                local_files_only=True,
            )
            processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
                local_files_only=True,
            )
        except OSError:
            # 2) 필요 시 다운로드 허용
            model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
            )
            processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
            )

        model = model.to(self.device).eval()

        self._cache[self.model_name] = (model, processor)
        return model, processor

    def emb_file_list(
        self, file_paths: List[str], batch_size: int
    ) -> List[List[float]]:
        """이미지 파일을 읽어 임베딩 벡터를 반환합니다.

        Args:
            file_paths(List[str]): 이미지 절대 경로 리스트

        Returns:
            List[List[float]]: 임베딩 벡터
        """
        if not file_paths:
            return []

        embeddings: List[List[float]] = []
        total_batches = (len(file_paths) + batch_size - 1) // batch_size

        for batch_paths in tqdm(
            self._chunks(file_paths, batch_size),
            total=total_batches,
            desc="Embedding images",
        ):
            images = []
            for path in batch_paths:
                with Image.open(path) as img:
                    rgb = ImageOps.exif_transpose(img).convert("RGB")
                    images.append(rgb)
            inputs = self.processor(
                images=images, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.inference_mode():
                image_emb = self.model.get_image_features(**inputs)

            embeddings.extend(image_emb.cpu().tolist())
        return embeddings

    def emb_query(self, query: str) -> List[float]:
        """텍스트 쿼리를 이미지 검색용 임베딩으로 변환합니다."""
        if not query:
            return []

        inputs = self.processor(
            text=[query], return_tensors="pt", padding=True
        ).to(self.device)

        with torch.inference_mode():
            text_emb = self.model.get_text_features(**inputs)

        embedding = text_emb.squeeze().cpu().tolist()
        return embedding
