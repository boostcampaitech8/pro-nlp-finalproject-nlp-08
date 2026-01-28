from typing import Dict, List, Optional, Tuple
import os

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


class SiglipEncoder:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv"}
    _cache: Dict[str, Tuple[AutoModel, AutoProcessor]] = {}

    def __init__(
        self,
        model_name: str = "google/siglip2-so400m-patch16-naflex",
        cache_dir: Optional[str] = None,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "hub"
        )
        self.model, self.processor = self._load_model()

    def _load_model(self) -> Tuple[AutoModel, AutoProcessor]:
        """모델을 캐시에서 로드하거나 새로 로드하여 캐시에 저장합니다."""
        if self.model_name in self._cache:
            return self._cache[self.model_name]

        model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        ).to(self.device)

        processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )

        self._cache[self.model_name] = (model, processor)
        return model, processor

    def create_emb_img(self, image_path: str) -> List[float]:
        """이미지 파일을 읽어 임베딩 벡터를 반환합니다.

        Args:
            image_path: 이미지 절대 경로

        Returns:
            임베딩 벡터 (List[float])
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        embedding = image_features.pooler_output.squeeze().cpu().tolist()
        return embedding

    def create_emb_txt(self, text_path: str) -> List[float]:
        """텍스트 파일을 읽어 임베딩 벡터를 반환합니다.

        Args:
            text_path: 텍스트 절대 경로

        Returns:
            임베딩 벡터 (List[float])
        """
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(
            self.device
        )

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        embedding = text_features.pooler_output.squeeze().cpu().tolist()
        return embedding

    def create_emb_list(self, files_path: List[str]) -> List[List[float]]:
        """입력으로 받은 모든 경로의 파일들의 임베딩을 반환합니다.

        Args:
            files_path: 절대 경로 파일 리스트

        Returns:
            임베딩된 결과값들 (List[List[float]])
            e.g. [[-1,1,-1.3, 1.2], [-1.5,2.7, -2.8]]
        """
        embedding_results: List[List[float]] = []

        for file_path in files_path:
            ext = os.path.splitext(file_path)[1].lower()

            if ext in self.IMAGE_EXTENSIONS:
                embedding = self.create_emb_img(file_path)
            elif ext in self.TEXT_EXTENSIONS:
                embedding = self.create_emb_txt(file_path)
            else:
                continue

            embedding_results.append(embedding)

        return embedding_results
