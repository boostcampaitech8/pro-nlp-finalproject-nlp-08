from typing import Dict, List, Optional, Tuple, Iterable
import os
import torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import snapshot_download

class SiglipEncoder:
    _cache: Dict[str, Tuple[AutoModel, AutoProcessor]] = {}

    def __init__(
        self,
        model_name: Optional[str] = "google/siglip2-so400m-patch16-naflex",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        project_root = Path(__file__).resolve().parents[2]
        self.cache_dir = project_root / "model" / "hf_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model, self.processor = self._load_model()
        self.type_func_map = {
            "image": self.create_emb_img,
            "text": self.create_emb_txt,
        }

    def _load_model(self) -> Tuple[AutoModel, AutoProcessor]:
        """모델을 model폴더에서 로드하거나 새로 로드하여 model폴더에 저장합니다."""
        if self.model_name in self._cache:
            return self._cache[self.model_name]
        
        org, name = self.model_name.split("/", 1)
        repo_dir = Path(self.cache_dir) / f"models--{org}--{name}"
        refs_main = repo_dir / "refs" / "main"
        
        snap: Path | None = None
        if refs_main.exists():
            commit = refs_main.read_text().strip()
            candidate = repo_dir / "snapshots" / commit
            if candidate.exists():
                snap = candidate

        if snap is None:
            snap_path = snapshot_download(
                repo_id=self.model_name,
                cache_dir=str(self.cache_dir),
                revision="main",
            )
            snap = Path(snap_path)
        
        model = AutoModel.from_pretrained(str(snap), local_files_only=True).to(self.device).eval()
        processor = AutoProcessor.from_pretrained(str(snap), local_files_only=True)
            
        self._cache[self.model_name] = (model, processor)
        return model, processor

    def _chuncks(self, iteralbe: Iterable, batch_size: int) -> Iterable[List]:
        """이터러블을 배치 크기 단위로 나눕니다."""
        iteralbe = list(iteralbe)
        for i in range(0, len(iteralbe), batch_size):
            yield iteralbe[i : i + batch_size]

    def create_emb_img(
        self, image_paths: List[str], batch_size: int
    ) -> List[List[float]]:
        """이미지 파일을 읽어 임베딩 벡터를 반환합니다.

        Args:
            image_path: 이미지 절대 경로

        Returns:
            임베딩 벡터
        """
        embeddings = []
        total_batches = (len(image_paths) + batch_size - 1) // batch_size

        for batch_paths in tqdm(
            self._chuncks(image_paths, batch_size),
            total=total_batches,
            desc="Embedding images",
        ):
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            images = [ImageOps.exif_transpose(image) for image in images]
            inputs = self.processor(
                images=images, return_tensors="pt", padding="max_length"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)

            image_emb = outputs.pooler_output
            embeddings.extend(image_emb.cpu().tolist())
        return embeddings

    def create_emb_txt(
        self, text_paths: List[str], batch_size: int
    ) -> List[List[float]]:
        """텍스트 파일들을 읽어 임베딩 벡터를 반환합니다.

        Args:
            texts(List[str]): 텍스트 쿼리 리스트
            batch_size(int): 배치 사이즈

        Returns:
            임베딩 벡터
        """
        if not text_paths:
            return []

        embedding: List[List[float]] = []
        total_batches = (len(text_paths) + batch_size - 1) // batch_size
        for batch_paths in tqdm(
            self._chuncks(text_paths, batch_size),
            total=total_batches,
            desc="Embedding texts",
        ):
            batch_texts: List[str] = []
            for text_path in batch_paths:
                with open(text_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    batch_texts.append(text)
            inputs = self.processor(
                text=batch_texts, return_tensors="pt", padding="max_length"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
            text_emb = outputs.pooler_output
            embedding.extend(text_emb.cpu().tolist())
        return embedding

    def create_emb_txt_query(self, text: str) -> List[float]:
        """텍스트를 읽어 임베딩 벡터를 반환합니다.

        Args:
            text: 텍스트 문자열

        Returns:
            임베딩 벡터
        """
        inputs = self.processor(
            text=[text], return_tensors="pt", padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)

        text_emb = outputs.pooler_output
        embedding = text_emb.squeeze().cpu().tolist()
        return embedding

    def create_emb_list(
        self,
        file_paths_dict: Dict[str, List[str]],
        batch_size: Optional[Dict[str, int]] = None,
    ) -> Dict[str, List[List[float]]]:
        """입력으로 받은 모든 경로의 파일들의 임베딩을 반환합니다.
        Args:
            file_paths_dict(Dict[str, List[str]]): 파일 경로 딕셔너리
            예: {"image": ["/path/to/image1.jpg", "/path/to/image2.png"],
                  "text": ["/path/to/text1.txt", "/path/to/text2.txt"]}
            batch_size(Optional[Dict[str, int]]): 배치 사이즈 딕셔너리
            예: {"image": 8, "text": 16}
        Returns:
            임베딩된 결과값들
        """
        DEFAULT_BATCH_SIZE = 8
        if batch_size is None:
            batch_size = {}

        embedding_results = defaultdict(list)

        for type_, files in file_paths_dict.items():
            func = self.type_func_map.get(type_)
            if func is None:
                continue
            embeddings = func(files, batch_size.get(type_, DEFAULT_BATCH_SIZE))
            embedding_results[type_] = embeddings

        return dict(embedding_results)
