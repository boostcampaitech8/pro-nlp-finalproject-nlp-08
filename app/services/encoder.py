from typing import Dict, List, Optional, Tuple
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

    def create_emb_img(self, image_path: str) -> List[float]:
        """이미지 파일을 읽어 임베딩 벡터를 반환합니다.

        Args:
            image_path: 이미지 절대 경로

        Returns:
            임베딩 벡터
        """
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        inputs = self.processor(
            images=[image], return_tensors="pt", padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)

        image_emb = outputs.pooler_output
        embedding = image_emb.squeeze().cpu().tolist()
        return embedding

    def create_emb_txt_query(self, text: str) -> List[float]:
        """텍스트 쿼리를 읽어 임베팅 벡터를 반환합니다.

        Args:
            text: 텍스트 쿼리

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

    def create_emb_txt(self, text_path: str) -> List[float]:
        """텍스트 파일을 읽어 임베딩 벡터를 반환합니다.

        Args:
            text_path: 텍스트 절대 경로

        Returns:
            임베딩 벡터
        """
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        inputs = self.processor(
            text=[text], return_tensors="pt", padding="max_length"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)

        text_emb = outputs.pooler_output
        embedding = text_emb.squeeze().cpu().tolist()
        return embedding

    def create_emb_list(
        self, files_path: Dict[str, List[str]]
    ) -> Dict[str, List[List[float]]]:
        """입력으로 받은 모든 경로의 파일들의 임베딩을 반환합니다.

        Args:
            files_path: 절대 경로 파일 리스트

        Returns:
            임베딩된 결과값들
        """
        embedding_results = defaultdict(list)

        for type_, files in files_path.items():
            func = self.type_func_map.get(type_)
            if func is None:
                continue
            for file_path in tqdm(files, desc=f"Embedding {type_} files"):
                embedding = func(file_path)
                embedding_results[type_].append(embedding)

        return dict(embedding_results)
