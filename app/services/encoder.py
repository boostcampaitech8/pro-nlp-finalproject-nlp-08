from collections import defaultdict
from typing import Dict, List, Optional, Set

from app.services.base_encoder import BaseEncoder
from app.services.image_encoder import ImageEncoder
from app.services.pdf_encoder import PdfEncoder
from app.services.text_encoder import TextEncoder


class Encoder:

    def __init__(
        self,
        cfg,
        target_types: Optional[Set[str]] = None,
    ):
        """Encoder 초기화.
        
        Args:
            cfg: 앱 설정 (model_name, allowed_extensions, batch_size 등)
            target_types: 로딩할 타입 집합 (예: {"image", "text", "docs"})
                         None이면 cfg.allowed_extensions의 모든 타입 로딩
        """
        self.encoders: Dict[str, BaseEncoder] = {}
        self.cfg = cfg

        model_names = dict(cfg.model_name)

        # target_types가 None이면 cfg.allowed_extensions의 키들 사용
        if target_types is None:
            target_types = set(cfg.allowed_extensions.keys())

        # text/docs 중 하나라도 필요하면 TextEncoder 생성
        need_text_encoder = bool(target_types & {"text", "docs"})
        text_encoder: Optional[TextEncoder] = None

        if need_text_encoder:
            text_encoder = TextEncoder(model_name=model_names["text"])
            if "text" in target_types:
                self.encoders["text"] = text_encoder
            if "docs" in target_types:
                self.encoders["docs"] = PdfEncoder(text_encoder=text_encoder, cfg=cfg)
        
        if "image" in target_types:
            self.encoders["image"] = ImageEncoder(model_name=model_names["image"])

    def create_query_embedding(self, query: str, type_: str) -> List[float]:
        """타입(image/text)에 맞는 쿼리 임베딩을 생성합니다."""
        encoder = self.encoders.get(type_)
        if encoder is None:
            raise KeyError(f"Unsupported encoder type: {type_}")
        return encoder.emb_query(query)

    def create_embedding_list(
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
            예: {"image": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
                  "text": [[0.5, 0.6, ...], [0.7, 0.8, ...]]}
        """
        # batch_size가 None이면 cfg에서 가져옴
        if batch_size is None:
            batch_size = dict(self.cfg.batch_size)

        embedding_results = defaultdict(list)

        for type_, files in file_paths_dict.items():
            encoder = self.encoders.get(type_)
            if encoder:
                b_size = batch_size.get(type_, 1)
                embeddings = encoder.emb_file_list(files, batch_size=b_size)
                embedding_results[type_].extend(embeddings)
        return dict(embedding_results)
