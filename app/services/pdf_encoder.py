from typing import List, Optional

import torch
from tqdm import tqdm

from app.services.base_encoder import BaseEncoder
from app.services.text_encoder import TextEncoder
from app.services.pdf_processor import pdf_to_combined_text


class PdfEncoder(BaseEncoder):
    """PDF 파일을 텍스트로 변환 후 임베딩하는 encoder.

    PDF에서 텍스트를 추출하고, 이미지/표/그래프는 VLM으로 캡셔닝하여
    텍스트로 변환한 뒤 TextEncoder로 임베딩합니다.
    """

    def __init__(
        self,
        text_encoder: TextEncoder,
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text_encoder = text_encoder

        # VLM 클라이언트는 처음 사용 시 초기화 (lazy loading)
        self._vlm_client = None

    def _get_vlm_client(self):
        """VLM 클라이언트를 반환합니다. 최초 호출 시 초기화됩니다."""
        if self._vlm_client is None and self.cfg is not None:
            from app.services.vlm_clova import ClovaStudioVLM
            from app.services.vlm_mock import MockVLM

            vlm_cfg = getattr(self.cfg, "vlm", None)
            provider = (
                (getattr(vlm_cfg, "provider", None) or "mock")
                if vlm_cfg
                else "mock"
            )
            if provider == "clova_studio":
                self._vlm_client = ClovaStudioVLM(self.cfg)
            else:
                self._vlm_client = MockVLM()
        return self._vlm_client

    def emb_file_list(
        self, file_paths: List[str], batch_size: int
    ) -> List[List[float]]:
        """PDF 파일들을 텍스트로 변환 후 임베딩합니다.

        Args:
            file_paths: PDF 파일 절대 경로 리스트
            batch_size: 배치 사이즈

        Returns:
            임베딩 벡터 리스트
        """
        if not file_paths:
            return []

        embeddings: List[List[float]] = []
        vlm_client = self._get_vlm_client()

        total_batches = (len(file_paths) + batch_size - 1) // batch_size

        for batch_paths in tqdm(
            self._chunks(file_paths, batch_size),
            total=total_batches,
            desc="Embedding PDFs",
        ):
            batch_texts: List[str] = []
            for pdf_path in batch_paths:
                # PDF를 텍스트로 변환 (이미지는 VLM으로 캡셔닝)
                combined_text = pdf_to_combined_text(
                    pdf_path=pdf_path,
                    cfg=self.cfg,
                    vlm_client=vlm_client,
                )
                batch_texts.append(combined_text if combined_text else "")

            # TextEncoder의 모델을 사용해서 임베딩
            with torch.no_grad():
                batch_embeddings = self.text_encoder.model.encode(
                    batch_texts, task="retrieval.passage"
                )
            embeddings.extend(batch_embeddings.tolist())

        return embeddings

    def emb_query(self, query: str) -> List[float]:
        """텍스트 쿼리를 임베딩합니다.

        PDF 검색 시 쿼리는 일반 텍스트이므로 TextEncoder를 그대로 사용합니다.

        Args:
            query: 텍스트 문자열

        Returns:
            임베딩 벡터
        """
        return self.text_encoder.emb_query(query)
