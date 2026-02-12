from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, List


class BaseEncoder(ABC):

    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        self.cache_dir = project_root / "model" / "hf_cache"

    def _chunks(self, iterable: Iterable, batch_size: int) -> Iterator[List]:
        """이터러블을 배치 크기 단위로 나눕니다."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        iterator = iter(iterable)
        while True:
            batch = list(islice(iterator, batch_size))
            if not batch:
                break
            yield batch

    @abstractmethod
    def emb_query(self, query: str) -> List[float]:
        """쿼리 문자열을 임베딩 벡터로 변환합니다."""
        raise NotImplementedError()

    @abstractmethod
    def emb_file_list(
        self, file_paths: List[str], batch_size: int
    ) -> List[List[float]]:
        """파일 경로 리스트를 임베딩 벡터 리스트로 변환합니다."""
        raise NotImplementedError()
