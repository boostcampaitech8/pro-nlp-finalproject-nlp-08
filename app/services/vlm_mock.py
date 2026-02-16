# VLM 목업: API 없이 이미지 → 텍스트 대체용
from typing import Optional


class MockVLM:
    """이미지를 텍스트로 변환하는 VLM API가 없을 때 사용하는 목업."""

    def describe_image(
        self,
        image_bytes: bytes,
        ext: str = "png",
        prompt: Optional[str] = None,
    ) -> str:
        """
        이미지 바이트를 받아 설명 텍스트를 반환합니다.
        목업이므로 고정 문구만 반환합니다.
        """
        return "[이미지/표/그래프 - VLM 설정 후 실제 설명으로 대체됩니다]"
