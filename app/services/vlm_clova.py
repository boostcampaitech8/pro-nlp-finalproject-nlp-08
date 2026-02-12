# Clova Studio VLM API (HCX-005) - 이미지 → 텍스트
# OpenAI 클라이언트로 v1/openai 호환 엔드포인트 호출 (image_url + data URI)
import logging
import base64
from typing import Optional
import io
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

# MIME 확장자 매핑 (jpeg → jpg 등)
_EXT_TO_MIME = {
    "jpg": "jpeg",
    "jpeg": "jpeg",
    "png": "png",
    "webp": "webp",
    "bmp": "bmp",
}


class ClovaStudioVLM:
    """
    Clova Studio HCX-005 Vision API를 사용해 이미지를 텍스트로 변환합니다.
    OpenAI 클라이언트로 v1/openai 호환 엔드포인트를 호출합니다.
    - base_url: {base}/v1/openai
    - 이미지: OpenAI 형식 — type "image_url", image_url.url = "data:image/xxx;base64,..."
    - config: clova_studio.api_base_url, model_name, max_tokens
    - api_key: .env / 환경변수 CLOVA_STUDIO_API_KEY 사용
    """

    def __init__(self, cfg):
        vlm_cfg = getattr(cfg, "vlm", None) or {}
        clova = getattr(vlm_cfg, "clova_studio", None) or {}
        self.base_url = getattr(clova, "api_base_url", None)
        self.model_name = getattr(clova, "model_name", None)
        self.api_key = os.getenv("CLOVA_STUDIO_API_KEY")
        self.max_tokens = int(getattr(clova, "max_tokens", 100))

        self._client: Optional[OpenAI] = None
        logger.info(
            "ClovaStudioVLM init: base_url=%s model=%s api_key_set=%s max_tokens=%s",
            self.base_url,
            self.model_name,
            bool(self.api_key),
            self.max_tokens,
        )

    def _get_client(self) -> Optional[OpenAI]:
        if not self.base_url or not self.api_key:
            logger.warning(
                "ClovaStudioVLM: base_url or api_key missing, client=None"
            )
            return None
        if self._client is None:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            logger.debug("ClovaStudioVLM: OpenAI client created")
        return self._client

    def describe_image(
        self,
        image,
        ext: str = "png",
        prompt: Optional[str] = None,
    ) -> str:
        """
        이미지를 data URL Base64로 넣어 HCX-005 Chat Completions에 보냅니다.
        image: PIL.Image이면 buffer.save(format="PNG", optimize=True) 후 base64,
               bytes이면 그대로 base64 인코딩.
        """
        client = self._get_client()
        if client is None:
            logger.warning(
                "describe_image: client is None, returning error message"
            )
            return "[이미지 - .env 의 CLOVA_STUDIO_API_KEY 를 설정해 주세요]"

        if hasattr(image, "save"):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            b64 = base64.b64encode(buffer.getvalue()).decode()
            mime = "png"
            img_info = f"PIL {image.size[0]}x{image.size[1]}"
        else:
            b64 = base64.b64encode(image).decode("utf-8")
            mime = _EXT_TO_MIME.get((ext or "png").lower().strip(), "png")
            img_info = f"bytes len={len(image)}"
        data_url = f"data:image/{mime};base64,{b64}"
        logger.info(
            "describe_image: %s, data_uri_len=%d", img_info, len(data_url)
        )

        text_prompt = (
            prompt
            or "이 이미지(그래프, 표, 다이어그램 등)의 내용을 설명하는 텍스트로 요약해 주세요."
        )

        # v1/openai: OpenAI 형식 (type "image_url", image_url.url)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ]

        try:
            logger.debug(
                "describe_image: calling chat.completions.create model=%s",
                self.model_name,
            )
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            out = (
                content.strip()
                if isinstance(content, str) and content.strip()
                else (str(content) if content else "")
            )
            logger.info("describe_image: success, response_len=%d", len(out))
            return out
        except Exception as e:
            err_str = str(e)
            hint = ""
            if "40001" in err_str:
                hint = " (40001: 파라미터 오류)"
            elif "40060" in err_str:
                hint = " (40060: 지원 형식 BMP, PNG, JPG, JPEG, WEBP)"
            elif "40061" in err_str:
                hint = " (40061: 이미지 크기 0초과 20MB 이하)"
            elif "40063" in err_str:
                hint = " (40063: 비율/해상도 제한)"
            logger.exception("describe_image: API error %s%s", e, hint)
            return f"[VLM 오류: {e}]{hint}"
