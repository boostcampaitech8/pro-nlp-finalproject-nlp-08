# Clova Studio VLM API (HCX-005) - 이미지 → 텍스트
# OpenAI 클라이언트로 v1/openai 호환 엔드포인트 호출 (image_url + data URI)
# 이미지 제한: 비율 1:5~5:1, 긴 변 ≤2240px, 짧은 변 ≥4px (에러 40063 방지)
import logging
import base64
from typing import Optional
import io
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

_CLOVA_MAX_SIDE = 2240
_CLOVA_MIN_SIDE = 4
_CLOVA_MAX_RATIO = 4.9  

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

    @staticmethod
    def _resize_for_clova(image) -> "Image":
        """
        Clova API 제한에 맞게 리사이즈: 비율 1:5~5:1, 긴 변 ≤2240, 짧은 변 ≥4.
        PIL.Image를 받아 새 Image 반환 (RGB). 극단적 비율은 캔버스에 맞춰 패딩.
        PDF 추출 이미지(가로로 긴 표 등)에서 40063 방지.
        """
        try:
            from PIL import Image, ImageOps
        except ImportError:
            return image
        if not hasattr(image, "resize"):
            return image

        img = image
        if hasattr(img, "convert"):
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            elif img.mode == "L":
                img = img.convert("RGB")
        img = ImageOps.exif_transpose(img)
        if hasattr(img, "convert") and img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if w <= 0 or h <= 0:
            return img

        long_side = max(w, h)
        short_side = min(w, h)
        if long_side > _CLOVA_MAX_SIDE:
            scale = _CLOVA_MAX_SIDE / long_side
            w, h = round(w * scale), round(h * scale)
            long_side = _CLOVA_MAX_SIDE
            short_side = min(w, h)
        short_side = max(short_side, _CLOVA_MIN_SIDE)
        if short_side < long_side / _CLOVA_MAX_RATIO:
            short_side = max(
                int(long_side / _CLOVA_MAX_RATIO), _CLOVA_MIN_SIDE
            )

        if w >= h:
            tw, th = int(long_side), int(short_side)
        else:
            tw, th = int(short_side), int(long_side)
        tw, th = max(tw, _CLOVA_MIN_SIDE), max(th, _CLOVA_MIN_SIDE)
        # 전송 직전 비율 재검증 (API 경계 오류 방지)
        final_long, final_short = max(tw, th), min(tw, th)
        if final_short < final_long / _CLOVA_MAX_RATIO:
            final_short = max(
                int(final_long / _CLOVA_MAX_RATIO), _CLOVA_MIN_SIDE
            )
            if tw >= th:
                tw, th = final_long, final_short
            else:
                tw, th = final_short, final_long
        if (img.size[0], img.size[1]) == (tw, th):
            return img.convert("RGB") if img.mode != "RGB" else img
        scale = min(tw / w, th / h)
        nw = max(_CLOVA_MIN_SIDE, round(w * scale))
        nh = max(_CLOVA_MIN_SIDE, round(h * scale))
        resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
        if nw < tw or nh < th:
            canvas = Image.new("RGB", (tw, th), (255, 255, 255))
            canvas.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
            resized = canvas
        # 최종 검증: 보낼 이미지가 제한 내인지
        rw, rh = resized.size
        if rw > _CLOVA_MAX_SIDE or rh > _CLOVA_MAX_SIDE:
            scale2 = _CLOVA_MAX_SIDE / max(rw, rh)
            resized = resized.resize(
                (
                    max(_CLOVA_MIN_SIDE, round(rw * scale2)),
                    max(_CLOVA_MIN_SIDE, round(rh * scale2)),
                ),
                Image.Resampling.LANCZOS,
            )
        rw, rh = resized.size
        ratio = max(rw, rh) / max(min(rw, rh), 1)
        if ratio > _CLOVA_MAX_RATIO:
            target_short = max(
                int(max(rw, rh) / _CLOVA_MAX_RATIO), _CLOVA_MIN_SIDE
            )
            if rw >= rh:
                canvas = Image.new("RGB", (rw, target_short), (255, 255, 255))
                canvas.paste(resized, (0, (target_short - rh) // 2))
            else:
                canvas = Image.new("RGB", (target_short, rh), (255, 255, 255))
                canvas.paste(resized, ((target_short - rw) // 2, 0))
            resized = canvas
        return resized

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
            image = self._resize_for_clova(image)
        else:
            try:
                from PIL import Image

                pil = Image.open(io.BytesIO(image)).convert("RGB")
                image = self._resize_for_clova(pil)
            except Exception as e:
                logger.warning(
                    "describe_image: bytes to PIL failed, sending as-is: %s", e
                )
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