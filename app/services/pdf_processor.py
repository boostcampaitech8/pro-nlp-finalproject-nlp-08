# PDF 처리: PyMuPDF(fitz)로 텍스트 추출 + 이미지/표/그래프는 VLM으로 텍스트화
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image
import io

from app.services.vlm_clova import ClovaStudioVLM
from app.services.vlm_mock import MockVLM


def _get_vlm_client(cfg):
    vlm_cfg = getattr(cfg, "vlm", None)
    provider = (
        (getattr(vlm_cfg, "provider", None) or "mock") if vlm_cfg else "mock"
    )
    if provider == "clova_studio":
        return ClovaStudioVLM(cfg)
    if provider == "mock":
        return MockVLM()
    raise ValueError(f"Unknown VLM provider: {provider}")


def extract_text_and_images(
    pdf_path: str,
    min_image_pixels: int = 1000,
) -> Tuple[str, List[Tuple[bytes, str]]]:
    """
    PDF에서 텍스트와 이미지(그래프/표 포함)를 추출합니다.

    Args:
        pdf_path: PDF 파일 절대 경로
        min_image_pixels: 최소 픽셀 수 (width*height). 이보다 작은 이미지는 스킵

    Returns:
        (전체_텍스트, [(이미지_bytes, 확장자), ...])
    """
    text_parts: List[str] = []
    images: List[Tuple[bytes, str]] = []

    doc = fitz.open(pdf_path)
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]

            page_text = page.get_text()
            if page_text.strip():
                text_parts.append(page_text)

            # extract_pdf_text_and_images.py와 동일: extract_image로 원본 바이트·확장자 반환
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                try:
                    base_img = doc.extract_image(xref)
                    img_bytes = base_img["image"]
                    ext = base_img["ext"]
                    w, h = base_img["width"], base_img["height"]
                    if w * h < min_image_pixels:
                        continue
                    if ext.lower() == "jpeg":
                        ext = "jpg"
                    images.append((img_bytes, ext))
                except Exception:
                    continue
    finally:
        doc.close()

    full_text = "\n\n".join(text_parts) if text_parts else ""
    return full_text, images


def image_to_text_vlm(
    images: List[Tuple[bytes, str]],
    vlm_client,
    prompt: str = "이 이미지(그래프, 표, 다이어그램 등)의 내용을 설명하는 텍스트로 요약해 주세요.",
) -> str:
    """
    이미지 리스트를 VLM에 넣어 각각 텍스트로 변환한 뒤 하나의 문자열로 합칩니다.
    bytes → PIL Image로 변환 후 describe_image에 넘깁니다 (PNG optimize 저장).
    """
    if not images or not vlm_client:
        return ""
    texts = []
    for img_bytes, _ in images:
        try:
            pil_image = Image.open(io.BytesIO(img_bytes))
            desc = vlm_client.describe_image(pil_image, prompt=prompt)
            if desc:
                texts.append(desc.strip())
        except Exception as e:
            texts.append(f"[이미지 설명 오류: {e}]")
    return "\n\n".join(texts)


def pdf_to_combined_text(
    pdf_path: str,
    cfg,
    vlm_client=None,
    min_image_pixels: Optional[int] = None,
) -> str:
    """
    PDF 경로를 받아:
    1) PyMuPDF로 추출한 텍스트
    2) 이미지/그래프/표는 VLM으로 설명 텍스트화
    두 부분을 합쳐 하나의 문자열로 반환합니다. 이 문자열을 텍스트 인코더에 넣어 임베딩합니다.

    Args:
        pdf_path: PDF 파일 절대 경로
        cfg: 앱 설정 (vlm, model 등)
        vlm_client: None이면 cfg에서 생성
        min_image_pixels: None이면 cfg.vlm.min_image_pixels 사용

    Returns:
        텍스트_추출본 + "\n\n" + VLM_이미지_설명 텍스트
    """
    vlm_cfg = getattr(cfg, "vlm", None)
    use_vlm = vlm_cfg and getattr(vlm_cfg, "enabled", True)
    min_px = min_image_pixels
    if min_px is None and vlm_cfg:
        min_px = getattr(vlm_cfg, "min_image_pixels", 1000) or 1000
    if min_px is None:
        min_px = 1000

    full_text, images = extract_text_and_images(
        pdf_path, min_image_pixels=min_px
    )

    vlm_text = ""
    if use_vlm and images:
        if vlm_client is None:
            vlm_client = _get_vlm_client(cfg)
        vlm_text = image_to_text_vlm(images, vlm_client)

    # 3가지 경우
    if full_text and vlm_text:
        return (
            full_text.rstrip() + "\n\n[이미지·표·그래프 설명]\n\n" + vlm_text
        )
    if vlm_text:
        return "[이미지·표·그래프 설명]\n\n" + vlm_text
    return full_text
