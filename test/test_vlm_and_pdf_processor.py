"""
vlm_mock / vlm_clova / pdf_processor / encoder 동작 확인용 스크립트.

- VLM은 config에 따라 동작합니다.
  - vlm.provider: mock  → MockVLM 사용 (API 호출 없음)
  - vlm.provider: clova_studio → Clova Studio API 사용 (HCX-005 등)
- 대상 경로: test/test_dir_pdf/ 안의 PDF. 없으면 임시 PDF로 테스트.
- vlm_clova 사용 시: .env 의 CLOVA_STUDIO_API_KEY

실행 (프로젝트 루트에서):
  python test/test_vlm_clova_describe.py
"""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 테스트 실행 시에도 .env 로드 (vlm_clova 가 os.getenv("CLOVA_STUDIO_API_KEY") 사용)
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

# PDF 테스트 대상 디렉터리 (test_dir_pdf 안에 .pdf 넣으면 그걸로 테스트)
TEST_DIR_PDF = ROOT / "test" / "test_dir_pdf"
# VLM describe_image 테스트용 이미지 (dog.png 또는 test_dir 내 jpg)
TEST_DIR = ROOT / "test"

import fitz
from omegaconf import OmegaConf

from app.services.pdf_processor import (
    extract_text_and_images,
    image_to_text_vlm,
    pdf_to_combined_text,
    _get_vlm_client,
)
from app.services.encoder import SiglipEncoder


def _get_test_image_path():
    """VLM describe_image 호출용 이미지 경로. 없으면 None."""
    dog = TEST_DIR / "graph.png"
    if dog.exists():
        return dog
    test_dir = TEST_DIR / "test_dir"
    if test_dir.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            imgs = sorted(test_dir.glob(ext))
            if imgs:
                return imgs[0]
    return None


def _get_test_pdf_path():
    """
    테스트에 쓸 PDF 경로 하나 반환.
    test_dir_pdf 안에 .pdf가 있으면 그중 하나, 없으면 임시 PDF 생성.
    Returns:
        (path: str|Path, is_temp: bool)  is_temp면 테스트 후 삭제해야 함.
    """
    if TEST_DIR_PDF.is_dir():
        pdfs = sorted(TEST_DIR_PDF.glob("*.pdf"))
        if pdfs:
            return pdfs[0], False
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        path = f.name
    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    page.insert_text((50, 50), "Test PDF for pipeline.", fontsize=12)
    doc.save(path)
    doc.close()
    return Path(path), True


def load_config_for_encoder():
    """config/config.yaml의 default를 그대로 사용. cfg 없으면 동작 안 함."""
    cfg_path = ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config 필요: {cfg_path}")
    raw = OmegaConf.load(cfg_path)
    # 앱과 동일: default 키 아래 설정 사용
    cfg = raw.get("default", raw)
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def test_extract_text_and_images():
    """test_dir_pdf 또는 임시 PDF로 extract_text_and_images() 동작 확인."""
    print("1. pdf_processor.extract_text_and_images 테스트")
    pdf_path, is_temp = _get_test_pdf_path()
    print(
        f"   대상: {pdf_path} (test_dir_pdf)"
        if not is_temp
        else f"   대상: 임시 PDF"
    )
    try:
        full_text, images = extract_text_and_images(
            str(pdf_path), min_image_pixels=100
        )
        assert isinstance(full_text, str)
        assert isinstance(images, list)
        print(
            f"   추출 텍스트 길이: {len(full_text)}, 이미지 개수: {len(images)}"
        )
        print(
            f"   텍스트 미리보기: {repr(full_text[:50]) if full_text else '(없음)'}"
        )
        print("   OK\n")
    finally:
        if is_temp:
            Path(pdf_path).unlink(missing_ok=True)


def test_pdf_to_combined_text():
    """test_dir_pdf 또는 임시 PDF + config cfg로 pdf_to_combined_text() 동작 확인."""
    print("2. pdf_processor.pdf_to_combined_text (config cfg) 테스트")
    cfg = load_config_for_encoder()
    pdf_path, is_temp = _get_test_pdf_path()
    print(
        f"   대상: {pdf_path} (test_dir_pdf)"
        if not is_temp
        else f"   대상: 임시 PDF"
    )
    try:
        combined = pdf_to_combined_text(str(pdf_path), cfg)
        assert isinstance(combined, str)
        print(f"   합친 텍스트 길이: {len(combined)}")
        print(
            f"   미리보기: {repr(combined[:80])}..."
            if len(combined) > 80
            else f"   미리보기: {repr(combined)}"
        )
        print("   OK\n")
    finally:
        if is_temp:
            Path(pdf_path).unlink(missing_ok=True)


def test_get_vlm_client():
    """config의 cfg로 _get_vlm_client()가 VLM 클라이언트를 반환하는지 확인."""
    print("3. pdf_processor._get_vlm_client (config cfg) 테스트")
    cfg = load_config_for_encoder()
    client = _get_vlm_client(cfg)
    assert client is not None
    assert hasattr(
        client, "describe_image"
    ), "VLM 클라이언트에 describe_image 필요"

    provider = getattr(getattr(cfg, "vlm", None), "provider", "mock") or "mock"
    if provider == "mock":
        out = client.describe_image(b"x", "png")
    else:
        # clova_studio: 실제 이미지로 describe_image 호출 (test/dog.png 또는 test_dir/*.jpg)
        img_path = _get_test_image_path()
        if img_path is not None:
            from PIL import Image

            pil_image = Image.open(img_path)
            out = client.describe_image(
                pil_image, ext=img_path.suffix.lstrip(".") or "png"
            )
            print(f"   이미지: {img_path.name}")
        else:
            out = client.describe_image(b"x", "png")
            print("   (테스트 이미지 없음, 더미 바이트 사용 → API 오류 가능)")

    assert isinstance(out, str)
    print(f"   client.describe_image() 반환: {repr(out[:50])}...")
    print("   OK\n")


def test_encoder_pdf():
    """test_dir_pdf 또는 임시 PDF로 PDF → encoder → 임베딩 벡터 전체 파이프라인 확인."""
    print(
        "4. encoder까지: PDF → pdf_to_combined_text + create_emb_txt_query 테스트"
    )
    pdf_path, is_temp = _get_test_pdf_path()
    print(
        f"   대상: {pdf_path} (test_dir_pdf)"
        if not is_temp
        else f"   대상: 임시 PDF"
    )
    try:
        cfg = load_config_for_encoder()
        model_name = str(
            getattr(cfg.model, "name", "google/siglip2-so400m-patch16-naflex")
        )
        encoder = SiglipEncoder(model_name=model_name)
        combined_text = pdf_to_combined_text(str(pdf_path), cfg)
        emb = encoder.create_emb_txt_query(combined_text)

        assert isinstance(emb, list), "임베딩은 list"
        assert len(emb) > 0, "임베딩 차원 > 0"
        assert all(isinstance(x, float) for x in emb), "원소는 float"
        print(f"   임베딩 차원: {len(emb)}")
        print(f"   샘플 값(앞 3개): {emb[:3]}")
        print("   OK\n")
    finally:
        if is_temp:
            Path(pdf_path).unlink(missing_ok=True)


def test_pdf_sample_text_image_embedding():
    """
    sample.pdf(또는 test_dir_pdf 내 PDF)로:
    1) 텍스트와 이미지 분리
    2) 텍스트 + Clova 이미지 설명 합치기
    3) 합친 문자열로 임베딩 생성 후 검증
    """
    print("5. sample.pdf: 텍스트·이미지 분리 → Clova 설명 합침 → 임베딩 검증")
    sample_pdf = TEST_DIR_PDF / "sample.pdf"
    pdf_path, is_temp = (
        (sample_pdf, False) if sample_pdf.exists() else _get_test_pdf_path()
    )
    print(f"   대상 PDF: {pdf_path}")

    try:
        cfg = load_config_for_encoder()
        vlm_cfg = getattr(cfg, "vlm", None)
        use_vlm = vlm_cfg and getattr(vlm_cfg, "enabled", True)
        min_px = (
            getattr(vlm_cfg, "min_image_pixels", 1000) or 1000
            if vlm_cfg
            else 1000
        )

        # 1) 텍스트와 이미지 분리
        full_text, images = extract_text_and_images(
            str(pdf_path), min_image_pixels=min_px
        )
        print(
            f"   [1] 텍스트 추출: {len(full_text)}자, 이미지: {len(images)}개"
        )

        # 2) 이미지 → VLM 설명 (config 기준: mock 또는 clova_studio)
        vlm_text = ""
        if use_vlm and images:
            vlm_client = _get_vlm_client(cfg)
            vlm_text = image_to_text_vlm(images, vlm_client)
            print(f"   [2] VLM 이미지 설명: {len(vlm_text)}자")
        else:
            print("   [2] VLM 비활성 또는 이미지 없음")

        # 3) 텍스트 + 이미지 설명 합치기 (pdf_to_combined_text와 동일 형식)
        if full_text and vlm_text:
            combined = (
                full_text.rstrip()
                + "\n\n[이미지·표·그래프 설명]\n\n"
                + vlm_text
            )
        elif vlm_text:
            combined = "[이미지·표·그래프 설명]\n\n" + vlm_text
        else:
            combined = full_text or " "
        print(f"   [3] 합친 문자열: {len(combined)}자")

        # 4) 임베딩 생성 및 검증
        model_name = str(
            getattr(cfg.model, "name", "google/siglip2-so400m-patch16-naflex")
        )
        encoder = SiglipEncoder(model_name=model_name)
        emb = encoder.create_emb_txt_query(combined)
        assert isinstance(emb, list), "임베딩은 list"
        assert len(emb) > 0, "임베딩 차원 > 0"
        assert all(isinstance(x, float) for x in emb), "원소는 float"

        print(f"   [4] 임베딩 차원: {len(emb)}, 샘플: {emb[:3]}")
        print("   OK — 텍스트·이미지 분리 후 합쳐서 임베딩 정상 동작\n")
    finally:
        if is_temp:
            Path(pdf_path).unlink(missing_ok=True)


def main():
    print("=" * 50)
    print("vlm_mock + pdf_processor + encoder 동작 확인")
    print("=" * 50)
    cfg = load_config_for_encoder()
    provider = getattr(getattr(cfg, "vlm", None), "provider", "mock") or "mock"
    print(f"VLM provider (config 기준): {provider}")
    print(f"PDF 대상 경로: {TEST_DIR_PDF}")
    print("  (1~4번: 여기에 .pdf 넣으면 그 파일로 테스트, 없으면 임시 PDF)")
    print(
        "  (3번 describe_image: test/graph.png 또는 test_dir 이미지, clova 시 env 사용)"
    )
    print(
        "  (5번 sample.pdf: 텍스트·이미지 분리 → VLM 설명 합침 → 임베딩 검증)\n"
    )
    try:
        test_extract_text_and_images()
        test_pdf_to_combined_text()
        test_get_vlm_client()
        test_encoder_pdf()
        test_pdf_sample_text_image_embedding()
        print("=" * 50)
        print("모든 확인 통과.")
        return 0
    except Exception as e:
        print(f"\n실패: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
