import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from app.services.encoder import SiglipEncoder


class TestSiglipEncoderUnit:
    """Mock을 사용한 단위 테스트"""

    @pytest.fixture
    def mock_encoder(self):
        """Mock된 SiglipEncoder 인스턴스 생성"""
        with patch.object(SiglipEncoder, "_load_model") as mock_load:
            mock_model = MagicMock()
            mock_processor = MagicMock()

            mock_output = MagicMock()
            mock_output.pooler_output = torch.randn(1, 1152)
            mock_model.get_image_features.return_value = mock_output
            mock_model.get_text_features.return_value = mock_output

            mock_processor.return_value = MagicMock()
            mock_processor.return_value.items.return_value = []

            mock_load.return_value = (mock_model, mock_processor)

            encoder = SiglipEncoder()
            return encoder

    @pytest.fixture
    def temp_image_file(self):
        """테스트용 임시 이미지 파일 생성"""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.new("RGB", (100, 100), color="red")
            img.save(f, format="JPEG")
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def temp_text_file(self):
        """테스트용 임시 텍스트 파일 생성"""
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w", encoding="utf-8"
        ) as f:
            f.write("This is a test text for embedding.")
            yield f.name
        os.unlink(f.name)

    def test_create_emb_img_returns_list(self, mock_encoder, temp_image_file):
        """create_emb_img가 List[float]를 반환하는지 테스트"""
        result = mock_encoder.create_emb_img(temp_image_file)

        assert isinstance(result, list)
        assert len(result) == 1152
        assert all(isinstance(x, float) for x in result)

    def test_create_emb_txt_returns_list(self, mock_encoder, temp_text_file):
        """create_emb_txt가 List[float]를 반환하는지 테스트"""
        result = mock_encoder.create_emb_txt(temp_text_file)

        assert isinstance(result, list)
        assert len(result) == 1152
        assert all(isinstance(x, float) for x in result)

    def test_create_emb_txt_query_returns_list(self, mock_encoder):
        """create_emb_txt_query가 List[float]를 반환하는지 테스트"""
        result = mock_encoder.create_emb_txt_query("A photo of a cat")

        assert isinstance(result, list)
        assert len(result) == 1152
        assert all(isinstance(x, float) for x in result)

    def test_create_emb_list_with_mixed_files(
        self, mock_encoder, temp_image_file, temp_text_file
    ):
        """create_emb_list가 이미지와 텍스트 파일을 모두 처리하는지 테스트"""
        files = [temp_image_file, temp_text_file]
        results = mock_encoder.create_emb_list(files)

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(emb, list) for emb in results)

    def test_create_emb_list_with_empty_list(self, mock_encoder):
        """빈 리스트 입력 시 빈 리스트를 반환하는지 테스트"""
        results = mock_encoder.create_emb_list([])

        assert results == []

    def test_create_emb_list_skips_unsupported_extensions(self, mock_encoder):
        """지원하지 않는 확장자는 건너뛰는지 테스트"""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"dummy content")
            unsupported_file = f.name

        try:
            results = mock_encoder.create_emb_list([unsupported_file])
            assert results == []
        finally:
            os.unlink(unsupported_file)

    def test_image_extensions_constant(self):
        """IMAGE_EXTENSIONS 상수가 올바르게 정의되어 있는지 테스트"""
        assert ".jpg" in SiglipEncoder.IMAGE_EXTENSIONS
        assert ".jpeg" in SiglipEncoder.IMAGE_EXTENSIONS
        assert ".png" in SiglipEncoder.IMAGE_EXTENSIONS

    def test_text_extensions_constant(self):
        """TEXT_EXTENSIONS 상수가 올바르게 정의되어 있는지 테스트"""
        assert ".txt" in SiglipEncoder.TEXT_EXTENSIONS
        assert ".md" in SiglipEncoder.TEXT_EXTENSIONS
        assert ".json" in SiglipEncoder.TEXT_EXTENSIONS
        assert ".csv" in SiglipEncoder.TEXT_EXTENSIONS


@pytest.mark.integration
class TestSiglipEncoderIntegration:
    """실제 모델을 사용한 통합 테스트 (GPU/시간 필요)"""

    @pytest.fixture(scope="class")
    def encoder(self):
        """실제 SiglipEncoder 인스턴스 생성"""
        return SiglipEncoder()

    @pytest.fixture
    def temp_image_file(self):
        """테스트용 임시 이미지 파일 생성"""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            img = Image.new("RGB", (224, 224), color="blue")
            img.save(f, format="JPEG")
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def temp_text_file(self):
        """테스트용 임시 텍스트 파일 생성"""
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w", encoding="utf-8"
        ) as f:
            f.write("A photo of a cat sitting on a couch.")
            yield f.name
        os.unlink(f.name)

    def test_real_image_embedding(self, encoder, temp_image_file):
        """실제 모델로 이미지 임베딩 테스트"""
        result = encoder.create_emb_img(temp_image_file)

        assert isinstance(result, list)
        assert len(result) == 1152

    def test_real_text_embedding(self, encoder, temp_text_file):
        """실제 모델로 텍스트 임베딩 테스트"""
        result = encoder.create_emb_txt(temp_text_file)

        assert isinstance(result, list)
        assert len(result) == 1152

    def test_real_text_query_embedding(self, encoder):
        """실제 모델로 텍스트 쿼리 임베딩 테스트"""
        result = encoder.create_emb_txt_query(
            "A photo of a cat sitting on a couch."
        )

        assert isinstance(result, list)
        assert len(result) == 1152

    def test_real_embedding_list(
        self, encoder, temp_image_file, temp_text_file
    ):
        """실제 모델로 리스트 임베딩 테스트"""
        files = [temp_image_file, temp_text_file]
        results = encoder.create_emb_list(files)

        assert len(results) == 2
        assert all(len(emb) == 1152 for emb in results)

    def test_model_caching(self, encoder):
        """모델이 캐시되는지 테스트"""
        assert encoder.model_name in SiglipEncoder._cache
