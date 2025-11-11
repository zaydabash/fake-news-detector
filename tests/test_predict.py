"""Tests for prediction functionality."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.predict.cli import FakeNewsPredictor


class TestFakeNewsPredictor:
    """Test fake news predictor."""

    @pytest.fixture
    def mock_predictor(self):
        """Create a mock predictor for testing."""
        predictor = Mock(spec=FakeNewsPredictor)
        predictor.predict_text.return_value = {
            "prediction": "Real",
            "confidence": 0.85,
            "probabilities": {"Real": 0.85, "Fake": 0.15},
            "processed_text": "processed text here",
        }
        return predictor

    def test_predict_text_structure(self, mock_predictor):
        """Test that predict_text returns expected structure."""
        result = mock_predictor.predict_text("test text")

        assert "prediction" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert isinstance(result["prediction"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["probabilities"], dict)

    def test_predict_batch(self, mock_predictor):
        """Test batch prediction functionality."""
        texts = ["text1", "text2", "text3"]
        results = []

        for i, text in enumerate(texts):
            result = mock_predictor.predict_text(text)
            result_copy = result.copy()
            result_copy["index"] = i
            results.append(result_copy)

        assert len(results) == len(texts)
        assert all("index" in result for result in results)
        # Verify indices are correct
        for i, result in enumerate(results):
            assert result["index"] == i

    @patch("src.predict.cli.joblib.load")
    @patch("src.predict.cli.TFIDFFeatureExtractor.load")
    def test_predictor_initialization(self, mock_vectorizer_load, mock_model_load):
        """Test predictor initialization."""
        mock_model = Mock()
        mock_vectorizer = Mock()
        mock_vectorizer.label_encoder.classes_ = ["Real", "Fake"]

        mock_model_load.return_value = mock_model
        mock_vectorizer_load.return_value = mock_vectorizer

        predictor = FakeNewsPredictor(Path("model.joblib"), Path("vectorizer.joblib"))

        assert predictor.model == mock_model
        assert predictor.vectorizer == mock_vectorizer
        assert predictor.class_names == ["Real", "Fake"]
