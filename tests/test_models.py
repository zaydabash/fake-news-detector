"""Tests for model training and evaluation."""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.train_niche import NicheModelTrainer


class TestNicheModelTrainer:
    """Test niche model trainer functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        texts = [
            "This is a real news article about politics",
            "Fake news spreads misinformation online",
            "Breaking news from reliable sources",
            "False claims about vaccines and health",
            "Verified information from experts",
            "Misleading headlines and clickbait",
        ]
        labels = ["scientific", "claim", "scientific", "claim", "scientific", "claim"]

        df = pd.DataFrame({"text": texts, "label": labels, "text_processed": texts})
        return df

    def test_initialization(self):
        """Test trainer initialization."""
        config = {"min_samples_per_class": 10, "max_features": 1000, "min_df": 2}
        trainer = NicheModelTrainer(niche="test_niche", config=config, random_state=42)
        assert trainer.niche == "test_niche"
        assert trainer.random_state == 42
        assert trainer.config == config

    @patch("src.models.train_niche.LogisticRegression")
    @patch("src.models.train_niche.LinearSVC")
    @patch("src.models.train_niche.cross_val_score")
    def test_train_models(self, mock_cv, mock_svm, mock_lr, sample_data):
        """Test model training."""
        config = {"min_samples_per_class": 10, "max_features": 1000, "min_df": 2}

        mock_lr_model = Mock()
        mock_lr_model.fit.return_value = None
        mock_lr_model.score.return_value = 0.95
        mock_lr_model.predict.return_value = [0, 1, 0, 1, 0, 1]
        mock_lr.return_value = mock_lr_model

        mock_svm_model = Mock()
        mock_svm_model.fit.return_value = None
        mock_svm_model.score.return_value = 0.93
        mock_svm_model.predict.return_value = [0, 1, 0, 1, 0, 1]
        mock_svm.return_value = mock_svm_model

        # Mock cross-validation scores
        mock_cv.return_value = np.array([0.9, 0.92, 0.91, 0.89, 0.93])

        trainer = NicheModelTrainer(niche="test_niche", config=config, random_state=42)

        X_train = np.random.rand(6, 10)
        y_train = np.array([0, 1, 0, 1, 0, 1])
        X_val = np.random.rand(2, 10)
        y_val = np.array([0, 1])

        trainer.train_models(X_train, y_train, X_val, y_val)

        assert "logistic_regression" in trainer.models
        assert "svm" in trainer.models
        assert "logistic_regression" in trainer.metrics
        assert "svm" in trainer.metrics
        assert trainer.best_model is not None
