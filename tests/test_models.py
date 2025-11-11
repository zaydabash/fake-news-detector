"""Tests for model training and evaluation."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.models.train_niche import NicheModelTrainer
from src.features.tfidf import TFIDFFeatureExtractor


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
            "Misleading headlines and clickbait"
        ]
        labels = ['scientific', 'claim', 'scientific', 'claim', 'scientific', 'claim']
        
        df = pd.DataFrame({
            'text': texts,
            'label': labels,
            'text_processed': texts
        })
        return df
    
    def test_initialization(self):
        """Test trainer initialization."""
        trainer = NicheModelTrainer(
            niche='test_niche',
            vectorizer=Mock(),
            random_state=42
        )
        assert trainer.niche == 'test_niche'
        assert trainer.random_state == 42
    
    @patch('src.models.train_niche.LogisticRegression')
    @patch('src.models.train_niche.LinearSVC')
    def test_train_models(self, mock_svm, mock_lr, sample_data):
        """Test model training."""
        mock_vectorizer = Mock()
        mock_vectorizer.transform.return_value = np.random.rand(6, 10)
        mock_vectorizer.label_encoder.transform.return_value = [0, 1, 0, 1, 0, 1]
        
        mock_lr_model = Mock()
        mock_lr_model.fit.return_value = None
        mock_lr_model.predict.return_value = [0, 1, 0, 1, 0, 1]
        mock_lr_model.predict_proba.return_value = np.array([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.9, 0.1],
            [0.2, 0.8],
            [0.85, 0.15],
            [0.25, 0.75]
        ])
        mock_lr.return_value = mock_lr_model
        
        mock_svm_model = Mock()
        mock_svm_model.fit.return_value = None
        mock_svm_model.predict.return_value = [0, 1, 0, 1, 0, 1]
        mock_svm.return_value = mock_svm_model
        
        trainer = NicheModelTrainer(
            niche='test_niche',
            vectorizer=mock_vectorizer,
            random_state=42
        )
        
        X_train = mock_vectorizer.transform(sample_data['text_processed'])
        y_train = mock_vectorizer.label_encoder.transform(sample_data['label'])
        X_test = X_train
        y_test = y_train
        
        models, metrics = trainer.train_models(X_train, y_train, X_test, y_test)
        
        assert 'logistic_regression' in models
        assert 'svm' in models
        assert 'accuracy' in metrics
        assert 'macro_f1' in metrics

