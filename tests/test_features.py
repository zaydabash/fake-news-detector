"""Tests for feature extraction."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.features.tfidf import TFIDFFeatureExtractor


class TestTFIDFFeatureExtractor:
    """Test TF-IDF feature extractor."""
    
    def test_feature_extractor_initialization(self):
        """Test feature extractor initialization."""
        extractor = TFIDFFeatureExtractor(max_features=1000)
        assert extractor.max_features == 1000
        assert extractor.ngram_range == (1, 2)
        assert extractor.min_df == 3
        assert not extractor.is_fitted
    
    def test_fit_transform(self):
        """Test fit and transform functionality."""
        # Sample data
        texts = pd.Series([
            "This is a real news article about politics",
            "Fake news spreads misinformation online",
            "Breaking news from reliable sources",
            "False claims about vaccines and health"
        ])
        labels = pd.Series([0, 1, 0, 1])
        
        extractor = TFIDFFeatureExtractor(max_features=50, min_df=1)
        
        # Fit and transform
        X, y = extractor.fit_transform(texts, labels)
        
        # Check shapes
        assert X.shape[0] == len(texts)
        assert X.shape[1] <= 50  # max_features
        assert y.shape[0] == len(labels)
        assert extractor.is_fitted
        
        # Check that features are sparse matrix
        assert hasattr(X, 'toarray')
        
        # Check labels
        assert set(y) == {0, 1}
    
    def test_transform_without_fit(self):
        """Test that transform fails without fitting."""
        texts = pd.Series(["test text"])
        labels = pd.Series([0])
        
        extractor = TFIDFFeatureExtractor()
        
        with pytest.raises(ValueError, match="must be fitted"):
            extractor.transform(texts, labels)
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        texts = pd.Series([
            "real news article",
            "fake news story"
        ])
        labels = pd.Series([0, 1])
        
        extractor = TFIDFFeatureExtractor(max_features=10, min_df=1)
        extractor.fit(texts, labels)
        
        feature_names = extractor.get_feature_names()
        assert len(feature_names) > 0
        assert isinstance(feature_names, np.ndarray)
    
    def test_label_encoding(self):
        """Test label encoding and decoding."""
        texts = pd.Series(["text1", "text2"])
        labels = pd.Series(["real", "fake"])
        
        extractor = TFIDFFeatureExtractor(min_df=1)
        extractor.fit(texts, labels)
        
        # Test encoding
        encoded = extractor.label_encoder.transform(labels)
        assert len(encoded) == len(labels)
        
        # Test decoding
        decoded = extractor.inverse_transform_labels(encoded)
        assert list(decoded) == list(labels)
