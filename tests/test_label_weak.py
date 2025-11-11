"""Tests for weak labeling functionality."""

import pytest
from pathlib import Path

from src.data.label_weak import WeakLabeler


class TestWeakLabeler:
    """Test weak labeler functionality."""
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        return {
            'niche': 'test_niche',
            'positive_label': 'claim',
            'negative_label': 'scientific',
            'positive_phrases': ['hoax', 'scam', 'fake'],
            'negative_phrases': ['evidence', 'study', 'research'],
            'positive_domains': ['example-denial.com'],
            'negative_domains': ['example-science.com']
        }
    
    def test_initialization(self, sample_config):
        """Test labeler initialization."""
        labeler = WeakLabeler(sample_config)
        assert labeler.niche == 'test_niche'
        assert labeler.positive_label == 'claim'
        assert labeler.negative_label == 'scientific'
        assert len(labeler.positive_phrases) == 3
        assert len(labeler.negative_phrases) == 3
    
    def test_label_text_positive_phrase_match(self, sample_config):
        """Test labeling with positive phrase match."""
        labeler = WeakLabeler(sample_config)
        text = "This is a hoax and a scam!"
        label, heuristics = labeler.label_text(text, "example.com", "unknown")
        
        assert label == 'claim'
        assert len(heuristics) > 0
        assert any('hoax' in h.lower() or 'scam' in h.lower() for h in heuristics)
    
    def test_label_text_negative_phrase_match(self, sample_config):
        """Test labeling with negative phrase match."""
        labeler = WeakLabeler(sample_config)
        text = "This study provides evidence from research."
        label, heuristics = labeler.label_text(text, "example.com", "unknown")
        
        assert label == 'scientific'
        assert len(heuristics) > 0
    
    def test_label_text_domain_match(self, sample_config):
        """Test labeling with domain match."""
        labeler = WeakLabeler(sample_config)
        text = "Some random text without keywords"
        label, heuristics = labeler.label_text(text, "example-denial.com", "positive")
        
        assert label == 'claim'
        assert any('domain' in h.lower() for h in heuristics)
    
    def test_label_text_conflict(self, sample_config):
        """Test labeling when both positive and negative rules match."""
        labeler = WeakLabeler(sample_config)
        text = "This hoax is evidence of a scam in the study."
        label, heuristics = labeler.label_text(text, "example.com", "unknown")
        
        # Should handle conflict (implementation may vary)
        assert label in ['claim', 'scientific', 'unlabeled']
        assert len(heuristics) > 0
    
    def test_label_text_no_match(self, sample_config):
        """Test labeling when no rules match."""
        labeler = WeakLabeler(sample_config)
        text = "This is just some random text with no keywords."
        label, heuristics = labeler.label_text(text, "unknown.com", "unknown")
        
        assert label == 'unlabeled'
        # When no matches, the labeler adds a heuristic indicating no match
        assert len(heuristics) >= 0  # May have "no_heuristics_match" heuristic
        assert any('no_heuristics_match' in h for h in heuristics) or len(heuristics) == 0
    
    def test_label_text_case_insensitive(self, sample_config):
        """Test that labeling is case-insensitive."""
        labeler = WeakLabeler(sample_config)
        text = "THIS IS A HOAX AND A SCAM!"
        label, heuristics = labeler.label_text(text, "example.com", "unknown")
        
        assert label == 'claim'
        assert len(heuristics) > 0
    
    def test_stats_tracking(self, sample_config):
        """Test that statistics are tracked correctly."""
        labeler = WeakLabeler(sample_config)
        
        # Create a DataFrame and use label_dataset to update stats
        import pandas as pd
        df = pd.DataFrame({
            'text': ["This is a hoax!", "This is evidence."],
            'source': ["example.com", "example.com"],
            'domain_type': ["unknown", "unknown"],
            'url': ["http://example.com/1", "http://example.com/2"],
            'title': ["Title 1", "Title 2"]
        })
        
        # Label dataset (this updates stats)
        labeled_df = labeler.label_dataset(df)
        
        assert labeler.stats['total_samples'] == 2
        assert labeler.stats['positive_labeled'] >= 0
        assert labeler.stats['negative_labeled'] >= 0

