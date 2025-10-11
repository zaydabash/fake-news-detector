"""Tests for data processing."""

import pytest
import pandas as pd
from pathlib import Path

from src.data.clean import clean_text, tokenize_text


class TestDataCleaning:
    """Test data cleaning functions."""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "This is a TEST text with UPPERCASE letters!"
        cleaned = clean_text(text)
        assert cleaned == "this is a test text with uppercase letters!"
    
    def test_clean_text_url_removal(self):
        """Test URL removal."""
        text = "Check out this website https://example.com for more info"
        cleaned = clean_text(text)
        assert "https://example.com" not in cleaned
        assert "website" in cleaned
    
    def test_clean_text_email_removal(self):
        """Test email removal."""
        text = "Contact us at test@example.com for support"
        cleaned = clean_text(text)
        assert "test@example.com" not in cleaned
        assert "contact us" in cleaned
    
    def test_clean_text_phone_removal(self):
        """Test phone number removal."""
        text = "Call us at (555) 123-4567 for assistance"
        cleaned = clean_text(text)
        assert "(555) 123-4567" not in cleaned
        assert "call us" in cleaned
    
    def test_clean_text_special_chars(self):
        """Test special character removal."""
        text = "This has @#$% special characters!!!"
        cleaned = clean_text(text)
        assert "@#$%" not in cleaned
        assert "this has special characters" in cleaned
    
    def test_clean_text_empty_input(self):
        """Test handling of empty input."""
        assert clean_text("") == ""
        assert clean_text(None) == ""
    
    def test_tokenize_text_basic(self):
        """Test basic tokenization."""
        text = "This is a test sentence with some words"
        tokens = tokenize_text(text, remove_stopwords=False)
        assert len(tokens) > 0
        assert "this" in tokens
        assert "test" in tokens
    
    def test_tokenize_text_stopwords(self):
        """Test tokenization with stopword removal."""
        text = "This is a test sentence with some words"
        tokens = tokenize_text(text, remove_stopwords=True)
        
        # Should remove common stopwords
        assert "this" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens
        assert "test" in tokens
        assert "sentence" in tokens
    
    def test_tokenize_text_short_tokens(self):
        """Test removal of very short tokens."""
        text = "a b c test word"
        tokens = tokenize_text(text, remove_stopwords=False)
        
        # Should remove tokens shorter than 3 characters
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "test" in tokens
        assert "word" in tokens
    
    def test_tokenize_text_empty(self):
        """Test tokenization of empty text."""
        assert tokenize_text("") == []
        assert tokenize_text(None) == []
