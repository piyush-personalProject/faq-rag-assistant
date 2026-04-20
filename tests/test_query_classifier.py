"""
Unit tests for QueryClassifier module.
"""

import pytest
from src.query_classifier import QueryClassifier


class TestQueryClassifier:
    """Tests for QueryClassifier.is_simple_query()"""
    
    def test_greeting_hi_is_simple(self):
        """Test that 'hi' is classified as simple query."""
        is_simple, response = QueryClassifier.is_simple_query("hi")
        assert is_simple is True
        assert response is not None
        assert "Hello" in response
    
    def test_greeting_hello_is_simple(self):
        """Test that 'hello' is classified as simple query."""
        is_simple, response = QueryClassifier.is_simple_query("hello")
        assert is_simple is True
        assert response is not None
    
    def test_greeting_hey_is_simple(self):
        """Test that 'hey' is classified as simple query."""
        is_simple, response = QueryClassifier.is_simple_query("hey")
        assert is_simple is True
    
    def test_greeting_with_punctuation(self):
        """Test that 'hi!' is still classified as simple."""
        is_simple, response = QueryClassifier.is_simple_query("hi!")
        assert is_simple is True
    
    def test_help_is_simple(self):
        """Test that 'help' is classified as simple query."""
        is_simple, response = QueryClassifier.is_simple_query("help")
        assert is_simple is True
        assert "account & billing" in response
    
    def test_question_mark_is_simple(self):
        """Test that '?' is classified as simple query."""
        is_simple, response = QueryClassifier.is_simple_query("?")
        assert is_simple is True
    
    def test_what_is_simple(self):
        """Test that 'what' is classified as simple query."""
        is_simple, response = QueryClassifier.is_simple_query("what")
        assert is_simple is True
    
    def test_real_question_not_simple(self):
        """Test that a real question is NOT classified as simple."""
        is_simple, response = QueryClassifier.is_simple_query("How do I track my order?")
        assert is_simple is False
        assert response is None
    
    def test_billing_question_not_simple(self):
        """Test that billing questions are NOT simple."""
        is_simple, response = QueryClassifier.is_simple_query("What is my billing address?")
        assert is_simple is False
    
    def test_case_insensitive(self):
        """Test that classification is case insensitive."""
        is_simple, _ = QueryClassifier.is_simple_query("HI")
        assert is_simple is True
        
        is_simple, _ = QueryClassifier.is_simple_query("Hello!")
        assert is_simple is True
    
    def test_whitespace_handling(self):
        """Test that extra whitespace is handled."""
        is_simple, _ = QueryClassifier.is_simple_query("  hi  ")
        assert is_simple is True
