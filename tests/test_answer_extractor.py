"""
Unit tests for AnswerExtractor module.
"""

import pytest
from src.answer_extractor import AnswerExtractor


class TestAnswerExtractor:
    """Tests for AnswerExtractor.extract()"""
    
    def test_extract_from_simple_qa(self):
        """Test extraction from a simple Q&A chunk."""
        chunks = [{
            "text": "Q: How do I track my order?\nA: You can track your order using the tracking link sent to your email.",
            "source": "test.txt",
            "score": 0.5
        }]
        
        answer = AnswerExtractor.extract("How do I track my order?", chunks)
        assert "track" in answer.lower()
        assert "order" in answer.lower()
    
    def test_extract_returns_empty_for_no_match(self):
        """Test that extraction returns empty string when no good match."""
        chunks = [{
            "text": "Q: What is the weather today?\nA: The weather is sunny.",
            "source": "test.txt",
            "score": 0.5
        }]
        
        answer = AnswerExtractor.extract("How do I return an item?", chunks)
        # Should return empty since no keywords match
        assert answer == ""
    
    def test_extract_handles_multiple_qa_blocks(self):
        """Test extraction from chunk with multiple Q&A blocks."""
        chunks = [{
            "text": "Q: How do I track my order?\nA: Use the tracking link.\n\nQ: How do I return an item?\nA: Contact support for a return label.",
            "source": "test.txt",
            "score": 0.5
        }]
        
        answer = AnswerExtractor.extract("How do I return an item?", chunks)
        assert "return" in answer.lower() or "contact" in answer.lower()
    
    def test_extract_truncates_long_answers(self):
        """Test that very long answers are truncated."""
        long_answer = "This is a very long answer. " * 50
        chunks = [{
            "text": f"Q: What is your policy?\nA: {long_answer}",
            "source": "test.txt",
            "score": 0.5
        }]
        
        answer = AnswerExtractor.extract("What is your policy?", chunks)
        assert len(answer) <= 210  # ~200 chars + ellipsis
    
    def test_extract_skips_short_answers(self):
        """Test that very short answers are skipped."""
        chunks = [{
            "text": "Q: Hi?\nA: Hi!",
            "source": "test.txt",
            "score": 0.5
        }]
        
        answer = AnswerExtractor.extract("Hi", chunks)
        assert answer == ""  # Both question and answer too short


class TestAnswerExtractorFormat:
    """Tests for AnswerExtractor.format_chunks_response()"""

    def test_format_returns_clean_answer(self):
        """Test that formatted response returns clean answer without source prefix."""
        chunks = [{
            "text": "Some text content.",
            "source": "test_faq.txt",
            "score": 0.5
        }]

        response = AnswerExtractor.format_chunks_response(chunks)
        assert response == "Some text content."

    def test_format_truncates_long_text(self):
        """Test that long chunk text is truncated."""
        # Use text without period in first 200 chars to ensure truncation with ellipsis
        long_text = "This is a very long piece of text without any period for a while " * 200
        chunks = [{
            "text": long_text,
            "source": "test.txt",
            "score": 0.5
        }]

        response = AnswerExtractor.format_chunks_response(chunks)
        # Should have ellipsis for truncated text
        assert "..." in response

    def test_format_respects_max_chunks(self):
        """Test that only max_chunks are returned."""
        chunks = [
            {"text": f"Text {i}.", "source": "test.txt", "score": 0.5}
            for i in range(5)
        ]

        response = AnswerExtractor.format_chunks_response(chunks, max_chunks=2)
        # Should only contain text from first chunk
        assert response == "Text 0."
