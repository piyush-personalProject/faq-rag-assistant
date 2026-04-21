"""
Unit tests for LLM module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.llm import LLMManager, AnswerOutputParser


class TestAnswerOutputParser:
    """Tests for AnswerOutputParser class."""
    
    def test_parse_with_answer_marker(self):
        """Test parsing when 'Answer:' marker is present."""
        parser = AnswerOutputParser()
        
        text = "Some reasoning here. Answer: This is the answer."
        result = parser.parse(text)
        
        assert result == "This is the answer."
    
    def test_parse_without_answer_marker(self):
        """Test parsing when no marker is present."""
        parser = AnswerOutputParser()
        
        text = "This is just a response without marker."
        result = parser.parse(text)
        
        # Should return stripped text
        assert len(result) > 0
    
    def test_parse_takes_first_line(self):
        """Test that parser takes only first line after marker."""
        parser = AnswerOutputParser()
        
        text = "Answer: First line\nSecond line\nThird line"
        result = parser.parse(text)
        
        assert result == "First line"
    
    def test_parse_short_output_fallback(self):
        """Test fallback for short outputs without marker."""
        parser = AnswerOutputParser()
        
        text = "Short"
        result = parser.parse(text)
        
        # Short output without marker returns original
        assert result == "Short"
    
    def test_parser_type(self):
        """Test parser type property."""
        parser = AnswerOutputParser()
        
        assert parser._type == "answer_output_parser"


class TestLLMManager:
    """Tests for LLMManager class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        with patch('src.llm.config') as mock:
            mock.LLM_MODEL = "distilgpt2"
            mock.LLM_MAX_TOKENS = 150
            mock.LLM_TEMPERATURE = 0.3
            yield mock
    
    def test_llm_manager_initialization(self):
        """Test LLMManager initializes without error."""
        # Mock the pipeline creation to avoid actual model loading
        with patch('src.llm.HuggingFacePipeline') as mock_pipeline:
            mock_pipeline.from_model_id.return_value = Mock()
            
            with patch('src.llm.LLMChain') as mock_chain:
                mock_chain.return_value = Mock()
                
                manager = LLMManager()
                
                assert manager is not None
    
    def test_is_available_property(self):
        """Test is_available property when pipeline is None."""
        manager = LLMManager()
        manager.pipeline = None
        manager.chain = None
        
        assert manager.is_available is False
    
    def test_generate_without_chain(self):
        """Test generate returns None when chain is not initialized."""
        manager = LLMManager()
        manager.chain = None
        
        result = manager.generate("test prompt")
        
        assert result is None
    
    def test_generate_with_chain(self):
        """Test generate calls chain invoke."""
        mock_chain = Mock()
        mock_chain.invoke.return_value = {"text": "Generated text"}
        
        manager = LLMManager()
        manager.chain = mock_chain
        
        result = manager.generate("test prompt")
        
        assert result == "Generated text"
        mock_chain.invoke.assert_called_once()
    
    def test_generate_with_context(self):
        """Test generate_with_context method."""
        mock_chain = Mock()
        mock_chain.invoke.return_value = {"text": "Context-based answer"}
        
        manager = LLMManager()
        manager.chain = mock_chain
        
        result = manager.generate_with_context("context text", "question")
        
        assert result == "Context-based answer"
        mock_chain.invoke.assert_called_once()
    
    def test_generate_with_reasoning(self):
        """Test generate_with_reasoning method."""
        mock_chain = Mock()
        mock_chain.invoke.return_value = {"text": "Step 1: Think... Final Answer: Done"}
        
        manager = LLMManager()
        manager.chain = mock_chain
        
        result = manager.generate_with_reasoning("context", "question")
        
        assert result is not None
        mock_chain.invoke.assert_called_once()
    
    def test_extract_answer_from_reasoning_with_final_marker(self):
        """Test extract_answer_from_reasoning with 'Final Answer:' marker."""
        manager = LLMManager()
        
        text = "Step 1: Analysis\nStep 2: More thinking\nFinal Answer: The final answer here"
        result = manager.extract_answer_from_reasoning(text)
        
        assert result == "The final answer here"
    
    def test_extract_answer_from_reasoning_with_answer_marker(self):
        """Test extract_answer_from_reasoning with 'Answer:' marker."""
        manager = LLMManager()
        
        text = "Some reasoning\nAnswer: The answer is here"
        result = manager.extract_answer_from_reasoning(text)
        
        assert result == "The answer is here"
    
    def test_extract_answer_from_reasoning_no_marker(self):
        """Test extract_answer_from_reasoning when no marker present."""
        manager = LLMManager()
        
        text = "Just some reasoning text without markers"
        result = manager.extract_answer_from_reasoning(text)
        
        # Should return full text when no marker found
        assert result == text
    
    def test_build_rag_prompt(self):
        """Test build_rag_prompt method."""
        manager = LLMManager()
        
        context = "Context about returns."
        question = "How do I return?"
        
        prompt = manager.build_rag_prompt(context, question)
        
        assert "Context:" in prompt
        assert "Question:" in prompt
        assert context in prompt
        assert question in prompt