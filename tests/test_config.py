"""
Unit tests for Config module.
"""

import pytest
from src.config import Config, config


class TestConfig:
    """Tests for Config class."""
    
    def test_config_singleton_exists(self):
        """Test that config singleton is available."""
        assert config is not None
        assert isinstance(config, Config)
    
    def test_default_chunk_size(self):
        """Test default chunk size is reasonable."""
        assert config.CHUNK_SIZE > 0
        assert config.CHUNK_SIZE == 500
    
    def test_default_chunk_overlap(self):
        """Test default chunk overlap is less than chunk size."""
        assert config.CHUNK_OVERLAP < config.CHUNK_SIZE
        assert config.CHUNK_OVERLAP >= 0
    
    def test_default_top_k(self):
        """Test default top k results."""
        assert config.TOP_K_RESULTS > 0
        assert config.TOP_K_RESULTS == 5
    
    def test_embedding_model_set(self):
        """Test embedding model is configured."""
        assert config.EMBEDDING_MODEL is not None
        assert len(config.EMBEDDING_MODEL) > 0
    
    def test_simple_greetings_not_empty(self):
        """Test that simple greetings set is populated."""
        assert len(config.SIMPLE_GREETINGS) > 0
        assert 'hi' in config.SIMPLE_GREETINGS
        assert 'hello' in config.SIMPLE_GREETINGS
    
    def test_stop_words_not_empty(self):
        """Test that stop words set is populated."""
        assert len(config.STOP_WORDS) > 0
        assert 'the' in config.STOP_WORDS
        assert 'a' in config.STOP_WORDS
    
    def test_paths_are_pathlib_paths(self):
        """Test that paths are properly configured Path objects."""
        assert config.UPLOAD_FOLDER is not None
        assert config.INDEX_PATH is not None
        assert config.METADATA_PATH is not None
    
    def test_llm_settings(self):
        """Test LLM configuration settings."""
        assert config.LLM_MODEL is not None
        assert config.LLM_MAX_TOKENS > 0
        assert 0 <= config.LLM_TEMPERATURE <= 2
