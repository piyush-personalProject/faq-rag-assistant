"""
Unit tests for RAG Engine module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from src.rag_engine import RAGEngine


class TestRAGEngine:
    """Tests for RAGEngine class."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock config for RAGEngine tests."""
        with patch('src.rag_engine.config') as mock:
            mock.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            mock.CHUNK_SIZE = 500
            mock.CHUNK_OVERLAP = 50
            mock.TOP_K_RESULTS = 5
            mock.INDEX_PATH = Path("embeddings/faiss.index")
            mock.EMBEDDINGS_DIR = Path("embeddings")
            yield mock
    
    def test_rag_engine_initialization(self, mock_config):
        """Test RAGEngine initializes embeddings and vectorstore."""
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss:
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 0
            mock_faiss.from_texts.return_value = mock_faiss_instance
            
            engine = RAGEngine()
            
            mock_emb.assert_called_once()
    
    def test_load_or_init_index_loads_existing(self, mock_config):
        """Test that _load_or_init_index loads existing index."""
        mock_config.INDEX_PATH.exists.return_value = True
        
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss:
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 100
            mock_faiss.load_local.return_value = mock_faiss_instance
            
            engine = RAGEngine()
            
            mock_faiss.load_local.assert_called_once()
    
    def test_load_or_init_index_creates_new(self, mock_config):
        """Test that _load_or_init_index creates new index when none exists."""
        mock_config.INDEX_PATH.exists.return_value = False
        mock_config.EMBEDDINGS_DIR.exists.return_value = False
        
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss:
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 0
            mock_faiss.from_texts.return_value = mock_faiss_instance
            
            engine = RAGEngine()
            
            mock_faiss.from_texts.assert_called()
    
    def test_chunk_text_splits_into_documents(self, mock_config):
        """Test _chunk_text creates Document objects from text."""
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss, \
             patch('src.rag_engine.RecursiveCharacterTextSplitter') as mock_splitter, \
             patch('src.rag_engine.Document') as mock_doc:
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 0
            mock_faiss.from_texts.return_value = mock_faiss_instance
            
            mock_splitter_instance = Mock()
            mock_splitter.return_value = mock_splitter_instance
            mock_splitter_instance.split_text.return_value = ["chunk1", "chunk2"]
            
            mock_doc.return_value = Mock()
            
            engine = RAGEngine()
            docs = engine._chunk_text("Long text to chunk", "test.txt")
            
            assert len(docs) == 2
    
    def test_ingest_folder_returns_error_when_no_files(self, mock_config):
        """Test ingest_folder returns error when no .txt files found."""
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss:
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 0
            mock_faiss.from_texts.return_value = mock_faiss_instance
            
            engine = RAGEngine()
            
            result = engine.ingest_folder("/nonexistent/path")
            
            assert result["status"] == "error"
            assert "No .txt files" in result["message"]
    
    def test_ingest_file_validates_extension(self, mock_config):
        """Test ingest_file rejects non-.txt files."""
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss:
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 0
            mock_faiss.from_texts.return_value = mock_faiss_instance
            
            engine = RAGEngine()
            
            result = engine.ingest_file("/path/to/file.pdf")
            
            assert result["status"] == "error"
            assert "not a .txt file" in result["message"]
    
    def test_retrieve_returns_empty_when_index_empty(self, mock_config):
        """Test retrieve returns empty list when index has no vectors."""
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss:
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 0
            mock_faiss.from_texts.return_value = mock_faiss_instance
            
            engine = RAGEngine()
            
            result = engine.retrieve("test query")
            
            assert result == []
    
    def test_retrieve_returns_results_with_scores(self, mock_config):
        """Test retrieve returns formatted results with scores."""
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss:
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 5
            mock_faiss.from_texts.return_value = mock_faiss_instance
            mock_faiss_instance.similarity_search_with_score.return_value = [
                (Mock(page_content="doc1", metadata={"source": "test.txt"}), 0.5),
                (Mock(page_content="doc2", metadata={"source": "test.txt"}), 0.3)
            ]
            
            engine = RAGEngine()
            results = engine.retrieve("test query", top_k=2)
            
            assert len(results) == 2
            assert results[0]["text"] == "doc1"
            assert results[0]["score"] == 0.5
            assert results[1]["text"] == "doc2"
            assert results[1]["score"] == 0.3
    
    def test_get_status_returns_correct_structure(self, mock_config):
        """Test get_status returns expected dict structure."""
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss:
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 10
            mock_faiss_instance.similarity_search.return_value = [
                Mock(metadata={"source": "file1.txt"}),
                Mock(metadata={"source": "file2.txt"})
            ]
            mock_faiss.from_texts.return_value = mock_faiss_instance
            
            engine = RAGEngine()
            status = engine.get_status()
            
            assert "total_chunks" in status
            assert "total_vectors" in status
            assert "sources" in status
            assert "embedding_model" in status
    
    def test_clear_index_resets_vectorstore(self, mock_config):
        """Test clear_index creates new empty vectorstore."""
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss:
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 100
            mock_faiss.from_texts.return_value = mock_faiss_instance
            
            engine = RAGEngine()
            engine.vectorstore = mock_faiss_instance
            
            result = engine.clear_index()
            
            assert result["status"] == "success"
            # Should have called from_texts to create new empty store