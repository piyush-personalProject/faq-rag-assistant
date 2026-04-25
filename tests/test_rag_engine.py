"""
Unit tests for RAG Engine module.
Tests hybrid search (BM25 + semantic) and cross-encoder reranking.
"""

import pytest
import numpy as np
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
            # Hybrid search and reranking config
            mock.USE_HYBRID_SEARCH = True
            mock.BM25_WEIGHT = 0.3
            mock.SEMANTIC_WEIGHT = 0.7
            mock.USE_RERANKING = True
            mock.RERANKER_MODEL = "BAAI/bge-reranker-base"
            mock.RERANK_TOP_K = 10
            mock.USE_SEMANTIC_CHUNKING = False
            mock.SEMANTIC_SIMILARITY_THRESHOLD = 0.5
            yield mock
    
    @pytest.fixture
    def engine_with_mocks(self, mock_config):
        """Create RAGEngine with all dependencies mocked."""
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss, \
             patch('src.rag_engine.BM25Okapi') as mock_bm25:
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 0
            mock_faiss.from_texts.return_value = mock_faiss_instance
            
            mock_bm25_instance = Mock()
            mock_bm25.return_value = mock_bm25_instance
            
            engine = RAGEngine()
            return engine, mock_emb_instance, mock_faiss_instance, mock_bm25_instance
    
    def test_rag_engine_initialization(self, mock_config):
        """Test RAGEngine initializes embeddings, vectorstore, and BM25 index."""
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss:
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 0
            mock_faiss.from_texts.return_value = mock_faiss_instance
            
            engine = RAGEngine()
            
            mock_emb.assert_called_once()
            assert engine.bm25_index is None  # Not initialized until first ingest
    
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
        """Test ingest_folder returns error when no supported files found."""
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
            assert "No supported files" in result["message"]
    
    def test_ingest_file_validates_extension(self, mock_config):
        """Test ingest_file rejects unsupported file extensions."""
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss:
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 0
            mock_faiss.from_texts.return_value = mock_faiss_instance
            
            engine = RAGEngine()
            
            result = engine.ingest_file("/path/to/file.xyz")
            
            assert result["status"] == "error"
            assert "unsupported format" in result["message"]
    
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
        mock_config.USE_HYBRID_SEARCH = False  # Disable hybrid to test semantic only
        
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
        """Test get_status returns expected dict structure including hybrid/reranking flags."""
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
            assert "bm25_enabled" in status
            assert "reranking_enabled" in status
    
    def test_clear_index_resets_vectorstore_and_bm25(self, mock_config):
        """Test clear_index creates new empty vectorstore and clears BM25 index."""
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss:
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 100
            mock_faiss.from_texts.return_value = mock_faiss_instance
            
            engine = RAGEngine()
            engine.vectorstore = mock_faiss_instance
            engine.bm25_index = Mock()  # Simulate existing BM25 index
            engine._corpus_texts = ["text1", "text2"]
            engine._corpus_metadatas = [{"source": "f1"}, {"source": "f2"}]
            
            result = engine.clear_index()
            
            assert result["status"] == "success"
            assert engine.bm25_index is None
            assert engine._corpus_texts == []
            assert engine._corpus_metadatas == []


class TestBM25Search:
    """Tests for BM25 keyword search functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock config for BM25 tests."""
        with patch('src.rag_engine.config') as mock:
            mock.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            mock.CHUNK_SIZE = 500
            mock.CHUNK_OVERLAP = 50
            mock.TOP_K_RESULTS = 5
            mock.INDEX_PATH = Path("embeddings/faiss.index")
            mock.EMBEDDINGS_DIR = Path("embeddings")
            mock.USE_HYBRID_SEARCH = True
            mock.BM25_WEIGHT = 0.3
            mock.SEMANTIC_WEIGHT = 0.7
            mock.USE_RERANKING = False
            mock.USE_SEMANTIC_CHUNKING = False
            mock.SEMANTIC_SIMILARITY_THRESHOLD = 0.5
            yield mock
    
    def test_bm25_search_returns_ranked_results(self, mock_config):
        """Test BM25 search returns results ranked by keyword relevance."""
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss, \
             patch('src.rag_engine.BM25Okapi') as mock_bm25_class:
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 0
            mock_faiss.from_texts.return_value = mock_faiss_instance
            
            # Setup BM25 mock
            mock_bm25_instance = Mock()
            mock_bm25_class.return_value = mock_bm25_instance
            mock_bm25_instance.get_scores.return_value = np.array([2.5, 1.0, 0.5])
            
            engine = RAGEngine()
            engine._corpus_texts = ["apple banana cherry", "dog cat bird", "fish whale dolphin"]
            engine._corpus_metadatas = [{"source": "f1"}, {"source": "f2"}, {"source": "f3"}]
            engine.bm25_index = mock_bm25_instance
            
            results = engine._bm25_search("apple", top_k=3)
            
            assert len(results) == 3
            assert results[0][0] == "apple banana cherry"  # Highest score
    
    def test_bm25_search_returns_empty_when_no_matches(self, mock_config):
        """Test BM25 search returns empty when no keywords match."""
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss, \
             patch('src.rag_engine.BM25Okapi') as mock_bm25_class:
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 0
            mock_faiss.from_texts.return_value = mock_faiss_instance
            
            mock_bm25_instance = Mock()
            mock_bm25_class.return_value = mock_bm25_instance
            mock_bm25_instance.get_scores.return_value = np.array([0.0, 0.0, 0.0])
            
            engine = RAGEngine()
            engine._corpus_texts = ["apple banana", "dog cat", "fish bird"]
            engine._corpus_metadatas = [{"source": "f1"}, {"source": "f2"}, {"source": "f3"}]
            engine.bm25_index = mock_bm25_instance
            
            results = engine._bm25_search("xyz", top_k=3)
            
            assert len(results) == 0


class TestHybridSearch:
    """Tests for hybrid search (BM25 + semantic) functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock config for hybrid search tests."""
        with patch('src.rag_engine.config') as mock:
            mock.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            mock.CHUNK_SIZE = 500
            mock.CHUNK_OVERLAP = 50
            mock.TOP_K_RESULTS = 5
            mock.INDEX_PATH = Path("embeddings/faiss.index")
            mock.EMBEDDINGS_DIR = Path("embeddings")
            mock.USE_HYBRID_SEARCH = True
            mock.BM25_WEIGHT = 0.3
            mock.SEMANTIC_WEIGHT = 0.7
            mock.USE_RERANKING = False
            mock.USE_SEMANTIC_CHUNKING = False
            mock.SEMANTIC_SIMILARITY_THRESHOLD = 0.5
            yield mock
    
    def test_hybrid_search_combines_bm25_and_semantic(self, mock_config):
        """Test hybrid search combines BM25 and semantic scores with weights."""
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss, \
             patch('src.rag_engine.BM25Okapi') as mock_bm25_class:
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 3
            mock_faiss.from_texts.return_value = mock_faiss_instance
            mock_faiss_instance.similarity_search_with_score.return_value = [
                (Mock(page_content="doc1", metadata={"source": "f1"}), 0.2),
                (Mock(page_content="doc2", metadata={"source": "f2"}), 0.5),
                (Mock(page_content="doc3", metadata={"source": "f3"}), 0.8)
            ]
            
            mock_bm25_instance = Mock()
            mock_bm25_class.return_value = mock_bm25_instance
            mock_bm25_instance.get_scores.return_value = np.array([3.0, 1.0, 0.5])
            
            engine = RAGEngine()
            engine._corpus_texts = ["doc1", "doc2", "doc3"]
            engine._corpus_metadatas = [{"source": "f1"}, {"source": "f2"}, {"source": "f3"}]
            engine.bm25_index = mock_bm25_instance
            
            results = engine._hybrid_search("test query", top_k=3)
            
            assert len(results) == 3
            # Check that results have both bm25_score and semantic_score
            for r in results:
                assert "bm25_score" in r
                assert "semantic_score" in r
                assert "score" in r
    
    def test_hybrid_search_sorts_by_combined_score(self, mock_config):
        """Test hybrid search sorts results by combined weighted score."""
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss, \
             patch('src.rag_engine.BM25Okapi') as mock_bm25_class:
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 2
            mock_faiss.from_texts.return_value = mock_faiss_instance
            # semantic scores (distance-based, lower is better)
            mock_faiss_instance.similarity_search_with_score.return_value = [
                (Mock(page_content="high semantic", metadata={"source": "f1"}), 0.1),
                (Mock(page_content="low semantic", metadata={"source": "f2"}), 0.9)
            ]
            
            mock_bm25_instance = Mock()
            mock_bm25_class.return_value = mock_bm25_instance
            # BM25 scores
            mock_bm25_instance.get_scores.return_value = np.array([0.5, 3.0])
            
            engine = RAGEngine()
            engine._corpus_texts = ["high semantic", "low semantic"]
            engine._corpus_metadatas = [{"source": "f1"}, {"source": "f2"}]
            engine.bm25_index = mock_bm25_instance
            
            results = engine._hybrid_search("query", top_k=2)
            
            # With weights (0.3 BM25 + 0.7 semantic), semantic has more influence
            # First result should have higher combined score
            assert results[0]["score"] >= results[1]["score"]


class TestCrossEncoderReranking:
    """Tests for cross-encoder reranking functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock config for reranking tests."""
        with patch('src.rag_engine.config') as mock:
            mock.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            mock.CHUNK_SIZE = 500
            mock.CHUNK_OVERLAP = 50
            mock.TOP_K_RESULTS = 5
            mock.INDEX_PATH = Path("embeddings/faiss.index")
            mock.EMBEDDINGS_DIR = Path("embeddings")
            mock.USE_HYBRID_SEARCH = True
            mock.BM25_WEIGHT = 0.3
            mock.SEMANTIC_WEIGHT = 0.7
            mock.USE_RERANKING = True
            mock.RERANKER_MODEL = "BAAI/bge-reranker-base"
            mock.USE_SEMANTIC_CHUNKING = False
            mock.SEMANTIC_SIMILARITY_THRESHOLD = 0.5
            yield mock
    
    def test_rerank_results_reorders_by_cross_encoder_scores(self, mock_config):
        """Test reranking reorders results based on cross-encoder relevance scores."""
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss, \
             patch('src.rag_engine.BM25Okapi'), \
             patch('src.rag_engine.CrossEncoder') as mock_cross_encoder:
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 0
            mock_faiss.from_texts.return_value = mock_faiss_instance
            
            # Setup cross-encoder mock
            mock_reranker = Mock()
            mock_cross_encoder.return_value = mock_reranker
            # Return scores in different order than input
            mock_reranker.predict.return_value = np.array([0.95, 0.5, 0.8])
            
            engine = RAGEngine()
            
            results = [
                {"text": "doc1", "source": "f1", "score": 0.6},
                {"text": "doc2", "source": "f2", "score": 0.7},
                {"text": "doc3", "source": "f3", "score": 0.5}
            ]
            
            reranked = engine._rerank_results("test query", results)
            
            # First result should be doc1 (highest rerank score 0.95)
            assert reranked[0]["text"] == "doc1"
            assert reranked[0]["rerank_score"] == 0.95
            assert reranked[1]["text"] == "doc3"
            assert reranked[2]["text"] == "doc2"
    
    def test_rerank_results_returns_original_when_reranker_fails(self, mock_config):
        """Test reranking returns original results when cross-encoder fails."""
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss, \
             patch('src.rag_engine.BM25Okapi'), \
             patch('src.rag_engine.CrossEncoder') as mock_cross_encoder:
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 0
            mock_faiss.from_texts.return_value = mock_faiss_instance
            
            mock_cross_encoder.side_effect = Exception("Model load failed")
            
            engine = RAGEngine()
            
            results = [
                {"text": "doc1", "source": "f1", "score": 0.6},
                {"text": "doc2", "source": "f2", "score": 0.7}
            ]
            
            reranked = engine._rerank_results("test query", results)
            
            # Should return original results when reranker fails
            assert len(reranked) == 2
            assert reranked[0]["text"] == "doc1"
            assert reranked[1]["text"] == "doc2"
    
    def test_rerank_results_returns_empty_when_no_results(self, mock_config):
        """Test reranking returns empty list when given no results."""
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss, \
             patch('src.rag_engine.BM25Okapi'):
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 0
            mock_faiss.from_texts.return_value = mock_faiss_instance
            
            engine = RAGEngine()
            
            reranked = engine._rerank_results("test query", [])
            
            assert reranked == []


class TestRetrieveIntegration:
    """Integration tests for the full retrieve pipeline."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock config for integration tests."""
        with patch('src.rag_engine.config') as mock:
            mock.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
            mock.CHUNK_SIZE = 500
            mock.CHUNK_OVERLAP = 50
            mock.TOP_K_RESULTS = 5
            mock.INDEX_PATH = Path("embeddings/faiss.index")
            mock.EMBEDDINGS_DIR = Path("embeddings")
            mock.USE_HYBRID_SEARCH = True
            mock.BM25_WEIGHT = 0.3
            mock.SEMANTIC_WEIGHT = 0.7
            mock.USE_RERANKING = True
            mock.RERANKER_MODEL = "BAAI/bge-reranker-base"
            mock.USE_SEMANTIC_CHUNKING = False
            mock.SEMANTIC_SIMILARITY_THRESHOLD = 0.5
            yield mock
    
    def test_retrieve_uses_hybrid_search_and_reranking(self, mock_config):
        """Test retrieve() uses both hybrid search and reranking when enabled."""
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss, \
             patch('src.rag_engine.BM25Okapi') as mock_bm25_class, \
             patch('src.rag_engine.CrossEncoder') as mock_cross_encoder:
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 2
            mock_faiss.from_texts.return_value = mock_faiss_instance
            mock_faiss_instance.similarity_search_with_score.return_value = [
                (Mock(page_content="doc1", metadata={"source": "f1"}), 0.3),
                (Mock(page_content="doc2", metadata={"source": "f2"}), 0.6)
            ]
            
            mock_bm25_instance = Mock()
            mock_bm25_class.return_value = mock_bm25_instance
            mock_bm25_instance.get_scores.return_value = np.array([2.0, 0.5])
            
            mock_reranker = Mock()
            mock_cross_encoder.return_value = mock_reranker
            mock_reranker.predict.return_value = np.array([0.9, 0.6])
            
            engine = RAGEngine()
            engine._corpus_texts = ["doc1", "doc2"]
            engine._corpus_metadatas = [{"source": "f1"}, {"source": "f2"}]
            engine.bm25_index = mock_bm25_instance
            
            results = engine.retrieve("test query", top_k=2)
            
            assert len(results) == 2
            assert "rerank_score" in results[0]
    
    def test_retrieve_respects_top_k_limit(self, mock_config):
        """Test retrieve() respects top_k parameter."""
        mock_config.USE_HYBRID_SEARCH = False
        mock_config.USE_RERANKING = False
        
        with patch('src.rag_engine.HuggingFaceEmbeddings') as mock_emb, \
             patch('src.rag_engine.FAISS') as mock_faiss:
            
            mock_emb_instance = Mock()
            mock_emb.return_value = mock_emb_instance
            
            mock_faiss_instance = Mock()
            mock_faiss_instance.index.ntotal = 10
            mock_faiss.from_texts.return_value = mock_faiss_instance
            mock_faiss_instance.similarity_search_with_score.return_value = [
                (Mock(page_content=f"doc{i}", metadata={"source": "f{i}"}), i * 0.1)
                for i in range(1, 6)
            ]
            
            engine = RAGEngine()
            
            results = engine.retrieve("test query", top_k=3)
            
            assert len(results) == 3
            assert results[0]["text"] == "doc1"
            assert results[0]["score"] == 0.6
            assert results[1]["text"] == "doc2"
            assert results[1]["score"] == 0.7
    
    def test_get_status_returns_correct_structure(self, mock_config):
