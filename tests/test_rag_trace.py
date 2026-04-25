"""
Tests for RAG tracer module.
Tests document-to-answer lineage tracking and optional LangSmith integration.
"""

import pytest
import time
from src.rag_trace import (
    RAGTracer, 
    ChunkContribution, 
    GenerationTrace,
    get_tracer,
    configure_langsmith_tracing
)


class TestChunkContribution:
    """Test ChunkContribution dataclass."""

    def test_creation(self):
        """Test creating a chunk contribution."""
        chunk = ChunkContribution(
            chunk_id="chunk_0",
            source="returns_and_refunds.txt",
            text_preview="Items can be returned within 30 days...",
            score=0.85,
            used_in_generation=True,
            token_overlap=0.6
        )
        
        assert chunk.chunk_id == "chunk_0"
        assert chunk.source == "returns_and_refunds.txt"
        assert chunk.score == 0.85
        assert chunk.used_in_generation is True

    def test_asdict(self):
        """Test converting to dictionary."""
        chunk = ChunkContribution(
            chunk_id="chunk_0",
            source="test.txt",
            text_preview="Test content",
            score=0.9,
            used_in_generation=False,
            token_overlap=0.3
        )
        
        d = chunk.__dict__
        assert d["chunk_id"] == "chunk_0"
        assert d["score"] == 0.9


class TestGenerationTrace:
    """Test GenerationTrace dataclass."""

    def test_creation(self):
        """Test creating a generation trace."""
        trace = GenerationTrace(
            trace_id="test_123",
            timestamp="2024-01-01T00:00:00",
            query="How do I return an item?",
            answer="Items can be returned within 30 days.",
            chunks=[],
            evaluation_metrics={"faithfulness": 0.9},
            generation_time_ms=150.5,
            llm_model="gpt2-medium"
        )
        
        assert trace.trace_id == "test_123"
        assert trace.query == "How do I return an item?"
        assert trace.generation_time_ms == 150.5

    def test_contributing_sources(self):
        """Test getting contributing sources."""
        trace = GenerationTrace(
            trace_id="test_456",
            timestamp="2024-01-01T00:00:00",
            query="Test query",
            answer="Test answer",
            chunks=[
                ChunkContribution("c1", "file1.txt", "text", 0.9, True, 0.5),
                ChunkContribution("c2", "file2.txt", "text", 0.8, False, 0.3),
                ChunkContribution("c3", "file1.txt", "text", 0.7, True, 0.4),
            ],
            evaluation_metrics={},
            generation_time_ms=100.0,
            llm_model="gpt2"
        )
        
        sources = trace.contributing_sources
        assert len(sources) == 2
        assert "file1.txt" in sources
        assert "file2.txt" in sources

    def test_to_dict(self):
        """Test serialization to dictionary."""
        trace = GenerationTrace(
            trace_id="test_789",
            timestamp="2024-01-01T00:00:00",
            query="Test",
            answer="Answer",
            chunks=[],
            evaluation_metrics={},
            generation_time_ms=50.0,
            llm_model="gpt2"
        )
        
        d = trace.to_dict()
        assert d["trace_id"] == "test_789"
        assert d["query"] == "Test"
        assert "chunks" in d

    def test_to_json(self):
        """Test JSON serialization."""
        trace = GenerationTrace(
            trace_id="test_json",
            timestamp="2024-01-01T00:00:00",
            query="Test",
            answer="Answer",
            chunks=[],
            evaluation_metrics={},
            generation_time_ms=50.0,
            llm_model="gpt2"
        )
        
        json_str = trace.to_json()
        assert "test_json" in json_str
        assert "Test" in json_str


class TestRAGTracer:
    """Test RAGTracer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tracer = RAGTracer(enable_file_tracing=False, enable_langsmith=False)

    def test_start_trace(self):
        """Test starting a new trace."""
        trace_id = self.tracer.start_trace("How do I return an item?")
        
        assert trace_id is not None
        assert trace_id.startswith("rag_")
        assert self.tracer._current_trace is not None
        assert self.tracer._current_trace.query == "How do I return an item?"

    def test_start_trace_with_custom_id(self):
        """Test starting trace with custom ID."""
        trace_id = self.tracer.start_trace("Test query", trace_id="custom_123")
        assert trace_id == "custom_123"

    def test_add_chunk(self):
        """Test adding chunks to trace."""
        self.tracer.start_trace("Test query")
        
        self.tracer.add_chunk(
            chunk_id="chunk_0",
            source="test.txt",
            text="Items can be returned within 30 days.",
            score=0.9,
            used=True,
            token_overlap=0.5
        )
        
        assert len(self.tracer._current_trace.chunks) == 1
        chunk = self.tracer._current_trace.chunks[0]
        assert chunk.chunk_id == "chunk_0"
        assert chunk.source == "test.txt"
        assert chunk.score == 0.9

    def test_add_chunk_no_active_trace(self):
        """Test adding chunk when no trace is active."""
        # Should not raise, just silently ignore
        self.tracer.add_chunk("chunk", "source", "text", 0.5)
        # No exception means success

    def test_mark_chunks_used(self):
        """Test marking chunks as used."""
        self.tracer.start_trace("Test")
        
        # Add some chunks
        for i in range(3):
            self.tracer.add_chunk(f"chunk_{i}", f"file{i}.txt", "text", 0.8)
        
        # Mark first and third as used
        self.tracer.mark_chunks_used(["chunk_0", "chunk_2"])
        
        chunks = self.tracer._current_trace.chunks
        assert chunks[0].used_in_generation is True
        assert chunks[1].used_in_generation is False
        assert chunks[2].used_in_generation is True

    def test_set_answer(self):
        """Test setting the final answer."""
        self.tracer.start_trace("Test query")
        self.tracer.set_answer("This is the final answer.")
        
        assert self.tracer._current_trace.answer == "This is the final answer."

    def test_set_evaluation(self):
        """Test setting evaluation metrics."""
        self.tracer.start_trace("Test")
        self.tracer.set_evaluation({
            "faithfulness": 0.9,
            "answer_relevance": 0.85,
            "context_precision": 0.8
        })
        
        assert self.tracer._current_trace.evaluation_metrics["faithfulness"] == 0.9

    def test_set_generation_time(self):
        """Test setting generation time."""
        self.tracer.start_trace("Test")
        self.tracer.set_generation_time(250.5)
        
        assert self.tracer._current_trace.generation_time_ms == 250.5

    def test_end_trace(self):
        """Test ending a trace."""
        self.tracer.start_trace("Test query")
        self.tracer.set_answer("Test answer")
        
        trace = self.tracer.end_trace()
        
        assert trace is not None
        assert trace.query == "Test query"
        assert trace.answer == "Test answer"
        assert self.tracer._current_trace is None  # Cleared

    def test_end_trace_no_active(self):
        """Test ending when no trace is active."""
        result = self.tracer.end_trace()
        assert result is None

    def test_get_trace(self):
        """Test retrieving a specific trace."""
        trace_id = self.tracer.start_trace("Query 1")
        self.tracer.set_answer("Answer 1")
        self.tracer.end_trace()
        
        # Start another trace
        self.tracer.start_trace("Query 2")
        self.tracer.set_answer("Answer 2")
        self.tracer.end_trace()
        
        # Retrieve first trace
        retrieved = self.tracer.get_trace(trace_id)
        assert retrieved is not None
        assert retrieved.query == "Query 1"

    def test_get_recent_traces(self):
        """Test getting recent traces."""
        for i in range(5):
            self.tracer.start_trace(f"Query {i}")
            self.tracer.set_answer(f"Answer {i}")
            self.tracer.end_trace()
        
        recent = self.tracer.get_recent_traces(limit=3)
        assert len(recent) == 3

    def test_get_traces_by_source(self):
        """Test filtering traces by source."""
        self.tracer.start_trace("Query")
        self.tracer.add_chunk("c1", "returns.txt", "text", 0.9, used=True)
        self.tracer.add_chunk("c2", "shipping.txt", "text", 0.8, used=False)
        self.tracer.set_answer("Answer")
        self.tracer.end_trace()
        
        returns_traces = self.tracer.get_traces_by_source("returns.txt")
        assert len(returns_traces) == 1
        
        shipping_traces = self.tracer.get_traces_by_source("shipping.txt")
        # Only traces where source contributed (used_in_generation=True)
        assert len(shipping_traces) == 0

    def test_get_trace_summary(self):
        """Test getting trace summary."""
        self.tracer.start_trace("Query 1")
        self.tracer.add_chunk("c1", "file1.txt", "text", 0.9, used=True)
        self.tracer.set_answer("Answer")
        self.tracer.set_evaluation({"faithfulness": 0.9})
        self.tracer.end_trace()
        
        summary = self.tracer.get_trace_summary()
        assert summary["total_traces"] == 1
        assert "file1.txt" in summary["sources_used"]
        assert "faithfulness" in summary["avg_evaluation_scores"]

    def test_clear_traces(self):
        """Test clearing in-memory traces."""
        self.tracer.start_trace("Query")
        self.tracer.set_answer("Answer")
        self.tracer.end_trace()
        
        assert len(self.tracer._traces) == 1
        
        self.tracer.clear_traces()
        
        assert len(self.tracer._traces) == 0

    def test_contributing_sources_excludes_unused(self):
        """Test that contributing_sources only includes used chunks."""
        trace = GenerationTrace(
            trace_id="test",
            timestamp="",
            query="",
            answer="",
            chunks=[
                ChunkContribution("c1", "file1.txt", "text", 0.9, used=True, overlap=0.5),
                ChunkContribution("c2", "file1.txt", "text", 0.8, used=False, overlap=0.3),
                ChunkContribution("c3", "file2.txt", "text", 0.7, used=False, overlap=0.2),
            ],
            evaluation_metrics={},
            generation_time_ms=100.0,
            llm_model=""
        )
        
        sources = trace.contributing_sources
        assert len(sources) == 1
        assert "file1.txt" in sources
        assert "file2.txt" not in sources  # Not used


class TestGetTracerSingleton:
    """Test singleton behavior."""

    def test_singleton_instance(self):
        """Test that get_tracer returns singleton."""
        tracer1 = get_tracer()
        tracer2 = get_tracer()
        
        assert tracer1 is tracer2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])