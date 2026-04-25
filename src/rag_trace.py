"""
Trace module for RAG document-to-answer lineage tracking.
Supports built-in logging and optional LangSmith integration.
"""

import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .config import config


@dataclass
class ChunkContribution:
    """Represents how a chunk contributed to the final answer."""
    chunk_id: str
    source: str
    text_preview: str  # First 100 chars of chunk
    score: float  # Relevance score
    used_in_generation: bool
    token_overlap: float  # How much this chunk contributed to answer


@dataclass
class GenerationTrace:
    """Complete trace of a RAG generation with document attribution."""
    trace_id: str
    timestamp: str
    query: str
    answer: str
    chunks: List[ChunkContribution]
    evaluation_metrics: Dict[str, float]
    generation_time_ms: float
    llm_model: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "query": self.query,
            "answer": self.answer,
            "chunks": [asdict(c) for c in self.chunks],
            "evaluation_metrics": self.evaluation_metrics,
            "generation_time_ms": self.generation_time_ms,
            "llm_model": self.llm_model
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @property
    def contributing_sources(self) -> List[str]:
        """Get list of unique sources that contributed to the answer."""
        return list({c.source for c in self.chunks if c.used_in_generation})


class RAGTracer:
    """
    Tracks document-to-answer lineage for RAG systems.
    
    Provides:
    - Built-in file-based trace logging
    - Optional LangSmith integration for cloud tracing
    - Query-level attribution showing which chunks led to which answer
    """

    def __init__(self, enable_file_tracing: bool = True, enable_langsmith: bool = False):
        self.enable_file_tracing = enable_file_tracing
        self.enable_langsmith = enable_langsmith and config.LANGSMITH_API_KEY
        
        self._traces: List[GenerationTrace] = []
        self._current_trace: Optional[GenerationTrace] = None
        
        # Lazy load LangSmith if enabled
        self._langsmith_client = None
        
        if self.enable_file_tracing:
            self._traces_dir = config.EMBEDDINGS_DIR / "traces"
            self._traces_dir.mkdir(parents=True, exist_ok=True)

    def _get_langsmith_client(self):
        """Lazy load LangSmith client."""
        if self._langsmith_client is None and self.enable_langsmith:
            try:
                from langsmith import Client
                self._langsmith_client = Client(api_key=config.LANGSMITH_API_KEY)
            except ImportError:
                print("[Trace] LangSmith not installed. Install with: pip install langsmith")
                self.enable_langsmith = False
            except Exception as e:
                print(f"[Trace] Failed to initialize LangSmith: {e}")
                self.enable_langsmith = False
        return self._langsmith_client

    def start_trace(self, query: str, trace_id: Optional[str] = None) -> str:
        """
        Start a new trace for a query.
        
        Args:
            query: The user's query
            trace_id: Optional custom trace ID (generated if not provided)
            
        Returns:
            The trace ID
        """
        if trace_id is None:
            trace_id = f"rag_{int(time.time() * 1000)}"
        
        self._current_trace = GenerationTrace(
            trace_id=trace_id,
            timestamp=datetime.utcnow().isoformat(),
            query=query,
            answer="",
            chunks=[],
            evaluation_metrics={},
            generation_time_ms=0.0,
            llm_model=config.LLM_MODEL
        )
        
        print(f"[Trace] Started trace {trace_id} for query: {query[:50]}...")
        return trace_id

    def add_chunk(
        self,
        chunk_id: str,
        source: str,
        text: str,
        score: float,
        used: bool = False,
        token_overlap: float = 0.0
    ) -> None:
        """
        Record a chunk's contribution to the current trace.
        
        Args:
            chunk_id: Unique identifier for the chunk
            source: Source document/file name
            text: Chunk text content (will be truncated to preview)
            score: Relevance score from retrieval
            used: Whether this chunk was used in generation
            token_overlap: Estimated overlap with final answer tokens
        """
        if self._current_trace is None:
            return
        
        preview = text[:100] + "..." if len(text) > 100 else text
        
        contribution = ChunkContribution(
            chunk_id=chunk_id,
            source=source,
            text_preview=preview,
            score=score,
            used_in_generation=used,
            token_overlap=token_overlap
        )
        
        self._current_trace.chunks.append(contribution)
        print(f"[Trace] Added chunk {chunk_id} from {source} (score={score:.3f}, used={used})")

    def mark_chunks_used(self, chunk_ids: List[str]) -> None:
        """
        Mark specific chunks as having been used in generation.
        
        Args:
            chunk_ids: List of chunk IDs that were passed to the LLM
        """
        if self._current_trace is None:
            return
        
        for chunk in self._current_trace.chunks:
            if chunk.chunk_id in chunk_ids:
                chunk.used_in_generation = True

    def set_answer(self, answer: str) -> None:
        """Set the final answer for the current trace."""
        if self._current_trace:
            self._current_trace.answer = answer

    def set_evaluation(self, metrics: Dict[str, float]) -> None:
        """Set evaluation metrics for the current trace."""
        if self._current_trace:
            self._current_trace.evaluation_metrics = metrics

    def set_generation_time(self, time_ms: float) -> None:
        """Set the generation time in milliseconds."""
        if self._current_trace:
            self._current_trace.generation_time_ms = time_ms

    def end_trace(self) -> Optional[GenerationTrace]:
        """
        Finalize and save the current trace.
        
        Returns:
            The completed GenerationTrace, or None if no trace was active
        """
        if self._current_trace is None:
            return None
        
        trace = self._current_trace
        
        print(f"[Trace] Ending trace {trace.trace_id} - query: {trace.query[:50]}..., chunks: {len(trace.chunks)}, answer length: {len(trace.answer)}")
        
        # Save to file
        if self.enable_file_tracing:
            self._save_trace_to_file(trace)
        
        # Send to LangSmith if enabled
        if self.enable_langsmith:
            self._send_to_langsmith(trace)
        
        # Store in memory
        self._traces.append(trace)
        
        # Clear current trace
        self._current_trace = None
        
        return trace

    def _save_trace_to_file(self, trace: GenerationTrace) -> None:
        """Save trace to JSON file."""
        try:
            filename = f"{trace.trace_id}.json"
            filepath = self._traces_dir / filename
            with open(filepath, "w") as f:
                f.write(trace.to_json())
        except Exception as e:
            print(f"[Trace] Failed to save trace to file: {e}")

    def _send_to_langsmith(self, trace: GenerationTrace) -> None:
        """Send trace to LangSmith for cloud tracing."""
        client = self._get_langsmith_client()
        if client is None:
            return
        
        try:
            # Create LangSmith run
            client.create_run(
                name=f"RAG Query: {trace.query[:50]}...",
                run_type="chain",
                inputs={
                    "query": trace.query,
                    "chunks": [c.text_preview for c in trace.chunks]
                },
                outputs={
                    "answer": trace.answer,
                    "contributing_sources": trace.contributing_sources
                },
                metadata={
                    "evaluation_metrics": trace.evaluation_metrics,
                    "generation_time_ms": trace.generation_time_ms,
                    "llm_model": trace.llm_model,
                    "chunk_count": len(trace.chunks),
                    "used_chunk_count": sum(1 for c in trace.chunks if c.used_in_generation)
                }
            )
        except Exception as e:
            print(f"[Trace] Failed to send trace to LangSmith: {e}")

    def get_trace(self, trace_id: str) -> Optional[GenerationTrace]:
        """Retrieve a specific trace by ID."""
        for trace in self._traces:
            if trace.trace_id == trace_id:
                return trace
        return None

    def get_recent_traces(self, limit: int = 10) -> List[GenerationTrace]:
        """Get the most recent traces."""
        return sorted(self._traces, key=lambda t: t.timestamp, reverse=True)[:limit]

    def get_traces_by_source(self, source: str) -> List[GenerationTrace]:
        """Get traces where a specific source contributed."""
        return [t for t in self._traces if source in t.contributing_sources]

    def get_trace_summary(self) -> Dict[str, Any]:
        """Get a summary of all traces."""
        if not self._traces:
            return {
                "total_traces": 0,
                "total_queries": 0,
                "sources_used": [],
                "avg_evaluation_scores": {}
            }
        
        all_sources = set()
        for trace in self._traces:
            all_sources.update(trace.contributing_sources)
        
        # Calculate average evaluation scores
        eval_keys = set()
        for trace in self._traces:
            eval_keys.update(trace.evaluation_metrics.keys())
        
        avg_scores = {}
        for key in eval_keys:
            scores = [t.evaluation_metrics.get(key, 0) for t in self._traces]
            avg_scores[key] = sum(scores) / len(scores) if scores else 0
        
        return {
            "total_traces": len(self._traces),
            "total_queries": len(self._traces),
            "sources_used": list(all_sources),
            "avg_evaluation_scores": avg_scores,
            "last_trace_time": self._traces[-1].timestamp if self._traces else None
        }

    def clear_traces(self) -> None:
        """Clear in-memory traces (does not delete files)."""
        self._traces.clear()


# Singleton instance
_tracer = None


def get_tracer() -> RAGTracer:
    """Get or create the singleton tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = RAGTracer(
            enable_file_tracing=True,
            enable_langsmith=False  # Disabled by default
        )
    return _tracer


def configure_langsmith_tracing(api_key: str, project_name: str = "knowledgeStore") -> None:
    """
    Configure LangSmith tracing.
    
    Args:
        api_key: LangSmith API key
        project_name: Project name for tracing
    """
    global _tracer
    config.LANGSMITH_API_KEY = api_key
    config.LANGSMITH_PROJECT = project_name
    
    if _tracer is None:
        _tracer = RAGTracer(enable_file_tracing=True, enable_langsmith=True)
    else:
        _tracer.enable_langsmith = bool(api_key)
        _tracer._langsmith_client = None  # Reset to reinitialize