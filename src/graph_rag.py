"""
LangGraph RAG implementation with self-correction capability.
Provides quality checks and regeneration for more accurate answers.
"""

import re
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from .rag_engine import RAGEngine
from .llm import LLMManager
from .answer_extractor import AnswerExtractor


class RAGState(TypedDict):
    """State passed between graph nodes."""
    query: str
    history: List[BaseMessage]
    retrieved_docs: list
    context: str
    answer: str
    quality_score: float
    attempts: int
    messages: list


class GraphRAG:
    """
    RAG engine using LangGraph for self-correction workflow.
    
    Flow: Retrieve → Generate → Quality Check → (Regenerate if poor) → Done
    """
    
    def __init__(self, rag_engine: RAGEngine, llm_manager: LLMManager):
        self.rag = rag_engine
        self.llm = llm_manager
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build and compile the RAG graph."""
        
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("quality_check", self._quality_check_node)
        workflow.add_node("regenerate", self._regenerate_node)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        # Sequential edges
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "quality_check")
        
        # Conditional routing based on quality
        workflow.add_conditional_edges(
            "quality_check",
            self._should_continue,
            {
                "good": END,
                "regenerate": "regenerate",
                "finalize": END
            }
        )
        
        # Regenerate loops back to quality check
        workflow.add_edge("regenerate", "quality_check")
        
        return workflow.compile()
    
    def _retrieve_node(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents based on query."""
        docs = self.rag.retrieve(state["query"], top_k=5)
        
        return {
            **state,
            "retrieved_docs": docs,
            "context": "\n\n".join([d["text"] for d in docs]),
            "messages": state["messages"] + [f"Retrieved {len(docs)} documents"]
        }
    
    def _generate_node(self, state: RAGState) -> RAGState:
        """Generate answer using LLM with retrieved context."""
        if not state["context"]:
            return {
                **state,
                "answer": "I don't have enough context to answer.",
                "quality_score": 0.0,
                "messages": state["messages"] + ["No context available"]
            }
        
        # Build conversation history context if available
        history_context = self._format_history(state["history"]) if state["history"] else ""
        
        answer = self.llm.generate_with_context(
            state["context"], 
            state["query"],
            history_context=history_context
        )
        
        # If LLM failed, returned empty, or returned gibberish, extract from chunks directly
        # Check if answer is useful: should be > 20 chars and contain relevant content
        answer_is_gibberish = (
            not answer or 
            len(answer) < 20 or 
            self._is_gibberish_answer(answer, state["context"])
        )
        
        if answer_is_gibberish:
            extracted = AnswerExtractor.extract(state["query"], state["retrieved_docs"])
            if extracted:
                answer = extracted
        
        # If still no good answer, try formatting chunks
        if not answer or len(answer) < 20:
            formatted = AnswerExtractor.format_chunks_response(state["retrieved_docs"])
            if formatted:
                answer = formatted
        
        return {
            **state,
            "answer": answer or "I couldn't generate a satisfactory answer.",
            "messages": state["messages"] + [AIMessage(content=answer or "")]
        }
    
    def _is_gibberish_answer(self, answer: str, context: str) -> bool:
        """Check if the LLM answer appears to be gibberish or unrelated."""
        if not answer or not context:
            return True
        
        # Get key content words from context (Q&A specific words)
        context_words = set(re.findall(r'\b\w{4,}\b', context.lower()))
        # Remove common stop words
        stop_words = {'what', 'your', 'have', 'from', 'this', 'with', 'will', 'been', 'they', 'their', 'there', 'when', 'where', 'which', 'about', 'some', 'would', 'could', 'should', 'into', 'only', 'other', 'then', 'than', 'very', 'also', 'after', 'before', 'such', 'each', 'more', 'most', 'other', 'some', 'these', 'those', 'what'}
        context_words -= stop_words
        
        answer_words = set(re.findall(r'\b\w{4,}\b', answer.lower()))
        answer_words -= stop_words
        
        # Check overlap - answer should share some words with context
        overlap = context_words & answer_words
        
        # If very low overlap (< 33% of context words), likely gibberish
        # Relaxed from //7 to //3 for better RAG performance with proper retrieval
        if len(context_words) > 0 and len(overlap) < max(1, len(context_words) // 3):
            return True
        
        # Check for repetitive patterns (hallucination sign)
        words = answer.split()
        if len(words) > 5:
            # Count word frequency
            word_counts = {}
            for w in words:
                w_lower = w.lower().strip('.,!?')
                word_counts[w_lower] = word_counts.get(w_lower, 0) + 1
            
            # If any word repeats more than 3 times, likely repetitive gibberish
            max_repeat = max(word_counts.values()) if word_counts else 0
            if max_repeat > 3 and len(words) > 10:
                return True
        
        return False
    
    def _format_history(self, history: List[BaseMessage]) -> str:
        """Format conversation history for inclusion in prompt."""
        if not history:
            return ""
        
        formatted = []
        for msg in history[-6:]:  # Last 6 messages to keep context manageable
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        
        return "\n".join(formatted)
    
    def _quality_check_node(self, state: RAGState) -> RAGState:
        """Evaluate if the generated answer meets quality threshold using heuristic."""
        # Simple heuristic-based quality check (no LLM call needed)
        answer = state.get("answer", "")
        context = state.get("context", "")
        
        if not answer or len(answer) < 20:
            score = 0.3
        else:
            # Check overlap between answer and context words
            context_words = set(re.findall(r'\b\w{4,}\b', context.lower()))
            answer_words = set(re.findall(r'\b\w{4,}\b', answer.lower()))
            stop_words = {'what', 'your', 'have', 'from', 'this', 'with', 'will', 'been', 'they', 'their', 'there', 'when', 'where', 'which', 'about', 'some', 'would', 'could', 'should', 'into', 'only', 'other', 'then', 'than', 'very', 'also', 'after', 'before', 'such', 'each', 'more', 'most', 'some', 'these', 'those'}
            context_words -= stop_words
            answer_words -= stop_words
            
            overlap = context_words & answer_words
            
            # Good overlap means answer is relevant to context
            if len(context_words) > 0:
                overlap_ratio = len(overlap) / len(context_words)
                score = min(0.9, overlap_ratio + 0.3)  # Scale: 0.3 to 0.9 based on overlap
            else:
                score = 0.5
        
        return {
            **state,
            "quality_score": score,
            "messages": state["messages"] + [f"Quality check: {score:.2f}"]
        }
    
    def _regenerate_node(self, state: RAGState) -> RAGState:
        """Regenerate answer with more context or different approach."""
        # Expand context with all retrieved documents
        expanded_context = state["context"]
        
        # Build conversation history context if available
        history_context = self._format_history(state["history"]) if state["history"] else ""
        
        # Try a more direct prompt with explicit instruction
        regenerate_prompt = f"""Based on this context, give a direct, accurate answer.
{history_context}
Context:
{expanded_context}

Question: {state['query']}

Instructions:
- If the context contains the answer, provide it clearly
- If the context doesn't contain enough information, say "I couldn't find enough information in the provided documents to answer this fully."
- Keep the answer concise and factual

Answer:"""
        
        answer = self.llm.generate(regenerate_prompt)
        
        # If LLM fails or returns gibberish, try extraction
        if not answer or len(answer) < 20 or self._is_gibberish_answer(answer, expanded_context):
            extracted = AnswerExtractor.extract(state["query"], state["retrieved_docs"])
            if extracted:
                answer = extracted
        
        # If still no good answer, format chunks
        if not answer or len(answer) < 20:
            formatted = AnswerExtractor.format_chunks_response(state["retrieved_docs"])
            if formatted:
                answer = formatted
        
        return {
            **state,
            "answer": answer or "I couldn't generate a satisfactory answer.",
            "attempts": state["attempts"] + 1,
            "messages": state["messages"] + [AIMessage(content=answer or "")]
        }
    
    def _should_continue(self, state: RAGState) -> str:
        """Determine next step based on quality score."""
        if state["quality_score"] >= 0.7:
            return "good"
        elif state["attempts"] < 2:
            return "regenerate"
        else:
            return "finalize"
    
    def query(self, user_query: str, history: List[BaseMessage] = None) -> dict:
        """
        Process a query through the self-correcting RAG graph.
        
        Args:
            user_query: The user's question
            history: Optional conversation history (list of BaseMessage objects)
            
        Returns:
            dict with 'answer', 'quality_score', 'attempts', 'sources'
        """
        # Convert list of dicts to BaseMessage objects if needed
        if history is None:
            history = []
        
        initial_state = {
            "query": user_query,
            "history": history,
            "retrieved_docs": [],
            "context": "",
            "answer": "",
            "quality_score": 0.0,
            "attempts": 0,
            "messages": [HumanMessage(content=user_query)]
        }
        
        result = self.graph.invoke(initial_state)
        
        # Extract sources from retrieved docs
        sources = list({d["source"] for d in result["retrieved_docs"]}) if result["retrieved_docs"] else []
        
        return {
            "answer": result["answer"],
            "quality_score": result["quality_score"],
            "attempts": result["attempts"] + 1,  # +1 because first attempt doesn't count as attempt
            "sources": sources,
            "retrieved_docs": result["retrieved_docs"]
        }


# Singleton instance (lazy initialization)
_graph_rag = None


def get_graph_rag() -> GraphRAG:
    """Get or create the singleton GraphRAG instance."""
    global _graph_rag
    if _graph_rag is None:
        from .routes import rag, llm
        _graph_rag = GraphRAG(rag, llm)
    return _graph_rag


def query_with_graph(user_query: str, history: List[BaseMessage] = None) -> dict:
    """
    Convenience function to query through the self-correcting graph.
    
    Args:
        user_query: The user's question
        history: Optional conversation history
        
    Returns:
        dict with 'answer', 'quality_score', 'attempts', 'sources'
    """
    graph_rag = get_graph_rag()
    return graph_rag.query(user_query, history)