"""
RAG Evaluator module.
Provides evaluation metrics for RAG systems: faithfulness, answer relevance, and context precision.
"""

import re
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass

from .config import config


@dataclass
class RAGEvaluationResult:
    """Container for RAG evaluation metrics."""
    faithfulness: float  # 0-1 score
    answer_relevance: float  # 0-1 score
    context_precision: float  # 0-1 score
    details: Dict


class RAGEvaluator:
    """
    Evaluates RAG system outputs for:
    - Faithfulness: Whether answer claims are supported by the context
    - Answer Relevance: Whether answer addresses the user's query
    - Context Precision: Whether retrieved contexts are relevant to answering the query
    """

    def __init__(self):
        self.stop_words = config.STOP_WORDS

    def _tokenize(self, text: str) -> Set[str]:
        """Extract meaningful tokens from text."""
        tokens = set(re.findall(r'\b\w{3,}\b', text.lower()))
        return tokens - self.stop_words

    def _get_ngrams(self, text: str, n: int) -> Set[str]:
        """Extract character n-grams for better matching."""
        text_clean = re.sub(r'[^\w\s]', '', text.lower())
        words = text_clean.split()
        return set(' '.join(words[i:i+n]) for i in range(len(words) - n + 1)) if len(words) >= n else set()

    def evaluate_faithfulness(
        self,
        answer: str,
        context: str
    ) -> Tuple[float, List[str]]:
        """
        Evaluate faithfulness: do the claims in the answer align with the context?
        
        Args:
            answer: Generated answer text
            context: Retrieved context text
            
        Returns:
            Tuple of (score 0-1, list of unfaithful claims)
        """
        if not answer or len(answer) < 10:
            return 0.0, ["Answer too short to evaluate"]

        if not context:
            return 0.0, ["No context provided"]

        # Tokenize both texts
        answer_tokens = self._tokenize(answer)
        context_tokens = self._tokenize(context)

        # Split answer into sentences
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]

        unfaithful_claims = []
        faithful_count = 0

        for sentence in sentences:
            if len(sentence.split()) < 3:
                continue

            sentence_tokens = self._tokenize(sentence)

            # Check if sentence tokens appear in context
            # Use Jaccard-like overlap calculation
            if sentence_tokens:
                overlap = len(sentence_tokens & context_tokens)
                coverage = overlap / len(sentence_tokens) if sentence_tokens else 0

                # A sentence is considered faithful if it has >50% token overlap with context
                # OR if it appears literally in context
                literal_match = sentence.lower() in context.lower()

                if coverage >= 0.5 or literal_match:
                    faithful_count += 1
                else:
                    unfaithful_claims.append(sentence)

        # Calculate score
        if not sentences:
            return 1.0, []  # Empty answer is trivially faithful

        score = faithful_count / len(sentences)
        return score, unfaithful_claims

    def evaluate_answer_relevance(
        self,
        answer: str,
        query: str
    ) -> Tuple[float, List[str]]:
        """
        Evaluate answer relevance: does the answer address the query?
        
        Args:
            answer: Generated answer text
            query: User's original query
            
        Returns:
            Tuple of (score 0-1, list of irrelevant parts)
        """
        if not answer or len(answer) < 10:
            return 0.0, ["Answer too short"]

        if not query:
            return 0.5, ["No query provided"]

        query_tokens = self._tokenize(query)
        answer_tokens = self._tokenize(answer)

        # Calculate overlap
        if not query_tokens:
            return 0.5, []

        overlap = query_tokens & answer_tokens
        coverage = len(overlap) / len(query_tokens)

        # Check for query keywords in answer
        query_keywords = set(w for w in query_tokens if len(w) > 4)
        keyword_hits = sum(1 for kw in query_keywords if kw in answer.lower())

        # Penalize if answer doesn't contain query terms
        if coverage < 0.2 and keyword_hits < len(query_keywords) * 0.3:
            irrelevant_parts = [f"Missing query terms: {query_keywords - answer_tokens}"]
            return 0.3, irrelevant_parts

        # Check for tangential information (sentences that don't match query)
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]

        irrelevant_sentences = []
        for sentence in sentences:
            if len(sentence.split()) < 5:
                continue
            sentence_tokens = self._tokenize(sentence)
            sentence_overlap = sentence_tokens & query_tokens
            if len(sentence_overlap) < len(query_tokens) * 0.15:
                irrelevant_sentences.append(sentence)

        # Base score on coverage, with penalties for irrelevant sentences
        base_score = min(0.95, coverage + 0.3)  # Boost slightly since having some overlap is good
        penalty = min(0.3, len(irrelevant_sentences) * 0.1)

        final_score = max(0.0, base_score - penalty)
        return final_score, irrelevant_sentences

    def evaluate_context_precision(
        self,
        contexts: List[Dict],
        query: str
    ) -> Tuple[float, List[int]]:
        """
        Evaluate context precision: how relevant are the retrieved contexts to the query?
        
        Args:
            contexts: List of retrieved document chunks
            query: User's original query
            
        Returns:
            Tuple of (score 0-1, list of irrelevant context indices)
        """
        if not contexts:
            return 0.0, []

        if not query:
            return 0.5, []  # Can't evaluate without query

        query_tokens = self._tokenize(query)

        precision_scores = []
        irrelevant_indices = []

        for i, ctx in enumerate(contexts):
            text = ctx.get("text", "")
            ctx_tokens = self._tokenize(text)

            if not ctx_tokens:
                precision_scores.append(0.0)
                irrelevant_indices.append(i)
                continue

            # Calculate precision for this context
            overlap = query_tokens & ctx_tokens
            precision = len(overlap) / len(ctx_tokens) if ctx_tokens else 0

            # Also check via n-gram matching for better semantic capture
            query_ngrams = self._get_ngrams(query, 2)
            ctx_ngrams = self._get_ngrams(text, 2)
            ngram_overlap = len(query_ngrams & ctx_ngrams) / len(query_ngrams) if query_ngrams else 0

            # Combined score
            combined = (precision * 0.6 + ngram_overlap * 0.4)

            precision_scores.append(combined)

            if combined < 0.1:  # Threshold for relevance
                irrelevant_indices.append(i)

        # Calculate mean precision (considering only relevant recalls matter)
        if precision_scores:
            # Context precision = mean of precision scores for relevant contexts
            # weighted by actual relevance
            relevant_count = len([s for s in precision_scores if s >= 0.1])
            if relevant_count > 0:
                score = sum(s for s in precision_scores if s >= 0.1) / relevant_count
            else:
                score = 0.0
        else:
            score = 0.0

        return score, irrelevant_indices

    def evaluate(
        self,
        answer: str,
        query: str,
        contexts: List[Dict]
    ) -> RAGEvaluationResult:
        """
        Run all three evaluation metrics.
        
        Args:
            answer: Generated answer text
            query: User's original query
            contexts: List of retrieved document chunks
            
        Returns:
            RAGEvaluationResult with all metrics
        """
        context_text = "\n".join(ctx.get("text", "") for ctx in contexts)

        # Compute each metric
        faithfulness, unfaithful = self.evaluate_faithfulness(answer, context_text)
        answer_relevance, irrelevant = self.evaluate_answer_relevance(answer, query)
        context_precision, irrelevant_ctx = self.evaluate_context_precision(contexts, query)

        return RAGEvaluationResult(
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_precision=context_precision,
            details={
                "unfaithful_claims": unfaithful,
                "irrelevant_sentences": irrelevant,
                "irrelevant_context_indices": irrelevant_ctx
            }
        )

    def get_overall_quality(self, result: RAGEvaluationResult) -> float:
        """
        Calculate overall quality score from individual metrics.
        
        Weighted average emphasizing faithfulness and answer relevance.
        """
        return (
            result.faithfulness * 0.4 +
            result.answer_relevance * 0.4 +
            result.context_precision * 0.2
        )


# Singleton instance
_evaluator = None


def get_evaluator() -> RAGEvaluator:
    """Get or create the singleton evaluator instance."""
    global _evaluator
    if _evaluator is None:
        _evaluator = RAGEvaluator()
    return _evaluator


def evaluate_rag_output(
    answer: str,
    query: str,
    contexts: List[Dict]
) -> RAGEvaluationResult:
    """
    Convenience function to evaluate a RAG output.
    
    Args:
        answer: Generated answer
        query: User's query
        contexts: Retrieved contexts
        
    Returns:
        RAGEvaluationResult with faithfulness, answer_relevance, context_precision
    """
    evaluator = get_evaluator()
    return evaluator.evaluate(answer, query, contexts)