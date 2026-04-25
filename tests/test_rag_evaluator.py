"""
Tests for RAG evaluator module.
Tests faithfulness, answer relevance, and context precision metrics.
"""

import pytest
from src.rag_evaluator import RAGEvaluator, evaluate_rag_output


class TestRAGEvaluator:
    """Test suite for RAGEvaluator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = RAGEvaluator()

    def test_faithfulness_with_grounded_answer(self):
        """Test faithfulness when answer is fully grounded in context."""
        context = "The return policy allows 30 days for returns. Items must be in original packaging."
        answer = "You can return items within 30 days if they are in original packaging."

        score, unfaithful = self.evaluator.evaluate_faithfulness(answer, context)
        
        assert score >= 0.8, f"Expected high faithfulness score, got {score}"
        assert len(unfaithful) == 0, f"Expected no unfaithful claims, got {unfaithful}"

    def test_faithfulness_with_ungrounded_answer(self):
        """Test faithfulness when answer contains claims not in context."""
        context = "The return policy allows 30 days for returns."
        answer = "You can return items within 60 days and get a full refund including shipping costs."

        score, unfaithful = self.evaluator.evaluate_faithfulness(answer, context)
        
        assert score < 0.5, f"Expected low faithfulness score, got {score}"
        assert len(unfaithful) > 0, "Expected unfaithful claims to be detected"

    def test_faithfulness_with_empty_answer(self):
        """Test faithfulness with empty or very short answer."""
        score, unfaithful = self.evaluator.evaluate_faithfulness("", "Some context")
        assert score == 0.0

        score, unfaithful = self.evaluator.evaluate_faithfulness("Hi", "Some context")
        assert score == 0.0

    def test_faithfulness_with_empty_context(self):
        """Test faithfulness with no context."""
        score, unfaithful = self.evaluator.evaluate_faithfulness("Some answer", "")
        assert score == 0.0

    def test_answer_relevance_with_relevant_answer(self):
        """Test answer relevance when answer addresses the query."""
        query = "How do I track my order?"
        answer = "You can track your order using the tracking link sent to your email."

        score, irrelevant = self.evaluator.evaluate_answer_relevance(answer, query)
        
        assert score >= 0.7, f"Expected high relevance score, got {score}"
        assert len(irrelevant) == 0

    def test_answer_relevance_with_irrelevant_answer(self):
        """Test answer relevance when answer doesn't address query."""
        query = "How do I track my order?"
        answer = "Our store is open Monday to Friday from 9am to 5pm."

        score, irrelevant = self.evaluator.evaluate_answer_relevance(answer, query)
        
        assert score < 0.5, f"Expected low relevance score, got {score}"

    def test_answer_relevance_with_partial_overlap(self):
        """Test answer relevance with partial query term overlap."""
        query = "What is the return policy for damaged items?"
        answer = "Items can be returned within 30 days."

        score, irrelevant = self.evaluator.evaluate_answer_relevance(answer, query)
        
        # Should have moderate score due to some overlap (return, days)
        assert 0.3 <= score < 0.8

    def test_context_precision_with_relevant_contexts(self):
        """Test context precision with relevant documents."""
        query = "How do I track my order?"
        contexts = [
            {"text": "You can track your order using the tracking link in your email."},
            {"text": "Tracking information is sent once your order ships."}
        ]

        score, irrelevant = self.evaluator.evaluate_context_precision(contexts, query)
        
        assert score >= 0.6, f"Expected high precision score, got {score}"
        assert len(irrelevant) == 0

    def test_context_precision_with_irrelevant_contexts(self):
        """Test context precision when contexts don't match query."""
        query = "How do I track my order?"
        contexts = [
            {"text": "Our store hours are 9am to 5pm Monday to Friday."},
            {"text": "We accept Visa and Mastercard credit cards."}
        ]

        score, irrelevant = self.evaluator.evaluate_context_precision(contexts, query)
        
        assert score < 0.3, f"Expected low precision score, got {score}"
        assert len(irrelevant) == 2

    def test_context_precision_with_empty_contexts(self):
        """Test context precision with no contexts."""
        score, irrelevant = self.evaluator.evaluate_context_precision([], "query")
        assert score == 0.0

    def test_evaluate_all_metrics(self):
        """Test full evaluation with all metrics."""
        query = "How do I return an item?"
        answer = "Items can be returned within 30 days with receipt."
        contexts = [
            {"text": "Return policy: items may be returned within 30 days of purchase with receipt."}
        ]

        result = self.evaluator.evaluate(answer, query, contexts)
        
        assert hasattr(result, 'faithfulness')
        assert hasattr(result, 'answer_relevance')
        assert hasattr(result, 'context_precision')
        assert hasattr(result, 'details')
        
        assert 0 <= result.faithfulness <= 1
        assert 0 <= result.answer_relevance <= 1
        assert 0 <= result.context_precision <= 1

    def test_get_overall_quality(self):
        """Test overall quality calculation."""
        from src.rag_evaluator import RAGEvaluationResult
        
        result = RAGEvaluationResult(
            faithfulness=0.9,
            answer_relevance=0.8,
            context_precision=0.7,
            details={}
        )
        
        quality = self.evaluator.get_overall_quality(result)
        
        # Expected: 0.9*0.4 + 0.8*0.4 + 0.7*0.2 = 0.36 + 0.32 + 0.14 = 0.82
        assert abs(quality - 0.82) < 0.01

    def test_tokenize_filters_stop_words(self):
        """Test that tokenization properly filters stop words."""
        tokens = self.evaluator._tokenize("The return policy for items is great")
        
        assert 'the' not in tokens
        assert 'for' not in tokens
        assert 'items' in tokens

    def test_ngram_extraction(self):
        """Test n-gram extraction for better matching."""
        ngrams = self.evaluator._get_ngrams("return policy for items", 2)
        
        assert 'return policy' in ngrams
        assert 'policy for' in ngrams
        assert 'for items' in ngrams


class TestEvaluateRagOutputFunction:
    """Test the convenience function."""

    def test_evaluate_rag_output_returns_result(self):
        """Test that convenience function returns proper result."""
        result = evaluate_rag_output(
            answer="Items can be returned within 30 days.",
            query="How do I return items?",
            contexts=[{"text": "Return policy: 30 day returns allowed."}]
        )
        
        assert result is not None
        assert hasattr(result, 'faithfulness')
        assert hasattr(result, 'answer_relevance')
        assert hasattr(result, 'context_precision')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])