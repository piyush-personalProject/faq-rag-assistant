"""
Query Classifier module.
Determines if a query is a simple greeting/non-informational or requires RAG.
"""

import re
from typing import Tuple, Optional

from .config import config


class QueryClassifier:
    """
    Classifies user queries to determine processing strategy.
    """
    
    @staticmethod
    def is_simple_query(message: str) -> Tuple[bool, Optional[str]]:
        """
        Check if query is a simple greeting or non-informational query.
        
        Args:
            message: User input message
            
        Returns:
            Tuple of (is_simple, simple_response)
            - is_simple: True if query should skip RAG
            - simple_response: Pre-defined response for simple queries, None otherwise
        """
        normalized = re.sub(r'[!?.,]+$', '', message.lower().strip())
        
        if normalized in config.SIMPLE_GREETINGS or len(normalized) <= 2:
            response = QueryClassifier._get_simple_response(normalized)
            return True, response
        
        return False, None
    
    @staticmethod
    def _get_simple_response(normalized: str) -> str:
        """Return contextual response for simple queries."""
        if normalized in ('hi', 'hello', 'hey'):
            return "Hello! How can I help you today? Ask me about account & billing, shipping, or returns."
        elif normalized == 'help':
            return "I can help you with questions about: account & billing, shipping & delivery, and returns & refunds. Just ask!"
        elif normalized in ('what', 'who'):
            return "I'm your FAQ assistant. Ask me specific questions about our services."
        elif normalized == '?':
            return "Ask me a question about our services and I'll find the answer from the knowledge base."
        else:
            return "Hello! How can I help you today?"
