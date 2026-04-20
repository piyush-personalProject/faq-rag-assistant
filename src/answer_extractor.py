"""
Answer Extractor module.
Extracts specific answers from retrieved chunks using keyword matching.
"""

import re
from typing import List, Dict, Optional

from .config import config


class AnswerExtractor:
    """
    Extracts relevant answers from retrieved document chunks.
    Uses Q&A pattern matching and keyword scoring.
    """
    
    @staticmethod
    def extract(query: str, chunks: List[Dict]) -> Optional[str]:
        """
        Extract a specific answer from retrieved chunks.
        
        Args:
            query: User's question
            chunks: List of retrieved document chunks
            
        Returns:
            Extracted answer text or None if no good match found
        """
        query_words = set(re.findall(r'\w+', query.lower())) - config.STOP_WORDS
        
        if not query_words:
            return ""
        
        best_answer = ""
        best_score = 0
        
        for chunk in chunks:
            text = chunk["text"]
            
            # Find Q&A blocks
            qa_blocks = re.split(r'(?=Q:\s*)', text)
            
            for block in qa_blocks:
                if not block.strip().startswith('Q:'):
                    continue
                
                parts = block.split('A:', 1)
                if len(parts) < 2:
                    continue
                
                question = parts[0].strip()
                answer = parts[1].strip()
                
                # Remove any trailing Q: that might be part of the answer
                if 'Q:' in answer:
                    answer = answer.split('Q:')[0].strip()
                
                # Skip if too short
                if len(question) < 5 or len(answer) < 5:
                    continue
                
                # Calculate match score with partial matching
                q_words = set(re.findall(r'\w+', question.lower())) - config.STOP_WORDS
                match_count = 0
                for qw in query_words:
                    for qword in q_words:
                        # Check for partial match (query word contains chunk word or vice versa)
                        if qw in qword or qword in qw:
                            match_count += 1
                            break
                
                a_words = set(re.findall(r'\w+', answer.lower())) - config.STOP_WORDS
                answer_match = 0
                for qw in query_words:
                    for aw in a_words:
                        if qw in aw or aw in qw:
                            answer_match += 1
                            break
                
                # Score = question matches * 2 + answer matches
                score = match_count * 2 + answer_match
                
                # Lower threshold: accept if we have any meaningful match
                if score > best_score and (match_count >= 1 or answer_match >= 1):
                    best_score = score
                    clean_answer = ' '.join(answer.split())
                    
                    # Truncate if too long
                    if len(clean_answer) > 200:
                        truncate_idx = clean_answer[:200].rfind('.')
                        if truncate_idx > 50:
                            clean_answer = clean_answer[:truncate_idx + 1]
                        else:
                            clean_answer = clean_answer[:200] + "..."
                    
                    best_answer = clean_answer
        
        if best_score > 0:
            return best_answer
        
        return ""
    
    @staticmethod
    def format_chunks_response(chunks: List[Dict], max_chunks: int = 1, max_chars: int = 200) -> str:
        """
        Format chunks into a concise response string.

        Args:
            chunks: List of document chunks
            max_chunks: Maximum number of chunks to include
            max_chars: Maximum characters per chunk

        Returns:
            Formatted response string
        """
        for c in chunks[:max_chunks]:
            text = c["text"]

            # Try to extract just the answer portion from Q&A blocks
            qa_blocks = re.split(r'(?=Q:\s*)', text)
            for block in qa_blocks:
                if block.strip().startswith('Q:'):
                    parts = block.split('A:', 1)
                    if len(parts) >= 2:
                        answer = parts[1].strip()
                        if 'Q:' in answer:
                            answer = answer.split('Q:')[0].strip()
                        if len(answer) > max_chars:
                            truncate_idx = answer[:max_chars].rfind('.')
                            if truncate_idx > 30:
                                answer = answer[:truncate_idx + 1]
                            else:
                                answer = answer[:max_chars] + "..."
                        return answer
            else:
                # No Q&A block found, truncate the text cleanly
                clean_text = ' '.join(text.split())
                if len(clean_text) > max_chars:
                    truncate_idx = clean_text[:max_chars].rfind('.')
                    if truncate_idx > 30:
                        clean_text = clean_text[:truncate_idx + 1]
                    else:
                        clean_text = clean_text[:max_chars] + "..."
                return clean_text

        return "I couldn't find a clear answer. Please try rephrasing your question."
