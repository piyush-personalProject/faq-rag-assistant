"""
LLM module.
Handles local LLM initialization and text generation.
"""

import os
import torch
from typing import Optional, List
from transformers import pipeline, set_seed

from .config import config


class LLMManager:
    """
    Manages local LLM for text generation.
    Falls back gracefully if LLM is unavailable.
    """
    
    def __init__(self):
        self.pipeline = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the LLM pipeline."""
        try:
            self.pipeline = pipeline(
                "text-generation",
                model=config.LLM_MODEL,
                device=-1,  # CPU
                torch_dtype=torch.float32
            )
        except Exception as e:
            print(f"[LLM] Warning: Could not load local LLM: {e}")
            self.pipeline = None
    
    @property
    def is_available(self) -> bool:
        """Check if LLM is loaded and available."""
        return self.pipeline is not None
    
    def generate(self, prompt: str) -> Optional[str]:
        """
        Generate text response from prompt.
        
        Args:
            prompt: Input prompt for generation
            
        Returns:
            Generated text or None if generation fails
        """
        if not self.pipeline:
            return None
        
        try:
            set_seed(42)
            outputs = self.pipeline(
                prompt,
                max_new_tokens=config.LLM_MAX_TOKENS,
                num_beams=1,
                do_sample=True,
                temperature=config.LLM_TEMPERATURE,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.pipeline.tokenizer.eos_token_id
            )
            
            generated_text = outputs[0]["generated_text"]
            
            # Extract just the answer part (after "Answer:")
            answer_start = generated_text.find("Answer:")
            if answer_start != -1:
                response_text = generated_text[answer_start + 7:].strip()
            else:
                response_text = generated_text[len(prompt):].strip()
            
            # Clean up any leftover context or repetition
            response_text = response_text.split("\n")[0].strip()
            if len(response_text) < 20:
                response_text = generated_text.strip()
            
            return response_text
            
        except Exception as e:
            print(f"[LLM] Generation error: {e}")
            return None
    
    def build_rag_prompt(self, context_text: str, question: str) -> str:
        """
        Build a RAG prompt with context and question.
        
        Args:
            context_text: Retrieved context chunks
            question: User question
            
        Returns:
            Formatted prompt string
        """
        return f"""Based on the following FAQ context, answer the user's question concisely and accurately.

Context:
{context_text}

Question: {question}

Answer:"""
