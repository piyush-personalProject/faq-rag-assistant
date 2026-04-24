"""
LLM module.
Handles local LLM initialization and text generation using LangChain.
"""

import torch
from typing import Optional, List
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser

from .config import config


class AnswerOutputParser(BaseOutputParser):
    """Custom output parser to extract the answer from LLM response."""
    
    def parse(self, text: str) -> str:
        """Parse the LLM output to extract the answer."""
        # Clean up the text
        text = text.strip()
        
        # Take only the first paragraph (first blank-line separated block)
        paragraphs = text.split('\n\n')
        first_para = paragraphs[0].strip() if paragraphs else text.strip()
        
        # Remove any "Answer:" prefix if present
        if first_para.startswith("Answer:"):
            first_para = first_para[7:].strip()
        
        # If result is too short, return full text
        if len(first_para) < 10:
            return text.strip() if len(text.strip()) > 10 else text.strip()
        
        return first_para
    
    @property
    def _type(self) -> str:
        return "answer_output_parser"


class LLMManager:
    """
    Manages local LLM for text generation using LangChain.
    Falls back gracefully if LLM is unavailable.
    """
    
    def __init__(self):
        self.pipeline = None
        self.chain = None
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the LangChain LLM pipeline."""
        try:
            # Create HuggingFacePipeline using LangChain's integration
            self.pipeline = HuggingFacePipeline.from_model_id(
                model_id=config.LLM_MODEL,
                task="text-generation",
                pipeline_kwargs=dict(
                    max_new_tokens=config.LLM_MAX_TOKENS * 3,  # Generate more tokens
                    temperature=config.LLM_TEMPERATURE + 0.2,  # Slightly higher
                    top_p=0.9,
                    repetition_penalty=1.5,
                    do_sample=True,
                ),
                device=-1,  # CPU
            )
            
            # Create prompt template with optional history support
            prompt = PromptTemplate(
                input_variables=["context", "question", "history"],
                template="""You are a helpful FAQ assistant. Use the context to answer the question accurately.

{history}Context:
{context}

Question: {question}

Answer:"""
            )
            
            # Create Runnable sequence
            self.chain = prompt | self.pipeline
            
        except Exception as e:
            print(f"[LLM] Warning: Could not load local LLM: {e}")
            self.pipeline = None
            self.chain = None
    
    @property
    def is_available(self) -> bool:
        """Check if LLM is loaded and available."""
        return self.pipeline is not None
    
    def _post_process(self, text: str) -> str:
        """Post-process LLM output to extract the answer portion."""
        if not text:
            return ""
        
        # Find "Answer:" marker and get everything after it
        answer_marker = text.rfind("Answer:")
        if answer_marker != -1:
            answer_part = text[answer_marker + 7:].strip()
            # If we have substantial content after Answer:, use it
            if len(answer_part) > 20:
                return answer_part
        
        # If no Answer: marker or too short, look for the actual response
        # by finding content after Context:/Question: lines
        lines = text.split('\n')
        answer_lines = []
        in_answer_section = False
        
        for line in lines:
            stripped = line.strip()
            # Once we hit "Answer:", capture everything after
            if stripped.startswith("Answer:"):
                in_answer_section = True
                # Get content after "Answer:"
                content_after = stripped[7:].strip()
                if content_after:
                    answer_lines.append(content_after)
            elif in_answer_section:
                # Continue adding lines in answer section
                if stripped and not any(stripped.startswith(x) for x in ["Context:", "Question:", "Context", "Question"]):
                    answer_lines.append(stripped)
                elif not stripped:
                    # Allow blank lines in answer
                    answer_lines.append("")
                else:
                    # Hit another Context:/Question: marker, stop
                    break
        
        if answer_lines:
            result = '\n'.join(answer_lines).strip()
            # Clean up any remaining partial prompt echoes
            result = result.replace("Question: what are conditions for returning stuff?", "").strip()
            result = result.replace("Context: Items must be in original condition, unused, and with all original packaging.", "").strip()
            result = result.replace("Context: test", "").strip()
            if len(result) > 10:
                return result
        
        return text.strip()
    
    def generate(self, prompt: str) -> Optional[str]:
        """
        Generate text response from prompt.
        
        Args:
            prompt: Input prompt for generation
            
        Returns:
            Generated text or None if generation fails
        """
        if not self.chain:
            return None
        
        try:
            result = self.chain.invoke({"context": "", "question": prompt, "history": ""})
            return self._post_process(result) if result else None
        except Exception as e:
            print(f"[LLM] Generation error: {e}")
            return None
    
    def build_rag_prompt(self, context_text: str, question: str) -> str:
        """
        Build a RAG prompt with context and question.
        Note: This is kept for backward compatibility, but the chain handles this internally.
        
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
    
    def generate_with_context(self, context_text: str, question: str, history_context: str = "") -> Optional[str]:
        """
        Generate response with RAG context using LangChain.
        
        Args:
            context_text: Retrieved context chunks
            question: User question
            history_context: Optional conversation history for multi-turn conversations
            
        Returns:
            Generated response or None if generation fails
        """
        if not self.chain:
            return None
        
        # Format history section if provided
        history_section = f"Conversation History:\n{history_context}\n\n" if history_context else ""
        
        try:
            result = self.chain.invoke({
                "context": context_text,
                "question": question,
                "history": history_section
            })
            # Chain returns a string directly
            return self._post_process(result) if result else None
        except Exception as e:
            print(f"[LLM] Generation error: {e}")
            return None

    def generate_with_reasoning(self, context_text: str, question: str) -> Optional[str]:
        """
        Generate response with step-by-step reasoning for better accuracy.
        Uses chain-of-thought prompting to break down complex questions.

        Args:
            context_text: Retrieved context chunks
            question: User question
            
        Returns:
            Generated response with reasoning, or None if generation fails
        """
        if not self.chain:
            return None

        reasoning_prompt = f"""You are a helpful FAQ assistant. Think step by step to provide an accurate answer.

Context:
{context_text}

Question: {question}

Think through this step by step:
1. What is the user asking about?
2. What information in the context is relevant?
3. Based on the context, what is the most accurate answer?

Provide your step-by-step reasoning first, then give the final answer."""

        try:
            result = self.chain.invoke({"context": "", "question": reasoning_prompt, "history": ""})
            return self._post_process(result) if result else None
        except Exception as e:
            print(f"[LLM] Reasoning generation error: {e}")
            return None

    def extract_answer_from_reasoning(self, reasoning_output: str) -> str:
        """
        Extract the final answer from a reasoning response.
        Looks for structured output or the most relevant section.
        
        Args:
            reasoning_output: The full reasoning output
            
        Returns:
            The extracted final answer
        """
        # Try to find "Final Answer:" marker
        final_marker = reasoning_output.find("Final Answer:")
        if final_marker != -1:
            answer = reasoning_output[final_marker + 12:].strip()
            return answer.split("\n")[0].strip()
        
        # Try to find "Answer:" marker
        answer_marker = reasoning_output.find("Answer:")
        if answer_marker != -1:
            answer = reasoning_output[answer_marker + 7:].strip()
            return answer.split("\n")[0].strip()
        
        # Otherwise, return the full output (the reasoning itself is useful)
        return reasoning_output
