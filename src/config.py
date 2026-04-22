"""
Configuration module for Knowledge Store.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""
    
    # Flask settings
    FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    
    # RAG settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # LLM settings
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt2-medium")
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 150))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.3))
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    UPLOAD_FOLDER = PROJECT_ROOT / "data" / "faq_docs"
    EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
    INDEX_PATH = EMBEDDINGS_DIR / "faiss.index"
    METADATA_PATH = EMBEDDINGS_DIR / "metadata.pkl"
    
    # Simple query patterns (greetings/non-informational)
    SIMPLE_GREETINGS = {
        'hi', 'hello', 'hey', 'hi!', 'hello!', 'hey!',
        'hi.', 'hello.', 'hey.', 'help', 'help me',
        '?', '??', '???', 'what', 'what?', 'who', 'who?'
    }
    
    # Stop words for query processing
    STOP_WORDS = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be',
        'do', 'does', 'did', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'how', 'what', 'when', 'where', 'why',
        'can', 'i', 'my', 'me', 'we', 'our', 'you', 'your',
        'have', 'has', 'get', 'got', 'claim', 'days', 'day'
    }


# Singleton config instance
config = Config()
