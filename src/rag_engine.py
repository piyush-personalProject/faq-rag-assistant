"""
RAG Engine module.
Handles text chunking, embedding generation, FAISS indexing, and retrieval using LangChain.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import config


class RAGEngine:
    """
    Core RAG engine for embedding generation, indexing, and retrieval.
    Uses LangChain for vector store and embeddings management.
    """
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        self.vectorstore = None
        self._load_or_init_index()
    
    def _load_or_init_index(self) -> None:
        """Load existing FAISS index or initialize a new one."""
        if config.INDEX_PATH.exists():
            try:
                self.vectorstore = FAISS.load_local(
                    str(config.EMBEDDINGS_DIR),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"[RAG] Could not load existing index: {e}")
                self.vectorstore = FAISS.from_texts(
                    ["initial"], 
                    self.embeddings,
                    metadatas=[{"source": "init"}]
                )
        else:
            # Create an empty vectorstore with a placeholder to initialize
            self.vectorstore = FAISS.from_texts(
                ["placeholder"], 
                self.embeddings,
                metadatas=[{"source": "placeholder"}]
            )
    
    def _save_index(self) -> None:
        """Persist FAISS index to disk."""
        config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(config.EMBEDDINGS_DIR))
    
    def _chunk_text(self, text: str, source: str) -> List[Document]:
        """Split text into overlapping chunks using LangChain text splitter."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )
        
        # Split the text
        splits = text_splitter.split_text(text)
        
        # Create Document objects with metadata
        documents = [
            Document(page_content=split, metadata={"source": source})
            for split in splits
        ]
        
        return documents
    
    def ingest_folder(self, folder_path: str) -> Dict:
        """
        Scan folder for .txt files, embed them, and add to FAISS index.
        
        Args:
            folder_path: Path to folder containing .txt files
            
        Returns:
            Summary dict with processing results
        """
        folder = Path(folder_path)
        txt_files = list(folder.rglob("*.txt"))
        
        if not txt_files:
            return {"status": "error", "message": "No .txt files found in folder."}
        
        all_documents = []
        for txt_file in txt_files:
            text = txt_file.read_text(encoding="utf-8", errors="ignore")
            documents = self._chunk_text(text, txt_file.name)
            all_documents.extend(documents)
        
        # Add documents to vectorstore
        self.vectorstore.add_documents(all_documents)
        self._save_index()
        
        return {
            "status": "success",
            "files_processed": len(txt_files),
            "chunks_added": len(all_documents),
            "total_chunks": self.vectorstore.index.ntotal,
        }
    
    def ingest_file(self, file_path: str) -> Dict:
        """
        Ingest a single .txt file into the index.
        
        Args:
            file_path: Path to the .txt file
            
        Returns:
            Summary dict with processing results
        """
        path = Path(file_path)
        if not path.exists() or path.suffix != ".txt":
            return {"status": "error", "message": "File not found or not a .txt file."}
        
        text = path.read_text(encoding="utf-8", errors="ignore")
        documents = self._chunk_text(text, path.name)
        
        self.vectorstore.add_documents(documents)
        self._save_index()
        
        return {
            "status": "success",
            "file": path.name,
            "chunks_added": len(documents),
            "total_chunks": self.vectorstore.index.ntotal,
        }
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Embed query and retrieve top-k relevant chunks.
        
        Args:
            query: Search query string
            top_k: Number of results to return (defaults to config.TOP_K_RESULTS)
            
        Returns:
            List of dicts with text, source, and score
        """
        if top_k is None:
            top_k = config.TOP_K_RESULTS
        
        if self.vectorstore.index.ntotal == 0:
            return []
        
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        return [
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "score": float(score),
            }
            for doc, score in results
        ]
    
    def get_status(self) -> Dict:
        """Return index status information."""
        if self.vectorstore is None:
            return {
                "total_chunks": 0,
                "total_vectors": 0,
                "sources": [],
                "embedding_model": config.EMBEDDING_MODEL,
            }
        
        # Get unique sources from the documents
        docs = self.vectorstore.similarity_search("*", k=10000)
        sources = list(set(doc.metadata.get("source", "unknown") for doc in docs))
        
        return {
            "total_chunks": self.vectorstore.index.ntotal,
            "total_vectors": self.vectorstore.index.ntotal,
            "sources": sources,
            "embedding_model": config.EMBEDDING_MODEL,
        }
    
    def clear_index(self) -> Dict:
        """Remove all data from the index."""
        # Create a new empty vectorstore
        self.vectorstore = FAISS.from_texts(
            ["placeholder"], 
            self.embeddings,
            metadatas=[{"source": "placeholder"}]
        )
        self._save_index()
        
        return {"status": "success", "message": "Index cleared."}
