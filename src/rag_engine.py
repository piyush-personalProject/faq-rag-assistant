"""
RAG Engine module.
Handles text chunking, embedding generation, FAISS indexing, and retrieval.
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .config import config


class RAGEngine:
    """
    Core RAG engine for embedding generation, indexing, and retrieval.
    """
    
    def __init__(self):
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.index = None
        self.chunks = []
        self.metadata = []
        self._load_or_init_index()
    
    def _load_or_init_index(self) -> None:
        """Load existing FAISS index or initialize a new one."""
        if config.INDEX_PATH.exists() and config.METADATA_PATH.exists():
            self.index = faiss.read_index(str(config.INDEX_PATH))
            with open(config.METADATA_PATH, "rb") as f:
                saved = pickle.load(f)
                self.chunks = saved["chunks"]
                self.metadata = saved["metadata"]
        else:
            dim = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(dim)
    
    def _save_index(self) -> None:
        """Persist FAISS index and metadata to disk."""
        config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(config.INDEX_PATH))
        with open(config.METADATA_PATH, "wb") as f:
            pickle.dump({"chunks": self.chunks, "metadata": self.metadata}, f)
    
    def _chunk_text(self, text: str, source: str) -> List[Dict]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        i = 0
        chunk_id = 0
        
        while i < len(words):
            chunk_words = words[i: i + config.CHUNK_SIZE]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "text": chunk_text,
                "source": source,
                "chunk_id": chunk_id,
            })
            chunk_id += 1
            i += config.CHUNK_SIZE - config.CHUNK_OVERLAP
        
        return chunks
    
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
        
        new_chunks = []
        for txt_file in txt_files:
            text = txt_file.read_text(encoding="utf-8", errors="ignore")
            file_chunks = self._chunk_text(text, txt_file.name)
            new_chunks.extend(file_chunks)
        
        texts = [c["text"] for c in new_chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        embeddings = np.array(embeddings, dtype="float32")
        
        self.index.add(embeddings)
        self.chunks.extend(texts)
        self.metadata.extend([
            {"source": c["source"], "chunk_id": c["chunk_id"]} 
            for c in new_chunks
        ])
        self._save_index()
        
        return {
            "status": "success",
            "files_processed": len(txt_files),
            "chunks_added": len(new_chunks),
            "total_chunks": len(self.chunks),
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
        new_chunks = self._chunk_text(text, path.name)
        texts = [c["text"] for c in new_chunks]
        embeddings = self.model.encode(texts, batch_size=32)
        embeddings = np.array(embeddings, dtype="float32")
        
        self.index.add(embeddings)
        self.chunks.extend(texts)
        self.metadata.extend([
            {"source": c["source"], "chunk_id": c["chunk_id"]} 
            for c in new_chunks
        ])
        self._save_index()
        
        return {
            "status": "success",
            "file": path.name,
            "chunks_added": len(new_chunks),
            "total_chunks": len(self.chunks),
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
            
        if self.index.ntotal == 0:
            return []
        
        q_embed = self.model.encode([query], batch_size=1)
        q_embed = np.array(q_embed, dtype="float32")
        distances, indices = self.index.search(q_embed, min(top_k, self.index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.chunks[idx],
                "source": self.metadata[idx]["source"],
                "score": float(dist),
            })
        
        return results
    
    def get_status(self) -> Dict:
        """Return index status information."""
        sources = list(set(m["source"] for m in self.metadata))
        return {
            "total_chunks": len(self.chunks),
            "total_vectors": self.index.ntotal,
            "sources": sources,
            "embedding_model": config.EMBEDDING_MODEL,
        }
    
    def clear_index(self) -> Dict:
        """Remove all data from the index."""
        dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []
        self.metadata = []
        
        if config.INDEX_PATH.exists():
            config.INDEX_PATH.unlink()
        if config.METADATA_PATH.exists():
            config.METADATA_PATH.unlink()
        
        return {"status": "success", "message": "Index cleared."}
