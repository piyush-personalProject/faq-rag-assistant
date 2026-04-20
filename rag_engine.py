"""
rag_engine.py
Core RAG engine: loads text files, chunks them, creates embeddings,
builds FAISS index, and retrieves relevant chunks for queries.
"""

import os
import json
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K = int(os.getenv("TOP_K_RESULTS", 5))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

INDEX_PATH = "embeddings/faiss.index"
METADATA_PATH = "embeddings/metadata.pkl"


class RAGEngine:
    def __init__(self):
        print(f"[RAG] Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.chunks = []       # list of text chunks
        self.metadata = []     # list of dicts: {source, chunk_id}
        self._load_or_init_index()

    # ------------------------------------------------------------------
    # Index persistence
    # ------------------------------------------------------------------

    def _load_or_init_index(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
            print("[RAG] Loading existing FAISS index...")
            self.index = faiss.read_index(INDEX_PATH)
            with open(METADATA_PATH, "rb") as f:
                saved = pickle.load(f)
                self.chunks = saved["chunks"]
                self.metadata = saved["metadata"]
            print(f"[RAG] Loaded {len(self.chunks)} chunks from index.")
        else:
            print("[RAG] No existing index found. Starting fresh.")
            dim = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(dim)

    def _save_index(self):
        os.makedirs("embeddings", exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump({"chunks": self.chunks, "metadata": self.metadata}, f)
        print("[RAG] Index saved.")

    # ------------------------------------------------------------------
    # Text chunking
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str, source: str) -> list[dict]:
        words = text.split()
        chunks = []
        i = 0
        chunk_id = 0
        while i < len(words):
            chunk_words = words[i: i + CHUNK_SIZE]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "text": chunk_text,
                "source": source,
                "chunk_id": chunk_id,
            })
            chunk_id += 1
            i += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_folder(self, folder_path: str) -> dict:
        """
        Scan a folder for .txt files, embed them, and add to FAISS index.
        Returns a summary dict.
        """
        folder = Path(folder_path)
        txt_files = list(folder.rglob("*.txt"))
        if not txt_files:
            return {"status": "error", "message": "No .txt files found in folder."}

        new_chunks = []
        for txt_file in txt_files:
            print(f"[RAG] Processing: {txt_file.name}")
            text = txt_file.read_text(encoding="utf-8", errors="ignore")
            file_chunks = self._chunk_text(text, txt_file.name)
            new_chunks.extend(file_chunks)

        # Build embeddings
        texts = [c["text"] for c in new_chunks]
        print(f"[RAG] Embedding {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        embeddings = np.array(embeddings, dtype="float32")

        # Add to index
        self.index.add(embeddings)
        self.chunks.extend(texts)
        self.metadata.extend([{"source": c["source"], "chunk_id": c["chunk_id"]} for c in new_chunks])
        self._save_index()

        return {
            "status": "success",
            "files_processed": len(txt_files),
            "chunks_added": len(new_chunks),
            "total_chunks": len(self.chunks),
        }

    def ingest_file(self, file_path: str) -> dict:
        """Ingest a single .txt file."""
        p = Path(file_path)
        if not p.exists() or p.suffix != ".txt":
            return {"status": "error", "message": "File not found or not a .txt file."}

        text = p.read_text(encoding="utf-8", errors="ignore")
        new_chunks = self._chunk_text(text, p.name)
        texts = [c["text"] for c in new_chunks]
        embeddings = self.model.encode(texts, batch_size=32)
        embeddings = np.array(embeddings, dtype="float32")

        self.index.add(embeddings)
        self.chunks.extend(texts)
        self.metadata.extend([{"source": c["source"], "chunk_id": c["chunk_id"]} for c in new_chunks])
        self._save_index()

        return {
            "status": "success",
            "file": p.name,
            "chunks_added": len(new_chunks),
            "total_chunks": len(self.chunks),
        }

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """
        Embed the query and retrieve top-k relevant chunks.
        Returns list of {text, source, score} dicts.
        """
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

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        sources = list(set(m["source"] for m in self.metadata))
        return {
            "total_chunks": len(self.chunks),
            "total_vectors": self.index.ntotal,
            "sources": sources,
            "embedding_model": EMBEDDING_MODEL,
        }

    def clear_index(self) -> dict:
        dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []
        self.metadata = []
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
        if os.path.exists(METADATA_PATH):
            os.remove(METADATA_PATH)
        return {"status": "success", "message": "Index cleared."}
