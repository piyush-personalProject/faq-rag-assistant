"""
RAG Engine module.
Handles text chunking, embedding generation, FAISS indexing, BM25 keyword search,
hybrid retrieval, and cross-encoder reranking using LangChain.
"""

import os
import json
import hashlib
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from .config import config


class RAGEngine:
    """
    Core RAG engine for embedding generation, indexing, and retrieval.
    Uses LangChain for vector store and embeddings management.
    Supports:
    - Semantic search via FAISS
    - Keyword search via BM25
    - Hybrid search combining BM25 + semantic
    - Cross-encoder reranking (BAAI/bge-reranker-base)
    """
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"}
        )
        self.vectorstore = None
        self.bm25_index: Optional[BM25Okapi] = None
        self._corpus_texts: List[str] = []
        self._corpus_metadatas: List[Dict] = []
        self._ingested_files: Set[str] = set()
        self._reranker_model = None
        self._load_or_init_index()
        self._load_ingested_files()
    
    def _get_reranker(self):
        """Lazy load the cross-encoder reranker model."""
        if self._reranker_model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker_model = CrossEncoder(config.RERANKER_MODEL)
            except Exception as e:
                print(f"[RAG] Could not load reranker model: {e}")
                return None
        return self._reranker_model
    
    def _get_ingestion_tracking_path(self) -> Path:
        """Return path to the ingestion tracking file."""
        return config.EMBEDDINGS_DIR / "ingested_files.json"
    
    def _load_ingested_files(self) -> None:
        """Load the set of previously ingested files from tracking file."""
        tracking_path = self._get_ingestion_tracking_path()
        if tracking_path.exists():
            try:
                with open(tracking_path, "r") as f:
                    data = json.load(f)
                    self._ingested_files = set(data.get("files", []))
            except (json.JSONDecodeError, IOError):
                self._ingested_files = set()
    
    def _save_ingested_files(self) -> None:
        """Save the set of ingested files to tracking file."""
        tracking_path = self._get_ingestion_tracking_path()
        config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(tracking_path, "w") as f:
            json.dump({"files": list(self._ingested_files)}, f)
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Generate a hash for file content to detect changes."""
        hasher = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except IOError:
            return ""
    
    def _is_file_ingested(self, file_path: Path) -> bool:
        """Check if file has been ingested and unchanged since then."""
        file_hash = self._get_file_hash(file_path)
        if not file_hash:
            return False
        return file_hash in self._ingested_files
    
    def _mark_file_ingested(self, file_path: Path) -> None:
        """Mark a file as ingested by storing its content hash."""
        file_hash = self._get_file_hash(file_path)
        if file_hash:
            self._ingested_files.add(file_hash)
            self._save_ingested_files()
    
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
    
    def _semantic_chunk_text(self, text: str, source: str) -> List[Document]:
        """
        Split text into semantically coherent chunks using embeddings.
        
        Algorithm:
        1. Split text into sentences
        2. Create embeddings for each sentence
        3. Group consecutive sentences into chunks based on semantic similarity
        4. Start a new chunk when similarity drops below threshold or max tokens exceeded
        """
        # Split text into sentences using simple regex
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [Document(page_content=text, metadata={"source": source})]
        
        # If only one sentence, return as single chunk
        if len(sentences) == 1:
            return [Document(page_content=text, metadata={"source": source})]
        
        # Get embeddings for all sentences
        embeddings = self.embeddings.embed_documents(sentences)
        embeddings = np.array(embeddings)
        
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = [embeddings[0]]
        
        for i in range(1, len(sentences)):
            sentence_emb = embeddings[i]
            
            # Calculate similarity with previous sentence
            sim = cosine_similarity(
                sentence_emb.reshape(1, -1),
                np.mean(current_embedding, axis=0).reshape(1, -1)
            )[0][0]
            
            # Estimate token count (rough: 1 token ≈ 4 chars)
            estimated_tokens = sum(len(s) for s in current_chunk) / 4
            sentence_tokens = len(sentences[i]) / 4
            
            # Start new chunk if similarity is too low or chunk is getting too large
            if sim < config.SEMANTIC_SIMILARITY_THRESHOLD or (estimated_tokens + sentence_tokens) > config.CHUNK_SIZE:
                # Finalize current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(Document(page_content=chunk_text, metadata={"source": source}))
                current_chunk = [sentences[i]]
                current_embedding = [sentence_emb]
            else:
                current_chunk.append(sentences[i])
                current_embedding.append(sentence_emb)
        
        # Add the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(Document(page_content=chunk_text, metadata={"source": source}))
        
        return chunks
    
    def _load_document(self, file_path: Path) -> List[Document]:
        """Load document using appropriate LangChain loader based on file extension."""
        suffix = file_path.suffix.lower()
        
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            # Apply chunking to PDF content
            return self._chunk_or_semantic_chunk(docs[0].page_content, file_path.name) if docs else []
        elif suffix in [".doc", ".docx"]:
            loader = UnstructuredWordDocumentLoader(str(file_path))
            docs = loader.load()
            # Apply chunking to document content
            return self._chunk_or_semantic_chunk(docs[0].page_content, file_path.name) if docs else []
        elif suffix == ".txt":
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            return self._chunk_or_semantic_chunk(text, file_path.name)
        else:
            return []
    
    def _chunk_or_semantic_chunk(self, text: str, source: str) -> List[Document]:
        """Choose between regular or semantic chunking based on config."""
        if config.USE_SEMANTIC_CHUNKING:
            return self._semantic_chunk_text(text, source)
        return self._chunk_text(text, source)
    
    def _rebuild_bm25_index(self) -> None:
        """Rebuild the BM25 index from current corpus."""
        if not self._corpus_texts:
            # If no corpus texts stored, extract from vectorstore
            docs = self.vectorstore.similarity_search("*", k=10000)
            self._corpus_texts = [doc.page_content for doc in docs]
            self._corpus_metadatas = [doc.metadata for doc in docs]
        
        if self._corpus_texts:
            # Tokenize corpus for BM25
            tokenized_corpus = [text.lower().split() for text in self._corpus_texts]
            self.bm25_index = BM25Okapi(tokenized_corpus)
        else:
            self.bm25_index = None
    
    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[str, float, Dict]]:
        """
        Perform BM25 keyword search.
        
        Returns:
            List of (text, score, metadata) tuples
        """
        if self.bm25_index is None:
            self._rebuild_bm25_index()
        
        if self.bm25_index is None or not self._corpus_texts:
            return []
        
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k results with scores
        top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0:
                results.append((
                    self._corpus_texts[idx],
                    float(bm25_scores[idx]),
                    self._corpus_metadatas[idx]
                ))
        
        return results
    
    def _semantic_search(self, query: str, top_k: int) -> List[Tuple[str, float, Dict]]:
        """
        Perform semantic search using FAISS.
        
        Returns:
            List of (text, score, metadata) tuples
        """
        if self.vectorstore.index.ntotal == 0:
            return []
        
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        
        # Normalize scores (lower distance = higher score)
        max_score = max(score for _, score in results) if results else 1.0
        min_score = min(score for _, score in results) if results else 0.0
        
        normalized_results = []
        for doc, score in results:
            # Convert distance to similarity score (invert and normalize)
            if max_score != min_score:
                normalized_score = 1.0 - (score - min_score) / (max_score - min_score)
            else:
                normalized_score = 1.0 if score == 0 else 0.0
            normalized_results.append((
                doc.page_content,
                float(normalized_score),
                doc.metadata
            ))
        
        return normalized_results
    
    def _hybrid_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Combine BM25 and semantic search with weighted scoring.
        
        Args:
            query: Search query string
            top_k: Number of results to return before reranking
            
        Returns:
            List of dicts with text, source, and combined score
        """
        # Get more results than needed for reranking
        retrieval_k = max(top_k * 3, config.RERANK_TOP_K)
        
        bm25_results = self._bm25_search(query, retrieval_k)
        semantic_results = self._semantic_search(query, retrieval_k)
        
        # Create score maps
        bm25_scores = {text: score for text, score, _ in bm25_results}
        semantic_scores = {text: score for text, score, _ in semantic_results}
        
        # Get all unique texts
        all_texts = set(bm25_scores.keys()) | set(semantic_scores.keys())
        
        # Normalize BM25 scores to 0-1 range
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
        min_bm25 = min(bm25_scores.values()) if bm25_scores else 0.0
        
        # Calculate combined scores
        combined_results = []
        for text in all_texts:
            bm25_score = bm25_scores.get(text, 0.0)
            semantic_score = semantic_scores.get(text, 0.0)
            
            # Normalize BM25 score
            if max_bm25 != min_bm25 and bm25_score > 0:
                norm_bm25 = (bm25_score - min_bm25) / (max_bm25 - min_bm25)
            else:
                norm_bm25 = 0.0
            
            # Combined score with weights
            combined_score = (
                config.BM25_WEIGHT * norm_bm25 +
                config.SEMANTIC_WEIGHT * semantic_score
            )
            
            # Get metadata - find the entry in bm25 or semantic results matching this text
            metadata = None
            for t, _, m in bm25_results:
                if t == text:
                    metadata = m
                    break
            if metadata is None:
                for t, _, m in semantic_results:
                    if t == text:
                        metadata = m
                        break
            
            combined_results.append({
                "text": text,
                "source": metadata.get("source", "unknown") if metadata else "unknown",
                "score": combined_score,
                "bm25_score": norm_bm25,
                "semantic_score": semantic_score,
            })
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        
        return combined_results[:top_k]
    
    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Rerank results using cross-encoder (BAAI/bge-reranker-base).
        
        Args:
            query: Original query string
            results: List of search results to rerank
            
        Returns:
            Reranked list of dicts with text, source, and rerank score
        """
        if not results or not config.USE_RERANKING:
            return results
        
        reranker = self._get_reranker()
        if reranker is None:
            return results
        
        try:
            # Prepare query-document pairs for cross-encoder
            pairs = [[query, result["text"]] for result in results]
            
            # Get relevance scores
            rerank_scores = reranker.predict(pairs)
            
            # Add rerank scores to results and resort
            for i, result in enumerate(results):
                result["rerank_score"] = float(rerank_scores[i])
                result["score"] = float(rerank_scores[i])  # Replace score with rerank score
            
            # Sort by rerank score
            results.sort(key=lambda x: x["rerank_score"], reverse=True)
            
        except Exception as e:
            print(f"[RAG] Reranking failed: {e}")
        
        return results
    
    def ingest_folder(self, folder_path: str) -> Dict:
        """
        Scan folder for .txt, .pdf, .doc, .docx files, embed them, and add to FAISS index.
        
        Args:
            folder_path: Path to folder containing supported files
            
        Returns:
            Summary dict with processing results
        """
        folder = Path(folder_path)
        supported_extensions = [".txt", ".pdf", ".doc", ".docx"]
        files = [f for f in folder.rglob("*") if f.suffix.lower() in supported_extensions]
        
        if not files:
            return {"status": "error", "message": f"No supported files found in folder. Supported: {supported_extensions}"}
        
        all_documents = []
        skipped_count = 0
        for file_path in files:
            if self._is_file_ingested(file_path):
                skipped_count += 1
                continue
            documents = self._load_document(file_path)
            if documents:
                # _load_document already returns properly chunked documents
                all_documents.extend(documents)
                self._mark_file_ingested(file_path)
        
        if not all_documents:
            return {"status": "error", "message": "No content could be extracted from files."}
        
        # Add documents to vectorstore
        self.vectorstore.add_documents(all_documents)
        
        # Update corpus texts for BM25
        for doc in all_documents:
            self._corpus_texts.append(doc.page_content)
            self._corpus_metadatas.append(doc.metadata)
        
        # Rebuild BM25 index
        self._rebuild_bm25_index()
        
        self._save_index()
        
        return {
            "status": "success",
            "files_processed": len(files) - skipped_count,
            "files_skipped": skipped_count,
            "chunks_added": len(all_documents),
            "total_chunks": self.vectorstore.index.ntotal,
        }
    
    def ingest_file(self, file_path: str) -> Dict:
        """
        Ingest a single .txt, .pdf, .doc, or .docx file into the index.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Summary dict with processing results
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        if not path.exists() or suffix not in [".txt", ".pdf", ".doc", ".docx"]:
            return {"status": "error", "message": f"File not found or unsupported format. Supported: .txt, .pdf, .doc, .docx"}
        
        # Check if file has already been ingested (skip duplicates)
        if self._is_file_ingested(path):
            return {
                "status": "skipped",
                "file": path.name,
                "reason": "File has not changed since last ingestion",
            }
        
        documents = self._load_document(path)
        if not documents:
            return {"status": "error", "message": "No content could be extracted from file."}
        
        chunked = self._chunk_text(documents[0].page_content, path.name)
        self.vectorstore.add_documents(chunked)
        
        # Update corpus texts for BM25
        for doc in chunked:
            self._corpus_texts.append(doc.page_content)
            self._corpus_metadatas.append(doc.metadata)
        
        # Rebuild BM25 index
        self._rebuild_bm25_index()
        
        self._save_index()
        self._mark_file_ingested(path)
        
        return {
            "status": "success",
            "file": path.name,
            "chunks_added": len(chunked),
            "total_chunks": self.vectorstore.index.ntotal,
        }
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve relevant chunks using hybrid search + reranking.
        
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
        
        # Use hybrid search if enabled
        if config.USE_HYBRID_SEARCH:
            results = self._hybrid_search(query, top_k * 3)  # Get more for reranking
        else:
            # Fall back to semantic-only search
            semantic_results = self._semantic_search(query, top_k)
            results = [
                {"text": text, "source": meta.get("source", "unknown"), "score": score}
                for text, score, meta in semantic_results
            ]
        
        # Apply reranking if enabled and we have results
        if results and config.USE_RERANKING:
            results = self._rerank_results(query, results)
        
        return results[:top_k]
    
    def get_status(self) -> Dict:
        """Return index status information."""
        if self.vectorstore is None:
            return {
                "total_chunks": 0,
                "total_vectors": 0,
                "sources": [],
                "embedding_model": config.EMBEDDING_MODEL,
                "bm25_enabled": config.USE_HYBRID_SEARCH,
                "reranking_enabled": config.USE_RERANKING,
            }
        
        # Get unique sources from the documents
        docs = self.vectorstore.similarity_search("*", k=10000)
        sources = list(set(doc.metadata.get("source", "unknown") for doc in docs))
        
        return {
            "total_chunks": self.vectorstore.index.ntotal,
            "total_vectors": self.vectorstore.index.ntotal,
            "sources": sources,
            "embedding_model": config.EMBEDDING_MODEL,
            "bm25_enabled": config.USE_HYBRID_SEARCH,
            "reranking_enabled": config.USE_RERANKING,
        }
    
    def clear_index(self) -> Dict:
        """Remove all data from the index."""
        # Create a new empty vectorstore
        self.vectorstore = FAISS.from_texts(
            ["placeholder"], 
            self.embeddings,
            metadatas=[{"source": "placeholder"}]
        )
        
        # Clear BM25 index
        self.bm25_index = None
        self._corpus_texts = []
        self._corpus_metadatas = []
        
        self._save_index()
        # Clear ingested files tracking
        self._ingested_files = set()
        self._save_ingested_files()
        
        return {"status": "success", "message": "Index cleared."}
