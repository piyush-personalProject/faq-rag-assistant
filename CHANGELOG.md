# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-04-24

### Added

- **Hybrid Search Pipeline**: BM25 keyword search + FAISS semantic search combined with weighted scoring
- **Cross-Encoder Reranking**: BAAI/bge-reranker-base for improved result ranking
- **Local RAG Pipeline**: FAISS-powered semantic search with sentence transformers
- **Local LLM**: Uses gpt2-medium for text generation (CPU-friendly)
- **Self-Correcting RAG**: LangGraph-powered quality checks with automatic regeneration
- **Conversation History**: Multi-turn dialogue support for contextual follow-up questions
- **REST API**: Streaming responses for real-time chat experience
- **MCP Server**: Model Context Protocol server for AI assistant integration
- **Comprehensive Tests**: Unit tests for all core components including hybrid search and reranking
- **Modular Architecture**: Clean separation of concerns for easy maintenance

### Features

- `src/rag_engine.py` - FAISS + BM25 indexing, hybrid retrieval, reranking
- `src/llm.py` - LLM text generation with reasoning
- `src/graph_rag.py` - LangGraph self-correcting RAG
- `src/query_classifier.py` - Query type detection
- `src/answer_extractor.py` - Answer extraction from chunks
- `src/routes.py` - Flask API routes
- `src/config.py` - Configuration management

### Documentation

- `README.md` - Project overview and quick start guide
- `ARCHITECTURE.md` - Detailed system architecture
- `docs/MCP_INTEGRATION.md` - MCP server integration guide

---

## [Unreleased]

### Planned

- GPU support for faster LLM inference
- Larger local models for improved accuracy
- Query routing based on query complexity
- Support for additional document formats (PDF, DOCX)

[1.0.0]: https://github.com/YOUR_USERNAME/knowledgeStore/releases/tag/v1.0.0
