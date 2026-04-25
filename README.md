# Knowledge Store - Enterprise RAG FAQ System

A modular, production-ready RAG (Retrieval Augmented Generation) FAQ assistant that uses local embeddings and LLMs for intelligent question answering with self-correction capabilities.

## Features

- **Hybrid Search Pipeline**: BM25 keyword search + FAISS semantic search combined with weighted scoring
- **Cross-Encoder Reranking**: BAAI/bge-reranker-base for improved result ranking
- **Local RAG Pipeline**: FAISS-powered semantic search with sentence transformers
- **Local LLM**: Uses gpt2-medium for text generation (CPU-friendly)
- **Self-Correcting RAG**: LangGraph-powered quality checks with automatic regeneration using faithfulness, answer relevance, and context precision metrics
- **Conversation History**: Multi-turn dialogue support for contextual follow-up questions
- **Modular Architecture**: Clean separation of concerns for easy maintenance
- **Comprehensive Tests**: Unit tests for all core components including hybrid search and reranking
- **REST API**: Streaming responses for real-time chat experience
- **MCP Server**: Model Context Protocol server for AI assistant integration

## Project Structure

```
knowledgeStore/
├── app.py                  # Flask application entry point
├── mcp_server.py           # MCP server for AI assistant integration
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
│
├── src/                    # Source modules
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── rag_engine.py       # FAISS + BM25 indexing, hybrid retrieval, reranking
│   ├── rag_evaluator.py    # RAG evaluation metrics (faithfulness, answer relevance, context precision)
│   ├── rag_trace.py       # Document-to-answer lineage tracing with optional LangSmith support
│   ├── llm.py              # LLM text generation with reasoning
│   ├── graph_rag.py        # LangGraph self-correcting RAG
│   ├── query_classifier.py # Query type detection
│   ├── answer_extractor.py # Answer extraction from chunks
│   └── routes.py           # Flask API routes
│
├── tests/                  # Test suite
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_query_classifier.py
│   ├── test_answer_extractor.py
│   ├── test_llm.py
│   ├── test_rag_engine.py  # Includes hybrid search + reranking tests
│   ├── test_rag_evaluator.py  # Tests for faithfulness, answer relevance, context precision
│   ├── test_rag_trace.py     # Tests for document-to-answer lineage tracing
│   └── test_graph_rag.py
│
├── docs/                   # Documentation
│   └── MCP_INTEGRATION.md
│
├── data/faq_docs/          # FAQ documents storage
├── embeddings/             # FAISS index + BM25 storage
└── templates/
    └── index.html          # Web UI
```

## Quick Start

### 1. Create Virtual Environment

```bash
# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create or edit `.env` file:

```env
# Flask
FLASK_PORT=5000
FLASK_DEBUG=True

# RAG Settings
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=5
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Hybrid Search & Reranking (enabled by default)
USE_HYBRID_SEARCH=True
BM25_WEIGHT=0.3
SEMANTIC_WEIGHT=0.7
USE_RERANKING=True
RERANKER_MODEL=BAAI/bge-reranker-base

# LLM Settings
LLM_MODEL=gpt2-medium
LLM_MAX_TOKENS=150
LLM_TEMPERATURE=0.3
```

### 4. Run the Application

```bash
python app.py
```

The server will start at `http://localhost:5000`

### 5. Run Tests

```bash
pytest tests/ -v
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve web UI |
| `/api/status` | GET | Get system status |
| `/api/ingest/folder` | POST | Ingest all .txt files from data/faq_docs |
| `/api/upload` | POST | Upload and ingest .txt files |
| `/api/clear` | POST | Clear the index |
| `/api/chat` | POST | Chat with the FAQ assistant |

### Chat Request Example

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I track my order?"}'
```

### Chat with History (Multi-turn Conversation)

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What about exchanges?",
    "history": [
      {"role": "user", "content": "How do returns work?"},
      {"role": "assistant", "content": "Items can be returned within 30 days with receipt."}
    ]
  }'
```

## Adding FAQ Content

Place `.txt` files in `data/faq_docs/` directory. The system will automatically ingest them on startup.

Example FAQ format:

```
Q: How do I track my order?
A: You can track your order using the tracking link sent to your email.

Q: What is your return policy?
A: You can return items within 30 days of purchase.
```

## Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | 500 | Characters per text chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `TOP_K_RESULTS` | 5 | Number of chunks to retrieve |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |
| `USE_HYBRID_SEARCH` | True | Enable BM25 + semantic hybrid search |
| `BM25_WEIGHT` | 0.3 | Weight for BM25 scores (0.0-1.0) |
| `SEMANTIC_WEIGHT` | 0.7 | Weight for semantic scores (0.0-1.0) |
| `USE_RERANKING` | True | Enable cross-encoder reranking |
| `RERANKER_MODEL` | BAAI/bge-reranker-base | Cross-encoder model for reranking |
| `LLM_MODEL` | gpt2-medium | Local LLM model |
| `LLM_MAX_TOKENS` | 150 | Max tokens in response |
| `LLM_TEMPERATURE` | 0.3 | LLM creativity (0-2) |

## Architecture

### Retrieval Pipeline (Hybrid Search + Reranking)

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         BM25 Keyword Search          │
│     (rank-bm25 + tokenized corpus)  │
└─────────────────┬───────────────────┘
                  │ scores
                  ▼
┌─────────────────────────────────────┐
│        FAISS Semantic Search        │
│   (sentence-transformers embeddings) │
└─────────────────┬───────────────────┘
                  │ scores
                  ▼
┌─────────────────────────────────────┐
│        Hybrid Score Combination     │
│  score = BM25_WEIGHT * norm_bm25    │
│        + SEMANTIC_WEIGHT * semantic │
└─────────────────┬───────────────────┘
                  │ top results
                  ▼
┌─────────────────────────────────────┐
│       Cross-Encoder Reranking      │
│   (BAAI/bge-reranker-base)         │
│   query-document pairs scored       │
└─────────────────┬───────────────────┘
                  │ reranked
                  ▼
         Final Results
```

### Standard Query Flow

```
User Input
    ↓
QueryClassifier (detects greetings vs FAQ queries)
    ↓
[Simple Query] → Return contextual greeting
[FAQ Query] → RAGEngine.retrieve() → Hybrid search + reranking
    ↓
[LLM Available] → LLMManager.generate() with context
[No LLM] → AnswerExtractor.extract() from chunks
    ↓
Streaming Response
```

### Self-Correcting RAG Flow (LangGraph)

```
User Query
    ↓
Classify Query → Simple? → Use fast path
    ↓ (complex)
[Retrieve Docs] → Hybrid search + reranking
    ↓
[Generate Answer] → LLM with context
    ↓
[Quality Check] → Score ≥ 0.7? → END (good)
    ↓ (no)
[Regenerate] → Attempt 2
    ↓
[Quality Check] → Still poor? → END (finalize)
    ↓ (no)
[Regenerate] → Attempt 3
    ↓
[END]
```

## Retrieval Components

### Hybrid Search (BM25 + Semantic)

The system combines two retrieval methods:

1. **BM25 Keyword Search**: Traditional keyword-based retrieval using `rank-bm25`
   - Tokenizes corpus and query
   - Calculates TF-IDF-like scores
   - Good for exact keyword matches

2. **FAISS Semantic Search**: Vector-based similarity search
   - Embeds text using `all-MiniLM-L6-v2` (384 dimensions)
   - Uses FAISS IndexFlatL2 for fast similarity search
   - Good for semantic/meaning-based matches

**Hybrid Combination**:
```
combined_score = BM25_WEIGHT * normalized_bm25_score + SEMANTIC_WEIGHT * semantic_score
```

### Cross-Encoder Reranking

After initial retrieval, results are reranked using `BAAI/bge-reranker-base`:

- Takes query-document pairs as input
- Outputs relevance scores (0-1)
- Reorders results by cross-encoder scores
- Better at capturing query-document relevance than embedding similarity alone

## RAG Evaluation Metrics

The system implements three key RAG evaluation metrics:

### Faithfulness
Measures whether the answer's claims are supported by the retrieved context.
- Analyzes sentence-level token overlap with context
- Identifies unfaithful claims that lack context grounding
- Score: 0-1 (higher is better)

### Answer Relevance  
Measures whether the answer addresses the user's query.
- Checks query term coverage in answer
- Identifies irrelevant sentences that don't relate to query
- Penalizes tangential information
- Score: 0-1 (higher is better)

### Context Precision
Measures whether retrieved contexts are relevant to answering the query.
- Evaluates each retrieved document for query term overlap
- Uses n-gram matching for better semantic capture
- Identifies irrelevant context indices
- Score: 0-1 (higher is better)

### Overall Quality Score
Weighted combination: `0.4 * faithfulness + 0.4 * answer_relevance + 0.2 * context_precision`

### Implementation
The [`RAGEvaluator`](src/rag_evaluator.py) class provides:
- [`evaluate_faithfulness()`](src/rag_evaluator.py:54) - Check answer-context alignment
- [`evaluate_answer_relevance()`](src/rag_evaluator.py:92) - Check answer-query alignment  
- [`evaluate_context_precision()`](src/rag_evaluator.py:132) - Check retrieval quality
- [`evaluate()`](src/rag_evaluator.py:174) - Run all metrics

The self-correcting RAG graph uses these metrics in its quality check node to determine if regeneration is needed.

## Document-to-Answer Lineage Tracing

The system tracks which documents/chunks contributed to each answer via [`rag_trace.py`](src/rag_trace.py):

### Features
- **Chunk-level attribution**: Records each retrieved chunk with score and usage status
- **File-based trace storage**: Traces saved as JSON to `embeddings/traces/` directory
- **Query-level lineage**: Shows which chunks led to which answer
- **Optional LangSmith integration**: Cloud tracing when `LANGSMITH_API_KEY` is set

### Configuration
```env
# Enable/disable tracing (default: True)
ENABLE_TRACING=True

# LangSmith integration (optional)
LANGSMITH_API_KEY=your_api_key_here
LANGSMITH_PROJECT=knowledgeStore
```

### Usage
```python
from src.rag_trace import get_tracer, configure_langsmith_tracing

# Enable LangSmith tracing
configure_langsmith_tracing("your-api-key", "my-project")

# Get traces for a specific source
tracer = get_tracer()
traces = tracer.get_traces_by_source("returns_and_refunds.txt")

# Get recent traces
recent = tracer.get_recent_traces(limit=10)

# Get trace summary with source usage statistics
summary = tracer.get_trace_summary()
```

### Trace Response
The `query()` method now returns a `trace_id` that can be used to retrieve the full trace:
```python
result = graph_rag.query("How do I return an item?")
print(result["trace_id"])  # e.g., "rag_1234567890"
```

## Self-Correction with LangGraph

The system uses LangGraph to implement self-correcting RAG:

1. **Retrieve**: Fetch relevant documents using hybrid search + reranking
2. **Generate**: Create initial answer using LLM
3. **Quality Check**: Evaluate answer using faithfulness, answer_relevance, and context_precision
4. **Regenerate** (if needed): If overall quality < 0.7 and attempts < 2, regenerate with modified prompt

This ensures better accuracy by catching and correcting poor responses using proper evaluation metrics.

## Development

### Running in Development Mode

```bash
FLASK_DEBUG=True python app.py
```

### Running Tests with Coverage

```bash
pytest tests/ -v --cov=src
```

### Adding Reasoning to LLM

The LLM module supports chain-of-thought reasoning:

```python
from src.llm import llm

# Generate with step-by-step reasoning
response = llm.generate_with_reasoning(context, question)

# Extract final answer from reasoning output
answer = llm.extract_answer_from_reasoning(response)
```

## Dependencies

Key new dependencies for hybrid search and reranking:

- **rank-bm25**: BM25 keyword search algorithm
- **sentence-transformers**: Cross-encoder models for reranking
