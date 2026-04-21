# Knowledge Store - Enterprise RAG FAQ System

A modular, production-ready RAG (Retrieval Augmented Generation) FAQ assistant that uses local embeddings and LLMs for intelligent question answering with self-correction capabilities.

## Features

- **Local RAG Pipeline**: FAISS-powered semantic search with sentence transformers
- **Local LLM**: Uses distilgpt2 for text generation (CPU-friendly)
- **Self-Correcting RAG**: LangGraph-powered quality checks with automatic regeneration
- **Modular Architecture**: Clean separation of concerns for easy maintenance
- **Comprehensive Tests**: Unit tests for all core components
- **REST API**: Streaming responses for real-time chat experience

## Project Structure

```
knowledgeStore/
├── app.py                  # Flask application entry point
├── requirements.txt       # Python dependencies
├── .env                    # Environment variables
│
├── src/                    # Source modules
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── rag_engine.py       # FAISS indexing and retrieval
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
│   ├── test_rag_engine.py
│   └── test_graph_rag.py
│
├── docs/                   # Documentation
│   └── LANGGRAPH_INTEGRATION.md
│
├── data/faq_docs/          # FAQ documents storage
├── embeddings/             # FAISS index storage
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

# LLM Settings
LLM_MODEL=distilgpt2
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
| `LLM_MODEL` | distilgpt2 | Local LLM model |
| `LLM_MAX_TOKENS` | 150 | Max tokens in response |
| `LLM_TEMPERATURE` | 0.3 | LLM creativity (0-2) |

## Architecture

### Standard Query Flow
```
User Input
    ↓
QueryClassifier (detects greetings vs FAQ queries)
    ↓
[Simple Query] → Return contextual greeting
[FAQ Query] → RAGEngine.retrieve() → FAISS search
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
[Retrieve Docs] → FAISS similarity search
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

## Self-Correction with LangGraph

The system uses LangGraph to implement self-correcting RAG:

1. **Retrieve**: Fetch relevant documents from FAISS
2. **Generate**: Create initial answer using LLM
3. **Quality Check**: Evaluate answer quality (0-1 scale)
4. **Regenerate** (if needed): If quality < 0.7 and attempts < 2, regenerate with modified prompt

This ensures better accuracy by catching and correcting poor responses.

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
