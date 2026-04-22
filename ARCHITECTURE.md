# Knowledge Store - RAG System Architecture

## Overview

This is a **Retrieval-Augmented Generation (RAG)** system that combines vector-based semantic search with a local Large Language Model (LLM) to provide intelligent question-answering over FAQ documents.

**Key Features:**
- All processing happens locally - no external API calls required
- Self-correcting RAG with LangGraph quality checks
- Chain-of-thought reasoning for better accuracy
- Multi-turn conversation support with history tracking
- MCP (Model Context Protocol) server for AI assistant integration

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT (Browser)                                │
│                                                                             │
│   ┌─────────────┐    ┌─────────────────┐    ┌──────────────────────────┐  │
│   │  HTML/CSS   │    │  JavaScript UI  │    │  Streaming Response      │  │
│   │  Frontend   │◄──►│  (index.html)   │◄──►│  (SSE - Server-Sent     │  │
│   │             │    │                 │    │   Events)               │  │
│   └─────────────┘    └─────────────────┘    └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼ HTTP/JSON
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FLASK BACKEND (app.py)                            │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                        API Endpoints                                 │  │
│   │   GET  /                    → Serve frontend (index.html)           │  │
│   │   GET  /api/status          → Knowledge base statistics              │  │
│   │   POST /api/chat            → Main Q&A endpoint                      │  │
│   │   POST /api/upload          → Upload FAQ files                        │  │
│   │   POST /api/ingest/folder   → Ingest all .txt files                  │  │
│   │   POST /api/clear           → Clear FAISS index                       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │               CHAT ENDPOINT FLOW (with Self-Correction)             │  │
│   │                                                                     │  │
│   │   1. Classify query (simple vs complex)                            │  │
│   │   2. Simple → Return greeting (fast path)                          │  │
│   │   3. Complex → GraphRAG.query() → LangGraph self-correcting flow   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                     ┌─────────────────┴─────────────────┐
                     ▼                                   ▼
┌─────────────────────────────┐       ┌─────────────────────────────────────┐
│      RAG ENGINE             │       │      LOCAL LLM                      │
│      (rag_engine.py)        │       │      (gpt2-medium)                  │
│                            │       │                                     │
│  ┌───────────────────────┐ │       │  • 82MB model                       │
│  │  SentenceTransformer   │ │       │  • CPU inference                   │
│  │  (all-MiniLM-L6-v2)    │ │       │  • Text generation                 │
│  │                       │ │       │  • No API calls                     │
│  │  • Embeddings: 384dim │ │       │                                     │
│  │  • Semantic search     │ │       └─────────────────────────────────────┘
│  └───────────────────────┘ │
│            │               │
│            ▼               │
│  ┌───────────────────────┐ │
│  │  FAISS Index           │ │
│  │  (IndexFlatL2)         │ │
│  │                       │ │
│  │  • Fast similarity    │ │
│  │    search             │ │
│  │  • Stores chunks       │ │
│  └───────────────────────┘ │
│            │               │
│            ▼               │
│  ┌───────────────────────┐ │
│  │  Text Chunks           │ │
│  │  (Metadata + Text)    │ │
│  └───────────────────────┘ │
└─────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LANGRAPH SELF-CORRECTION LAYER                         │
│                      (graph_rag.py)                                         │
│                                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌───────────┐  │
│   │  Retrieve   │───▶│  Generate   │───▶│  Quality    │───▶│  Done?    │  │
│   │  (FAISS)    │    │  (LLM)      │    │  Check      │    │           │  │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────┬─────┘  │
│                                                              │         │
│                   ┌──────────────────────────────────────────┘         │
│                   ▼                                                  │
│            ┌─────────────┐    ┌─────────────┐                        │
│            │  Regenerate │───▶│  Quality    │◀──────────────────────│
│            │  (retry)    │    │  Check      │                        │
│            └─────────────┘    └─────────────┘                        │
│                   │                                                │
│                   └────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                         │
│                                                                             │
│   ┌─────────────────────┐        ┌─────────────────────────────────────┐  │
│   │  FAQ Documents       │        │  Persisted Index                    │  │
│   │  (data/faq_docs/)   │        │  (embeddings/)                       │  │
│   │                     │        │                                     │  │
│   │  • .txt files       │        │  • faiss.index (FAISS binary)       │  │
│   │  • Chunked & indexed │        │  • metadata.pkl (chunks + meta)     │  │
│   └─────────────────────┘        └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Frontend (templates/index.html)

**Technology:** Vanilla JavaScript, HTML5, CSS3

**Features:**
- Dark-themed modern UI
- Real-time streaming responses (SSE)
- File upload with drag-and-drop
- Status indicators for knowledge base
- Chat history with conversation management

**Key Functions:**
```javascript
sendMessage()      // Sends query to /api/chat, handles streaming
refreshStatus()    // Polls /api/status every 15 seconds
uploadFiles()      // Handles file uploads to /api/upload
```

### 2. Flask Backend (app.py)

**Technology:** Python 3, Flask, Flask-CORS

**API Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serve frontend HTML |
| `/api/status` | GET | Return knowledge base statistics |
| `/api/chat` | POST | Process Q&A with RAG + LLM |
| `/api/upload` | POST | Upload .txt files |
| `/api/ingest/folder` | POST | Ingest all files in FAQ folder |
| `/api/clear` | POST | Clear the FAISS index |

**Chat Endpoint Flow:**
```
1. Validate user message
2. Classify query (simple vs complex)
3. Simple query → Return greeting response (fast path)
4. Complex query → Use GraphRAG (LangGraph self-correcting flow)
5. Stream response back via SSE
```

### 3. Self-Correcting RAG Engine (graph_rag.py)

**Technology:** LangGraph, StateGraph, TypedDict

**Self-Correction Flow:**
```
Retrieve → Generate → Quality Check → (Regenerate if poor) → Done
```

**State (RAGState):**
```python
{
    "query": str,           # User question
    "history": list,         # Conversation history (BaseMessage objects)
    "retrieved_docs": list, # FAISS results
    "context": str,         # Combined text from docs
    "answer": str,          # Generated response
    "quality_score": float, # 0.0 to 1.0
    "attempts": int,        # Regeneration attempts (max 2)
    "messages": list        # Conversation messages
}
```

**Quality Threshold:** 0.7 (regenerates if below)

**Max Regeneration Attempts:** 2

### 4. RAG Engine (rag_engine.py)

**Technology:** LangChain, LangChain-Community, HuggingFaceEmbeddings, FAISS

**Components:**

#### 4.1 Embedding Model
- **Model:** `all-MiniLM-L6-v2` via LangChain's `HuggingFaceEmbeddings`
- **Dimensions:** 384
- **Purpose:** Convert text into semantic vector embeddings

#### 4.2 Text Chunking
```python
CHUNK_SIZE = 500 characters
CHUNK_OVERLAP = 50 characters
```
- FAQ files are split into overlapping chunks using LangChain's `RecursiveCharacterTextSplitter`
- Preserves context at chunk boundaries

#### 4.3 FAISS Vector Store
- **Type:** LangChain's `FAISS` vectorstore (wraps FAISS IndexFlatL2)
- **Purpose:** Fast similarity search on embeddings
- **Persistence:** Saved to `embeddings/` directory

#### 4.4 Retrieval Process
```python
# LangChain simplifies retrieval
results = vectorstore.similarity_search_with_score(query, k=5)
```

### 5. Local LLM (gpt2-medium)

**Technology:** LangChain HuggingFace Pipeline

**Model Details:**
- **Size:** 82 million parameters (gpt2-medium)
- **Memory:** ~500MB RAM
- **Inference Device:** CPU
- **Generation Settings:**
  - `max_new_tokens=150`
  - `temperature=0.3` (low for factual Q&A)
  - `top_p=0.9`
  - `repetition_penalty=1.2`

**LangChain Integration:**
- Uses `HuggingFacePipeline` from `langchain_huggingface`
- Uses `LLMChain` for prompt handling
- Uses `PromptTemplate` for structured prompts

### 6. Reasoning Capabilities (llm.py)

**Chain-of-Thought Prompting:**
```python
# generate_with_reasoning() adds step-by-step thinking
llm.generate_with_reasoning(context, question)
# Output includes reasoning steps + final answer
```

### 7. Conversation History (Multi-turn Support)

**History Format:**
- Client sends `history` array in chat request: `[{"role": "user"|"assistant", "content": "..."}]`
- Server converts to `HumanMessage`/`AIMessage` objects from LangChain

**History Processing:**
```python
# In graph_rag.py - _format_history()
def _format_history(self, history: List[BaseMessage]) -> str:
    # Formats last 6 messages into prompt context
    formatted = []
    for msg in history[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)
```

**Integration Points:**
- `routes.py`: Converts incoming history dicts to LangChain messages
- `graph_rag.py`: Passes history through RAGState; formats for LLM prompts
- `llm.py`: Includes history context in `generate_with_context()`

---

## Data Flow Example

### Question: "How long does a refund take?"

```
1. CLIENT
   └── "How long does a refund take?" 
       └── POST /api/chat

2. FLASK (QueryClassifier)
   └── Classify as "complex" (not a simple greeting)
   └── Call graph_rag.query()

3. LANGRAPH SELF-CORRECTION FLOW
   a. RETRIEVE
       └── rag.retrieve() → FAISS search
       └── Returns 5 nearest chunks
   
   b. GENERATE
       └── llm.generate_with_context(context, question)
       └── Initial answer: "Refunds take about a week"
   
   c. QUALITY CHECK
       └── Evaluates answer quality: 0.6 (below threshold)
       └── Decision: "regenerate"
   
   d. REGENERATE (attempt 1)
       └── More explicit prompt
       └── New answer: "Refunds are processed within 3-5 business days"
   
   e. QUALITY CHECK (attempt 2)
       └── Score: 0.85 (above threshold)
       └── Decision: "good" → END

4. CLIENT (SSE Streaming)
   └── data: {"type": "sources", "sources": ["returns_and_refunds.txt"]}
   └── data: {"type": "token", "text": "Refund"}
   └── data: {"type": "token", "text": "are"}
   └── ...
   └── data: {"type": "done"}
```

---

## File Structure

```
knowledgeStore/
├── app.py                    # Flask backend
├── rag_engine.py             # RAG engine (embeddings + FAISS)
├── mcp_server.py            # MCP server for AI assistant integration
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuration management
│   ├── rag_engine.py        # FAISS indexing and retrieval
│   ├── llm.py               # LLM with reasoning capabilities
│   ├── graph_rag.py         # LangGraph self-correcting RAG
│   ├── query_classifier.py  # Query type detection
│   ├── answer_extractor.py  # Answer extraction from chunks
│   └── routes.py           # Flask API routes
├── docs/
│   └── MCP_INTEGRATION.md
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_query_classifier.py
│   ├── test_answer_extractor.py
│   ├── test_llm.py
│   ├── test_rag_engine.py
│   └── test_graph_rag.py
├── templates/
│   └── index.html            # Frontend UI
├── data/
│   └── faq_docs/           # FAQ document storage
│       ├── account_and_billing.txt
│       ├── returns_and_refunds.txt
│       └── shipping_and_delivery.txt
├── embeddings/              # Persisted FAISS index
│   ├── faiss.index
│   └── metadata.pkl
├── .env                     # Environment variables
├── requirements.txt         # Python dependencies
└── ARCHITECTURE.md          # This document
```

---

## Dependencies

```
# Web framework
flask>=3.0.0
flask-cors>=4.0.0
werkzeug>=3.0.0

# LangChain
langchain>=0.2.0
langchain-community>=0.2.0
langchain-huggingface>=0.0.1

# LangGraph (self-correcting RAG)
langgraph>=0.2.0

# Embeddings and vector store
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0
numpy>=1.26.0

# LLM
transformers>=4.40.0
torch>=2.0.0

# Utilities
python-dotenv>=1.0.0
httpx>=0.27.0

# MCP (Model Context Protocol)
mcp>=1.0.0

# Testing
pytest>=8.2.0
```

---

## Configuration (.env)

```env
# App Settings
FLASK_PORT=5000
FLASK_DEBUG=False

# RAG Settings
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=5
EMBEDDING_MODEL=all-MiniLM-L6-v2

# LLM Settings (local model - no API needed)
LLM_MODEL=gpt2-medium
LLM_MAX_TOKENS=150
LLM_TEMPERATURE=0.3
```

---

## Advantages of This Architecture

1. **Privacy:** All data stays local - no external API calls
2. **Self-Correction:** LangGraph ensures poor answers are regenerated
3. **Speed:** FAISS provides sub-millisecond similarity search
4. **Cost-effective:** No API usage fees
5. **Offline-capable:** Works without internet (after initial model download)
6. **Reasoning:** Chain-of-thought prompting for complex questions

---

## Limitations

1. **CPU-only LLM:** Slower generation than GPU-based solutions
2. **Small model:** gpt2-medium has limited reasoning capabilities
3. **Quality threshold:** May regenerate valid answers occasionally
4. **Chunk size:** May miss cross-chunk context

---

## Future Improvements

1. **Better LLM:** Use `gpt2-medium` or `opt-350m` for improved accuracy
2. **GPU support:** Enable CUDA for faster inference
3. **Query routing:** Different flows based on query complexity
4. **Reranking:** Add a cross-encoder for better result ranking
5. **Hybrid search:** Combine keyword and semantic search
6. **MCP Server:** Expose RAG tools via MCP protocol for AI assistant integration (see [`docs/MCP_INTEGRATION.md`](docs/MCP_INTEGRATION.md))

---

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ -v --cov=src
```

Test files:
- `test_config.py` - Configuration management
- `test_query_classifier.py` - Query classification
- `test_answer_extractor.py` - Answer extraction
- `test_llm.py` - LLM generation and reasoning
- `test_rag_engine.py` - FAISS indexing and retrieval
- `test_graph_rag.py` - LangGraph self-correction

---

*Document generated: 2026-04-22*
*System: knowledgeStore RAG Application with LangGraph Self-Correction and MCP Server*
