# Knowledge Store - RAG System Architecture

## Overview

This is a **Retrieval-Augmented Generation (RAG)** system that combines vector-based semantic search with a local Large Language Model (LLM) to provide intelligent question-answering over FAQ documents.

**Key Feature:** All processing happens locally - no external API calls required.

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
│   │   POST /api/clear           → Clear FAISS index                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│                                      ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                     CHAT ENDPOINT FLOW                             │  │
│   │                                                                     │  │
│   │   1. Receive user query                                             │  │
│   │   2. Retrieve relevant chunks from FAISS                            │  │
│   │   3. Build prompt with context                                      │  │
│   │   4. Generate response via local LLM                                │  │
│   │   5. Stream response to client (SSE)                                 │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
┌─────────────────────────────┐       ┌─────────────────────────────────────┐
│      RAG ENGINE             │       │      LOCAL LLM                      │
│      (rag_engine.py)        │       │      (distilgpt2)                   │
│                             │       │                                     │
│  ┌───────────────────────┐ │       │  • 82MB model                       │
│  │  SentenceTransformer   │ │       │  • CPU inference                     │
│  │  (all-MiniLM-L6-v2)    │ │       │  • Text generation                  │
│  │                       │ │       │  • No API calls                      │
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
│  │  • Stores 30 chunks   │ │
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
│                         DATA LAYER                                          │
│                                                                             │
│   ┌─────────────────────┐        ┌─────────────────────────────────────┐  │
│   │  FAQ Documents       │        │  Persisted Index                     │  │
│   │  (data/faq_docs/)   │        │  (embeddings/)                       │  │
│   │                     │        │                                      │  │
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
2. Call rag.retrieve(user_query) → get top-k chunks
3. Build context prompt with retrieved chunks
4. Call LLM pipeline with prompt
5. Stream response back via SSE
```

### 3. RAG Engine (rag_engine.py)

**Technology:** SentenceTransformers, FAISS, NumPy

**Components:**

#### 3.1 Embedding Model
- **Model:** `all-MiniLM-L6-v2`
- **Dimensions:** 384
- **Purpose:** Convert text into semantic vector embeddings

#### 3.2 Text Chunking
```python
CHUNK_SIZE = 500 words
CHUNK_OVERLAP = 50 words
```
- FAQ files are split into overlapping chunks
- Preserves context at chunk boundaries

#### 3.3 FAISS Index
- **Type:** IndexFlatL2 (brute-force L2 distance)
- **Purpose:** Fast similarity search on embeddings
- **Persistence:** Saved to `embeddings/faiss.index`

#### 3.4 Retrieval Process
```python
query_embedding = model.encode(user_query)
distances, indices = index.search(query_embedding, k=5)
results = [chunks[idx] for idx in indices]
```

### 4. Local LLM (distilgpt2)

**Technology:** HuggingFace Transformers

**Model Details:**
- **Size:** 82 million parameters
- **Memory:** ~500MB RAM
- **Inference Device:** CPU
- **Generation Settings:**
  - `max_new_tokens=150`
  - `temperature=0.3` (low for factual Q&A)
  - `top_p=0.9`
  - `repetition_penalty=1.2`

**Generation Prompt Template:**
```
Based on the following FAQ context, answer the user's question concisely and accurately.

Context:
{retrieved_chunks}

Question: {user_question}

Answer:
```

---

## Data Flow Example

### Question: "How long does a refund take?"

```
1. CLIENT
   └── "How long does a refund take?" 
       └── POST /api/chat

2. FLASK
   └── rag.retrieve("How long does a refund take?", top_k=5)

3. RAG ENGINE
   └── Encode query → [0.123, -0.456, ...] (384-dim)
   └── FAISS search → Returns 5 nearest chunks
   └── Example chunk:
       "Q: How long does a refund take?
        A: Once we receive your returned item, 
        refunds are processed within 3-5 business days."

4. LLM (distilgpt2)
   └── Input: Context + Question
   └── Output: "Refunds are typically processed within 
               3-5 business days after receiving the returned item."

5. CLIENT (SSE Streaming)
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
├── templates/
│   └── index.html            # Frontend UI
├── data/
│   └── faq_docs/            # FAQ document storage
│       ├── account_and_billing.txt
│       ├── returns_and_refunds.txt
│       └── shipping_and_delivery.txt
├── embeddings/               # Persisted FAISS index
│   ├── faiss.index
│   └── metadata.pkl
├── .env                     # Environment variables
├── requirements.txt         # Python dependencies
└── ARCHITECTURE.md          # This document
```

---

## Dependencies

```
flask>=3.0.0
flask-cors>=4.0.0
dotenv>=1.0.0
sentence-transformers>=3.0.0
faiss-cpu>=1.8.0
numpy>=1.26.0
torch>=2.0.0
transformers>=4.40.0
werkzeug>=3.0.0
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

# LLM Settings (not used - local model)
# ANTHROPIC_API_KEY=  # Not needed anymore
```

---

## Advantages of This Architecture

1. **Privacy:** All data stays local - no external API calls
2. **Speed:** FAISS provides sub-millisecond similarity search
3. **Cost-effective:** No API usage fees
4. **Customizable:** Easy to swap embedding models or LLMs
5. **Offline-capable:** Works without internet (after initial model download)

## Limitations

1. **CPU-only LLM:** Slower generation than GPU-based solutions
2. **Small model:** distilgpt2 has limited reasoning capabilities
3. **Chunk size:** May miss cross-chunk context
4. **No fine-tuning:** Generic model may not understand domain specifics

---

## Future Improvements

1. **Better LLM:** Use `gpt2-medium` or `opt-350m` for improved accuracy
2. **GPU support:** Enable CUDA for faster inference
3. **Hybrid search:** Combine keyword and semantic search
4. **Reranking:** Add a cross-encoder for better result ranking
5. **Caching:** Cache frequent queries for instant responses

---

*Document generated: 2026-04-19*
*System: knowledgeStore RAG Application*
