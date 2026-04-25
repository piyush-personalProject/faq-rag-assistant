# Building a Self-Correcting RAG FAQ System: From Beginner to Professional

A comprehensive guide to building production-ready retrieval-augmented generation systems with self-correction capabilities.

---

## Table of Contents

1. [What is RAG?](#1-what-is-rag)
2. [Why Build a Local RAG System?](#2-why-build-a-local-rag-system)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Core Components Explained](#4-core-components-explained)
5. [The Self-Correction Mechanism](#5-the-self-correction-mechanism)
6. [Building Your First FAQ Bot](#6-building-your-first-faq-bot)
7. [Professional Deployment Considerations](#7-professional-deployment-considerations)
8. [Performance Benchmarks](#8-performance-benchmarks)
9. [Performance Metrics: Why Each Metric Matters](#9-performance-metrics-why-each-metric-matters)
10. [Design Decisions Explained](#10-design-decisions-explained)
11. [Hallucination Prevention Architecture](#11-hallucination-prevention-architecture)
12. [Tracing System: Document-to-Answer Lineage](#12-tracing-system-document-to-answer-lineage)
13. [Private Processing: Your Data Stays Local](#13-private-processing-your-data-stays-local)
14. [Common Pitfalls and Solutions](#14-common-pitfalls-and-solutions)

---

## 1. What is RAG?

**RAG (Retrieval-Augmented Generation)** combines two powerful techniques:

- **Retrieval**: Finding relevant documents from a knowledge base using semantic search
- **Generation**: Using an LLM to synthesize an answer from the retrieved context

### Beginner Analogy

Think of RAG like a student taking an exam:
1. **Retrieval** = Open-book access to reference materials
2. **Generation** = Writing an answer using that material

The alternative (pure LLM) is like a closed-book exam—the model must rely solely on its training data, which may be outdated or inaccurate.

### Why RAG Matters

| Approach | Pros | Cons |
|----------|------|------|
| **Pure LLM** | Fluent responses | May hallucinate; no access to private data |
| **Keyword Search + LLM** | Access to documents | No semantic understanding |
| **RAG** | Accurate, grounded answers | Adds complexity |

---

## 2. Why Build a Local RAG System?

### Privacy Requirements

Many organizations cannot send data to third-party APIs due to:
- GDPR, HIPAA, or SOC2 compliance
- Proprietary business data
- Customer PII protection

### Cost Optimization

Commercial LLM APIs charge per token:
- OpenAI GPT-4: ~$0.03/1K tokens (input)
- Anthropic Claude: ~$0.015/1K tokens (input)

A local system has **zero per-query costs** after initial infrastructure investment.

### Offline Capability

Local systems work without internet access—critical for:
- Internal corporate tools
- Air-gapped environments
- Remote locations with limited connectivity

---

## 3. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENT                                │
│                   (Browser/App)                             │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/JSON + SSE
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    FLASK BACKEND                             │
│                                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│   │   Upload    │    │    Chat      │    │    Status    │  │
│   │   Endpoint  │    │   Endpoint   │    │   Endpoint   │  │
│   └──────────────┘    └──────┬───────┘    └──────────────┘  │
└─────────────────────────────┼───────────────────────────────┘
                              │
         ┌────────────────────┴────────────────────┐
         ▼                                         ▼
┌─────────────────────┐              ┌─────────────────────────┐
│     RAG ENGINE       │              │      LOCAL LLM          │
│   (FAISS Search)    │              │   (gpt2-medium)        │
│                     │              │                         │
│  1. Embed query     │              │  • 82M parameters      │
│  2. Search index    │              │  • CPU inference       │
│  3. Return chunks   │─────────────▶│  • Text generation    │
└─────────────────────┘              └─────────────────────────┘
         │
         │ (LangGraph Workflow)
         ▼
┌─────────────────────────────────────────────────────────────┐
│               SELF-CORRECTION LAYER                          │
│                                                              │
│   Retrieve → Generate → Quality Check → [Regenerate?]       │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Core Components Explained

### 4.1 The RAG Engine ([`src/rag_engine.py`](src/rag_engine.py))

The RAG engine handles **document indexing and retrieval**.

#### Embeddings

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
# Output: 384-dimensional vectors
```

This converts text into mathematical vectors where **semantically similar texts are close in vector space**.

#### Vector Store (FAISS)

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    documents=text_chunks,
    embedding=embedding_model
)
```

FAISS (Facebook AI Similarity Search) provides **sub-millisecond similarity search** over millions of vectors.

#### Text Chunking

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Characters per chunk
    chunk_overlap=50     # Overlap to preserve context
)
chunks = splitter.split_text(document)
```

### 4.2 The Local LLM ([`src/llm.py`](src/llm.py))

```python
from langchain_huggingface import HuggingFacePipeline

llm_pipeline = HuggingFacePipeline.from_model_id(
    model_id="gpt2-medium",
    pipeline_kwargs={
        "max_new_tokens": 150,
        "temperature": 0.3,  # Low for factual Q&A
    }
)
```

| Model | Parameters | Size | Use Case |
|-------|-----------|------|----------|
| gpt2 | 124M | ~500MB | Lightweight tasks |
| gpt2-medium | 355M | ~1.5GB | Balanced |
| gpt2-large | 774M | ~3GB | Higher quality |

### 4.3 Query Classifier ([`src/query_classifier.py`](src/query_classifier.py))

Not every query needs full RAG processing:

```python
SIMPLE_PATTERNS = [
    "hello", "hi", "hey",
    "thanks", "thank you",
    "goodbye", "bye"
]

def classify_query(query: str) -> QueryType:
    # Simple greetings → Fast path (no RAG)
    # FAQ questions → Full RAG pipeline
```

This optimization reduces latency for simple interactions.

---

## 5. The Self-Correction Mechanism

This is where the system gets **intelligent**. Using LangGraph, the system automatically evaluates and improves its own answers.

### The Flow

```
┌─────────────┐
│  RETRIEVE   │ ← Fetch relevant documents from FAISS
└──────┬──────┘
       ▼
┌─────────────┐
│  GENERATE   │ ← LLM creates initial answer
└──────┬──────┘
       ▼
┌─────────────┐
│ QUALITY     │ ← Evaluate answer quality (0.0 - 1.0)
│   CHECK     │
└──────┬──────┘
       │
       ├─── Score ≥ 0.7 ──▶ DONE ✓
       │
       └─── Score < 0.7 ──▶ REGENERATE (max 2 attempts)
```

### Quality Scoring

The quality checker evaluates:
1. **Relevance**: Does the answer address the question?
2. **Grounding**: Is the answer supported by retrieved context?
3. **Coherence**: Is the answer fluent and well-structured?

### Implementation ([`src/graph_rag.py`](src/graph_rag.py))

```python
from langgraph.graph import StateGraph

class RAGState(TypedDict):
    query: str
    retrieved_docs: List[Document]
    answer: str
    quality_score: float
    attempts: int

workflow = StateGraph(RAGState)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)
workflow.add_node("quality_check", evaluate_quality)
workflow.add_node("regenerate", regenerate_answer)
```

### Why Self-Correction Matters

| Without Self-Correction | With Self-Correction |
|------------------------|-----------------------|
| Single attempt | Up to 3 attempts |
| Potentially poor answers delivered | Poor answers regenerated |
| Requires human review | Automatic quality assurance |

---

## 6. Building Your First FAQ Bot

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Create FAQ Documents

Place `.txt` files in `data/faq_docs/`:

```
Q: How do I track my order?
A: You can track your order using the tracking link 
   sent to your email once your order ships.

Q: What is your return policy?
A: Items can be returned within 30 days of purchase 
   with a valid receipt.
```

### Step 3: Run the Application

```bash
python app.py
```

### Step 4: Query via API

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I track my order?"}'
```

### Multi-turn Conversation

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What about exchanges?",
    "history": [
      {"role": "user", "content": "How do returns work?"},
      {"role": "assistant", "content": "Items can be returned within 30 days."}
    ]
  }'
```

---

## 7. Professional Deployment Considerations

### Scaling

| Component | Scaling Strategy |
|-----------|-----------------|
| Flask API | Add Gunicorn workers |
| FAISS Index | Use `IndexIVFFlat` for large datasets |
| LLM | Upgrade to larger models or add GPU |

### Monitoring

Key metrics to track:
- **Retrieval latency**: Time to fetch from FAISS
- **Generation latency**: Time for LLM to respond
- **Quality scores**: Distribution of self-correction scores
- **Regeneration rate**: % of answers that required regeneration

### Security

- Run behind VPN for internal tools
- Add authentication to API endpoints
- Sanitize file uploads
- Rate limiting on public deployments

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Run Tests
  run: pytest tests/ -v --cov=src
  
- name: Deploy
  run: docker build -t knowledge-store .
```

---

## 8. Performance Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| FAISS search | ~5ms | 100K documents |
| LLM generation | ~30s | CPU inference |
| Total (cached) | ~30-45s | First query |
| Total (warm) | ~31s | Subsequent queries |

*Note: LLM times are for gpt2-medium on CPU. GPU inference is 10-50x faster.*

---

## 9. Performance Metrics: Why Each Metric Matters

The system uses three core evaluation metrics defined in [`src/rag_evaluator.py`](src/rag_evaluator.py:13):

### 9.1 Faithfulness (0-1 score)

**What it measures**: Whether the generated answer's claims are supported by the retrieved context.

**Why it matters**: Faithfulness prevents hallucination—the AI generating plausible-sounding but incorrect information. When faithfulness is low, the model is "making things up" that aren't in the source material.

**How it works**:
```python
# The evaluator tokenizes both answer and context
answer_tokens = self._tokenize(answer)
context_tokens = self._tokenize(context)

# Each sentence must have >50% token overlap with context
# OR appear literally in the context
if coverage >= 0.5 or literal_match:
    faithful_count += 1
```

**What this catches**: "Based on our analysis, the product ships in 2-3 days" when the context only says "ships within a week."

### 9.2 Answer Relevance (0-1 score)

**What it measures**: Whether the answer actually addresses the user's query.

**Why it matters**: An answer can be faithful (all claims match context) but irrelevant if it doesn't address what was asked.

**How it works**:
```python
# Check overlap between query terms and answer terms
query_tokens = self._tokenize(query)
answer_tokens = self._tokenize(answer)
coverage = len(overlap) / len(query_tokens)

# Penalize tangential sentences that don't match query
for sentence in sentences:
    sentence_overlap = sentence_tokens & query_tokens
    if len(sentence_overlap) < len(query_tokens) * 0.15:
        irrelevant_sentences.append(sentence)
```

**What this catches**: User asks about return policy, but answer explains shipping process.

### 9.3 Context Precision (0-1 score)

**What it measures**: How relevant the retrieved documents are to answering the query.

**Why it matters**: Even the best LLM can't produce good answers if fed irrelevant context. This metric identifies when retrieval is failing.

**How it works**:
```python
# Score each retrieved chunk against the query
for ctx in contexts:
    overlap = query_tokens & ctx_tokens
    precision = len(overlap) / len(ctx_tokens)
    
    # Combined with n-gram matching for semantic capture
    combined = (precision * 0.6 + ngram_overlap * 0.4)
```

**What this catches**: Query about exchanges returns shipping FAQ documents instead of return policy.

### Overall Quality Score

```python
def get_overall_quality(self, result: RAGEvaluationResult) -> float:
    return (
        result.faithfulness * 0.4 +      # Most important - prevent hallucinations
        result.answer_relevance * 0.4 +  # Must address the question
        result.context_precision * 0.2    # Retrieval quality matters
    )
```

Faithfulness and relevance get equal weight (40% each) because you need both: an answer that's both truthful AND relevant. Context precision gets 20% since poor retrieval is usually caught by the other two metrics failing anyway.

---

## 10. Design Decisions Explained

### 10.1 Why FAISS for Vector Search?

FAISS (Facebook AI Similarity Search) was chosen over alternatives for specific reasons:

| Alternative | Why FAISS Wins |
|-------------|----------------|
| Pinecone/Qdrant | Requires external service; not fully local |
| ChromaDB | Simpler but slower for large datasets |
| Simple cosine similarity | O(n) scan—too slow at scale |
| Elasticsearch | Overkill for vector use case |

FAISS provides **sub-millisecond search** over millions of vectors using:
- **IVF (Inverted File Index)**: Partitions vectors into clusters; searches only relevant clusters
- **HNSW (Hierarchical Navigable Small World)**: Graph-based approach for very fast approximate search

```python
# From src/config.py line 24
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dimensional embeddings
```

The `all-MiniLM-L6-v2` model was chosen because:
- 384 dimensions: good balance of precision vs. memory
- 42MB size: runs easily on CPU
- 6x faster than larger models with 92% accuracy

### 10.2 Why LangGraph for Self-Correction?

The self-correction loop uses LangGraph's state machine approach ([`src/graph_rag.py`](src/graph_rag.py:31)):

```python
workflow = StateGraph(RAGState)
workflow.add_node("retrieve", self._retrieve_node)
workflow.add_node("generate", self._generate_node)
workflow.add_node("quality_check", self._quality_check_node)
workflow.add_node("regenerate", self._regenerate_node)
```

**Why state machine over simple function calls**:
- **Visualization**: Can debug the exact flow of any query
- **Conditional routing**: Quality check branches to "good" → END or "regenerate" → loop
- **State persistence**: Each node receives and modifies the full state dict
- **Testability**: Each node can be tested in isolation

### 10.3 Why Hybrid Search with Reranking?

```python
# From src/config.py lines 31-36
USE_HYBRID_SEARCH = True        # Combine keyword + semantic
BM25_WEIGHT = 0.3               # 30% keyword matching
SEMANTIC_WEIGHT = 0.7           # 70% vector similarity
USE_RERANKING = True            # Re-order results for quality
RERANKER_MODEL = "BAAI/bge-reranker-base"
```

**Why hybrid**:
- Pure semantic search misses exact matches ("order 12345")
- Pure keyword search misses synonyms ("tracking" vs "delivery status")
- Hybrid combines both for robust retrieval

**Why reranking**:
- Initial retrieval is fast but approximate
- Cross-encoder reranker does deeper relevance scoring
- "BAAI/bge-reranker-base" achieves high precision without GPU

### 10.4 Why Token-Based Chunking with Overlap?

```python
# From src/config.py lines 21-22
CHUNK_SIZE = 500          # Characters per chunk
CHUNK_OVERLAP = 50        # Preserve context across boundaries
```

**Why 500 characters**:
- Small enough to be precise (doesn't mix topics)
- Large enough to contain complete Q&A pairs
- Aligns with embedding model's optimal input length

**Why 50 character overlap**:
- A question at the end of one chunk might need context from the start of the next
- 50 chars = ~12 words = enough context for continuity
- Too much overlap wastes vector space with redundant embeddings

---

## 11. Hallucination Prevention Architecture

Hallucination (confident but incorrect responses) is the #1 failure mode for RAG systems. This system fights it at multiple levels:

### 11.1 Detection Layer (in [`src/graph_rag.py`](src/graph_rag.py:150))

```python
def _is_gibberish_answer(self, answer: str, context: str) -> bool:
    # Check 1: Does answer share vocabulary with context?
    overlap = context_words & answer_words
    if len(overlap) < len(context_words) // 3:  # Less than 33% overlap
        return True  # Likely hallucinated
    
    # Check 2: Is answer repetitive? (sign of generation failure)
    word_counts = {w.lower(): count for w in answer.split()}
    if max_repeat > 3 and len(words) > 10:
        return True  # Repetitive gibberish
```

### 11.2 Quality Threshold Layer

```python
def _should_continue(self, state: RAGState) -> str:
    if state["quality_score"] >= 0.7:  # Quality threshold
        return "good"
    elif state["attempts"] < 2:        # Max 2 regeneration attempts
        return "regenerate"
```

Answers below 0.7 quality trigger regeneration. The threshold was set at 0.7 based on testing:
- Higher (0.8+): Causes excessive regeneration loops
- Lower (0.6): Lets poor answers through too often

### 11.3 Fallback Extraction Layer

When LLM generation fails, the system falls back to direct answer extraction:

```python
# From src/graph_rag.py line 134
if answer_is_gibberish:
    extracted = AnswerExtractor.extract(state["query"], state["retrieved_docs"])
    if extracted:
        answer = extracted  # Use extracted answer instead

# If extraction also fails, format chunks as response
if not answer or len(answer) < 20:
    formatted = AnswerExtractor.format_chunks_response(state["retrieved_docs"])
```

This ensures **something valid** is always returned—even if not elegant.

### 11.4 Why Temperature is Locked Low

```python
# From src/config.py line 41
LLM_TEMPERATURE = 0.3  # Low for factual Q&A
```

High temperature (>0.7) increases creativity but also hallucinations. For FAQ answering, we want:
- Correct facts from context
- Consistent formatting
- No made-up details

---

## 12. Tracing System: Document-to-Answer Lineage

The tracing system in [`src/rag_trace.py`](src/rag_trace.py:62) provides full visibility into how answers are generated:

### 12.1 What Gets Traced

```python
@dataclass
class GenerationTrace:
    trace_id: str           # Unique identifier for this query
    timestamp: str          # When the query occurred
    query: str              # Original user question
    answer: str             # Final generated answer
    chunks: List[ChunkContribution]  # Which documents contributed
    evaluation_metrics: Dict[str, float]  # Quality scores
    generation_time_ms: float  # How long it took
    llm_model: str          # Which model generated the answer
```

### 12.2 Chunk Attribution

```python
@dataclass
class ChunkContribution:
    chunk_id: str           # "chunk_0", "chunk_1", etc.
    source: str             # "shipping_and_delivery.txt"
    text_preview: str       # First 100 chars of chunk
    score: float            # Relevance score from retrieval
    used_in_generation: bool  # Did this chunk actually get used?
    token_overlap: float    # How much contributed to final answer
```

### 12.3 Why Tracing Matters

**Debugging**: When a user complains "your answer is wrong," you can:
1. Look up their trace by trace_id
2. See exactly which documents were retrieved
3. Check quality scores at each step
4. Identify if hallucination or retrieval failure

**Compliance**: Regulated industries need to show:
- "We retrieved document X to answer your question"
- "Our confidence score was 0.85, above threshold"

**Optimization**: Aggregate traces to find:
- Which source documents cause most regenerations?
- What's the average quality score trend over time?
- Which queries consistently fail?

### 12.4 File-Based vs LangSmith Tracing

```python
# From src/rag_trace.py line 72
def __init__(self, enable_file_tracing: bool = True, enable_langsmith: bool = False):
```

**File tracing** (default):
- Saves JSON files to `embeddings/traces/`
- Works offline, no external dependencies
- Queryable via `get_trace()` and `get_recent_traces()`

**LangSmith tracing** (optional):
- Cloud service for centralized monitoring
- Enables team collaboration on quality issues
- Requires `LANGSMITH_API_KEY` environment variable

```python
# To enable LangSmith, call:
configure_langsmith_tracing(api_key="your-key-here", project_name="knowledgeStore")
```

---

## 13. Private Processing: Your Data Stays Local

### 13.1 Why Private Processing Matters

| Data Leakage Risk | What It Means |
|-------------------|----------------|
| User queries sent to OpenAI | Customer questions exposed to third party |
| Documents uploaded to external API | Proprietary content in vendor systems |
| Conversation history stored externally | Full context of business operations exposed |

### 13.2 How This System Avoids External Services

**Embedding Generation**:
```python
# From src/rag_engine.py
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # Runs locally via sentence-transformers
)
```

All embeddings generated locally—no data sent anywhere.

**LLM Inference**:
```python
# From src/llm.py
from langchain_huggingface import HuggingFacePipeline

llm_pipeline = HuggingFacePipeline.from_model_id(
    model_id="gpt2-medium"  # 355M parameters, runs on CPU
)
```

Full text generation happens locally. The model is downloaded once and runs entirely on your infrastructure.

**No External API Calls**:
```python
# The system only contacts:
# - HuggingFace Hub (one-time model download)
# - Your local file system (document storage)
# - Nothing else
```

### 13.3 Configuration for Air-Gapped Environments

```python
# From src/config.py
# All paths are relative to project root—works offline
PROJECT_ROOT = Path(__file__).parent.parent
UPLOAD_FOLDER = PROJECT_ROOT / "data" / "faq_docs"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

# No cloud dependencies by default
LANGSMITH_API_KEY = ""  # Empty unless explicitly configured
```

### 13.4 Data Flow Privacy

```
User Query → Flask Backend → Local Embedding → FAISS Search → Local LLM → Response
     ↑              ↓               ↓              ↓            ↓         ↓
   YOU          LOCAL ONLY      LOCAL ONLY     LOCAL ONLY   LOCAL ONLY   YOU
```

Every step runs on your infrastructure. No intermediate services, no data leaves your network.

---

## 14. Common Pitfalls and Solutions

### Problem: Poor Retrieval Quality

**Symptoms**: Irrelevant documents returned

**Solutions**:
- Increase `TOP_K_RESULTS` to retrieve more candidates
- Adjust chunk size (smaller = more precise, larger = more context)
- Fine-tune embedding model on your domain

### Problem: Hallucinated Answers

**Symptoms**: LLM generates plausible but incorrect info

**Solutions**:
- Lower temperature (0.1-0.3 for factual Q&A)
- Increase quality threshold
- Add source citations to responses
- Use stronger LLM model

### Problem: Slow Response Times

**Symptoms**: Users wait too long

**Solutions**:
- Use streaming responses (SSE) for perceived speed
- Cache frequent queries
- Upgrade to GPU inference
- Pre-compile FAISS index on startup

### Problem: Context Truncation

**Symptoms**: Answers ignore earlier parts of conversation

**Solutions**:
- Increase context window
- Limit conversation history length
- Use summarization for long histories

---

## Conclusion

This RAG FAQ system demonstrates production-ready patterns:
- **Semantic search** with FAISS
- **Self-correcting generation** with LangGraph
- **Local processing** for privacy and cost savings
- **Modular architecture** for easy customization

Whether you're a **beginner** learning RAG concepts or a **professional** building enterprise knowledge bases, these patterns scale from prototype to production.

### Next Steps

1. Clone the repository and run locally
2. Add your own FAQ documents
3. Experiment with different embedding models
4. Integrate with your application via REST API
5. Consider GPU acceleration for production

---

*Originally published: 2026-04-23*
*System: knowledgeStore RAG FAQ Assistant*
