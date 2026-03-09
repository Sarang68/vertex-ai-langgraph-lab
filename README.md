# ☕ Vertex AI RAG & Agentic AI — Production Lab

> Sunday Morning. Coffee. Cloud Shell. Let's Talk Vertex AI.

A hands-on, end-to-end production lab demonstrating how to build and deploy a
LangGraph-based multi-agent RAG system on Google Cloud Vertex AI — from environment
setup through production API deployment with monitoring and cost guardrails.

Built by [Sarang Mahatwo](https://www.linkedin.com/in/sarang-mahatwo) —
Distinguished Engineer | AI Architect | LangGraph + Vertex AI

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│   PDFs · Web Pages · GCS Buckets · Internal Docs               │
└────────────────────────┬────────────────────────────────────────┘
                         │ loaders.py + gcs_sink.py
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   CHUNKING & EMBEDDING                          │
│   strategies.py (Fixed / Semantic / Hierarchical)              │
│   vertex_embedder.py  →  text-embedding-004 (768 dims)         │
└──────────┬─────────────────────────────┬───────────────────────┘
           │                             │
           ▼                             ▼
┌──────────────────────┐    ┌────────────────────────────────────┐
│   VECTOR STORES      │    │         GRAPH DB                   │
│  vertex_vector_      │    │   neo4j_builder.py                 │
│    search.py         │    │   Entity extraction + Cypher       │
│  pinecone_index.py   │    │   GraphRAG traversal               │
└──────────┬───────────┘    └───────────────┬────────────────────┘
           │                                │
           └──────────────┬─────────────────┘
                          │ hybrid_search.py
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              RETRIEVAL & GENERATION                             │
│   hybrid_search.py  (Dense + BM25 + RRF fusion + Reranking)    │
│   generator.py      (Grounded generation + Citation tracking)  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│               AGENTIC ORCHESTRATION                             │
│   rag_agent.py  (LangGraph StateGraph + ToolNode)              │
│   guardrails.py (Token budget + Cost controls + Caching)       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              EVALUATION & MONITORING                            │
│   ragas_eval.py  (Faithfulness / Relevancy / Recall)           │
│   metrics.py     (Retrieval metrics + Vertex Eval Service)     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              PRODUCTION DEPLOYMENT                              │
│   main.py   (FastAPI + CORS + Auth + Rate limiting)            │
│   Cloud Run / Vertex AI Reasoning Engine                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
vertex-ai-langgraph-lab/
│
├── validate_setup.py              # Phase 0: Environment & connectivity validation
│
├── ingestion/
│   ├── loaders.py                 # Phase 1: Multi-source document loaders (PDF, web, GCS)
│   └── gcs_sink.py                # Phase 1: GCS + Pub/Sub ingestion pipeline
│
├── chunking/
│   └── strategies.py              # Phase 2: Fixed / Semantic / Hierarchical chunking
│
├── embeddings/
│   └── vertex_embedder.py         # Phase 2: Async batch embedding via text-embedding-004
│
├── vector_db/
│   ├── vertex_vector_search.py    # Phase 3: Vertex AI Vector Search index management
│   └── pinecone_index.py          # Phase 3: Pinecone serverless index (multi-cloud alt)
│
├── retrieval/
│   ├── hybrid_search.py           # Phase 4: Dense + BM25 hybrid search with RRF fusion
│   └── generator.py               # Phase 4: Grounded response generation + citations
│
├── agents/
│   └── rag_agent.py               # Phase 5: LangGraph multi-agent RAG orchestration
│
├── cost/
│   └── guardrails.py              # Phase 5: Token budget, cost controls, LRU caching
│
├── graph/
│   └── neo4j_builder.py           # Phase 6: Neo4j GraphRAG — entity extraction + traversal
│
├── evaluation/
│   ├── ragas_eval.py              # Phase 7: RAGAS evaluation (faithfulness, relevancy, recall)
│   └── metrics.py                 # Phase 7: Retrieval metrics + Vertex Eval Service
│
├── api/
│   └── main.py                    # Phase 8: FastAPI production API + middleware + auth
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🗺️ Lab Modules — Phase by Phase

### Phase 0 — Environment Setup
| Script | Purpose |
|--------|---------|
| `validate_setup.py` | Validates Vertex AI connectivity, Gemini model access, ADC authentication, and all package imports before any lab module runs |

**Key lessons from this phase:**
- Org policy `constraints/iam.disableServiceAccountKeyCreation` blocks key files → use ADC
- `gemini-1.5-flash` returns 404 on new projects — all Gemini 1.x retired; use `gemini-2.0-flash-001`
- `vertexai.generative_models` deprecated June 2025 → migrate to `google-genai` SDK

---

### Phase 1 — Data Ingestion & ETL
| Script | Purpose |
|--------|---------|
| `ingestion/loaders.py` | Multi-source document loaders: PDF (PyMuPDF), web (BeautifulSoup4), GCS, with metadata extraction and content hashing for deduplication |
| `ingestion/gcs_sink.py` | Async GCS upload + Pub/Sub event trigger pipeline for streaming ingestion into the RAG corpus |

---

### Phase 2 — Chunking & Embeddings
| Script | Purpose |
|--------|---------|
| `chunking/strategies.py` | Three chunking strategies: Fixed-size with overlap (tiktoken), Semantic (embedding similarity boundaries), Hierarchical (parent-child chunks for long docs) |
| `embeddings/vertex_embedder.py` | Async batch embedding using `text-embedding-004` (768 dims). Supports RETRIEVAL_DOCUMENT and RETRIEVAL_QUERY task types. Batched for throughput. |

**Chunking decision guide:**
- **Fixed-size**: Structured content, product specs, FAQs — fast, predictable
- **Semantic**: Narrative docs, legal text, reports — preserves meaning across splits
- **Hierarchical**: Documents >20 pages — index small chunks, retrieve parent for context

---

### Phase 3 — Vector DB & Indexing
| Script | Purpose |
|--------|---------|
| `vector_db/vertex_vector_search.py` | Creates and manages Vertex AI Vector Search ANN index. Handles batch upsert, index deployment to endpoint, and similarity queries |
| `vector_db/pinecone_index.py` | Pinecone serverless index as multi-cloud alternative. Demonstrates upsert, query, and namespace-based access control |

**Vector DB selection guide:**
| DB | Best For |
|----|---------|
| Vertex Vector Search | GCP-native, >100M vectors, tight Gemini integration |
| Pinecone | Multi-cloud, fast startup, managed ops |
| pgvector | Already on Postgres/Cloud SQL, <10M vectors |
| Weaviate | Built-in hybrid search, GraphQL interface |

---

### Phase 4 — Retrieval & Reranking
| Script | Purpose |
|--------|---------|
| `retrieval/hybrid_search.py` | Hybrid search combining dense vector (Vertex) + sparse BM25 keyword search, fused via Reciprocal Rank Fusion (RRF). Cross-encoder reranking on top-N results. |
| `retrieval/generator.py` | Grounded response generation with citation tracking. Structured output with Pydantic. Groundedness validation before returning response. |

**Why hybrid search matters in enterprise RAG:**
Dense search handles semantic queries ("stores underperforming forecast"). BM25 handles
exact-match queries (store IDs, SKUs, product codes). Enterprise data has both.
RRF fusion: `score = α × vector_score + (1-α) × bm25_score`

---

### Phase 5 — Tool-Using Agents
| Script | Purpose |
|--------|---------|
| `agents/rag_agent.py` | LangGraph `StateGraph` multi-agent RAG orchestration. Supervisor + specialist pattern. TypedDict state, ToolNode, conditional routing, interrupt() for human-in-the-loop |
| `cost/guardrails.py` | Production cost controls: token budget enforcement, LRU response caching (hashlib-based), per-user rate limiting, cost estimation before execution |

**Agent architecture:**
```
Supervisor Agent (gemini-2.0-flash-001)
    ├── Retrieval Specialist  → hybrid_search + generator
    ├── Graph Specialist      → neo4j_builder traversal
    └── Synthesis Agent       → final grounded response
```

---

### Phase 6 — Graph DB Integration
| Script | Purpose |
|--------|---------|
| `graph/neo4j_builder.py` | GraphRAG pipeline: LLM-based entity extraction from documents → Neo4j node/relationship creation → Cypher-based graph traversal → context fusion with vector retrieval |

**GraphRAG vs pure vector RAG:**
Vector search finds *similar* documents. Graph traversal finds *related* entities.
For supply chain, org hierarchy, or regulatory dependency queries — graph wins.

```cypher
-- Example: multi-hop relationship query
MATCH (supplier:Supplier)-[:SUPPLIES]->(p:Product)-[:ALLOCATED_TO]->(s:Store)
WHERE s.region = 'Northeast'
RETURN supplier.name, p.name, count(s) AS store_count
```

---

### Phase 7 — Evaluation Framework
| Script | Purpose |
|--------|---------|
| `evaluation/ragas_eval.py` | Full RAGAS evaluation suite: faithfulness, answer_relevancy, context_recall, context_precision. Runs against golden dataset. CI/CD integration ready. |
| `evaluation/metrics.py` | Custom retrieval metrics (MRR, NDCG, Hit@K) + Vertex Eval Service integration for groundedness, tool_name_match, trajectory_exact_match |

**RAGAS → Vertex Eval mapping:**
| RAGAS | Vertex Eval | What it measures |
|-------|-------------|-----------------|
| faithfulness | groundedness | Claims grounded in retrieved context? |
| answer_relevancy | response_quality/relevance | Does answer address the question? |
| context_recall | rag_context_recall | Were relevant docs retrieved? |
| context_precision | rag_context_precision | Signal-to-noise in retrieved chunks |

---

### Phase 8 — Deploy & Observe
| Script | Purpose |
|--------|---------|
| `api/main.py` | Production FastAPI application with CORS middleware, Pydantic request/response schemas, dependency injection for the RAG agent, error handling, and health endpoints. Ready for Cloud Run deployment. |

**Deployment targets:**
```bash
# Cloud Run (serverless)
gcloud run deploy rag-agent-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated

# Vertex AI Reasoning Engine (managed agent runtime)
reasoning_engines.ReasoningEngine.create(
    RAGAgentWrapper(),
    requirements=["langgraph", "langchain-google-vertexai"],
)
```

---

## ⚠️ Key Production Lessons from This Lab

**1. Org Policy blocks service account keys in Google Workspace**
```bash
# ❌ Blocked: constraints/iam.disableServiceAccountKeyCreation
gcloud iam service-accounts keys create ./sa-key.json ...

# ✅ Use ADC instead
gcloud auth application-default login
unset GOOGLE_APPLICATION_CREDENTIALS
```

**2. Gemini 1.x model aliases return 404 on new projects**
```python
# ❌ Retired — 404 Not Found
model = GenerativeModel("gemini-1.5-flash")

# ✅ Use versioned stable ID
model = GenerativeModel("gemini-2.0-flash-001")
```

**3. vertexai.generative_models SDK deprecated**
```python
# ❌ Deprecated June 2025, removal June 2026
from vertexai.generative_models import GenerativeModel

# ✅ New unified SDK
from google import genai
client = genai.Client(vertexai=True, project="your-project", location="us-central1")
```

**4. Install numpy before ragas to avoid scikit-network build failure**
```bash
pip install numpy          # must come first
pip install ragas          # now builds cleanly
```

**5. Package name mismatches**
| Import | pip install |
|--------|------------|
| `fitz` | `pymupdf` |
| `bs4` | `beautifulsoup4` |
| `pubsub_v1` | `google-cloud-pubsub` |

**6. venv must be activated — always verify before running scripts**
```bash
source ~/.venv/bin/activate
echo $VIRTUAL_ENV           # must show /home/sarang/.venv
python3 validate_setup.py
```

---

## 🚀 Quick Start

### Prerequisites
- Google Cloud project with billing enabled (Google Workspace billing account supported)
- Python 3.12
- Google Cloud SDK (`gcloud`)
- GitHub account

### Setup

```bash
# Clone
git clone https://github.com/Sarang68/vertex-ai-langgraph-lab.git
cd vertex-ai-langgraph-lab

# Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install (numpy first — avoids scikit-network build error)
pip install numpy
pip install -r requirements.txt

# Authenticate
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

# Enable APIs
gcloud services enable aiplatform.googleapis.com \
  storage.googleapis.com \
  pubsub.googleapis.com \
  cloudresourcemanager.googleapis.com

# Validate everything works
python3 validate_setup.py
```

### Run Lab Modules in Order

```bash
python3 ingestion/loaders.py          # Phase 1: Load documents
python3 ingestion/gcs_sink.py         # Phase 1: Push to GCS + Pub/Sub
python3 chunking/strategies.py        # Phase 2: Chunk documents
python3 embeddings/vertex_embedder.py # Phase 2: Generate embeddings
python3 vector_db/vertex_vector_search.py  # Phase 3: Index in Vertex
python3 retrieval/hybrid_search.py    # Phase 4: Test retrieval
python3 agents/rag_agent.py           # Phase 5: Run RAG agent
python3 graph/neo4j_builder.py        # Phase 6: Build knowledge graph
python3 evaluation/ragas_eval.py      # Phase 7: Evaluate quality
uvicorn api.main:app --reload         # Phase 8: Start production API
```

---

## 🔑 LangGraph → Vertex AI Concept Map

| LangGraph | Vertex AI Equivalent |
|-----------|---------------------|
| `ChatOpenAI` / `ChatAnthropic` | `ChatVertexAI` / `google-genai` client |
| `StateGraph` + TypedDict | ADK `LlmAgent` + Session state |
| `MemorySaver` / `SqliteSaver` | `InMemorySessionService` / Firestore |
| `graph.invoke()` / `graph.stream()` | `runner.run()` / `runner.run_live()` |
| `interrupt()` + `Command.RESUME` | `before_tool_callback` returning override |
| FAISS / Chroma | Vertex Vector Search / RAG Engine |
| LangSmith traces | Cloud Trace + Vertex Eval Service |
| Supervisor node pattern | ADK `LlmAgent` with `sub_agents=[]` |

---

## 💡 The Core Insight

> Your LangGraph code stays largely intact — you're mostly swapping the LLM provider
> and adding Vertex's managed infrastructure layer around it.

Enterprise AI deployment is **30% architecture, 70% plumbing**.
The agent code is the easy part. IAM, org policies, versioned model IDs,
deprecated SDKs, venv management — that's where projects succeed or fail.

---

## 📦 Tech Stack

- **Agent Framework**: LangGraph 0.2.28 + LangChain 0.2.16
- **LLM**: Gemini 2.0 Flash (`gemini-2.0-flash-001`) via Vertex AI
- **Embeddings**: `text-embedding-004` (768 dims)
- **Vector DBs**: Vertex AI Vector Search, Pinecone
- **Graph DB**: Neo4j (Cypher)
- **RAG Evaluation**: RAGAS + Vertex Eval Service
- **API**: FastAPI + Uvicorn
- **Cloud**: GCP — Vertex AI, Cloud Run, GCS, Pub/Sub, Cloud Logging
- **Auth**: Application Default Credentials (ADC)

---

## 🔗 Related Projects

- [Device Launch Planning Agent](https://github.com/Sarang68) — LangGraph multi-agent for 1,800+ Verizon retail stores (3 weeks → 3 days)
- [Telecom Retail Agentic AI with SAP](https://github.com/Sarang68) — LangGraph + SAP Retail ECC/S4HANA integration
- [AWS Glue + Kafka + Bedrock Lab](https://github.com/Sarang68/glue-kafka-spark-lab)

---

*Built on a Sunday morning in Google Cloud Shell ☕*
*Real errors. Real fixes. Real production lessons.*

