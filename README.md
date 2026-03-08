# ☕ Vertex AI + LangGraph Lab

> Sunday Morning. Coffee. Cloud Shell. Let's Talk Vertex AI.

A hands-on lab demonstrating how to migrate a LangGraph multi-agent system
to Google Cloud Vertex AI — from environment setup to RAG pipeline deployment.

Built by [Sarang Mahatwo](https://www.linkedin.com/in/sarang-mahatwo) —
Distinguished Engineer | AI Architect | LangGraph + Vertex AI

---

## 🎯 What This Lab Covers

| Module | Description |
|--------|-------------|
| `validate_setup.py` | Verify Vertex AI connectivity + Gemini model access |
| `loaders.py` | Document loaders for RAG pipeline (PDF, web, GCS) |
| `gcs_sink.py` | GCS + PubSub ingestion pipeline for RAG corpus |

---

## 🏗️ Architecture
```
LangGraph Agent (Python)
        ↓
ChatVertexAI (gemini-2.0-flash-001)
        ↓
Vertex AI Reasoning Engine (managed runtime)
        ↓
RAG Engine ←→ Vertex Vector Search
        ↓
Vertex Eval Service (groundedness, tool_name_match)
```

---

## 🚀 Quick Start

### Prerequisites
- Google Cloud project with billing enabled
- Python 3.12+
- Google Cloud SDK (`gcloud`)

### 1. Clone the repo
```bash
git clone https://github.com/Sarang68/vertex-ai-langgraph-lab.git
cd vertex-ai-langgraph-lab
```

### 2. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install numpy  # install first to avoid build errors
pip install -r requirements.txt
```

### 4. Authenticate
```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 5. Validate setup
```bash
python3 validate_setup.py
```

---

## ⚠️ Key Lessons from This Lab

**1. Org Policy blocks service account keys**
Google Workspace orgs enforce `constraints/iam.disableServiceAccountKeyCreation`.
Use Application Default Credentials (ADC) instead:
```bash
gcloud auth application-default login
```

**2. Gemini 1.5 models return 404 on new projects**
All Gemini 1.0 and 1.5 models are retired. New projects only have access
to Gemini 2.x. Always use versioned model IDs:
```python
# ❌ Broken
model = GenerativeModel("gemini-1.5-flash")

# ✅ Working
model = GenerativeModel("gemini-2.0-flash-001")
```

**3. vertexai.generative_models is deprecated**
Deprecated June 2025, removal June 2026. Migrate to `google-genai` SDK:
```python
from google import genai
client = genai.Client(vertexai=True, project="your-project", location="us-central1")
```

**4. Package name mismatches**
| Import | Install |
|--------|---------|
| `fitz` | `pymupdf` |
| `bs4` | `beautifulsoup4` |

---

## 🔑 LangGraph → Vertex AI Concept Map

| LangGraph | Vertex AI |
|-----------|-----------|
| `ChatOpenAI` | `ChatVertexAI` |
| `StateGraph` | ADK `LlmAgent` |
| `MemorySaver` | `InMemorySessionService` |
| `graph.invoke()` | `runner.run()` |
| `interrupt()` | `before_tool_callback` |
| FAISS/Chroma | Vertex RAG Engine |
| LangSmith | Cloud Trace + Vertex Eval |

---

## 💡 The Key Insight

> Your LangGraph code stays largely intact. You're mostly swapping the
> LLM provider and adding Vertex's managed infrastructure layer around it.

---

## 📦 Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Vertex AI](https://img.shields.io/badge/Vertex%20AI-1.71-orange)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2.28-green)
![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Platform-red)

---

## 🔗 Related

- [LinkedIn Post](https://www.linkedin.com/in/sarang-mahatwo)
- [Device Launch Planning Agent](https://github.com/Sarang68)
- [Telecom Retail Agentic AI with SAP](https://github.com/Sarang68)

---

*Built on a Sunday morning in Google Cloud Shell ☕*
