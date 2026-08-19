# SickleGuide

> Evidence-grounded clinical AI assistant for Sickle Cell Disease (SCD), with hybrid retrieval, knowledge-graph reasoning, reranking, grounding checks, citations, and a full-stack web interface.

## What is SickleGuide?

SickleGuide is designed to help users explore Sickle Cell Disease information from a controlled knowledge base rather than relying on unsupported model knowledge.

The system combines:

- Dense semantic retrieval with BGE-M3
- BM25 lexical retrieval
- Knowledge-graph retrieval
- Reciprocal Rank Fusion (RRF)
- BGE reranking
- Evidence fusion
- Grounded LLM generation
- Claim/evidence grounding review
- Citation validation
- Clinical-safety checks
- FastAPI backend
- React + Vite frontend
- Persistent Chroma vector storage

> **Clinical safety:** SickleGuide is an information/research aid, not a substitute for a qualified healthcare professional. Outputs should be verified against the cited source material and appropriate clinical guidance.

---

## Architecture

```text
PDF / Knowledge Base
        │
        ▼
Document Ingestion
        │
        ├── PDF Loading
        ├── Cleaning
        ├── Markdown Conversion
        ├── Section-Aware Chunking
        └── Metadata Enrichment
        │
        ├───────────────┬────────────────┐
        ▼               ▼                ▼
   BGE-M3           BM25             Graph
   Dense Search     Search           Retrieval
        │               │                │
        └───────────────┼────────────────┘
                        ▼
                 Hybrid Retrieval
                       (RRF)
                        │
                        ▼
                    Reranker
                        │
                        ▼
                  Final Evidence
                        │
                        ▼
                Grounded LLM Answer
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
       Grounding Review    Citation Validation
              │                   │
              └─────────┬─────────┘
                        ▼
                 Final Answer + Sources
                        │
                        ▼
                 FastAPI REST API
                        │
                        ▼
                  React Frontend
```

## Main features

### Chat

- ChatGPT-style conversation interface
- Collapsible sidebar
- Automatic short chat titles
- Conversation history
- Streaming-style answer rendering
- Evidence/source display
- Grounding and citation status
- Copy-friendly answers

### Knowledge Base

Upload a PDF from the Knowledge Base page. The backend processes it through loading, cleaning, Markdown conversion, chunking, metadata enrichment, persistent vector indexing, and runtime retriever refresh. Uploaded chunks are immediately available to dense and BM25 retrieval.

> Current upload indexing is designed around the existing retrieval pipeline. If graph entities/relationships are extracted from new documents in your deployment, rebuild/refresh the graph index as part of that ingestion workflow.

### Knowledge Graph

- Interactive graph visualization
- Drag, pan and zoom
- Node selection
- Node details and metadata
- Relationship exploration

### Evaluation

The evaluation interface is organized around the project rubric:

1. Retrieval Quality
2. Answer Grounding & Faithfulness
3. System Architecture & Full-Stack Implementation
4. Evaluation & Metrics Implementation
5. Clinical Safety & Responsible AI
6. Presentation, Communication & Live Demo
7. Innovation & Out-of-the-Box Thinking

Automatic evaluation currently includes retrieval and grounding-oriented metrics where supported by the evaluation pipeline; architecture, presentation and innovation are demo/reviewer criteria rather than pretending they are purely numerical metrics.

---

## Project structure

```text
SickleGuide/
├── api/
│   ├── main.py
│   └── routes/
│       ├── chat.py
│       ├── data.py
│       ├── evaluation.py
│       ├── graph.py
│       ├── health.py
│       └── search.py
├── frontend/
│   ├── src/
│   ├── package.json
│   └── ...
├── src/
│   ├── chunking/
│   ├── evaluation/
│   ├── generation/
│   ├── graph/
│   ├── ingestion/
│   └── retrieval/
├── data/
│   ├── raw/
│   └── processed/
├── .env.example
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Local setup

### 1. Clone

```bash
git clone https://github.com/Nasser-10/SickleGuide.git
cd SickleGuide
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
copy .env.example .env
```

For macOS/Linux:

```bash
cp .env.example .env
```

Adjust `.env` if your Ollama server, model, paths, or retrieval settings differ.

### 5. Install and run Ollama

SickleGuide currently uses Ollama for generation through LangChain. Install Ollama separately, make sure it is running, and install the model configured by `LLM_MODEL`.

Default configuration:

```text
LLM_MODEL=qwen2.5:7b
LLM_BASE_URL=http://localhost:11434
```

### 6. Start the backend

From the repository root:

```bash
uvicorn api.main:app --reload
```

Backend:

```text
http://localhost:8000
```

API docs:

```text
http://localhost:8000/docs
```

### 7. Start the frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The Vite development server normally runs at:

```text
http://localhost:5173
```

### 8. Verify health

Open:

```text
http://localhost:8000/api/v1/health
```

---

## Docker

The repository includes a backend Dockerfile and a Docker Compose configuration.

```bash
docker compose up --build
```

Backend is exposed on port `8000`. The frontend container is configured for port `3000` in the current Compose setup.

> Ollama is normally run separately on the host unless you explicitly add an Ollama service/container to your deployment.

---

## Environment variables

See `.env.example` for the complete documented configuration.

Important variables include:

| Variable | Purpose | Default |
|---|---|---|
| `LLM_MODEL` | Ollama generation model | `qwen2.5:7b` |
| `LLM_BASE_URL` | Ollama endpoint | `http://localhost:11434` |
| `LLM_TEMPERATURE` | Generation temperature | `0.0` |
| `LLM_NUM_PREDICT` | Maximum generated tokens | `1200` |
| `EMBEDDING_MODEL` | Dense embedding model | `BAAI/bge-m3` |
| `RERANKER_MODEL` | Reranker model | `BAAI/bge-reranker-v2-m3` |
| `CHROMA_PATH` | Persistent vector DB path | `./data/processed/chroma` |
| `CHUNKS_PATH` | Processed chunks file | `./data/processed/chunks.json` |
| `GRAPH_PATH` | Knowledge graph file | `./data/processed/graph.json` |
| `DENSE_K` | Dense retrieval candidates | `15` |
| `BM25_K` | BM25 candidates | `15` |
| `GRAPH_K` | Graph candidates | `15` |
| `CANDIDATE_K` | Hybrid candidate pool | `20` |
| `FINAL_K` | Final evidence count | `5` |
| `API_HOST` | FastAPI host | `0.0.0.0` |
| `API_PORT` | FastAPI port | `8000` |
| `VITE_API_BASE_URL` | Frontend API base URL | `http://localhost:8000/api/v1` |

Never commit a real `.env` file or API secrets.

---

## Evaluation rubric

SickleGuide is evaluated against:

- **Retrieval Quality** — relevance and ranking of retrieved evidence.
- **Answer Grounding & Faithfulness** — whether important medical claims are supported by retrieved evidence.
- **System Architecture & Full-Stack Implementation** — ingestion, retrieval, graph, API and frontend integration.
- **Evaluation & Metrics Implementation** — reproducible retrieval and answer-quality measurements.
- **Clinical Safety & Responsible AI** — evidence-first behavior, uncertainty handling and safe failure.
- **Presentation, Communication & Live Demo** — clarity, usability and reliable demonstration.
- **Innovation & Out-of-the-Box Thinking** — graph retrieval, evidence fusion, grounding checks and transparent evidence exploration.

---

## Safety note

SickleGuide is intended for educational, research and evidence-exploration purposes. It does not diagnose patients, replace clinicians, or provide individualized medical treatment decisions. Users should consult qualified healthcare professionals and verify important information against the cited evidence.

## Team

**Clinova**

## Project

**SickleGuide**
