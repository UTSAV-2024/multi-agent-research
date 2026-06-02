# Multi-Agent Research System

A modular multi-agent AI research pipeline that performs:
- web search
- article retrieval
- content chunking
- fact extraction
- cross-source verification
- report generation

The system is designed using a service-oriented modular architecture to simulate real-world AI backend systems.

---

# Architecture

## Current Pipeline

```
User Query
   ↓
Search Agent            — DuckDuckGo search
   ↓
Content Fetch Agent     — Trafilatura article extraction
   ↓
Chunking Pipeline       — Overlapping text chunks → MongoDB
   ↓
Summarizer Agent        — LLM fact extraction from source content
   ↓
Fact-Check Agent        — Cross-source verification
   ↓
Report Agent            — Final research report generation
   ↓
API Response
```

## Directory Structure

```
multi-agent-research/
│
├── agents/
│   ├── search_agent.py
│   ├── content_fetch_agent.py
│   ├── summarizer_agent.py
│   ├── factcheck_agent.py
│   └── report_agent.py
│
├── api/
│   ├── routes/
│   │   ├── health.py
│   │   ├── research.py
│   │   ├── history.py
│   │   └── reports.py
│   └── router.py
│
├── config/
│   └── settings.py
│
├── db/
│   ├── collections/
│   │   ├── reports_collection.py
│   │   └── chunks_collection.py
│   ├── database.py
│   └── mongodb.py
│
├── middleware/
│   ├── request_id.py
│   ├── request_logging.py
│   └── request_timing.py
│
├── orchestrator/
│   └── workflow.py
│
├── repositories/
│   ├── report_repository.py
│   └── chunk_repository.py
│
├── schemas/
│   └── research_schema.py
│
├── services/
│   ├── llm_service.py
│   ├── chunking_service.py
│   └── embedding_service.py
│
├── utils/
│   ├── response_builder.py
│   ├── json_utils.py
│   ├── exceptions.py
│   └── logger.py
│
├── tests/
│   ├── test_health.py
│   ├── test_research.py
│   ├── test_history.py
│   └── test_reports.py
│
├── main.py
├── requirements.txt
└── .env
```

---

# Features

- Multi-agent orchestration
- Web search using DuckDuckGo
- Full article content extraction (Trafilatura)
- Overlapping content chunking (foundation for retrieval)
- Structured JSON-based fact extraction
- Cross-source fact verification
- Automated research report generation
- Modular backend architecture (FastAPI)
- MongoDB persistence (reports & chunks)
- Logging system
- Fault-tolerant JSON parsing
- Paginated history API
- Report detail & delete APIs

---

# Chunk Flow

After content is fetched from web sources, it passes through the chunking pipeline:

1. **Input**: Raw article text (up to ~5000 chars per source)
2. **Chunking**: `chunk_text()` splits text into overlapping segments (500 char chunks, 100 char overlap)
3. **Storage**: Each chunk saved to `chunks` MongoDB collection
4. **Downstream**: Summarizer agent uses original source content

---

# Embedding Flow (Planned)

1. Each stored chunk will be embedded into a vector
2. ChromaDB will store chunk_id → vector mappings
3. Queries will be matched against stored vectors
4. Top-K chunks used as summarizer context

---

# Vector Search Plan

1. Integrate ChromaDB
2. Implement `EmbeddingService` with a concrete model
3. Add embedding step to chunking pipeline
4. Create vector search endpoint
5. Use retrieved chunks as augmented context

---

# API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Root status |
| GET | `/health` | Health check |
| POST | `/research` | Run research pipeline |
| GET | `/history` | Paginated report history |
| GET | `/reports/{id}` | Get report by ID |
| DELETE | `/reports/{id}` | Delete report |

---

# Tech Stack

- **Python** 3.14
- **FastAPI** — Web framework
- **MongoDB** (Motor) — Primary data store
- **Groq API** — LLM inference
- **Llama 3.1** — Default model
- **DuckDuckGo Search** (DDGS) — Web search
- **Trafilatura** — Article extraction
- **Pytest** — Testing

---

# Key Engineering Concepts

- Agent orchestration
- Retrieval pipelines
- Structured outputs
- Fault tolerance
- Service-oriented architecture
- Logging systems
- JSON reliability handling
- Content chunking
- Embedding service abstraction
- Repository pattern

---

# Run Locally

```
pip install -r requirements.txt
```

Create `.env`:
```
GROQ_API_KEY=your_api_key
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=multi_agent_platform
```

Run:
```
python -m uvicorn app.main:app --reload
```

Run tests:
```
python -m pytest tests/ -v
```