# Multi-Agent Research Platform

An evidence-backed, production-oriented multi-agent research system that autonomously searches the web, retrieves and verifies information across sources, generates grounded reports with deterministic citations, and continuously evaluates retrieval and evidence quality.

---

## Overview

The Multi-Agent Research Platform is designed to simulate how modern AI research systems operate in production environments.

Given a user query, the platform:

* Searches the web for relevant sources
* Extracts article content
* Chunks and persists information
* Generates embeddings and stores them in a vector database
* Retrieves evidence using hybrid retrieval
* Verifies facts across sources
* Produces grounded reports with deterministic citations
* Evaluates retrieval quality and evidence quality using benchmark suites
* Persists evaluation results for historical analysis

The goal of this project is not simply to build a RAG application, but to demonstrate strong software engineering practices around reliability, observability, evaluation, and maintainability.

---

# System Architecture

```text
User Query
    ↓
FastAPI API Layer
    ↓
Workflow Orchestrator
    ↓
Search Agent
    ↓
Content Fetch Agent
    ↓
Chunking Pipeline
    ↓
Embedding Service
    ↓
MongoDB Persistence
    ↓
ChromaDB Vector Store
    ↓
Hybrid Retrieval
    ↓
Evidence Service
    ↓
Evidence-Based Factchecking
    ↓
Deterministic Citations
    ↓
Hierarchical Report Generation
    ↓
Grounded Research Report
```

---

# Key Features

## Multi-Agent Research Pipeline

* Automated web research workflow
* Modular agent orchestration
* Concurrent execution support
* Graceful degradation during failures

---

## Retrieval-Augmented Generation (RAG)

* ChromaDB-backed vector retrieval
* Hybrid retrieval (semantic + keyword)
* Stable chunk identifiers
* Deterministic ranking
* Source deduplication
* Domain diversity tracking

---

## Evidence-Based Fact Verification

* Cross-source evidence retrieval
* Evidence-supported fact checking
* Confidence scoring
* Supporting evidence attribution
* Citation generation

---

## Deterministic Citations

* Code-generated citations
* Stable ordering
* Deduplicated references
* Source traceability

---

## Hierarchical Report Generation

Reports are generated through independent sections:

* Executive Summary
* Key Findings
* Evidence Analysis
* Limitations & Uncertainty
* Sources & Citations

Partial failures do not terminate report generation.

---

## Evaluation Infrastructure

### Retrieval Evaluation

Measures:

* Retrieval latency
* Chunk counts
* URL diversity
* Domain diversity
* Hybrid retrieval scores
* Stability metrics
* Jaccard similarity

Benchmark dataset:

* 35 benchmark queries
* 5 failure scenarios

---

### Evidence Evaluation

Measures:

* Support ratios
* Coverage scores
* Evidence-per-fact ratios
* Citation counts
* Confidence distributions
* Source diversity

---

### Historical Evaluation Tracking

Evaluation results are persisted for trend analysis.

Supported capabilities:

* Evaluation history
* Pagination
* Benchmark persistence
* Future dashboard support

---

# API Endpoints

## Research

| Method | Endpoint        | Description                   |
| ------ | --------------- | ----------------------------- |
| POST   | `/research`     | Execute the research workflow |
| GET    | `/history`      | Retrieve report history       |
| GET    | `/reports/{id}` | Get report details            |
| DELETE | `/reports/{id}` | Delete a report               |

---

## Evaluation

| Method | Endpoint                     | Description                 |
| ------ | ---------------------------- | --------------------------- |
| POST   | `/api/v1/evaluate/retrieval` | Run retrieval benchmarks    |
| POST   | `/api/v1/evaluate/evidence`  | Evaluate evidence quality   |
| GET    | `/api/v1/evaluate/history`   | Retrieve evaluation history |
| GET    | `/api/v1/evaluate/summary`   | System evaluation snapshot  |

---

## Health

| Method | Endpoint  | Description  |
| ------ | --------- | ------------ |
| GET    | `/health` | Health check |
| GET    | `/`       | Root status  |

---

# Technology Stack

### Backend

* Python 3.14
* FastAPI
* Uvicorn

### Datastores

* MongoDB
* ChromaDB

### AI & Retrieval

* Groq API
* Llama Models
* Sentence Transformers
* Hybrid Retrieval

### Web Intelligence

* DDGS (DuckDuckGo Search)
* Trafilatura

### Testing

* Pytest

---

# Engineering Highlights

* Service-oriented architecture
* Repository pattern
* Modular agent design
* Dependency injection
* Request ID propagation
* Structured logging
* Fault tolerance
* Retry mechanisms
* Timeout handling
* Graceful degradation
* Evaluation-driven development
* Historical benchmarking

---

# Testing

Current automated test status:

```text
392 passing tests
```

Coverage includes:

* API endpoints
* Retrieval services
* Evidence services
* Evaluation frameworks
* Persistence layers
* Failure scenarios
* Timeout handling
* Historical evaluation
* Report generation

Run tests:

```bash
pytest
```

---

# Running Locally

## Prerequisites

* Python 3.14+
* MongoDB
* Docker (recommended)
* Groq API Key

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd multi-agent-research
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=multi_agent_platform
CHROMA_PERSIST_DIR=./chroma_data
```

Start the application:

```bash
uvicorn app.main:app --reload
```

---

# Docker Support

Docker Compose support is included.

Run:

```bash
docker compose build
docker compose up
```

The platform automatically provisions:

* FastAPI application
* MongoDB
* Persistent ChromaDB storage

---

# Current Status

## Backend Status

Production-oriented backend complete.

Implemented:

* Multi-agent research workflow
* Hybrid retrieval
* Evidence-backed fact verification
* Deterministic citations
* Hierarchical reporting
* Retrieval evaluation
* Evidence evaluation
* Historical evaluation persistence
* Dockerization
* High-resolution observability metrics

---

# Future Work

Planned enhancements include:

* GitHub Actions CI/CD
* Documentation website
* Evaluation dashboards
* Frontend interface
* Cloud deployment

---

# License

This project is intended for educational, research, and portfolio purposes.
