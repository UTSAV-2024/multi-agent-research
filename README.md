# Multi-Agent Research System

A modular multi-agent AI research pipeline that performs:
- web search
- article retrieval
- fact extraction
- cross-source verification
- report generation

The system is designed using a service-oriented modular architecture to simulate real-world AI backend systems.

---

# Features

- Multi-agent orchestration
- Web search using DuckDuckGo
- Full article content extraction
- Structured JSON-based fact extraction
- Cross-source fact verification
- Automated research report generation
- Modular backend architecture
- Logging system
- Fault-tolerant JSON parsing

---

# Architecture

```text
User Query
   ↓
Search Agent
   ↓
Content Fetch Agent
   ↓
Summarizer Agent
   ↓
Fact-Check Agent
   ↓
Report Agent
   ↓
Final Research Report

# directory Structure

multi-agent-research/
│
├── agents/
│   ├── search_agent.py
│   ├── content_fetch_agent.py
│   ├── summarizer_agent.py
│   ├── factcheck_agent.py
│   └── report_agent.py
│
├── services/
│   └── llm_service.py
│
├── utils/
│   ├── json_utils.py
│   └── logger.py
│
├── orchestrator/
│   └── workflow.py
│
├── config/
│   └── settings.py
│
├── main.py
├── requirements.txt
└── .env

Agents
Search Agent

Finds relevant web sources using DuckDuckGo search.

Content Fetch Agent

Retrieves and cleans article content using Trafilatura.

Summarizer Agent

Extracts structured facts from retrieved articles using LLMs.

Fact-Check Agent

Cross-references facts across sources to identify:

confirmed facts
disputed facts
low-confidence claims
Report Agent

Generates a final structured research report.

Tech Stack
Python
Groq API
Llama 3.1
DuckDuckGo Search (DDGS)
Trafilatura
Modular Backend Architecture
Key Engineering Concepts
Agent orchestration
Retrieval pipelines
Structured outputs
Fault tolerance
Service-oriented architecture
Logging systems
JSON reliability handling


Example Topics
Operation Paperclip
MKUltra
NVIDIA Blackwell
AI Agent Frameworks
OpenAI vs Anthropic
Future Improvements
Async agent execution
Parallel workflows
Planner agents
Source credibility scoring
Vector database integration
Citation tracking
FastAPI backend
Docker deployment
Observability dashboard


Run Locally
Install dependencies
pip install -r requirements.txt
Add environment variables

Create .env

GROQ_API_KEY=your_api_key
Run
python main.py