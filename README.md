<div align="center">

# 🧠 Ontology Reasoning System

**Think-on-Graph 3.0 + MACER Framework**

*Meta-cognitive Adaptive Chain-of-thought with Evidence-based Reasoning*

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-green.svg)](https://github.com/langchain-ai/langgraph)
[![Neo4j](https://img.shields.io/badge/Neo4j-Knowledge_Graph-008CC1.svg)](https://neo4j.com/)

[English](#overview) | [한국어](#개요)

**Built with ❤️ by Kit4Some & [sapiens.team](https://sapiens.team)**

</div>

---

## Overview

Ontology Reasoning System is a next-generation knowledge graph reasoning engine that goes far beyond traditional RAG (Retrieval-Augmented Generation). It implements **Think-on-Graph (ToG) 3.0** with the **MACER framework** — a meta-cognitive reasoning pipeline that adaptively explores, validates, and synthesizes evidence from structured knowledge graphs.

### Why Not Just RAG?

| Aspect | Traditional RAG | Ontology Reasoning |
|--------|----------------|-------------------|
| **Reasoning** | Vector similarity + LLM | Meta-cognitive 4-stage pipeline |
| **Query Handling** | Static, single-pass | Adaptive refinement & decomposition |
| **Evidence Validation** | Basic relevance | 5-component scoring + contradiction detection |
| **Multi-hop Questions** | LLM-dependent hallucination | Explicit path tracking & bridge entity detection |
| **Temporal Reasoning** | Ignored | Native temporal alignment & event sequencing |
| **Failure Transparency** | "I don't know" | Detailed confidence classification & gap analysis |

---

---

## ✨ Key Features

### 🔄 MACER Reasoning Pipeline
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Constructor │ -> │  Retriever  │ -> │  Reflector  │ -> │  Responser  │
│             │    │             │    │   (loop)    │    │             │
│ Entity      │    │ 5 Evidence  │    │ Sufficiency │    │ Synthesis   │
│ Extraction  │    │ Strategies  │    │ Assessment  │    │ & Answer    │
└─────────────┘    └─────────────┘    └──────┬──────┘    └─────────────┘
                                             │
                                    EXPLORE / FOCUS / REFINE / BACKTRACK
```

### 📊 5 Evidence Collection Strategies
- **Vector Search**: Semantic similarity on entity/chunk embeddings
- **Graph Traversal**: Multi-hop structural exploration
- **Community Summaries**: High-level contextual retrieval
- **Text2Cypher**: Natural language to Cypher with self-healing
- **Hybrid Mode**: Intelligent combination of all strategies

### 🎯 Advanced Evidence Scoring
- **Entity Overlap** (35%): Jaccard similarity matching
- **Relationship Match** (25%): Graph structure alignment
- **Temporal Alignment** (20%): Date/time context validation
- **Answer Presence** (10%): Direct answer detection
- **Negative Evidence** (10%): Contradiction & negation detection

### 🌐 Additional Capabilities
- **Multilingual**: Full Korean/English support with optimized fuzzy matching
- **LLM Failover**: Automatic cascade (OpenAI → Anthropic → Azure → Ollama)
- **Incremental Updates**: Delta-based graph modifications with change tracking
- **Ontology Schema**: Entity type inheritance, predicate cardinality, domain profiles
- **SSE Streaming**: Real-time progress for long-running operations

---

## 🏗️ Architecture

### MACER Pipeline Flow
```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INTENT CLASSIFICATION                        │
│         (KNOWLEDGE | GREETING | SMALL_TALK | SYSTEM)           │
└─────────────────────────────────────────────────────────────────┘
    │ KNOWLEDGE
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ CONSTRUCTOR                                                     │
│ • Extract topic entities (multilingual NLP)                     │
│ • Vector + Full-text entity retrieval                          │
│ • Build seed subgraph with 1-3 hop neighbors                   │
│ • Detect bridge entities for multi-hop questions               │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ RETRIEVER                                                       │
│ • Execute 5 evidence collection strategies                      │
│ • Rank evidence with 5-component scoring                       │
│ • Track evidence chains for provenance                         │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ REFLECTOR (Meta-cognitive Core)                    ◄────┐      │
│ • Assess sufficiency (0.0 - 1.0)                        │      │
│ • Evaluate: Completeness, Coverage, Consistency         │      │
│ • Decide: EXPLORE | FOCUS | REFINE | BACKTRACK | CONCLUDE      │
│ • Evolve query if needed ───────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
    │ CONCLUDE
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ RESPONSER                                                       │
│ • Synthesize evidence into facts/inferences                    │
│ • Generate natural language answer                             │
│ • Provide confidence: CONFIDENT | PROBABLE | UNCERTAIN         │
│ • Include reasoning explanation & evidence attribution         │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
Final Answer with Confidence + Explanation + Sources
```

### SDDI Ingestion Pipeline
```
Documents (JSON, PDF, MD, CSV, XML, YAML, HTML, DOCX)
    │
    ▼
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Ingest  │ → │  Chunk   │ → │ Extract  │ → │  Embed   │ → │  Load    │
│          │   │          │   │ Entities │   │          │   │ to Neo4j │
│ Loaders  │   │ Smart    │   │ Relations│   │ 1536-dim │   │ Bulk     │
│ Encoding │   │ Overlap  │   │ LLM-based│   │ Vectors  │   │ Upsert   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

---

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/your-org/ontology-reasoning.git
cd ontology-reasoning

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start all services
docker-compose up -d

# Verify
curl http://localhost:8000/api/health
```

**Access Points:**
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Neo4j Browser: http://localhost:7474
- LangGraph Studio: http://localhost:8123

### Manual Setup

```bash
# Python 3.11+ required
pip install -e ".[dev]"

# Start Neo4j separately (Docker or native)
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:5.15

# Configure and run
cp .env.example .env
uvicorn src.api.main:app --reload
```

### Desktop App (Electron)

```bash
cd desktop
npm install
npm run dev      # Development mode
npm run build    # Production build
```

**Features:**
- Session-based chat history with auto-save
- Expandable reasoning process view
- Real-time streaming with step-by-step updates
- Multi-session management (create, switch, delete)
- Dark/Light theme support

---

## 📡 API Reference

### Query Endpoints
```bash
# Synchronous reasoning query
POST /api/query
{
  "query": "What is the relationship between Entity A and Entity B?",
  "max_iterations": 5
}

# SSE streaming with step-by-step updates
POST /api/query/stream
```

### Ingestion Endpoints
```bash
# Upload files (up to 1GB)
POST /api/ingest
Content-Type: multipart/form-data

# Stream ingestion progress
GET /api/ingest/{job_id}/stream
```

### Graph Operations
```bash
# Natural language to Cypher
POST /api/text2cypher
{
  "query": "Find all employees who work in Seoul",
  "execute": true
}

# Raw Cypher execution
POST /api/cypher
{
  "query": "MATCH (n:Entity) RETURN n LIMIT 10"
}
```

### System Endpoints
```bash
GET /api/health          # Health check
GET /api/stats           # Graph statistics
GET /api/schema          # Neo4j schema
GET /api/ontology        # Export ontology (JSON-LD, Turtle, JSON)
```

Full API documentation available at `/docs` (Swagger UI).

---

## ⚙️ Configuration

Create `.env` file from `.env.example`:

```env
# Neo4j Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password

# LLM Provider (openai | anthropic | azure | local)
LLM_PROVIDER=openai
LLM_OPENAI_API_KEY=sk-...
LLM_ANTHROPIC_API_KEY=sk-ant-...  # Optional fallback

# Model Selection
LLM_REASONING_MODEL=gpt-4o-mini
LLM_EMBEDDING_MODEL=text-embedding-3-small

# Deterministic Response Settings
LLM_TEMPERATURE=0.0
LLM_SEED=42
LLM_TOP_P=1.0

# Reasoning Parameters
TOG_MAX_REASONING_DEPTH=5
TOG_CONFIDENCE_THRESHOLD=0.7
```

---

## 🛠️ Development

### Commands

```bash
# Install with dev dependencies
make dev

# Run tests
make test           # All tests
make test-unit      # Unit tests only
make test-cov       # With coverage report

# Code quality
make lint-fix       # Lint with auto-fix
make format         # Format code
make typecheck      # Type checking
make check          # All checks

# Docker operations
make docker-up      # Start services
make docker-down    # Stop services
make db-setup       # Initialize Neo4j schema
make health         # Health check
```

### Project Structure

```
src/
├── api/              # FastAPI endpoints
├── config/           # Pydantic settings
├── graph/            # Neo4j client & operations
├── llm/              # LLM provider with failover
├── sddi/             # Data ingestion pipeline
│   ├── document_loaders/
│   ├── extractors/
│   └── loaders/
├── tog/              # MACER reasoning agents
│   ├── agents/       # Constructor, Retriever, Reflector, Responser
│   ├── temporal_reasoning.py
│   └── negative_evidence.py
├── text2cypher/      # NL to Cypher generation
├── validation/       # Pipeline validation framework
└── workflow/         # LangGraph orchestration

desktop/              # Electron desktop app
tests/                # Unit & integration tests
```

---

## 🔬 Advanced Features

### Temporal Reasoning
```python
from src.tog.temporal_reasoning import compute_enhanced_temporal_alignment

result = compute_enhanced_temporal_alignment(
    query="What happened before 2023?",
    evidence_text="The event occurred in January 2022..."
)
# Returns: {score, alignment_type, temporal_match, temporal_consistency}
```

### Negative Evidence Detection
```python
from src.tog.negative_evidence import analyze_evidence_polarity

polarity = analyze_evidence_polarity(
    evidence="The company did NOT acquire the startup.",
    query="Did the company acquire the startup?"
)
# Returns: NEGATIVE with contradiction score
```

### Incremental Graph Updates
```python
from src.sddi.pipeline import SDDIPipeline

pipeline = SDDIPipeline(
    llm=llm,
    embeddings=embeddings,
    use_incremental_loading=True,
)

# Get change report after ingestion
delta = pipeline.get_last_delta_report()
# DeltaReport: new_entities, modified_entities, unchanged, deleted
```

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards
- Python 3.11+ with type hints
- Ruff for linting and formatting
- MyPy for type checking
- Pytest for testing

---

## 🌟 About

<div align="center">

**Created by Kit4Some **

in collaboration with **[sapiens.team](https://sapiens.team)**

*Building the future of intelligent systems*

</div>

We believe in the power of open source to accelerate innovation. Ontology Reasoning System is our contribution to the AI community — a production-ready framework for building knowledge-intensive applications that reason, not just retrieve.

### Our Philosophy
- **Transparency**: Every reasoning step is traceable
- **Reliability**: Confidence scores you can trust
- **Extensibility**: Modular architecture for customization
- **Community**: Built together, better together

### Connect With Us
- 🌐 Website: [https://sapiens.team](https://sapiens.team)
- 📧 Email: gkemqk7@gmail.com
- 💬 Discussions: GitHub Issues & Discussions

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Star ⭐ this repository if you find it useful!**

Made with 🧠 by Kit4Some & [sapiens.team](https://sapiens.team)

</div>
