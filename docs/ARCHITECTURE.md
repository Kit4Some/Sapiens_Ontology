# Ontology Reasoning System - Architecture Documentation

> **Version**: 0.1.0
> **Last Updated**: 2025-11-28
> **Framework**: Think-on-Graph (ToG) 3.0 with MACER

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Directory Structure](#2-directory-structure)
3. [Core Modules](#3-core-modules)
4. [Data Flow Architecture](#4-data-flow-architecture)
5. [State Management](#5-state-management)
6. [External Integrations](#6-external-integrations)
7. [API Reference](#7-api-reference)
8. [Configuration System](#8-configuration-system)
9. [Design Patterns](#9-design-patterns)
10. [Scalability & Performance](#10-scalability--performance)

---

## 1. System Overview

### 1.1 Purpose

Ontology Reasoning System은 지식 그래프 기반 질의응답 시스템으로, **Think-on-Graph (ToG) 3.0**과 **MACER (Meta-cognitive Adaptive Chain-of-thought with Evidence-based Reasoning)** 프레임워크를 구현합니다.

### 1.2 Key Capabilities

| Capability | Description |
|------------|-------------|
| **Knowledge Graph Reasoning** | Neo4j 기반 그래프 탐색 및 추론 |
| **Multi-Format Ingestion** | JSON, PDF, DOCX, Markdown 등 11개 포맷 지원 |
| **Semantic Search** | Vector similarity + Full-text 하이브리드 검색 |
| **Natural Language to Cypher** | 자연어 질의를 Cypher 쿼리로 변환 |
| **Multi-LLM Support** | OpenAI, Anthropic, Azure, Local LLM 지원 |
| **Real-time Streaming** | SSE 기반 실시간 진행상황 스트리밍 |

### 1.3 Technology Stack

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend/Client                         │
│                  (Electron Desktop App)                      │
├─────────────────────────────────────────────────────────────┤
│                      API Layer                               │
│              FastAPI + Uvicorn (Async)                       │
├─────────────────────────────────────────────────────────────┤
│                   Orchestration Layer                        │
│                     LangGraph                                │
├──────────────────┬──────────────────┬───────────────────────┤
│  MACER Reasoning │  SDDI Pipeline   │   Text2Cypher         │
│  (ToG Agents)    │  (Ingestion)     │   (NL→Query)          │
├──────────────────┴──────────────────┴───────────────────────┤
│                    Data Layer                                │
│              Neo4j (Graph Database)                          │
├─────────────────────────────────────────────────────────────┤
│                  Infrastructure                              │
│     LLM Providers │ Redis Cache │ Celery Workers            │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
src/
├── api/                          # FastAPI REST API
│   └── main.py                   # 메인 애플리케이션 (3,209 lines)
│
├── config/                       # 설정 관리
│   └── settings.py               # Pydantic Settings
│
├── core/                         # 핵심 유틸리티
│   ├── cache.py                  # Multi-layer 캐싱 (L1/L2)
│   └── circuit_breaker.py        # 서킷 브레이커
│
├── graph/                        # Neo4j 통합
│   ├── neo4j_client.py           # 메인 클라이언트 (1,062 lines)
│   ├── schema.py                 # 그래프 스키마
│   ├── query_optimizer.py        # 쿼리 최적화
│   ├── community.py              # 커뮤니티 탐지
│   ├── backup/                   # 백업/복구
│   ├── integrity/                # 무결성 검사
│   ├── lifecycle/                # 소프트 삭제
│   └── migrations/               # 스키마 마이그레이션
│
├── llm/                          # LLM 추상화
│   ├── provider.py               # Multi-provider + Failover (701 lines)
│   └── cached_embeddings.py      # 임베딩 캐싱
│
├── observability/                # 모니터링
│   ├── logging.py                # 구조화 로깅 (structlog)
│   ├── metrics.py                # Prometheus 메트릭
│   ├── tracing.py                # OpenTelemetry
│   ├── alerting.py               # 알림 (Slack, PagerDuty)
│   └── correlation.py            # 요청 추적
│
├── security/                     # 보안
│   ├── authentication.py         # JWT + API Key 인증
│   └── authorization.py          # RBAC 권한 관리
│
├── sddi/                         # Data Integration Pipeline
│   ├── pipeline.py               # 메인 파이프라인
│   ├── state.py                  # 상태 모델
│   ├── reliable_pipeline.py      # 신뢰성 래퍼
│   ├── document_loaders/         # 포맷별 로더 (11종)
│   ├── chunkers/                 # 텍스트 청킹
│   ├── extractors/               # 엔티티/관계 추출
│   ├── embedders/                # 임베딩 생성
│   ├── loaders/                  # Neo4j 로딩
│   ├── reliability/              # 에러 처리/재시도
│   └── distributed/              # Celery 분산 처리
│
├── text2cypher/                  # NL → Cypher 변환
│   ├── generator.py              # 쿼리 생성기 (19,299 lines)
│   ├── validator.py              # 쿼리 검증
│   ├── prompts.py                # 프롬프트 템플릿
│   └── examples.py               # Few-shot 예제
│
├── tog/                          # MACER Framework
│   ├── state.py                  # MACERState 정의
│   ├── prompts.py                # 에이전트 프롬프트
│   └── agents/                   # 4-Agent 시스템
│       ├── constructor.py        # 토픽 추출 + 시드 서브그래프
│       ├── retriever.py          # 증거 수집
│       ├── reflector.py          # 메타인지 제어 (핵심)
│       └── responser.py          # 답변 생성
│
├── validation/                   # 파이프라인 검증
│   ├── step_validators.py        # 단계별 검증
│   ├── pipeline_validator.py     # E2E 검증
│   └── context_extractor.py      # 컨텍스트 추출
│
└── workflow/                     # LangGraph 워크플로우
    └── graph.py                  # MACER 오케스트레이션
```

---

## 3. Core Modules

### 3.1 MACER Reasoning Framework (`src/tog/`)

Think-on-Graph 3.0 기반의 4-에이전트 추론 시스템:

```
┌─────────────────────────────────────────────────────────────┐
│                    MACER Pipeline                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Intent Classification]                                     │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │ Constructor │───▶│  Retriever  │◀──▶│  Reflector  │      │
│  │             │    │             │    │   (Core)    │      │
│  │ - 토픽 추출  │    │ - 증거 수집  │    │ - 충분성 평가│      │
│  │ - 시드 그래프│    │ - 멀티홉 탐색│    │ - 쿼리 진화  │      │
│  └─────────────┘    └─────────────┘    └──────┬──────┘      │
│                                               │              │
│                                               ▼              │
│                                        ┌─────────────┐      │
│                                        │  Responser  │      │
│                                        │             │      │
│                                        │ - 답변 합성  │      │
│                                        │ - 신뢰도 계산│      │
│                                        └─────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

#### Constructor Agent
- **역할**: 질문에서 토픽 엔티티 추출, 시드 서브그래프 구성
- **기능**:
  - 다국어 지원 (한국어/영어)
  - Vector + Full-text 하이브리드 검색
  - N-gram 유사도 기반 퍼지 매칭
  - 1-hop 이웃 확장

#### Retriever Agent
- **역할**: 다중 전략 증거 수집
- **전략**:
  - Vector Similarity Search
  - Multi-hop Graph Traversal
  - Community Summary Search
  - Text2Cypher NL Query

#### Reflector Agent (Core)
- **역할**: 메타인지 제어 및 추론 방향 결정
- **기능**:
  - 충분성 점수 평가 (0.0-1.0)
  - 쿼리 진화 (분해/정제)
  - 서브그래프 확장/가지치기
  - 액션 결정: `EXPLORE | FOCUS | REFINE | BACKTRACK | CONCLUDE`

#### Responser Agent
- **역할**: 최종 답변 생성
- **출력**:
  - 자연어 답변
  - 신뢰도 점수
  - 추론 설명
  - 답변 유형: `DIRECT | INFERRED | UNCERTAIN | NO_DATA`

### 3.2 SDDI Pipeline (`src/sddi/`)

Schema-Driven Data Integration 파이프라인:

```
┌─────────────────────────────────────────────────────────────┐
│                    SDDI Pipeline                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [File Upload]                                               │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   Loader    │───▶│   Chunker   │───▶│  Extractor  │      │
│  │             │    │             │    │             │      │
│  │ - 11 포맷   │    │ - 구조 인식  │    │ - Entity    │      │
│  │ - 스트리밍   │    │ - 오버랩    │    │ - Relation  │      │
│  └─────────────┘    └─────────────┘    └──────┬──────┘      │
│                                               │              │
│       ┌───────────────────────────────────────┘              │
│       ▼                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │  Embedder   │───▶│ Neo4j Load  │───▶│  Validate   │      │
│  │             │    │             │    │             │      │
│  │ - 1536-dim  │    │ - Bulk Load │    │ - Quality   │      │
│  │ - Caching   │    │ - Indexing  │    │ - Metrics   │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### Supported Formats

| Format | Loader | Features |
|--------|--------|----------|
| JSON | `json_loader.py` | ijson 스트리밍 (10MB+) |
| PDF | `pdf_loader.py` | PyMuPDF 추출 |
| DOCX | `docx_loader.py` | python-docx |
| Markdown | `markdown_loader.py` | 구조 보존 |
| CSV | `csv_loader.py` | 다중 인코딩 |
| XML | `xml_loader.py` | lxml 파싱 |
| YAML | `yaml_loader.py` | 안전 로딩 |
| HTML | `html_loader.py` | BeautifulSoup |
| JSON-LD | `jsonld_loader.py` | 시맨틱 웹 |
| Text | `text_loader.py` | Plain text |

### 3.3 Graph Layer (`src/graph/`)

```python
class OntologyGraphClient:
    """Neo4j 비동기 클라이언트"""

    # Schema
    async def create_indexes()           # 인덱스 생성
    async def setup_vector_index()       # 벡터 인덱스 (1536-dim)
    async def setup_fulltext_index()     # 전문 검색 인덱스

    # CRUD
    async def create_entity()
    async def get_entity()
    async def update_entity()
    async def delete_entity()

    # Search
    async def vector_search()            # 벡터 유사도 검색
    async def fulltext_search()          # 전문 검색
    async def hybrid_search()            # 하이브리드 검색

    # Graph Operations
    async def get_neighbors()            # 이웃 노드 조회
    async def find_paths()               # 경로 탐색
    async def get_subgraph()             # 서브그래프 추출
```

#### Neo4j Schema

```cypher
// Node Labels
(:Entity {
  id: STRING,
  name: STRING,
  type: STRING,
  description: STRING,
  embedding: LIST<FLOAT>,  // 1536-dim
  created_at: DATETIME,
  updated_at: DATETIME
})

(:Chunk {
  id: STRING,
  text: STRING,
  embedding: LIST<FLOAT>,
  document_id: STRING,
  position: INTEGER
})

(:Community {
  id: STRING,
  summary: STRING,
  embedding: LIST<FLOAT>,
  member_count: INTEGER
})

// Relationships
[:RELATES_TO {type: STRING, confidence: FLOAT}]
[:CONTAINS]
[:MENTIONS]
[:BELONGS_TO]
[:PARENT]
[:DERIVED_FROM]
[:SUPPORTS]

// Indexes
CREATE VECTOR INDEX entity_embedding FOR (e:Entity) ON (e.embedding)
  OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}

CREATE FULLTEXT INDEX entity_fulltext FOR (e:Entity) ON EACH [e.name, e.description]
```

### 3.4 LLM Provider Chain (`src/llm/`)

```
┌─────────────────────────────────────────────────────────────┐
│                   LLM Provider Chain                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Request ──▶ Health Check ──▶ Primary Provider               │
│                    │                │                        │
│                    │                ▼                        │
│                    │         ┌──────────────┐                │
│                    │         │   OpenAI     │◀── Primary     │
│                    │         │  (GPT-4o)    │                │
│                    │         └──────┬───────┘                │
│                    │                │ Fail?                  │
│                    │                ▼                        │
│                    │         ┌──────────────┐                │
│                    │         │  Anthropic   │◀── Fallback 1  │
│                    │         │(Claude 3.5)  │                │
│                    │         └──────┬───────┘                │
│                    │                │ Fail?                  │
│                    │                ▼                        │
│                    │         ┌──────────────┐                │
│                    │         │ Azure OpenAI │◀── Fallback 2  │
│                    │         └──────┬───────┘                │
│                    │                │ Fail?                  │
│                    │                ▼                        │
│                    │         ┌──────────────┐                │
│                    └────────▶│  Local LLM   │◀── Fallback 3  │
│                              │  (Ollama)    │                │
│                              └──────────────┘                │
│                                                              │
│  Features:                                                   │
│  - Auto failover on API errors                               │
│  - Exponential backoff retry                                 │
│  - Per-provider health tracking                              │
│  - Latency & success rate metrics                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow Architecture

### 4.1 Query Pipeline

```
User Query
    │
    ▼
┌───────────────────┐
│ Intent Classifier │──▶ GREETING/SMALL_TALK ──▶ Direct Response
└─────────┬─────────┘
          │ KNOWLEDGE
          ▼
┌───────────────────┐
│    Constructor    │
│                   │
│ 1. 질문 분석       │
│ 2. 토픽 엔티티 추출│
│ 3. 시드 서브그래프 │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐     ┌───────────────────┐
│    Retriever      │◀───▶│    Reflector      │
│                   │     │                   │
│ - Vector Search   │     │ 1. 충분성 평가     │
│ - Graph Traversal │     │ 2. 쿼리 진화      │
│ - Community Search│     │ 3. 서브그래프 조정 │
│ - Text2Cypher     │     │ 4. 종료 결정      │
└───────────────────┘     └─────────┬─────────┘
                                    │
                          ┌─────────┴─────────┐
                          │                   │
                     Continue            Terminate
                          │                   │
                          ▼                   ▼
                    Loop Back          ┌───────────────┐
                                       │   Responser   │
                                       │               │
                                       │ - 증거 합성   │
                                       │ - 답변 생성   │
                                       │ - 신뢰도 계산 │
                                       └───────┬───────┘
                                               │
                                               ▼
                                        Final Response
```

### 4.2 Ingestion Pipeline

```
File Upload (max 1GB)
    │
    ▼
┌───────────────────┐
│  Format Detection │
│  & Loader Select  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Content Extract  │
│                   │
│ - Streaming (10MB+)│
│ - Multi-encoding  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│    Chunking       │
│                   │
│ - Size: 1000 chars│
│ - Overlap: 200    │
│ - Structure-aware │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Entity Extraction │
│                   │
│ - LLM-based       │
│ - Batch processing│
│ - Confidence score│
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│Relation Extraction│
│                   │
│ - Predicate ID    │
│ - Entity linking  │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Embedding Gen     │
│                   │
│ - 1536-dim        │
│ - OpenAI API      │
│ - Caching         │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   Neo4j Load      │
│                   │
│ - Bulk insert     │
│ - Index creation  │
│ - Relationships   │
└─────────┬─────────┘
          │
          ▼
    Knowledge Graph
```

---

## 5. State Management

### 5.1 MACERState

```python
class MACERState(TypedDict):
    """MACER 추론 상태"""

    # Query Tracking
    original_query: str
    current_query: str
    query_history: list[QueryEvolution]

    # Graph Tracking
    topic_entities: list[TopicEntity]
    retrieved_entities: list[dict]
    current_subgraph: SubGraph
    subgraph_history: list[SubGraph]

    # Evidence Tracking
    evidence: list[Evidence]
    evidence_rankings: dict[str, float]

    # Reasoning Tracking
    reasoning_path: list[ReasoningStep]
    sufficiency_score: float          # 0.0 - 1.0
    iteration: int
    max_iterations: int
    should_terminate: bool

    # Output
    final_answer: str | None
    confidence: float
    explanation: str
    answer_type: AnswerType           # DIRECT/INFERRED/UNCERTAIN/NO_DATA

    # Metadata
    metadata: MACERMetadata
```

### 5.2 SDDIPipelineState

```python
class SDDIPipelineState(TypedDict):
    """SDDI 파이프라인 상태"""

    # Input
    raw_data: list[RawDocument]

    # Processing Stages
    chunks: list[TextChunk]
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
    triplets: list[Triplet]
    embeddings: dict[str, list[float]]

    # Output
    load_status: LoadStatus           # PENDING/IN_PROGRESS/COMPLETED/FAILED
    load_result: LoadResult

    # Metadata
    pipeline_id: str
    current_step: str
    errors: list[str]
    progress: float
```

### 5.3 SubGraph Model

```python
class SubGraph(BaseModel):
    """서브그래프 표현"""

    nodes: list[SubGraphNode] = []
    edges: list[SubGraphEdge] = []
    center_entity_id: str | None = None

    def node_count(self) -> int
    def edge_count(self) -> int
    def get_node(self, node_id: str) -> SubGraphNode | None
    def get_neighbors(self, node_id: str) -> list[SubGraphNode]
    def add_node(self, node: SubGraphNode) -> None
    def add_edge(self, edge: SubGraphEdge) -> None
    def merge(self, other: SubGraph) -> SubGraph
    def prune_low_relevance(self, threshold: float) -> SubGraph
```

---

## 6. External Integrations

### 6.1 Neo4j

| Aspect | Configuration |
|--------|---------------|
| Connection | `bolt://localhost:7687` |
| Pool Size | 50 connections |
| Query Timeout | 30s (default), 120s (max) |
| Vector Index | 1536-dim, cosine similarity |
| Full-text Index | Multilingual support |

### 6.2 LLM Providers

| Provider | Models | Use Case |
|----------|--------|----------|
| OpenAI | GPT-4o, GPT-4o-mini | Primary reasoning |
| Anthropic | Claude 3.5 Sonnet | Fallback, extraction |
| Azure OpenAI | Custom deployment | Enterprise |
| Ollama | Llama 3.2, etc. | Local/offline |

### 6.3 Embeddings

| Model | Dimensions | Provider |
|-------|------------|----------|
| text-embedding-3-small | 1536 | OpenAI |

### 6.4 Infrastructure

| Component | Technology | Purpose |
|-----------|------------|---------|
| Cache L1 | In-memory LRU | Fast access |
| Cache L2 | Redis | Distributed cache |
| Task Queue | Celery | Distributed processing |
| Message Broker | Redis | Task communication |

---

## 7. API Reference

### 7.1 Query Endpoints

```http
POST /api/query
Content-Type: application/json

{
  "query": "What is the relationship between Entity A and Entity B?",
  "max_iterations": 5,
  "confidence_threshold": 0.7
}

Response:
{
  "answer": "...",
  "confidence": 0.85,
  "evidence": [...],
  "reasoning_path": [...]
}
```

```http
POST /api/query/stream
Content-Type: application/json
Accept: text/event-stream

Response: SSE Stream
event: step
data: {"step": "constructor", "progress": 0.2, "message": "..."}

event: step
data: {"step": "retriever", "progress": 0.5, "message": "..."}

event: complete
data: {"answer": "...", "confidence": 0.85}
```

### 7.2 Ingestion Endpoints

```http
POST /api/ingest
Content-Type: multipart/form-data

files: [file1.json, file2.pdf]

Response:
{
  "job_id": "uuid",
  "status": "processing"
}
```

```http
GET /api/ingest/{job_id}/stream
Accept: text/event-stream

Response: SSE Stream with progress updates
```

### 7.3 System Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | 헬스 체크 |
| `/api/stats` | GET | 그래프 통계 |
| `/api/schema` | GET | Neo4j 스키마 |
| `/api/cypher` | POST | Raw Cypher 실행 |
| `/api/text2cypher` | POST | NL→Cypher 변환 |

### 7.4 Observability Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/metrics` | GET | Prometheus 메트릭 |
| `/api/observability/alerts` | GET | 활성 알림 |
| `/api/cache/stats` | GET | 캐시 통계 |

---

## 8. Configuration System

### 8.1 Settings Hierarchy

```python
class Settings(BaseSettings):
    """루트 설정"""

    # Application
    app_name: str = "Ontology Reasoning System"
    environment: Literal["development", "staging", "production"]
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]

    # Sub-settings
    neo4j: Neo4jSettings
    llm: LLMSettings
    tog: ToGSettings
    workflow: WorkflowSettings
    api: APISettings
    ingestion: IngestionSettings
    distributed: DistributedSettings
    observability: ObservabilitySettings
    cache: CacheSettings
```

### 8.2 Environment Variables

```bash
# Application
APP_NAME=Ontology Reasoning System
ENVIRONMENT=development
LOG_LEVEL=INFO

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<secret>
NEO4J_MAX_CONNECTION_POOL_SIZE=50

# LLM
LLM_PROVIDER=anthropic
LLM_OPENAI_API_KEY=<secret>
LLM_ANTHROPIC_API_KEY=<secret>
LLM_REASONING_MODEL=claude-sonnet-4-5-20250929
LLM_TEMPERATURE=0.0

# ToG MACER
TOG_MAX_REASONING_DEPTH=5
TOG_EXPLORATION_WIDTH=10
TOG_CONFIDENCE_THRESHOLD=0.7

# API
API_HOST=0.0.0.0
API_PORT=8000
API_MAX_UPLOAD_SIZE_MB=1024
```

---

## 9. Design Patterns

### 9.1 Async/Await Pattern

모든 I/O 작업은 비동기로 처리:

```python
async def process_query(query: str) -> Response:
    # Neo4j 비동기 쿼리
    entities = await neo4j_client.search_entities(query)

    # LLM 비동기 호출
    response = await llm.ainvoke(prompt)

    return response
```

### 9.2 State-Based Orchestration (LangGraph)

```python
# Workflow 정의
workflow = StateGraph(MACERState)

workflow.add_node("constructor", constructor_agent)
workflow.add_node("retriever", retriever_agent)
workflow.add_node("reflector", reflector_agent)
workflow.add_node("responser", responser_agent)

# 조건부 라우팅
workflow.add_conditional_edges(
    "reflector",
    should_continue,
    {
        "continue": "retriever",
        "respond": "responser"
    }
)

graph = workflow.compile()
```

### 9.3 Multi-Provider Failover

```python
class LLMProviderChain:
    def __init__(self):
        self.providers = [
            OpenAIProvider(),
            AnthropicProvider(),
            AzureProvider(),
            LocalProvider()
        ]

    async def invoke(self, prompt: str) -> str:
        for provider in self.providers:
            if provider.is_healthy():
                try:
                    return await provider.invoke(prompt)
                except Exception:
                    provider.mark_unhealthy()
                    continue
        raise AllProvidersFailedError()
```

### 9.4 Multi-Layer Caching

```python
class MultiLayerCache:
    def __init__(self):
        self.l1 = LRUCache(max_size=1000)      # In-memory
        self.l2 = RedisCache()                  # Distributed

    async def get(self, key: str) -> Any:
        # L1 체크
        if value := self.l1.get(key):
            return value

        # L2 체크
        if value := await self.l2.get(key):
            self.l1.set(key, value)  # L1에 승격
            return value

        return None
```

---

## 10. Scalability & Performance

### 10.1 Horizontal Scaling

| Component | Scaling Strategy |
|-----------|-----------------|
| API | Multiple Uvicorn workers |
| SDDI Pipeline | Celery workers |
| Neo4j | Connection pooling |
| Cache | Redis cluster |

### 10.2 Performance Optimizations

| Optimization | Impact |
|--------------|--------|
| Vector indexing | O(log n) semantic search |
| Query result caching | 90%+ cache hit rate |
| Embedding caching | API call 감소 |
| Batch processing | LLM 호출 최적화 |
| Streaming JSON | 대용량 파일 메모리 효율 |
| Connection pooling | DB 연결 재사용 |

### 10.3 Resource Limits

| Resource | Default | Max |
|----------|---------|-----|
| Query timeout | 30s | 120s |
| Upload size | 1GB | Configurable |
| Connection pool | 50 | Configurable |
| L1 cache | 100MB | Configurable |
| Concurrent chunks | 5 | Configurable |

### 10.4 Monitoring Metrics

```
# Prometheus Metrics

# Query latency
ontology_query_duration_seconds{step="constructor|retriever|reflector|responser"}

# LLM provider health
llm_provider_health{provider="openai|anthropic|azure|local"}

# Cache hit rate
cache_hit_rate{layer="l1|l2", type="query|embedding|subgraph"}

# Ingestion throughput
ingestion_documents_total
ingestion_entities_total
ingestion_relations_total
```

---

## Appendix

### A. Quick Start

```bash
# 1. Dependencies 설치
pip install -e ".[dev]"

# 2. 환경 변수 설정
cp .env.example .env
# Edit .env with your settings

# 3. Neo4j 시작
docker-compose up -d neo4j

# 4. API 서버 시작
uvicorn src.api.main:app --reload

# 5. 헬스 체크
curl http://localhost:8000/api/health
```

### B. Development Commands

```bash
# Testing
pytest                          # All tests
pytest tests/unit/ -v           # Unit tests
pytest --cov=src                # Coverage

# Code Quality
ruff check src/ --fix           # Lint
ruff format src/                # Format
mypy src/                       # Type check

# Docker
docker-compose up -d            # Start services
docker-compose logs -f api      # View logs
```

### C. Troubleshooting

| Issue | Solution |
|-------|----------|
| Neo4j connection failed | Check `NEO4J_URI` and credentials |
| LLM timeout | Increase `LLM_TIMEOUT` or check API key |
| Import errors | Run `pip install -e ".[dev]"` |
| Memory issues | Reduce batch size or enable streaming |

---

> **Document maintained by**: Development Team
> **Last review**: 2025-11-28
