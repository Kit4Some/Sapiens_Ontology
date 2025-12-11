"""
Few-Shot Example Store for Text2Cypher.

Manages example queries and provides semantic similarity-based retrieval.
"""

import json
from pathlib import Path

import structlog
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class CypherExample(BaseModel):
    """A single Text2Cypher example."""

    question: str = Field(..., description="Natural language question")
    cypher: str = Field(..., description="Corresponding Cypher query")
    description: str = Field(default="", description="Optional description")
    category: str = Field(default="general", description="Query category")
    complexity: str = Field(default="simple", description="simple/medium/complex")
    embedding: list[float] | None = Field(default=None, description="Question embedding")


# =============================================================================
# Default Examples for Ontology Schema
# =============================================================================

DEFAULT_EXAMPLES: list[dict[str, str]] = [
    # Entity Lookup
    {
        "question": "Find all entities of type Person",
        "cypher": "MATCH (e:Entity {type: 'PERSON'}) RETURN e.id, e.name, e.description LIMIT 25",
        "category": "entity_lookup",
        "complexity": "simple",
    },
    {
        "question": "What entities are named Microsoft?",
        "cypher": "MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower('Microsoft') RETURN e.id, e.name, e.type, e.description",
        "category": "entity_lookup",
        "complexity": "simple",
    },
    {
        "question": "Find entity by ID abc123",
        "cypher": "MATCH (e:Entity {id: $entity_id}) RETURN e",
        "category": "entity_lookup",
        "complexity": "simple",
    },
    # Relationship Queries
    {
        "question": "What is the relationship between Entity A and Entity B?",
        "cypher": """MATCH (a:Entity)-[r]->(b:Entity)
WHERE toLower(a.name) CONTAINS toLower($entity_a) AND toLower(b.name) CONTAINS toLower($entity_b)
RETURN a.name as source, type(r) as relationship, r.predicate as predicate, b.name as target""",
        "category": "relationship",
        "complexity": "medium",
    },
    {
        "question": "Show all relationships for entity named Google",
        "cypher": """MATCH (e:Entity)-[r]-(other:Entity)
WHERE toLower(e.name) CONTAINS toLower('Google')
RETURN e.name as entity, type(r) as rel_type, r.predicate as predicate, other.name as related_entity
LIMIT 50""",
        "category": "relationship",
        "complexity": "medium",
    },
    {
        "question": "Find entities that work for Microsoft",
        "cypher": """MATCH (person:Entity)-[r:RELATES_TO]->(org:Entity)
WHERE r.predicate IN ['WORKS_FOR', 'EMPLOYED_BY'] AND toLower(org.name) CONTAINS toLower('Microsoft')
RETURN person.name as employee, r.predicate as relation, org.name as organization""",
        "category": "relationship",
        "complexity": "medium",
    },
    # Path Finding
    {
        "question": "Find the shortest path between Entity A and Entity B",
        "cypher": """MATCH (a:Entity), (b:Entity)
WHERE toLower(a.name) CONTAINS toLower($entity_a) AND toLower(b.name) CONTAINS toLower($entity_b)
MATCH path = shortestPath((a)-[*..5]-(b))
RETURN [node in nodes(path) | node.name] as path_nodes, length(path) as path_length""",
        "category": "path",
        "complexity": "complex",
    },
    {
        "question": "How are Elon Musk and OpenAI connected?",
        "cypher": """MATCH (a:Entity), (b:Entity)
WHERE toLower(a.name) CONTAINS toLower('Elon Musk') AND toLower(b.name) CONTAINS toLower('OpenAI')
MATCH path = shortestPath((a)-[*..4]-(b))
RETURN [node in nodes(path) | node.name] as entities,
       [rel in relationships(path) | rel.predicate] as relationships""",
        "category": "path",
        "complexity": "complex",
    },
    # Aggregation
    {
        "question": "Count entities by type",
        "cypher": "MATCH (e:Entity) RETURN e.type as entity_type, count(*) as count ORDER BY count DESC",
        "category": "aggregation",
        "complexity": "simple",
    },
    {
        "question": "What are the most connected entities?",
        "cypher": """MATCH (e:Entity)-[r]-()
RETURN e.name as entity, e.type as type, count(r) as connection_count
ORDER BY connection_count DESC
LIMIT 10""",
        "category": "aggregation",
        "complexity": "medium",
    },
    {
        "question": "How many relationships exist between organizations?",
        "cypher": """MATCH (a:Entity {type: 'ORGANIZATION'})-[r:RELATES_TO]->(b:Entity {type: 'ORGANIZATION'})
RETURN count(r) as relationship_count""",
        "category": "aggregation",
        "complexity": "medium",
    },
    # Chunk/Text Queries
    {
        "question": "Find chunks that mention artificial intelligence",
        "cypher": """MATCH (c:Chunk)
WHERE toLower(c.text) CONTAINS toLower('artificial intelligence')
RETURN c.id, c.text, c.source
LIMIT 10""",
        "category": "text_search",
        "complexity": "simple",
    },
    {
        "question": "What entities are mentioned in document doc123?",
        "cypher": """MATCH (c:Chunk {source: $doc_id})-[:CONTAINS]->(e:Entity)
RETURN DISTINCT e.name as entity, e.type as type
ORDER BY e.name""",
        "category": "text_search",
        "complexity": "medium",
    },
    {
        "question": "Get context chunks for entity named Tesla",
        "cypher": """MATCH (e:Entity)<-[:CONTAINS]-(c:Chunk)
WHERE toLower(e.name) CONTAINS toLower('Tesla')
RETURN c.text as context, c.source as source
LIMIT 5""",
        "category": "text_search",
        "complexity": "medium",
    },
    # Community/Hierarchy
    {
        "question": "What communities exist at level 1?",
        "cypher": "MATCH (c:Community {level: 1}) RETURN c.id, c.summary, c.member_count ORDER BY c.member_count DESC",
        "category": "community",
        "complexity": "simple",
    },
    {
        "question": "Which community does entity X belong to?",
        "cypher": """MATCH (e:Entity)-[:BELONGS_TO]->(c:Community)
WHERE toLower(e.name) CONTAINS toLower($entity_name)
RETURN e.name as entity, c.id as community_id, c.level as level, c.summary as community_summary""",
        "category": "community",
        "complexity": "medium",
    },
    # Neighbor Exploration
    {
        "question": "Find 2-hop neighbors of entity Apple",
        "cypher": """MATCH (start:Entity)-[r1]-(hop1:Entity)-[r2]-(hop2:Entity)
WHERE toLower(start.name) CONTAINS toLower('Apple') AND start <> hop2
RETURN DISTINCT start.name as source, hop1.name as hop1, hop2.name as hop2
LIMIT 50""",
        "category": "exploration",
        "complexity": "complex",
    },
    {
        "question": "What people are connected to organizations in the tech industry?",
        "cypher": """MATCH (p:Entity {type: 'PERSON'})-[r:RELATES_TO]->(o:Entity {type: 'ORGANIZATION'})
WHERE toLower(o.name) CONTAINS 'tech' OR toLower(o.description) CONTAINS 'technology'
RETURN p.name as person, r.predicate as relation, o.name as organization
LIMIT 25""",
        "category": "exploration",
        "complexity": "complex",
    },
    # Property Filtering
    {
        "question": "Find high-confidence entities",
        "cypher": "MATCH (e:Entity) WHERE e.confidence >= 0.9 RETURN e.name, e.type, e.confidence ORDER BY e.confidence DESC LIMIT 20",
        "category": "filtering",
        "complexity": "simple",
    },
    {
        "question": "Find relations with confidence above 0.8",
        "cypher": """MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE r.confidence >= 0.8
RETURN a.name as source, r.predicate as relation, b.name as target, r.confidence
ORDER BY r.confidence DESC
LIMIT 25""",
        "category": "filtering",
        "complexity": "medium",
    },
]


class FewShotExampleStore:
    """
    Few-shot example store with semantic similarity retrieval.

    Stores Text2Cypher examples and retrieves the most relevant ones
    based on question similarity using embeddings.
    """

    def __init__(
        self,
        embeddings: Embeddings | None = None,
        examples: list[dict[str, str]] | None = None,
    ) -> None:
        """
        Initialize the example store.

        Args:
            embeddings: Embedding model for semantic similarity
            examples: Initial examples (uses DEFAULT_EXAMPLES if None)
        """
        self._embeddings = embeddings
        self._examples: list[CypherExample] = []
        self._embeddings_computed = False

        # Load initial examples
        initial_examples = examples or DEFAULT_EXAMPLES
        for ex in initial_examples:
            self.add_example(
                question=ex["question"],
                cypher=ex["cypher"],
                category=ex.get("category", "general"),
                complexity=ex.get("complexity", "simple"),
                description=ex.get("description", ""),
            )

    def add_example(
        self,
        question: str,
        cypher: str,
        category: str = "general",
        complexity: str = "simple",
        description: str = "",
    ) -> None:
        """Add a new example to the store."""
        example = CypherExample(
            question=question,
            cypher=cypher,
            category=category,
            complexity=complexity,
            description=description,
        )
        self._examples.append(example)
        self._embeddings_computed = False  # Reset flag

    async def compute_embeddings(self) -> None:
        """Compute embeddings for all examples."""
        if not self._embeddings:
            logger.warning("No embedding model provided, skipping embedding computation")
            return

        if self._embeddings_computed:
            return

        questions = [ex.question for ex in self._examples]
        embeddings = await self._embeddings.aembed_documents(questions)

        for example, embedding in zip(self._examples, embeddings, strict=True):
            example.embedding = embedding

        self._embeddings_computed = True
        logger.info("Computed embeddings for examples", count=len(self._examples))

    async def get_similar_examples(
        self,
        question: str,
        k: int = 3,
        category: str | None = None,
    ) -> list[CypherExample]:
        """
        Retrieve k most similar examples to the given question.

        Args:
            question: Query question
            k: Number of examples to return
            category: Optional category filter

        Returns:
            List of most similar examples
        """
        if not self._embeddings:
            # Fall back to random selection if no embeddings
            filtered = self._examples
            if category:
                filtered = [ex for ex in filtered if ex.category == category]
            return filtered[:k]

        await self.compute_embeddings()

        # Embed the query
        query_embedding = await self._embeddings.aembed_query(question)

        # Calculate similarities
        scored_examples = []
        for example in self._examples:
            if category and example.category != category:
                continue

            if example.embedding:
                similarity = self._cosine_similarity(query_embedding, example.embedding)
                scored_examples.append((similarity, example))

        # Sort by similarity (descending)
        scored_examples.sort(key=lambda x: x[0], reverse=True)

        return [ex for _, ex in scored_examples[:k]]

    def get_examples_by_category(
        self,
        category: str,
        limit: int = 5,
    ) -> list[CypherExample]:
        """Get examples filtered by category."""
        filtered = [ex for ex in self._examples if ex.category == category]
        return filtered[:limit]

    def get_all_examples(self) -> list[CypherExample]:
        """Get all stored examples."""
        return self._examples.copy()

    def format_examples_for_prompt(
        self,
        examples: list[CypherExample],
    ) -> str:
        """Format examples for prompt injection."""
        lines = []
        for i, ex in enumerate(examples, 1):
            lines.append(f"Example {i}:")
            lines.append(f"Question: {ex.question}")
            lines.append(f"Cypher: {ex.cypher}")
            lines.append("")
        return "\n".join(lines)

    def to_dict_list(
        self,
        examples: list[CypherExample] | None = None,
    ) -> list[dict[str, str]]:
        """Convert examples to list of dicts for LangChain few-shot prompts."""
        examples = examples or self._examples
        return [{"question": ex.question, "cypher": ex.cypher} for ex in examples]

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product: float = sum(a * b for a, b in zip(vec1, vec2, strict=True))
        norm1: float = sum(a * a for a in vec1) ** 0.5
        norm2: float = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    def save_to_file(self, path: str | Path) -> None:
        """Save examples to a JSON file."""
        path = Path(path)
        data = [ex.model_dump(exclude={"embedding"}) for ex in self._examples]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Examples saved", path=str(path), count=len(data))

    @classmethod
    def load_from_file(
        cls,
        path: str | Path,
        embeddings: Embeddings | None = None,
    ) -> "FewShotExampleStore":
        """Load examples from a JSON file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        store = cls(embeddings=embeddings, examples=[])
        for ex in data:
            store.add_example(
                question=ex["question"],
                cypher=ex["cypher"],
                category=ex.get("category", "general"),
                complexity=ex.get("complexity", "simple"),
                description=ex.get("description", ""),
            )

        logger.info("Examples loaded", path=str(path), count=len(data))
        return store


# =============================================================================
# Example Categories
# =============================================================================

EXAMPLE_CATEGORIES = {
    "entity_lookup": "Finding specific entities by name, type, or ID",
    "relationship": "Querying relationships between entities",
    "path": "Finding paths and connections between entities",
    "aggregation": "Counting, grouping, and statistical queries",
    "text_search": "Searching text content in chunks",
    "community": "Community and hierarchy queries",
    "exploration": "Multi-hop exploration and neighbor queries",
    "filtering": "Filtering by properties like confidence",
}
