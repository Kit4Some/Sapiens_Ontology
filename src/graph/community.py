"""
Community Detection Module.

Implements hierarchical community detection for the knowledge graph
using Neo4j GDS Louvain algorithm or fallback Python implementation.

Communities help provide high-level context during reasoning by:
- Grouping related entities
- Generating community summaries
- Enabling multi-scale graph exploration
"""

from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from src.graph.neo4j_client import OntologyGraphClient, get_ontology_client

logger = structlog.get_logger(__name__)


# Community summary generation prompt
COMMUNITY_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert at summarizing groups of related concepts.

Given a list of entities and their relationships within a community,
generate a concise summary (2-3 sentences) that captures:
1. The main theme or topic of this group
2. Key entities and how they relate
3. The significance of this grouping

Be specific and informative. Avoid generic statements."""
    ),
    (
        "human",
        """Community members:
{members}

Relationships within community:
{relationships}

Generate a summary:"""
    ),
])


class CommunityDetector:
    """
    Detects and manages communities in the knowledge graph.

    Uses Neo4j GDS Louvain algorithm when available, with fallback
    to a simpler connected components approach.
    """

    def __init__(
        self,
        client: OntologyGraphClient | None = None,
        llm: BaseChatModel | None = None,
        min_community_size: int = 3,
        max_levels: int = 3,
    ) -> None:
        """
        Initialize the community detector.

        Args:
            client: Neo4j client instance
            llm: LLM for generating community summaries
            min_community_size: Minimum entities to form a community
            max_levels: Maximum hierarchy levels to detect
        """
        self._client = client or get_ontology_client()
        self._llm = llm
        self._min_size = min_community_size
        self._max_levels = max_levels

    async def detect_communities(
        self,
        use_gds: bool = True,
        generate_summaries: bool = True,
    ) -> dict[str, Any]:
        """
        Detect communities in the knowledge graph.

        Args:
            use_gds: Try to use Neo4j GDS library
            generate_summaries: Generate LLM summaries for communities

        Returns:
            Dict with community detection results
        """
        await self._client.connect()

        # Check if GDS is available
        gds_available = await self._check_gds_available() if use_gds else False

        if gds_available:
            logger.info("Using Neo4j GDS for community detection")
            result = await self._detect_with_gds()
        else:
            logger.info("Using fallback community detection (connected components)")
            result = await self._detect_with_fallback()

        # Generate summaries if requested and LLM is available
        if generate_summaries and self._llm and result.get("communities"):
            await self._generate_summaries(result["communities"])

        return result

    async def _check_gds_available(self) -> bool:
        """Check if Neo4j GDS library is available."""
        try:
            result = await self._client.execute_cypher(
                "RETURN gds.version() as version"
            )
            if result:
                logger.info("GDS available", version=result[0].get("version"))
                return True
        except Exception as e:
            logger.debug("GDS not available", error=str(e))
        return False

    async def _detect_with_gds(self) -> dict[str, Any]:
        """Detect communities using Neo4j GDS Louvain algorithm."""
        try:
            # Create in-memory graph projection
            projection_query = """
            CALL gds.graph.project(
                'community_graph',
                'Entity',
                {
                    RELATES_TO: {
                        orientation: 'UNDIRECTED',
                        properties: ['confidence']
                    }
                }
            )
            YIELD graphName, nodeCount, relationshipCount
            RETURN graphName, nodeCount, relationshipCount
            """

            projection_result = await self._client.execute_cypher(projection_query)
            if not projection_result:
                raise RuntimeError("Failed to create graph projection")

            node_count = projection_result[0].get("nodeCount", 0)
            rel_count = projection_result[0].get("relationshipCount", 0)

            logger.info(
                "Graph projection created",
                nodes=node_count,
                relationships=rel_count,
            )

            # Run Louvain community detection
            louvain_query = """
            CALL gds.louvain.stream('community_graph', {
                maxLevels: $max_levels,
                includeIntermediateCommunities: true
            })
            YIELD nodeId, communityId, intermediateCommunityIds
            RETURN gds.util.asNode(nodeId).id AS entity_id,
                   gds.util.asNode(nodeId).name AS entity_name,
                   communityId,
                   intermediateCommunityIds
            """

            communities_result = await self._client.execute_cypher(
                louvain_query,
                {"max_levels": self._max_levels}
            )

            # Process results into community structure
            communities = self._process_gds_results(communities_result)

            # Store communities in Neo4j
            stored_count = await self._store_communities(communities)

            # Clean up projection
            await self._client.execute_cypher(
                "CALL gds.graph.drop('community_graph', false)"
            )

            return {
                "method": "gds_louvain",
                "communities": communities,
                "community_count": len(communities),
                "stored_count": stored_count,
                "levels": self._max_levels,
            }

        except Exception as e:
            logger.error("GDS community detection failed", error=str(e))
            # Clean up projection if it exists
            try:
                await self._client.execute_cypher(
                    "CALL gds.graph.drop('community_graph', false)"
                )
            except Exception:
                pass
            # Fall back to simpler method
            return await self._detect_with_fallback()

    async def _detect_with_fallback(self) -> dict[str, Any]:
        """
        Detect communities using connected components approach.

        This is a simpler fallback when GDS is not available.
        """
        # Find connected components using path traversal
        components_query = """
        MATCH (e:Entity)
        WHERE NOT EXISTS {
            MATCH (e)<-[:RELATES_TO*]-(prev:Entity)
            WHERE prev.id < e.id
        }
        WITH e
        CALL {
            WITH e
            MATCH path = (e)-[:RELATES_TO*0..5]-(connected:Entity)
            RETURN DISTINCT connected
        }
        WITH e as root, collect(DISTINCT connected) as members
        WHERE size(members) >= $min_size
        RETURN root.id as root_id,
               [m in members | {id: m.id, name: m.name, type: m.type}] as members,
               size(members) as member_count
        ORDER BY member_count DESC
        LIMIT 50
        """

        result = await self._client.execute_cypher(
            components_query,
            {"min_size": self._min_size}
        )

        # Process into community structure
        communities = []
        for idx, row in enumerate(result or []):
            community = {
                "id": f"community_{idx}",
                "level": 0,
                "members": row.get("members", []),
                "member_count": row.get("member_count", 0),
                "root_id": row.get("root_id"),
                "summary": "",
            }
            communities.append(community)

        # Store communities
        stored_count = await self._store_communities(communities)

        return {
            "method": "connected_components",
            "communities": communities,
            "community_count": len(communities),
            "stored_count": stored_count,
            "levels": 1,
        }

    def _process_gds_results(self, results: list[dict]) -> list[dict[str, Any]]:
        """Process GDS Louvain results into community structure."""
        if not results:
            return []

        # Group entities by community ID at each level
        community_members: dict[int, list[dict]] = {}

        for row in results:
            community_id = row.get("communityId")
            if community_id is None:
                continue

            if community_id not in community_members:
                community_members[community_id] = []

            community_members[community_id].append({
                "id": row.get("entity_id"),
                "name": row.get("entity_name"),
            })

        # Create community objects
        communities = []
        for comm_id, members in community_members.items():
            if len(members) >= self._min_size:
                communities.append({
                    "id": f"community_{comm_id}",
                    "level": 0,  # Base level
                    "members": members,
                    "member_count": len(members),
                    "summary": "",
                })

        return communities

    async def _store_communities(self, communities: list[dict]) -> int:
        """Store detected communities in Neo4j."""
        if not communities:
            return 0

        stored_count = 0

        for community in communities:
            try:
                # Create Community node
                create_query = """
                MERGE (c:Community {id: $id})
                SET c.level = $level,
                    c.summary = $summary,
                    c.member_count = $member_count
                RETURN c.id as created_id
                """

                await self._client.execute_cypher(create_query, {
                    "id": community["id"],
                    "level": community.get("level", 0),
                    "summary": community.get("summary", ""),
                    "member_count": community.get("member_count", 0),
                })

                # Create BELONGS_TO relationships
                for member in community.get("members", []):
                    link_query = """
                    MATCH (e:Entity {id: $entity_id})
                    MATCH (c:Community {id: $community_id})
                    MERGE (e)-[:BELONGS_TO]->(c)
                    """
                    await self._client.execute_cypher(link_query, {
                        "entity_id": member.get("id"),
                        "community_id": community["id"],
                    })

                stored_count += 1

            except Exception as e:
                logger.warning(
                    "Failed to store community",
                    community_id=community.get("id"),
                    error=str(e),
                )

        logger.info("Communities stored", count=stored_count)
        return stored_count

    async def _generate_summaries(self, communities: list[dict]) -> None:
        """Generate LLM summaries for communities."""
        if not self._llm:
            return

        for community in communities:
            try:
                members = community.get("members", [])
                if len(members) < 2:
                    continue

                # Get member names
                member_names = [m.get("name", m.get("id", "Unknown")) for m in members[:20]]
                members_text = ", ".join(member_names)

                # Get relationships within community
                member_ids = [m.get("id") for m in members if m.get("id")]
                if member_ids:
                    rel_query = """
                    MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                    WHERE a.id IN $ids AND b.id IN $ids
                    RETURN a.name + ' -> ' + COALESCE(r.predicate, 'relates to') + ' -> ' + b.name as rel
                    LIMIT 10
                    """
                    rel_result = await self._client.execute_cypher(rel_query, {"ids": member_ids})
                    relationships_text = "\n".join([r.get("rel", "") for r in (rel_result or [])])
                else:
                    relationships_text = "No explicit relationships found"

                # Generate summary
                chain = COMMUNITY_SUMMARY_PROMPT | self._llm
                response = await chain.ainvoke({
                    "members": members_text,
                    "relationships": relationships_text,
                })

                summary = response.content if hasattr(response, "content") else str(response)

                # Update community with summary
                community["summary"] = summary

                # Store summary in Neo4j
                await self._client.execute_cypher(
                    "MATCH (c:Community {id: $id}) SET c.summary = $summary",
                    {"id": community["id"], "summary": summary}
                )

                logger.debug("Community summary generated", community_id=community["id"])

            except Exception as e:
                logger.warning(
                    "Failed to generate community summary",
                    community_id=community.get("id"),
                    error=str(e),
                )

    async def get_community_context(
        self,
        entity_ids: list[str],
        include_summaries: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get community context for given entities.

        Args:
            entity_ids: List of entity IDs
            include_summaries: Whether to include community summaries

        Returns:
            List of relevant communities with their summaries
        """
        await self._client.connect()

        query = """
        MATCH (e:Entity)-[:BELONGS_TO]->(c:Community)
        WHERE e.id IN $entity_ids
        RETURN DISTINCT c.id as id,
                        c.summary as summary,
                        c.level as level,
                        c.member_count as member_count,
                        collect(e.name) as matching_entities
        ORDER BY c.level, c.member_count DESC
        """

        results = await self._client.execute_cypher(query, {"entity_ids": entity_ids})

        communities = []
        for row in (results or []):
            comm = {
                "id": row.get("id"),
                "level": row.get("level", 0),
                "member_count": row.get("member_count", 0),
                "matching_entities": row.get("matching_entities", []),
            }
            if include_summaries:
                comm["summary"] = row.get("summary", "")
            communities.append(comm)

        return communities

    async def rebuild_communities(
        self,
        clear_existing: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Rebuild all communities from scratch.

        Args:
            clear_existing: Delete existing communities first
            **kwargs: Additional arguments for detect_communities

        Returns:
            Community detection results
        """
        await self._client.connect()

        if clear_existing:
            # Delete existing community relationships
            await self._client.execute_cypher(
                "MATCH ()-[r:BELONGS_TO]->() DELETE r"
            )
            # Delete existing community nodes
            await self._client.execute_cypher(
                "MATCH (c:Community) DELETE c"
            )
            logger.info("Existing communities cleared")

        return await self.detect_communities(**kwargs)
