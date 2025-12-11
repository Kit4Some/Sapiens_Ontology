"""
Configurable Entity and Relation Extractors.

Schema-driven extractors that use dynamic configuration:
- No hardcoded entity types or predicates
- Domain profile support
- Dynamic prompt generation
- Pluggable schema customization
"""

import hashlib
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.sddi.state import ExtractedEntity, ExtractedRelation, TextChunk, Triplet
from src.sddi.extractors.schema import (
    DomainProfileManager,
    ExtractionSchema,
    DynamicPromptBuilder,
    get_profile_manager,
)

logger = structlog.get_logger(__name__)


class EntityExtractionOutput(BaseModel):
    """Schema for LLM entity extraction output."""
    entities: list[dict[str, Any]] = Field(description="List of extracted entities")


class RelationExtractionOutput(BaseModel):
    """Schema for LLM relation extraction output."""
    relations: list[dict[str, Any]] = Field(description="List of extracted relations")


class ConfigurableEntityExtractor:
    """
    Schema-driven entity extractor.

    Uses dynamic schema configuration instead of hardcoded entity types.
    Supports domain profiles for specialized extraction.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        schema: ExtractionSchema | None = None,
        profile_name: str | None = None,
        min_confidence: float | None = None,
        language: str | None = None,
    ) -> None:
        """
        Initialize configurable entity extractor.

        Args:
            llm: LangChain chat model
            schema: Pre-built extraction schema
            profile_name: Domain profile name (e.g., "technology", "medical")
            min_confidence: Override minimum confidence threshold
            language: Override language setting
        """
        self._llm = llm
        self._parser = JsonOutputParser(pydantic_object=EntityExtractionOutput)

        # Load schema
        if schema:
            self._schema = schema
        else:
            manager = get_profile_manager()
            self._schema = manager.create_schema(profile_name)

        # Build prompts
        self._prompt_builder = DynamicPromptBuilder(self._schema)
        self._language = language or self._schema.primary_language

        # Create extraction chain
        prompt_template = self._prompt_builder.build_entity_extraction_prompt(
            include_examples=True,
            language=self._language,
        )
        self._chain = prompt_template.template | self._llm | self._parser

        # Configuration
        self._min_confidence = min_confidence or self._schema.min_entity_confidence

        logger.info(
            "Configurable entity extractor initialized",
            entity_types=len(self._schema.entity_registry.get_all()),
            min_confidence=self._min_confidence,
            language=self._language,
        )

    @property
    def schema(self) -> ExtractionSchema:
        return self._schema

    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """Generate a deterministic entity ID."""
        key = f"{entity_type}:{name.lower().strip()}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _resolve_entity_type(self, type_str: str) -> str:
        """Resolve entity type using schema registry."""
        resolved = self._schema.resolve_entity_type(type_str)

        # Fallback to OTHER if not found
        if not self._schema.entity_registry.contains(resolved):
            return "OTHER"

        return resolved

    async def extract(
        self,
        chunk: TextChunk,
    ) -> list[ExtractedEntity]:
        """
        Extract entities from a single text chunk.

        Args:
            chunk: Text chunk to process

        Returns:
            List of extracted entities
        """
        try:
            logger.debug(
                "Starting entity extraction",
                chunk_id=chunk.id,
                text_length=len(chunk.text),
            )
            result = await self._chain.ainvoke({"text": chunk.text})
            entities_data = result.get("entities", [])

        except Exception as e:
            logger.error(
                "Entity extraction failed",
                chunk_id=chunk.id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return []

        entities = []
        for entity_data in entities_data:
            try:
                # Resolve type using schema
                raw_type = entity_data.get("type", "OTHER")
                entity_type = self._resolve_entity_type(raw_type)

                confidence = float(entity_data.get("confidence", 0.8))
                if confidence < self._min_confidence:
                    continue

                name = entity_data.get("name", "").strip()
                if not name:
                    continue

                entity = ExtractedEntity(
                    id=self._generate_entity_id(name, entity_type),
                    name=name,
                    type=entity_type,
                    description=entity_data.get("description", ""),
                    aliases=entity_data.get("aliases", []),
                    chunk_ids=[chunk.id],
                    confidence=confidence,
                    properties=entity_data.get("properties", {}),
                )
                entities.append(entity)

            except Exception as e:
                logger.warning("Failed to parse entity", error=str(e))
                continue

        logger.info(
            "Entities extracted",
            chunk_id=chunk.id,
            entity_count=len(entities),
        )
        return entities

    async def extract_batch(
        self,
        chunks: list[TextChunk],
        deduplicate: bool = True,
        max_concurrent: int = 3,
    ) -> list[ExtractedEntity]:
        """
        Extract entities from multiple chunks with concurrency.

        Args:
            chunks: List of text chunks
            deduplicate: Whether to merge duplicate entities
            max_concurrent: Max concurrent LLM calls

        Returns:
            List of extracted entities
        """
        import asyncio

        all_entities: list[ExtractedEntity] = []
        total_chunks = len(chunks)

        for i in range(0, total_chunks, max_concurrent):
            batch_chunks = chunks[i:i + max_concurrent]
            batch_num = (i // max_concurrent) + 1
            total_batches = (total_chunks + max_concurrent - 1) // max_concurrent

            logger.info(
                "Processing entity batch",
                batch=f"{batch_num}/{total_batches}",
                chunks=len(batch_chunks),
            )

            tasks = [self.extract(chunk) for chunk in batch_chunks]

            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=120.0,
                )

                for result in results:
                    if isinstance(result, Exception):
                        logger.warning("Chunk extraction failed", error=str(result))
                    elif result:
                        all_entities.extend(result)

            except asyncio.TimeoutError:
                logger.error("Entity batch timed out", batch=batch_num)

        if deduplicate:
            all_entities = self._deduplicate_entities(all_entities)

        logger.info(
            "Batch entity extraction completed",
            chunks=total_chunks,
            entities=len(all_entities),
        )
        return all_entities

    def _deduplicate_entities(
        self,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedEntity]:
        """Merge duplicate entities by ID."""
        entity_map: dict[str, ExtractedEntity] = {}

        for entity in entities:
            if entity.id in entity_map:
                existing = entity_map[entity.id]
                entity_map[entity.id] = ExtractedEntity(
                    id=entity.id,
                    name=existing.name,
                    type=existing.type,
                    description=existing.description or entity.description,
                    aliases=list(set(existing.aliases + entity.aliases)),
                    chunk_ids=list(set(existing.chunk_ids + entity.chunk_ids)),
                    confidence=max(existing.confidence, entity.confidence),
                    properties={**existing.properties, **entity.properties},
                )
            else:
                entity_map[entity.id] = entity

        return list(entity_map.values())


class ConfigurableRelationExtractor:
    """
    Schema-driven relation extractor.

    Uses dynamic predicate configuration instead of hardcoded predicates.
    Supports type constraints and predicate validation.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        schema: ExtractionSchema | None = None,
        profile_name: str | None = None,
        min_confidence: float | None = None,
        predicate_categories: list[str] | None = None,
        validate_types: bool = True,
    ) -> None:
        """
        Initialize configurable relation extractor.

        Args:
            llm: LangChain chat model
            schema: Pre-built extraction schema
            profile_name: Domain profile name
            min_confidence: Override minimum confidence
            predicate_categories: Limit to specific predicate categories
            validate_types: Validate source/target types against predicate constraints
        """
        self._llm = llm
        self._parser = JsonOutputParser(pydantic_object=RelationExtractionOutput)

        # Load schema
        if schema:
            self._schema = schema
        else:
            manager = get_profile_manager()
            self._schema = manager.create_schema(profile_name)

        # Build prompts
        self._prompt_builder = DynamicPromptBuilder(self._schema)
        self._predicate_categories = predicate_categories

        # Create extraction chain
        prompt_template = self._prompt_builder.build_relation_extraction_prompt(
            include_examples=True,
            categories=predicate_categories,
        )
        self._chain = prompt_template.template | self._llm | self._parser

        # Configuration
        self._min_confidence = min_confidence or self._schema.min_relation_confidence
        self._validate_types = validate_types

        logger.info(
            "Configurable relation extractor initialized",
            predicates=len(self._schema.predicate_registry.get_all()),
            min_confidence=self._min_confidence,
            categories=predicate_categories,
        )

    @property
    def schema(self) -> ExtractionSchema:
        return self._schema

    def _generate_relation_id(
        self,
        source: str,
        target: str,
        predicate: str,
    ) -> str:
        """Generate a deterministic relation ID."""
        key = f"{source.lower()}:{predicate}:{target.lower()}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _normalize_predicate(self, predicate: str) -> str:
        """Normalize and resolve predicate name."""
        # Basic normalization
        normalized = predicate.strip().upper().replace(" ", "_").replace("-", "_")
        while "__" in normalized:
            normalized = normalized.replace("__", "_")

        # Resolve using schema
        return self._schema.resolve_predicate(normalized)

    def _format_entities_for_prompt(
        self,
        entities: list[ExtractedEntity],
    ) -> str:
        """Format entity list for prompt injection."""
        lines = []
        for entity in entities:
            aliases_str = f" (aliases: {', '.join(entity.aliases)})" if entity.aliases else ""
            lines.append(f"- {entity.name} [{entity.type}]{aliases_str}")
        return "\n".join(lines)

    def _find_entity_by_name(
        self,
        name: str,
        entities: list[ExtractedEntity],
    ) -> ExtractedEntity | None:
        """Find entity by name or alias."""
        name_lower = name.lower().strip()
        for entity in entities:
            if entity.name.lower() == name_lower:
                return entity
            if any(alias.lower() == name_lower for alias in entity.aliases):
                return entity
        return None

    async def extract(
        self,
        chunk: TextChunk,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedRelation]:
        """
        Extract relations from a text chunk given known entities.

        Args:
            chunk: Text chunk to process
            entities: List of entities found in this chunk

        Returns:
            List of extracted relations
        """
        if len(entities) < 2:
            logger.debug("Not enough entities", chunk_id=chunk.id)
            return []

        # Filter entities relevant to this chunk
        chunk_entities = [e for e in entities if chunk.id in e.chunk_ids]
        if len(chunk_entities) < 2:
            chunk_entities = entities[:20]  # Limit to prevent token overflow

        entities_str = self._format_entities_for_prompt(chunk_entities)

        try:
            result = await self._chain.ainvoke({
                "text": chunk.text,
                "entities": entities_str,
            })
            relations_data = result.get("relations", [])

        except Exception as e:
            logger.error(
                "Relation extraction failed",
                chunk_id=chunk.id,
                error=str(e),
            )
            return []

        relations = []
        for rel_data in relations_data:
            try:
                source_name = rel_data.get("source", "").strip()
                target_name = rel_data.get("target", "").strip()
                predicate = self._normalize_predicate(rel_data.get("predicate", "RELATED_TO"))
                confidence = float(rel_data.get("confidence", 0.7))

                # Find entities
                source_entity = self._find_entity_by_name(source_name, chunk_entities)
                target_entity = self._find_entity_by_name(target_name, chunk_entities)

                if not source_entity or not target_entity:
                    logger.debug(
                        "Skipping relation with unknown entity",
                        source=source_name,
                        target=target_name,
                    )
                    continue

                # Validate types if enabled
                if self._validate_types:
                    is_valid, error_msg = self._schema.predicate_registry.validate_relation(
                        predicate,
                        source_entity.type,
                        target_entity.type,
                    )
                    if not is_valid:
                        logger.debug(
                            "Relation failed type validation",
                            predicate=predicate,
                            error=error_msg,
                        )
                        continue

                # Filter by confidence
                if confidence < self._min_confidence:
                    continue

                relation = ExtractedRelation(
                    id=self._generate_relation_id(
                        source_entity.id, target_entity.id, predicate
                    ),
                    source_entity=source_entity.id,
                    target_entity=target_entity.id,
                    predicate=predicate,
                    description=rel_data.get("description", ""),
                    chunk_ids=[chunk.id],
                    confidence=confidence,
                    properties={
                        "source_name": source_entity.name,
                        "target_name": target_entity.name,
                        "source_type": source_entity.type,
                        "target_type": target_entity.type,
                    },
                )
                relations.append(relation)

            except Exception as e:
                logger.warning("Failed to parse relation", error=str(e))
                continue

        logger.info(
            "Relations extracted",
            chunk_id=chunk.id,
            relation_count=len(relations),
        )
        return relations

    async def extract_batch(
        self,
        chunks: list[TextChunk],
        entities: list[ExtractedEntity],
        deduplicate: bool = True,
        max_concurrent: int = 3,
    ) -> list[ExtractedRelation]:
        """
        Extract relations from multiple chunks with concurrency.

        Args:
            chunks: List of text chunks
            entities: All entities extracted from chunks
            deduplicate: Whether to merge duplicate relations
            max_concurrent: Max concurrent LLM calls

        Returns:
            List of extracted relations
        """
        import asyncio

        all_relations: list[ExtractedRelation] = []
        total_chunks = len(chunks)

        for i in range(0, total_chunks, max_concurrent):
            batch_chunks = chunks[i:i + max_concurrent]
            batch_num = (i // max_concurrent) + 1
            total_batches = (total_chunks + max_concurrent - 1) // max_concurrent

            logger.info(
                "Processing relation batch",
                batch=f"{batch_num}/{total_batches}",
            )

            tasks = [self.extract(chunk, entities) for chunk in batch_chunks]

            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=120.0,
                )

                for result in results:
                    if isinstance(result, Exception):
                        logger.warning("Relation extraction failed", error=str(result))
                    elif result:
                        all_relations.extend(result)

            except asyncio.TimeoutError:
                logger.error("Relation batch timed out", batch=batch_num)

        if deduplicate:
            all_relations = self._deduplicate_relations(all_relations)

        logger.info(
            "Batch relation extraction completed",
            chunks=total_chunks,
            relations=len(all_relations),
        )
        return all_relations

    def _deduplicate_relations(
        self,
        relations: list[ExtractedRelation],
    ) -> list[ExtractedRelation]:
        """Merge duplicate relations."""
        relation_map: dict[str, ExtractedRelation] = {}

        for relation in relations:
            if relation.id in relation_map:
                existing = relation_map[relation.id]
                relation_map[relation.id] = ExtractedRelation(
                    id=relation.id,
                    source_entity=existing.source_entity,
                    target_entity=existing.target_entity,
                    predicate=existing.predicate,
                    description=existing.description or relation.description,
                    chunk_ids=list(set(existing.chunk_ids + relation.chunk_ids)),
                    confidence=max(existing.confidence, relation.confidence),
                    properties={**existing.properties, **relation.properties},
                )
            else:
                relation_map[relation.id] = relation

        return list(relation_map.values())

    def relations_to_triplets(
        self,
        relations: list[ExtractedRelation],
        entities: list[ExtractedEntity],
    ) -> list[Triplet]:
        """Convert relations to knowledge graph triplets."""
        entity_map = {e.id: e for e in entities}

        triplets = []
        for relation in relations:
            source = entity_map.get(relation.source_entity)
            target = entity_map.get(relation.target_entity)

            if not source or not target:
                continue

            triplets.append(Triplet(
                subject=source.name,
                predicate=relation.predicate,
                object=target.name,
                subject_type=source.type,
                object_type=target.type,
                confidence=relation.confidence,
                source_chunk_id=relation.chunk_ids[0] if relation.chunk_ids else None,
            ))

        return triplets


# =============================================================================
# Factory Functions
# =============================================================================

def create_entity_extractor(
    llm: BaseChatModel,
    profile: str = "default",
    **kwargs,
) -> ConfigurableEntityExtractor:
    """
    Create a configurable entity extractor.

    Args:
        llm: LangChain chat model
        profile: Domain profile name (default, technology, medical, finance, legal, academic)
        **kwargs: Additional configuration

    Returns:
        Configured entity extractor
    """
    return ConfigurableEntityExtractor(
        llm=llm,
        profile_name=profile if profile != "default" else None,
        **kwargs,
    )


def create_relation_extractor(
    llm: BaseChatModel,
    profile: str = "default",
    **kwargs,
) -> ConfigurableRelationExtractor:
    """
    Create a configurable relation extractor.

    Args:
        llm: LangChain chat model
        profile: Domain profile name
        **kwargs: Additional configuration

    Returns:
        Configured relation extractor
    """
    return ConfigurableRelationExtractor(
        llm=llm,
        profile_name=profile if profile != "default" else None,
        **kwargs,
    )


def create_extractors_for_domain(
    llm: BaseChatModel,
    domain: str,
    **kwargs,
) -> tuple[ConfigurableEntityExtractor, ConfigurableRelationExtractor]:
    """
    Create matched entity and relation extractors for a domain.

    Args:
        llm: LangChain chat model
        domain: Domain name (technology, medical, finance, legal, academic)
        **kwargs: Shared configuration

    Returns:
        Tuple of (entity_extractor, relation_extractor)
    """
    manager = get_profile_manager()
    schema = manager.create_schema(domain)

    entity_extractor = ConfigurableEntityExtractor(
        llm=llm,
        schema=schema,
        **kwargs,
    )

    relation_extractor = ConfigurableRelationExtractor(
        llm=llm,
        schema=schema,
        **kwargs,
    )

    return entity_extractor, relation_extractor
