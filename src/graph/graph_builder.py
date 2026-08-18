from typing import Dict, List, Optional, Tuple

import networkx as nx
from langchain_core.documents import Document

from src.graph.entities import (
    EntityType,
    MedicalEntity,
    create_entity,
)
from src.graph.relations import (
    MedicalRelation,
    RelationType,
    create_relation,
)


class MedicalKnowledgeGraph:
    """
    In-memory medical knowledge graph for SickleGuide.

    Nodes:
        MedicalEntity

    Edges:
        MedicalRelation

    Every node and edge keeps provenance through chunk IDs
    and citations.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()

        self.entities: Dict[
            str,
            MedicalEntity,
        ] = {}

        self.relations: Dict[
            str,
            MedicalRelation,
        ] = {}

    # ============================================================
    # Entity operations
    # ============================================================

    def add_entity(
        self,
        entity: MedicalEntity,
    ) -> MedicalEntity:
        """
        Add or merge an entity.
        """

        if not isinstance(
            entity,
            MedicalEntity,
        ):
            raise TypeError(
                "entity must be a MedicalEntity"
            )

        entity_id = entity.entity_id

        if entity_id in self.entities:

            self.entities[
                entity_id
            ].merge(entity)

        else:

            self.entities[
                entity_id
            ] = entity

            self.graph.add_node(
                entity_id,
                name=entity.name,
                entity_type=(
                    entity.entity_type.value
                ),
                aliases=list(
                    entity.aliases
                ),
                description=entity.description,
            )

        return self.entities[
            entity_id
        ]

    # ============================================================
    # Relation operations
    # ============================================================

    def add_relation(
        self,
        relation: MedicalRelation,
    ) -> MedicalRelation:
        """
        Add or merge a relation.

        Source and target entities must exist.
        """

        if not isinstance(
            relation,
            MedicalRelation,
        ):
            raise TypeError(
                "relation must be a MedicalRelation"
            )

        if (
            relation.source_entity_id
            not in self.entities
        ):
            raise ValueError(
                "Source entity does not exist: "
                f"{relation.source_entity_id}"
            )

        if (
            relation.target_entity_id
            not in self.entities
        ):
            raise ValueError(
                "Target entity does not exist: "
                f"{relation.target_entity_id}"
            )

        relation_id = (
            relation.relation_id
        )

        if relation_id in self.relations:

            self.relations[
                relation_id
            ].merge(relation)

        else:

            self.relations[
                relation_id
            ] = relation

            self.graph.add_edge(
                relation.source_entity_id,
                relation.target_entity_id,
                key=relation.relation_type.value,
                relation_id=relation.relation_id,
                relation_type=(
                    relation.relation_type.value
                ),
                confidence=relation.confidence,
                citations=list(
                    relation.citations
                ),
                source_chunks=list(
                    relation.source_chunks
                ),
                evidence_text=(
                    relation.evidence_text
                ),
            )

        return self.relations[
            relation_id
        ]

    # ============================================================
    # Construction helpers
    # ============================================================

    def add_relation_from_entities(
        self,
        source_entity: MedicalEntity,
        target_entity: MedicalEntity,
        relation_type: RelationType,
        chunk_id: Optional[str] = None,
        citation: Optional[str] = None,
        evidence_text: str = "",
        confidence: float = 1.0,
        properties: Optional[
            Dict
        ] = None,
    ) -> MedicalRelation:
        """
        Convenience method to add an entity pair + relation.
        """

        self.add_entity(
            source_entity
        )

        self.add_entity(
            target_entity
        )

        relation = create_relation(
            source_entity_id=(
                source_entity.entity_id
            ),
            target_entity_id=(
                target_entity.entity_id
            ),
            relation_type=relation_type,
            chunk_id=chunk_id,
            citation=citation,
            evidence_text=evidence_text,
            confidence=confidence,
            properties=properties,
        )

        self.add_relation(
            relation
        )

        return relation

    # ============================================================
    # Retrieval helpers
    # ============================================================

    def get_entity(
        self,
        entity_id: str,
    ) -> Optional[MedicalEntity]:
        """
        Retrieve an entity by ID.
        """

        return self.entities.get(
            entity_id
        )

    def get_relation(
        self,
        relation_id: str,
    ) -> Optional[MedicalRelation]:
        """
        Retrieve a relation by ID.
        """

        return self.relations.get(
            relation_id
        )

    def neighbors(
        self,
        entity_id: str,
    ) -> List[MedicalEntity]:
        """
        Return neighboring entities.
        """

        if entity_id not in self.graph:
            return []

        neighbor_ids = set(
            self.graph.successors(
                entity_id
            )
        )

        neighbor_ids.update(
            self.graph.predecessors(
                entity_id
            )
        )

        return [
            self.entities[node_id]
            for node_id in neighbor_ids
            if node_id in self.entities
        ]

    def relations_for_entity(
        self,
        entity_id: str,
    ) -> List[MedicalRelation]:
        """
        Return all incoming and outgoing relations
        for an entity.
        """

        if entity_id not in self.graph:
            return []

        results = []

        for relation in self.relations.values():

            if (
                relation.source_entity_id
                == entity_id
                or relation.target_entity_id
                == entity_id
            ):
                results.append(
                    relation
                )

        return results

    # ============================================================
    # Graph statistics
    # ============================================================

    def entity_count(self) -> int:
        return len(
            self.entities
        )

    def relation_count(self) -> int:
        return len(
            self.relations
        )

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    # ============================================================
    # Serialization
    # ============================================================

    def to_dict(self) -> Dict:
        """
        Convert the graph into a JSON-serializable dictionary.
        """

        return {
            "entities": [
                entity.to_dict()
                for entity in self.entities.values()
            ],
            "relations": [
                relation.to_dict()
                for relation in self.relations.values()
            ],
        }


# ============================================================
# Graph construction
# ============================================================

def build_graph_from_facts(
    facts: List[Dict],
) -> MedicalKnowledgeGraph:
    """
    Build a graph from structured facts.

    Expected fact format:

    {
        "source": {
            "name": "Hydroxyurea",
            "type": "drug"
        },
        "relation": "treats",
        "target": {
            "name": "Sickle Cell Disease",
            "type": "disease"
        },
        "chunk_id": "221",
        "citation": "...",
        "evidence_text": "...",
        "confidence": 0.95
    }
    """

    graph = MedicalKnowledgeGraph()

    for fact in facts:

        source = fact.get(
            "source"
        )

        target = fact.get(
            "target"
        )

        relation = fact.get(
            "relation"
        )

        if not source or not target:
            continue

        if not relation:
            continue

        source_entity = create_entity(
            name=source["name"],
            entity_type=EntityType(
                source["type"]
            ),
            chunk_id=fact.get(
                "chunk_id"
            ),
            citation=fact.get(
                "citation"
            ),
            confidence=fact.get(
                "confidence",
                1.0,
            ),
        )

        target_entity = create_entity(
            name=target["name"],
            entity_type=EntityType(
                target["type"]
            ),
            chunk_id=fact.get(
                "chunk_id"
            ),
            citation=fact.get(
                "citation"
            ),
            confidence=fact.get(
                "confidence",
                1.0,
            ),
        )

        try:
            relation_type = (
                RelationType(
                    relation
                )
            )
        except ValueError:
            relation_type = (
                RelationType.OTHER
            )

        graph.add_relation_from_entities(
            source_entity=source_entity,
            target_entity=target_entity,
            relation_type=relation_type,
            chunk_id=fact.get(
                "chunk_id"
            ),
            citation=fact.get(
                "citation"
            ),
            evidence_text=fact.get(
                "evidence_text",
                "",
            ),
            confidence=fact.get(
                "confidence",
                1.0,
            ),
        )

    return graph


def build_graph_from_documents(
    documents: List[Document],
) -> MedicalKnowledgeGraph:
    """
    Placeholder construction layer.

    The actual LLM-based entity/relation extraction will be
    implemented separately through the graph pipeline.

    For now, this function creates a graph containing the
    document/chunk provenance as graph context without
    hallucinating medical relationships.
    """

    graph = MedicalKnowledgeGraph()

    for document in documents:

        metadata = document.metadata

        source = metadata.get(
            "source",
            "unknown",
        )

        page = metadata.get(
            "page_number",
            0,
        )

        chunk_id = metadata.get(
            "chunk_id",
            "unknown",
        )

        # Document-level provenance node only.
        entity = create_entity(
            name=(
                f"{source} "
                f"Page {page}"
            ),
            entity_type=EntityType.GUIDELINE,
            chunk_id=str(
                chunk_id
            ),
            citation=metadata.get(
                "citation"
            ),
            confidence=1.0,
            properties={
                "source": source,
                "page_number": page,
            },
        )

        graph.add_entity(
            entity
        )

    return graph