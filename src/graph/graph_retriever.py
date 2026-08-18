from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import json
import re

from langchain_core.documents import Document

from src.graph.entities import (
    EntityType,
    MedicalEntity,
)

from src.graph.relations import (
    MedicalRelation,
    RelationType,
)

from src.graph.graph_builder import (
    MedicalKnowledgeGraph,
)


# ============================================================
# Retrieval configuration
# ============================================================

DEFAULT_MAX_SEED_ENTITIES = 3
DEFAULT_MAX_HOPS = 2
DEFAULT_MAX_ENTITIES = 20
DEFAULT_MAX_DOCUMENTS = 10

MIN_ENTITY_NAME_LENGTH = 4
MIN_EVIDENCE_LENGTH = 12


# ============================================================
# Generic / noisy entity names
# ============================================================

GENERIC_ENTITY_NAMES = {
    "treatment",
    "treatments",
    "management",
    "disease",
    "condition",
    "patient",
    "patients",
    "guideline",
    "recommendation",
    "recommendations",
    "therapy",
    "therapies",
    "outcome",
    "outcomes",
    "sign",
    "symptom",
    "procedure",
    "test",
    "other",
}


STOPWORDS = {
    "what",
    "which",
    "when",
    "where",
    "who",
    "how",
    "are",
    "the",
    "for",
    "with",
    "from",
    "does",
    "this",
    "that",
    "these",
    "those",
    "and",
    "or",
    "can",
    "may",
    "should",
    "used",
    "use",
    "about",
    "into",
    "over",
    "under",
    "patient",
    "patients",
}


# ============================================================
# Query intent
# ============================================================

TREATMENT_TERMS = {
    "treat",
    "treats",
    "treatment",
    "treatments",
    "therapy",
    "therapies",
    "management",
    "manage",
    "managed",
    "medication",
    "medications",
    "drug",
    "drugs",
    "recommend",
    "recommended",
    "recommendation",
    "recommendations",
}


DIAGNOSIS_TERMS = {
    "diagnose",
    "diagnosis",
    "diagnostic",
    "screening",
    "screen",
    "test",
    "testing",
    "laboratory",
    "lab",
    "imaging",
    "mri",
}


COMPLICATION_TERMS = {
    "complication",
    "complications",
    "adverse",
    "side",
    "risk",
    "risks",
    "toxicity",
}


MONITORING_TERMS = {
    "monitor",
    "monitoring",
    "follow",
    "follow-up",
    "surveillance",
    "check",
    "checks",
}


# ============================================================
# Text helpers
# ============================================================

def _normalize(text: str) -> str:
    text = (
        str(text)
        .lower()
        .strip()
    )

    text = re.sub(
        r"[^\w\s-]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _meaningful_tokens(text: str) -> List[str]:
    normalized = _normalize(text)

    return [
        token
        for token in normalized.split()
        if (
            len(token) >= 3
            and token not in STOPWORDS
        )
    ]


def detect_query_intent(query: str) -> str:
    """
    Detect coarse clinical intent.
    """

    tokens = {
        token
        for token in _normalize(query).split()
        if token not in STOPWORDS
    }

    if tokens & TREATMENT_TERMS:
        return "treatment"

    if tokens & DIAGNOSIS_TERMS:
        return "diagnosis"

    if tokens & COMPLICATION_TERMS:
        return "complication"

    if tokens & MONITORING_TERMS:
        return "monitoring"

    return "general"


# ============================================================
# Chunk quality / bibliography detection
# ============================================================

def _reference_features(text: str) -> Dict[str, float]:
    """
    Estimate whether a chunk is dominated by bibliography/reference
    content.

    Returns interpretable signals instead of a single boolean.
    """

    if not text or not text.strip():
        return {
            "reference_markers": 0.0,
            "doi_count": 0.0,
            "url_count": 0.0,
            "year_count": 0.0,
            "numbered_refs": 0.0,
            "author_year": 0.0,
            "citation_density": 0.0,
        }

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    normalized = _normalize(text)

    reference_markers = 0

    # Strong section indicators.
    strong_reference_patterns = [
        r"^\s*references\s*$",
        r"^\s*bibliography\s*$",
        r"^\s*references and notes\s*$",
        r"^\s*literature cited\s*$",
    ]

    for line in lines[:12]:
        for pattern in strong_reference_patterns:
            if re.search(
                pattern,
                line,
                flags=re.IGNORECASE,
            ):
                reference_markers += 5

    # DOI / URL / years.
    doi_count = len(
        re.findall(
            r"\bdoi\b|10\.\d{4,9}/\S+",
            normalized,
        )
    )

    url_count = len(
        re.findall(
            r"https?://|www\.",
            normalized,
        )
    )

    year_count = len(
        re.findall(
            r"\b(?:19|20)\d{2}\b",
            normalized,
        )
    )

    # Numbered bibliography entries:
    # "132. Author..."
    numbered_refs = len(
        re.findall(
            r"(?m)^\s*\d{1,4}\.\s+[A-Z][A-Za-z-]+",
            text,
        )
    )

    # Another common PDF-reference pattern:
    # "132. Foo..." can also be embedded inline.
    inline_numbered_refs = len(
        re.findall(
            r"\b\d{1,4}\.\s+[A-Z][A-Za-z-]+\s+[A-Z]",
            text,
        )
    )

    numbered_refs = max(
        numbered_refs,
        inline_numbered_refs,
    )

    # Author + year pattern.
    author_year = len(
        re.findall(
            r"\b[A-Z][A-Za-z-]+,\s*[A-Z][A-Za-z.-]*\s+.*?\b(?:19|20)\d{2}\b",
            text,
        )
    )

    # Citation density:
    # References tend to contain many years/DOIs/URLs relative to text size.
    text_len = max(
        len(normalized),
        1,
    )

    citation_density = (
        (
            doi_count * 80
            + url_count * 60
            + year_count * 8
            + numbered_refs * 20
            + author_year * 20
        )
        / text_len
    )

    return {
        "reference_markers": float(
            reference_markers
        ),
        "doi_count": float(
            doi_count
        ),
        "url_count": float(
            url_count
        ),
        "year_count": float(
            year_count
        ),
        "numbered_refs": float(
            numbered_refs
        ),
        "author_year": float(
            author_year
        ),
        "citation_density": float(
            citation_density
        ),
    }


def _is_reference_or_bibliography(text: str) -> bool:
    """
    Strong bibliography detection.

    The filter is deliberately conservative for medical content.
    """

    features = _reference_features(text)

    if features["reference_markers"] >= 5:
        return True

    # Very strong reference pattern.
    if (
        features["numbered_refs"] >= 5
        and features["year_count"] >= 5
    ):
        return True

    if (
        features["author_year"] >= 4
        and features["year_count"] >= 6
    ):
        return True

    if (
        features["doi_count"] >= 2
        and features["year_count"] >= 5
    ):
        return True

    if (
        features["url_count"] >= 3
        and features["year_count"] >= 5
    ):
        return True

    # Short citation-heavy chunks.
    text_len = len(
        _normalize(text)
    )

    if (
        text_len < 1800
        and features["citation_density"] >= 0.09
    ):
        return True

    return False


def _is_boilerplate_or_cover(text: str) -> bool:
    """
    Reject legal/copyright/cover-image boilerplate.
    """

    normalized = _normalize(text)

    boilerplate_terms = [
        "all rights reserved",
        "creative commons",
        "under the terms of this licence",
        "copyright",
        "cover photo",
        "freepik",
        "the responsibility for the interpretation",
        "liable for damages",
    ]

    matched = sum(
        term in normalized
        for term in boilerplate_terms
    )

    return matched >= 2


def _clinical_content_score(text: str) -> float:
    """
    Estimate usefulness as clinical evidence.

    Higher = more likely to be substantive clinical content.
    """

    if not text or not text.strip():
        return 0.0

    if _is_reference_or_bibliography(text):
        return 0.0

    if _is_boilerplate_or_cover(text):
        return 0.0

    normalized = _normalize(text)

    score = 1.0

    if len(normalized) >= 300:
        score += 0.4

    if len(normalized) >= 700:
        score += 0.3

    if "##" in text or "###" in text:
        score += 0.4

    clinical_terms = {
        "patient",
        "patients",
        "recommendation",
        "recommended",
        "treatment",
        "therapy",
        "dose",
        "monitoring",
        "transfusion",
        "hydroxyurea",
        "sickle",
        "clinical",
        "management",
        "guideline",
        "evidence",
    }

    clinical_hits = sum(
        term in normalized
        for term in clinical_terms
    )

    score += min(
        1.2,
        clinical_hits * 0.12,
    )

    if len(normalized) < 180:
        score -= 0.5

    return max(
        0.0,
        score,
    )


def _evidence_supported_by_chunk(
    relation: MedicalRelation,
    chunk_text: str,
) -> bool:
    """
    Verify that relation evidence is actually grounded
    in the candidate chunk.
    """

    evidence = (
        relation.evidence_text
        or ""
    ).strip()

    if len(evidence) < MIN_EVIDENCE_LENGTH:
        return False

    evidence_norm = _normalize(
        evidence
    )

    chunk_norm = _normalize(
        chunk_text
    )

    if (
        not evidence_norm
        or not chunk_norm
    ):
        return False

    if evidence_norm in chunk_norm:
        return True

    evidence_tokens = set(
        _meaningful_tokens(
            evidence
        )
    )

    chunk_tokens = set(
        _meaningful_tokens(
            chunk_text
        )
    )

    if not evidence_tokens:
        return False

    overlap = (
        evidence_tokens
        & chunk_tokens
    )

    ratio = (
        len(overlap)
        / len(evidence_tokens)
    )

    return ratio >= 0.55


def _query_terms_match(
    query: str,
    text: str,
) -> float:
    """
    Measure how strongly the chunk matches meaningful
    query terms.
    """

    query_tokens = set(
        _meaningful_tokens(
            query
        )
    )

    text_tokens = set(
        _meaningful_tokens(
            text
        )
    )

    if not query_tokens:
        return 0.0

    overlap = (
        query_tokens
        & text_tokens
    )

    return (
        len(overlap)
        / len(query_tokens)
    )


# ============================================================
# Entity helpers
# ============================================================

def _is_valid_entity(
    entity: MedicalEntity,
) -> bool:
    name = _normalize(
        entity.name
    )

    if not name:
        return False

    if len(name) < MIN_ENTITY_NAME_LENGTH:
        return False

    if name in GENERIC_ENTITY_NAMES:
        return False

    return True


def _canonical_entity_key(
    entity: MedicalEntity,
) -> str:
    normalized = _normalize(
        entity.name
    )

    if normalized:
        return normalized

    aliases = [
        _normalize(alias)
        for alias in entity.aliases
    ]

    return (
        aliases[0]
        if aliases
        else ""
    )


# ============================================================
# Graph Retriever
# ============================================================

class GraphRetriever:

    def __init__(
        self,
        graph: MedicalKnowledgeGraph,
        documents: Optional[
            List[Document]
        ] = None,
    ):

        if not isinstance(
            graph,
            MedicalKnowledgeGraph,
        ):
            raise TypeError(
                "graph must be a MedicalKnowledgeGraph"
            )

        self.graph = graph

        self.documents_by_chunk: Dict[
            str,
            Document,
        ] = {}

        if documents:

            for document in documents:

                chunk_id = (
                    document.metadata.get(
                        "chunk_id"
                    )
                )

                if chunk_id is not None:

                    self.documents_by_chunk[
                        str(chunk_id)
                    ] = document

    # ========================================================
    # Entity matching
    # ========================================================

    def _score_entity(
        self,
        query: str,
        entity: MedicalEntity,
    ) -> float:

        if not _is_valid_entity(
            entity
        ):
            return 0.0

        query_norm = _normalize(
            query
        )

        entity_name = _normalize(
            entity.name
        )

        query_tokens = set(
            _meaningful_tokens(query)
        )

        entity_tokens = set(
            _meaningful_tokens(
                entity.name
            )
        )

        score = 0.0

        if (
            entity_name
            and entity_name in query_norm
        ):
            score += 20.0

        for alias in entity.aliases:

            alias_norm = _normalize(alias)

            if (
                alias_norm
                and alias_norm in query_norm
            ):
                score += 15.0

        overlap = (
            query_tokens
            & entity_tokens
        )

        if overlap:

            score += (
                3.0
                * len(overlap)
            )

            if entity_tokens:
                score += (
                    3.0
                    * (
                        len(overlap)
                        / len(entity_tokens)
                    )
                )

        type_bonus = {
            EntityType.DISEASE: 4.0,
            EntityType.CONDITION: 3.0,
            EntityType.DRUG: 3.0,
            EntityType.TREATMENT: 3.0,
            EntityType.INTERVENTION: 2.0,
            EntityType.PROCEDURE: 1.5,
            EntityType.COMPLICATION: 1.5,
            EntityType.LAB_TEST: 1.0,
            EntityType.IMAGING: 1.0,
            EntityType.PREGNANCY: 1.0,
            EntityType.GUIDELINE: 0.5,
        }

        score += type_bonus.get(
            entity.entity_type,
            0.0,
        )

        score *= max(
            0.5,
            min(
                1.0,
                entity.confidence,
            ),
        )

        return score

    def find_entities(
        self,
        query: str,
        max_entities: int = DEFAULT_MAX_SEED_ENTITIES,
    ) -> List[str]:

        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "query must be a string"
            )

        query = query.strip()

        if not query:
            return []

        scored = []

        for entity_id, entity in (
            self.graph.entities.items()
        ):

            score = self._score_entity(
                query,
                entity
            )

            if score <= 0:
                continue

            scored.append(
                (
                    entity_id,
                    score,
                    entity,
                )
            )

        scored.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        selected = []
        seen_canonical = set()

        for (
            entity_id,
            score,
            entity,
        ) in scored:

            if score < 4.0:
                continue

            canonical = (
                _canonical_entity_key(
                    entity
                )
            )

            if (
                not canonical
                or canonical
                in seen_canonical
            ):
                continue

            seen_canonical.add(
                canonical
            )

            selected.append(
                entity_id
            )

            if (
                len(selected)
                >= max_entities
            ):
                break

        return selected

    # ========================================================
    # Traversal
    # ========================================================

    def traverse(
        self,
        entity_id: str,
        max_hops: int = DEFAULT_MAX_HOPS,
        max_entities: int = DEFAULT_MAX_ENTITIES,
    ) -> List[str]:

        if entity_id not in self.graph.entities:
            return []

        if max_hops < 0:
            raise ValueError(
                "max_hops cannot be negative"
            )

        if max_entities <= 0:
            raise ValueError(
                "max_entities must be greater than 0"
            )

        visited = {
            entity_id
        }

        frontier = [
            entity_id
        ]

        ordered = [
            entity_id
        ]

        for _ in range(max_hops):

            next_frontier = []

            for current in frontier:

                for neighbor in (
                    self.graph.neighbors(current)
                ):

                    neighbor_id = (
                        neighbor.entity_id
                    )

                    if neighbor_id in visited:
                        continue

                    neighbor_entity = (
                        self.graph.get_entity(
                            neighbor_id
                        )
                    )

                    if (
                        neighbor_entity is None
                        or not _is_valid_entity(
                            neighbor_entity
                        )
                    ):
                        continue

                    visited.add(
                        neighbor_id
                    )

                    ordered.append(
                        neighbor_id
                    )

                    next_frontier.append(
                        neighbor_id
                    )

                    if (
                        len(ordered)
                        >= max_entities
                    ):
                        return ordered

            frontier = next_frontier

            if not frontier:
                break

        return ordered

    # ========================================================
    # Relations
    # ========================================================

    def collect_relations(
        self,
        entity_ids: List[str],
    ) -> List[MedicalRelation]:

        selected = set(
            entity_ids
        )

        relations = []

        for relation in (
            self.graph.relations.values()
        ):

            if (
                relation.source_entity_id
                in selected
                or
                relation.target_entity_id
                in selected
            ):
                relations.append(
                    relation
                )

        return relations

    # ========================================================
    # Relation scoring
    # ========================================================

    @staticmethod
    def _base_relation_priority(
        relation: MedicalRelation,
    ) -> float:

        priority = {
            RelationType.TREATS: 12.0,
            RelationType.RECOMMENDED_FOR: 12.0,
            RelationType.CONTRAINDICATED_FOR: 12.0,
            RelationType.PREVENTS: 10.0,
            RelationType.MANAGES: 10.0,
            RelationType.REQUIRES_TEST: 9.0,
            RelationType.MONITORS: 9.0,
            RelationType.REDUCES: 8.0,
            RelationType.IMPROVES: 8.0,
            RelationType.HAS_COMPLICATION: 7.0,
            RelationType.HAS_ADVERSE_EVENT: 7.0,
            RelationType.CAUSES: 6.0,
            RelationType.RISK_FACTOR_FOR: 6.0,
            RelationType.APPLIES_TO: 5.0,
            RelationType.ASSOCIATED_WITH: 4.0,
            RelationType.RELATED_TO: 1.0,
        }

        return priority.get(
            relation.relation_type,
            0.5,
        )

    def _relation_score(
        self,
        relation: MedicalRelation,
        query_intent: str,
    ) -> float:

        score = (
            self._base_relation_priority(
                relation
            )
        )

        if (
            query_intent == "treatment"
            and relation.relation_type
            in {
                RelationType.TREATS,
                RelationType.RECOMMENDED_FOR,
                RelationType.MANAGES,
                RelationType.PREVENTS,
                RelationType.REDUCES,
                RelationType.IMPROVES,
                RelationType.CONTRAINDICATED_FOR,
            }
        ):
            score += 15.0

        elif (
            query_intent == "diagnosis"
            and relation.relation_type
            in {
                RelationType.REQUIRES_TEST,
                RelationType.MEASURED_BY,
                RelationType.MONITORS,
            }
        ):
            score += 15.0

        elif (
            query_intent == "monitoring"
            and relation.relation_type
            in {
                RelationType.MONITORS,
                RelationType.REQUIRES_TEST,
                RelationType.MEASURED_BY,
            }
        ):
            score += 15.0

        elif (
            query_intent == "complication"
            and relation.relation_type
            in {
                RelationType.HAS_COMPLICATION,
                RelationType.HAS_ADVERSE_EVENT,
                RelationType.CAUSES,
                RelationType.RISK_FACTOR_FOR,
            }
        ):
            score += 15.0

        confidence = max(
            0.5,
            min(
                1.0,
                relation.confidence,
            ),
        )

        return score * confidence

    # ========================================================
    # Candidate chunk quality
    # ========================================================

    def _score_chunk_for_relation(
        self,
        relation: MedicalRelation,
        chunk: Document,
        query: str,
        query_intent: str,
    ) -> float:
        """
        Final score for relation-supported chunk.

        This deliberately punishes bibliography-heavy chunks.
        """

        text = (
            chunk.page_content
            or ""
        ).strip()

        if not text:
            return 0.0

        if not _evidence_supported_by_chunk(
            relation,
            text,
        ):
            return 0.0

        clinical_score = (
            _clinical_content_score(
                text
            )
        )

        if clinical_score <= 0:
            return 0.0

        query_match = (
            _query_terms_match(
                query,
                text,
            )
        )

        relation_score = (
            self._relation_score(
                relation,
                query_intent,
            )
        )

        features = _reference_features(
            text
        )

        # Extra penalty for citation-heavy content.
        citation_penalty = 1.0

        if features[
            "numbered_refs"
        ] >= 3:
            citation_penalty *= 0.35

        if features[
            "author_year"
        ] >= 3:
            citation_penalty *= 0.45

        if features[
            "doi_count"
        ] >= 1:
            citation_penalty *= 0.65

        # Clinical query term coverage is useful.
        query_bonus = (
            1.0
            + 1.5
            * query_match
        )

        return (
            relation_score
            * clinical_score
            * citation_penalty
            * query_bonus
        )

    def _rank_evidence_chunks(
        self,
        relations: List[MedicalRelation],
        query: str,
        query_intent: str,
        max_documents: int,
    ) -> List[
        Tuple[str, float]
    ]:
        """
        Rank relation-supported source chunks.
        """

        chunk_scores: Dict[
            str,
            float,
        ] = {}

        chunk_relation_counts: Dict[
            str,
            int,
        ] = {}

        for relation in relations:

            for chunk_id in (
                relation.source_chunks
            ):

                chunk_id = str(
                    chunk_id
                )

                document = (
                    self.documents_by_chunk.get(
                        chunk_id
                    )
                )

                if document is None:
                    continue

                score = (
                    self._score_chunk_for_relation(
                        relation=relation,
                        chunk=document,
                        query=query,
                        query_intent=query_intent,
                    )
                )

                if score <= 0:
                    continue

                chunk_scores[
                    chunk_id
                ] = (
                    chunk_scores.get(
                        chunk_id,
                        0.0,
                    )
                    + score
                )

                chunk_relation_counts[
                    chunk_id
                ] = (
                    chunk_relation_counts.get(
                        chunk_id,
                        0,
                    )
                    + 1
                )

        # Bonus when one clinical chunk supports
        # several validated relations.
        for chunk_id, count in (
            chunk_relation_counts.items()
        ):

            chunk_scores[
                chunk_id
            ] += min(
                count,
                3,
            ) * 3.0

        ranked = sorted(
            chunk_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return ranked[
            :max_documents
        ]

    # ========================================================
    # Main retrieval
    # ========================================================

    def retrieve(
        self,
        query: str,
        max_hops: int = DEFAULT_MAX_HOPS,
        max_entities: int = DEFAULT_MAX_ENTITIES,
        max_documents: int = DEFAULT_MAX_DOCUMENTS,
    ) -> List[Document]:

        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "query must be a string"
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty"
            )

        query_intent = (
            detect_query_intent(
                query
            )
        )

        seed_entities = (
            self.find_entities(
                query
            )
        )

        if not seed_entities:
            return []

        # ----------------------------------------------------
        # Traverse
        # ----------------------------------------------------

        selected_entities = []
        seen_entities = set()

        for seed in seed_entities:

            traversed = (
                self.traverse(
                    seed,
                    max_hops=max_hops,
                    max_entities=max_entities,
                )
            )

            for entity_id in traversed:

                if (
                    entity_id
                    in seen_entities
                ):
                    continue

                seen_entities.add(
                    entity_id
                )

                selected_entities.append(
                    entity_id
                )

                if (
                    len(selected_entities)
                    >= max_entities
                ):
                    break

            if (
                len(selected_entities)
                >= max_entities
            ):
                break

        # ----------------------------------------------------
        # Relations
        # ----------------------------------------------------

        relations = (
            self.collect_relations(
                selected_entities
            )
        )

        relations = [
            relation
            for relation in relations
            if (
                relation.relation_type
                != RelationType.OTHER
            )
        ]

        relations.sort(
            key=lambda relation: (
                self._relation_score(
                    relation,
                    query_intent,
                )
            ),
            reverse=True,
        )

        # ----------------------------------------------------
        # Evidence ranking
        # ----------------------------------------------------

        ranked_chunks = (
            self._rank_evidence_chunks(
                relations=relations,
                query=query,
                query_intent=query_intent,
                max_documents=max_documents,
            )
        )

        results = []

        for chunk_id, score in (
            ranked_chunks
        ):

            document = (
                self.documents_by_chunk.get(
                    str(chunk_id)
                )
            )

            if document is None:
                continue

            supporting_relations = []

            for relation in relations:

                if str(chunk_id) not in {
                    str(cid)
                    for cid
                    in relation.source_chunks
                }:
                    continue

                if not _evidence_supported_by_chunk(
                    relation,
                    document.page_content,
                ):
                    continue

                # Do not show relation if source itself
                # is bibliographic/boilerplate.
                if (
                    _clinical_content_score(
                        document.page_content
                    )
                    <= 0
                ):
                    continue

                supporting_relations.append(
                    relation.relation_type.value
                )

            if not supporting_relations:
                continue

            metadata = (
                document.metadata.copy()
            )

            metadata[
                "retrieval_method"
            ] = "graph"

            metadata[
                "graph_query_intent"
            ] = query_intent

            metadata[
                "graph_seed_entities"
            ] = list(
                seed_entities
            )

            metadata[
                "graph_entities"
            ] = list(
                selected_entities
            )

            metadata[
                "graph_supporting_relations"
            ] = supporting_relations

            metadata[
                "graph_score"
            ] = float(score)

            results.append(
                Document(
                    page_content=(
                        document.page_content
                    ),
                    metadata=metadata,
                )
            )

            if (
                len(results)
                >= max_documents
            ):
                break

        return results

    # ========================================================
    # Structured context
    # ========================================================

    def get_context(
        self,
        query: str,
        max_hops: int = DEFAULT_MAX_HOPS,
        max_entities: int = DEFAULT_MAX_ENTITIES,
    ) -> Dict:

        query_intent = (
            detect_query_intent(
                query
            )
        )

        seed_entities = (
            self.find_entities(
                query
            )
        )

        selected_entities = []
        seen_entities = set()

        for seed in seed_entities:

            traversed = (
                self.traverse(
                    seed,
                    max_hops=max_hops,
                    max_entities=max_entities,
                )
            )

            for entity_id in traversed:

                if (
                    entity_id
                    in seen_entities
                ):
                    continue

                seen_entities.add(
                    entity_id
                )

                selected_entities.append(
                    entity_id
                )

                if (
                    len(selected_entities)
                    >= max_entities
                ):
                    break

            if (
                len(selected_entities)
                >= max_entities
            ):
                break

        entities = []

        for entity_id in (
            selected_entities
        ):

            entity = self.graph.get_entity(
                entity_id
            )

            if (
                entity is None
                or not _is_valid_entity(
                    entity
                )
            ):
                continue

            entities.append(
                entity.to_dict()
            )

        relations = (
            self.collect_relations(
                selected_entities
            )
        )

        relations = [
            relation
            for relation in relations
            if (
                relation.relation_type
                != RelationType.OTHER
            )
        ]

        relations.sort(
            key=lambda relation: (
                self._relation_score(
                    relation,
                    query_intent,
                )
            ),
            reverse=True,
        )

        return {
            "query": query,
            "query_intent": query_intent,
            "seed_entities": seed_entities,
            "entities": entities,
            "relations": [
                relation.to_dict()
                for relation in relations[:20]
            ],
        }


# ============================================================
# Persistence
# ============================================================

def load_graph(
    graph_path: str = "data/processed/graph.json",
) -> MedicalKnowledgeGraph:

    path = Path(
        graph_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Graph file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    graph = MedicalKnowledgeGraph()

    # --------------------------------------------------------
    # Entities
    # --------------------------------------------------------

    for item in data.get(
        "entities",
        [],
    ):

        entity = MedicalEntity(
            entity_id=item[
                "entity_id"
            ],
            name=item[
                "name"
            ],
            entity_type=EntityType(
                item["entity_type"]
            ),
            aliases=item.get(
                "aliases",
                [],
            ),
            description=item.get(
                "description",
                "",
            ),
            source_chunks=item.get(
                "source_chunks",
                [],
            ),
            citations=item.get(
                "citations",
                [],
            ),
            confidence=item.get(
                "confidence",
                1.0,
            ),
            properties=item.get(
                "properties",
                {},
            ),
        )

        graph.add_entity(
            entity
        )

    # --------------------------------------------------------
    # Relations
    # --------------------------------------------------------

    for item in data.get(
        "relations",
        [],
    ):

        relation = MedicalRelation(
            relation_id=item[
                "relation_id"
            ],
            source_entity_id=item[
                "source_entity_id"
            ],
            target_entity_id=item[
                "target_entity_id"
            ],
            relation_type=RelationType(
                item["relation_type"]
            ),
            confidence=item.get(
                "confidence",
                1.0,
            ),
            source_chunks=item.get(
                "source_chunks",
                [],
            ),
            citations=item.get(
                "citations",
                [],
            ),
            evidence_text=item.get(
                "evidence_text",
                "",
            ),
            properties=item.get(
                "properties",
                {},
            ),
        )

        graph.add_relation(
            relation
        )

    return graph


# ============================================================
# Factory
# ============================================================

def create_graph_retriever(
    graph: MedicalKnowledgeGraph,
    documents: Optional[
        List[Document]
    ] = None,
) -> GraphRetriever:

    return GraphRetriever(
        graph=graph,
        documents=documents,
    )