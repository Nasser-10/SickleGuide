from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ============================================================
# Relation Types
# ============================================================

class RelationType(str, Enum):
    """
    Medical relation types used by the SickleGuide graph.
    """

    TREATS = "treats"
    PREVENTS = "prevents"
    MANAGES = "manages"

    CAUSES = "causes"
    ASSOCIATED_WITH = "associated_with"

    PRESENTS_WITH = "presents_with"
    HAS_SYMPTOM = "has_symptom"
    HAS_SIGN = "has_sign"

    HAS_COMPLICATION = "has_complication"
    HAS_ADVERSE_EVENT = "has_adverse_event"

    RECOMMENDED_FOR = "recommended_for"
    NOT_RECOMMENDED_FOR = "not_recommended_for"

    CONTRAINDICATED_FOR = "contraindicated_for"

    MONITORS = "monitors"
    REQUIRES_TEST = "requires_test"
    MEASURED_BY = "measured_by"

    IMPROVES = "improves"
    REDUCES = "reduces"
    INCREASES = "increases"

    RISK_FACTOR_FOR = "risk_factor_for"

    APPLIES_TO = "applies_to"
    HAS_AGE_GROUP = "has_age_group"
    HAS_POPULATION = "has_population"

    RELATED_TO = "related_to"

    SUPPORTED_BY = "supported_by"
    DERIVED_FROM = "derived_from"

    PART_OF = "part_of"
    SUBTYPE_OF = "subtype_of"

    OCCURS_DURING = "occurs_during"

    OTHER = "other"


# ============================================================
# Graph Relation
# ============================================================

@dataclass
class MedicalRelation:
    """
    Represents a directed relationship between two medical entities.

    Example:

        drug:hydroxyurea
              |
              | treats
              v
        disease:sickle_cell_disease
    """

    relation_id: str

    source_entity_id: str

    target_entity_id: str

    relation_type: RelationType

    confidence: float = 1.0

    source_chunks: List[str] = field(
        default_factory=list
    )

    citations: List[str] = field(
        default_factory=list
    )

    evidence_text: str = ""

    properties: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self):
        """
        Validate and normalize the relation.
        """

        self.relation_id = (
            self.relation_id.strip()
        )

        self.source_entity_id = (
            self.source_entity_id.strip()
        )

        self.target_entity_id = (
            self.target_entity_id.strip()
        )

        if not self.relation_id:
            raise ValueError(
                "relation_id cannot be empty"
            )

        if not self.source_entity_id:
            raise ValueError(
                "source_entity_id cannot be empty"
            )

        if not self.target_entity_id:
            raise ValueError(
                "target_entity_id cannot be empty"
            )

        if isinstance(
            self.relation_type,
            str,
        ):
            self.relation_type = RelationType(
                self.relation_type
            )

        self.confidence = max(
            0.0,
            min(
                1.0,
                float(self.confidence),
            ),
        )

        self.source_chunks = (
            self._unique_strings(
                self.source_chunks
            )
        )

        self.citations = (
            self._unique_strings(
                self.citations
            )
        )

    @staticmethod
    def _unique_strings(
        values: List[str],
    ) -> List[str]:
        """
        Remove duplicates while preserving order.
        """

        seen = set()
        result = []

        for value in values:

            if value is None:
                continue

            value = str(value).strip()

            if not value:
                continue

            if value in seen:
                continue

            seen.add(value)
            result.append(value)

        return result

    def add_evidence(
        self,
        chunk_id: Optional[str] = None,
        citation: Optional[str] = None,
        evidence_text: Optional[str] = None,
    ) -> None:
        """
        Attach provenance/evidence to the relation.
        """

        if chunk_id:

            chunk_id = str(
                chunk_id
            ).strip()

            if (
                chunk_id
                and chunk_id
                not in self.source_chunks
            ):
                self.source_chunks.append(
                    chunk_id
                )

        if citation:

            citation = citation.strip()

            if (
                citation
                and citation
                not in self.citations
            ):
                self.citations.append(
                    citation
                )

        if (
            evidence_text
            and not self.evidence_text
        ):
            self.evidence_text = (
                evidence_text.strip()
            )

    def merge(
        self,
        other: "MedicalRelation",
    ) -> "MedicalRelation":
        """
        Merge another occurrence of the same relation.
        """

        if not isinstance(
            other,
            MedicalRelation,
        ):
            raise TypeError(
                "other must be a MedicalRelation"
            )

        if (
            self.relation_id
            != other.relation_id
        ):
            raise ValueError(
                "Cannot merge relations "
                "with different relation IDs"
            )

        for chunk_id in other.source_chunks:

            if (
                chunk_id
                not in self.source_chunks
            ):
                self.source_chunks.append(
                    chunk_id
                )

        for citation in other.citations:

            if (
                citation
                not in self.citations
            ):
                self.citations.append(
                    citation
                )

        if (
            not self.evidence_text
            and other.evidence_text
        ):
            self.evidence_text = (
                other.evidence_text
            )

        self.confidence = max(
            self.confidence,
            other.confidence,
        )

        self.properties.update(
            other.properties
        )

        return self

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Convert relation to a JSON-serializable dictionary.
        """

        data = asdict(self)

        data["relation_type"] = (
            self.relation_type.value
        )

        return data


# ============================================================
# Relation Helpers
# ============================================================

def build_relation_id(
    source_entity_id: str,
    relation_type: RelationType,
    target_entity_id: str,
) -> str:
    """
    Create a stable relation identifier.

    Example:

        drug:hydroxyurea
        +
        treats
        +
        disease:sickle_cell_disease

        ->
        drug:hydroxyurea|treats|disease:sickle_cell_disease
    """

    if not source_entity_id:
        raise ValueError(
            "source_entity_id cannot be empty"
        )

    if not target_entity_id:
        raise ValueError(
            "target_entity_id cannot be empty"
        )

    if isinstance(
        relation_type,
        str,
    ):
        relation_type = RelationType(
            relation_type
        )

    return (
        f"{source_entity_id}"
        f"|{relation_type.value}"
        f"|{target_entity_id}"
    )


def create_relation(
    source_entity_id: str,
    target_entity_id: str,
    relation_type: RelationType,
    chunk_id: Optional[str] = None,
    citation: Optional[str] = None,
    evidence_text: str = "",
    confidence: float = 1.0,
    properties: Optional[
        Dict[str, Any]
    ] = None,
) -> MedicalRelation:
    """
    Create a canonical MedicalRelation.
    """

    relation_id = build_relation_id(
        source_entity_id=source_entity_id,
        relation_type=relation_type,
        target_entity_id=target_entity_id,
    )

    relation = MedicalRelation(
        relation_id=relation_id,
        source_entity_id=source_entity_id,
        target_entity_id=target_entity_id,
        relation_type=relation_type,
        confidence=confidence,
        evidence_text=evidence_text,
        properties=properties or {},
    )

    relation.add_evidence(
        chunk_id=chunk_id,
        citation=citation,
        evidence_text=evidence_text,
    )

    return relation