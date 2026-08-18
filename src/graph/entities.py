from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ============================================================
# Entity Types
# ============================================================

class EntityType(str, Enum):
    """
    Supported medical entity types for SickleGuide Graph RAG.
    """

    DISEASE = "disease"
    CONDITION = "condition"

    SYMPTOM = "symptom"
    SIGN = "sign"

    DRUG = "drug"
    TREATMENT = "treatment"
    INTERVENTION = "intervention"
    PROCEDURE = "procedure"

    LAB_TEST = "lab_test"
    IMAGING = "imaging"

    COMPLICATION = "complication"
    ADVERSE_EVENT = "adverse_event"

    CONTRAINDICATION = "contraindication"
    RISK_FACTOR = "risk_factor"

    BIOMARKER = "biomarker"
    LAB_VALUE = "lab_value"

    POPULATION = "population"
    AGE_GROUP = "age_group"

    PREGNANCY = "pregnancy"

    OUTCOME = "outcome"
    EVIDENCE = "evidence"

    GUIDELINE = "guideline"
    RECOMMENDATION = "recommendation"

    ORGAN = "organ"
    GENE = "gene"
    PROTEIN = "protein"

    OTHER = "other"


# ============================================================
# Graph Entity
# ============================================================

@dataclass
class MedicalEntity:
    """
    Canonical medical entity used as a Graph RAG node.

    Attributes:
        entity_id:
            Stable identifier used inside the graph.

        name:
            Human-readable entity name.

        entity_type:
            Medical entity category.

        aliases:
            Alternative names / abbreviations.

        description:
            Optional normalized description.

        source_chunks:
            IDs of chunks supporting this entity.

        citations:
            Human-readable source citations.

        confidence:
            Extraction / normalization confidence.

        properties:
            Additional entity-specific attributes.
    """

    entity_id: str

    name: str

    entity_type: EntityType

    aliases: List[str] = field(
        default_factory=list
    )

    description: str = ""

    source_chunks: List[str] = field(
        default_factory=list
    )

    citations: List[str] = field(
        default_factory=list
    )

    confidence: float = 1.0

    properties: Dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self):
        """
        Validate and normalize the entity.
        """

        self.entity_id = (
            self.entity_id.strip()
        )

        self.name = (
            self.name.strip()
        )

        if not self.entity_id:
            raise ValueError(
                "entity_id cannot be empty"
            )

        if not self.name:
            raise ValueError(
                "name cannot be empty"
            )

        if isinstance(
            self.entity_type,
            str,
        ):
            self.entity_type = EntityType(
                self.entity_type
            )

        self.confidence = max(
            0.0,
            min(
                1.0,
                float(self.confidence),
            ),
        )

        self.aliases = self._unique_strings(
            self.aliases
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
        Remove empty and duplicate strings
        while preserving order.
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

    def add_source(
        self,
        chunk_id: str,
        citation: Optional[str] = None,
    ) -> None:
        """
        Attach provenance to the entity.
        """

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

    def add_alias(
        self,
        alias: str,
    ) -> None:
        """
        Add an alias safely.
        """

        alias = alias.strip()

        if (
            alias
            and alias.lower()
            != self.name.lower()
            and alias
            not in self.aliases
        ):
            self.aliases.append(
                alias
            )

    def merge(
        self,
        other: "MedicalEntity",
    ) -> "MedicalEntity":
        """
        Merge another occurrence of the same entity
        into this canonical entity.
        """

        if not isinstance(
            other,
            MedicalEntity,
        ):
            raise TypeError(
                "other must be a MedicalEntity"
            )

        if (
            self.entity_id
            != other.entity_id
        ):
            raise ValueError(
                "Cannot merge entities "
                "with different entity IDs"
            )

        for alias in other.aliases:
            self.add_alias(alias)

        for chunk_id in other.source_chunks:
            self.add_source(
                chunk_id
            )

        for citation in other.citations:

            if citation not in self.citations:
                self.citations.append(
                    citation
                )

        if (
            not self.description
            and other.description
        ):
            self.description = (
                other.description
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
        Convert entity to a JSON-serializable dictionary.
        """

        data = asdict(self)

        data["entity_type"] = (
            self.entity_type.value
        )

        return data


# ============================================================
# Entity Helpers
# ============================================================

def normalize_entity_name(
    name: str,
) -> str:
    """
    Normalize entity names for stable graph IDs.

    Example:
        "Sickle Cell Disease"
        -> "sickle_cell_disease"
    """

    if not isinstance(
        name,
        str,
    ):
        raise TypeError(
            "name must be a string"
        )

    normalized = (
        name.strip()
        .lower()
    )

    result = []

    previous_was_separator = False

    for char in normalized:

        if char.isalnum():

            result.append(char)
            previous_was_separator = False

        elif not previous_was_separator:

            result.append("_")
            previous_was_separator = True

    return "".join(result).strip("_")


def build_entity_id(
    name: str,
    entity_type: EntityType,
) -> str:
    """
    Build a stable graph node ID.

    Format:

        <entity_type>:<normalized_name>

    Example:

        disease:sickle_cell_disease
        drug:hydroxyurea
    """

    if isinstance(
        entity_type,
        str,
    ):
        entity_type = EntityType(
            entity_type
        )

    normalized_name = (
        normalize_entity_name(name)
    )

    if not normalized_name:
        raise ValueError(
            "Cannot build entity ID "
            "from empty name"
        )

    return (
        f"{entity_type.value}:"
        f"{normalized_name}"
    )


def create_entity(
    name: str,
    entity_type: EntityType,
    chunk_id: Optional[str] = None,
    citation: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    description: str = "",
    confidence: float = 1.0,
    properties: Optional[
        Dict[str, Any]
    ] = None,
) -> MedicalEntity:
    """
    Create a canonical MedicalEntity.
    """

    entity_id = build_entity_id(
        name=name,
        entity_type=entity_type,
    )

    entity = MedicalEntity(
        entity_id=entity_id,
        name=name.strip(),
        entity_type=entity_type,
        aliases=aliases or [],
        description=description,
        confidence=confidence,
        properties=properties or {},
    )

    if chunk_id:
        entity.add_source(
            chunk_id=chunk_id,
            citation=citation,
        )

    return entity