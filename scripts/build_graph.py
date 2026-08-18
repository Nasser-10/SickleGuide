from pathlib import Path
from typing import List, Optional
import json
import os
import sys
import traceback

from pydantic import BaseModel, Field


# ============================================================
# Project root
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


PROCESSED_DIR = ROOT_DIR / "data" / "processed"

INPUT_FILE = (
    PROCESSED_DIR / "chunks.json"
)

OUTPUT_FILE = (
    PROCESSED_DIR / "graph.json"
)


# ============================================================
# Configuration
# ============================================================

OLLAMA_MODEL = os.getenv(
    "SICKLE_GRAPH_MODEL",
    "qwen2.5:7b",
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

MAX_CHUNKS = int(
    os.getenv(
        "SICKLE_GRAPH_MAX_CHUNKS",
        "0",
    )
)

MAX_ENTITIES_PER_CHUNK = 10
MAX_RELATIONS_PER_CHUNK = 6


# ============================================================
# Allowed Graph Vocabulary
# ============================================================

ALLOWED_ENTITY_TYPES = [
    "disease",
    "condition",
    "symptom",
    "sign",
    "drug",
    "treatment",
    "intervention",
    "procedure",
    "lab_test",
    "imaging",
    "complication",
    "adverse_event",
    "contraindication",
    "risk_factor",
    "biomarker",
    "lab_value",
    "population",
    "age_group",
    "pregnancy",
    "outcome",
    "guideline",
    "recommendation",
]


ALLOWED_RELATION_TYPES = [
    "treats",
    "prevents",
    "manages",
    "causes",
    "associated_with",
    "presents_with",
    "has_symptom",
    "has_sign",
    "has_complication",
    "has_adverse_event",
    "recommended_for",
    "not_recommended_for",
    "contraindicated_for",
    "monitors",
    "requires_test",
    "measured_by",
    "improves",
    "reduces",
    "increases",
    "risk_factor_for",
    "applies_to",
    "has_age_group",
    "has_population",
    "occurs_during",
    "related_to",
]


# ============================================================
# Structured Extraction Schema
# ============================================================

class ExtractedEntity(BaseModel):
    name: str = Field(
        description=(
            "Canonical medical entity name."
        ),
    )

    entity_type: str = Field(
        description=(
            "One of the allowed medical entity "
            "types provided in the prompt."
        ),
    )

    aliases: List[str] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "At most 3 important aliases or "
            "abbreviations."
        ),
    )


class ExtractedRelation(BaseModel):
    source: str = Field(
        description=(
            "Exact source entity name from the "
            "entities list."
        ),
    )

    target: str = Field(
        description=(
            "Exact target entity name from the "
            "entities list."
        ),
    )

    relation: str = Field(
        description=(
            "One of the allowed clinical relation types."
        ),
    )

    evidence: str = Field(
        max_length=350,
        description=(
            "Short supporting evidence from the passage."
        ),
    )

    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
    )


class MedicalExtraction(BaseModel):
    entities: List[ExtractedEntity] = Field(
        default_factory=list,
        max_length=MAX_ENTITIES_PER_CHUNK,
    )

    relations: List[ExtractedRelation] = Field(
        default_factory=list,
        max_length=MAX_RELATIONS_PER_CHUNK,
    )


# ============================================================
# LLM
# ============================================================

def create_extraction_model():
    """
    Create structured-output LLM for graph extraction.
    """

    from langchain_ollama import ChatOllama

    print(
        f"Using Ollama model: {OLLAMA_MODEL}",
        flush=True,
    )

    print(
        f"Ollama endpoint: {OLLAMA_BASE_URL}",
        flush=True,
    )

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
        num_predict=1800,
    )

    return llm.with_structured_output(
        MedicalExtraction,
        method="json_schema",
    )


# ============================================================
# Prompt
# ============================================================

SYSTEM_PROMPT = f"""
You are a conservative medical knowledge-graph extraction system.

Extract ONLY clinically meaningful facts explicitly supported
by the supplied medical passage.

ALLOWED ENTITY TYPES:
{", ".join(ALLOWED_ENTITY_TYPES)}

ALLOWED RELATIONS:
{", ".join(ALLOWED_RELATION_TYPES)}

STRICT RULES:

1. Extract at most {MAX_ENTITIES_PER_CHUNK} entities.
2. Extract at most {MAX_RELATIONS_PER_CHUNK} relations.
3. Prefer clinically important entities over administrative,
   bibliographic, author, methodology, or software entities.
4. Ignore references, bibliography, author names, page numbers,
   guideline-production workflow, and document formatting.
5. Do NOT create "same_as", "component_of", "prepared",
   "used", "summarizes", "suggests", "helps", or other
   relations outside the allowed list.
6. Only create a relation when the passage explicitly supports it.
7. source and target MUST exactly match names from the entity list.
8. Every relation MUST have short supporting evidence from the passage.
9. Do NOT infer a treatment relation merely because two entities
   appear in the same paragraph.
10. For recommendations, prefer:
       intervention/drug -> recommended_for -> population/condition
11. For contraindications, prefer:
       intervention/drug -> contraindicated_for -> condition/population
12. For monitoring, prefer:
       test/lab -> monitors -> condition/intervention
13. Ignore low-value generic concepts such as:
       disease severity, considerations, critical outcomes,
       clinical decision-making, methodology, literature review.
14. Return empty lists when there is insufficient evidence.

The goal is a SMALL, HIGH-PRECISION clinical graph,
not maximum extraction.
"""


def build_prompt(
    text: str,
) -> str:
    return f"""
{SYSTEM_PROMPT}

MEDICAL PASSAGE:

{text}
"""


def extract_from_chunk(
    structured_llm,
    text: str,
) -> MedicalExtraction:

    result = structured_llm.invoke(
        build_prompt(text)
    )

    if isinstance(
        result,
        MedicalExtraction,
    ):
        return result

    if isinstance(
        result,
        dict,
    ):
        return MedicalExtraction.model_validate(
            result
        )

    raise TypeError(
        "Unexpected structured output: "
        f"{type(result)}"
    )


# ============================================================
# Normalization
# ============================================================

def normalize_entity_type(
    entity_type: str,
) -> str:

    value = (
        entity_type.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    aliases = {
        "disease_or_condition": "disease",
        "condition": "condition",
        "medication": "drug",
        "medicine": "drug",
        "therapy": "treatment",
        "therapeutic": "treatment",
        "test": "lab_test",
        "laboratory_test": "lab_test",
        "imaging_test": "imaging",
        "adverse_effect": "adverse_event",
        "risk": "risk_factor",
        "age": "age_group",
    }

    value = aliases.get(
        value,
        value,
    )

    if value in ALLOWED_ENTITY_TYPES:
        return value

    return "condition"


def normalize_relation_type(
    relation: str,
) -> Optional[str]:

    value = (
        relation.strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    aliases = {
        "treat": "treats",
        "treated_by": "treats",
        "prevent": "prevents",
        "manage": "manages",
        "managed_by": "manages",
        "cause": "causes",
        "caused_by": "causes",
        "associated": "associated_with",
        "symptoms": "has_symptom",
        "has_symptoms": "has_symptom",
        "has_signs": "has_sign",
        "complicates": "has_complication",
        "adverse_effect": "has_adverse_event",
        "recommended": "recommended_for",
        "not_recommended": "not_recommended_for",
        "contraindicated": "contraindicated_for",
        "monitor": "monitors",
        "requires_monitoring": "monitors",
        "requires_lab": "requires_test",
        "measure": "measured_by",
        "improve": "improves",
        "reduce": "reduces",
        "increase": "increases",
        "risk_for": "risk_factor_for",
        "applies": "applies_to",
        "during": "occurs_during",
        "related": "related_to",
    }

    value = aliases.get(
        value,
        value,
    )

    if value not in ALLOWED_RELATION_TYPES:
        return None

    return value


# ============================================================
# Build Graph
# ============================================================

def build_graph():

    from src.graph.entities import (
        EntityType,
        create_entity,
    )

    from src.graph.relations import (
        RelationType,
        create_relation,
    )

    from src.graph.graph_builder import (
        MedicalKnowledgeGraph,
    )

    print("=" * 70)
    print(
        "SickleGuide - Graph Construction",
        flush=True,
    )
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {INPUT_FILE}"
        )

    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        chunk_records = json.load(file)

    if not isinstance(
        chunk_records,
        list,
    ):
        raise ValueError(
            "chunks.json must contain a list"
        )

    if MAX_CHUNKS > 0:
        chunk_records = (
            chunk_records[:MAX_CHUNKS]
        )

    print(
        f"Input chunks: {len(chunk_records)}",
        flush=True,
    )

    print(
        "\nLoading structured extraction model...",
        flush=True,
    )

    structured_llm = (
        create_extraction_model()
    )

    graph = MedicalKnowledgeGraph()

    error_count = 0
    retry_count = 0

    for index, record in enumerate(
        chunk_records,
        start=1,
    ):

        text = (
            record.get(
                "page_content",
                "",
            )
            or ""
        ).strip()

        metadata = record.get(
            "metadata",
            {},
        )

        chunk_id = str(
            metadata.get(
                "chunk_id",
                index - 1,
            )
        )

        citation = metadata.get(
            "citation",
            "",
        )

        if not text:
            continue

        # ------------------------------------------------------
        # Extraction
        # ------------------------------------------------------

        try:

            extraction = extract_from_chunk(
                structured_llm,
                text,
            )

        except Exception as first_error:

            retry_count += 1

            print(
                f"[RETRY] Chunk {chunk_id}",
                flush=True,
            )

            try:

                # Compact fallback prompt.
                retry_prompt = f"""
Extract ONLY clinically important entities and
clinical relations from this medical passage.

Maximum:
- 6 entities
- 3 relations

Allowed relations:
{", ".join(ALLOWED_RELATION_TYPES)}

Ignore references, methodology, authors,
software, generic concepts, and administrative content.

Every relation must:
- use entities from the entity list
- use an allowed relation
- include short evidence

Passage:
{text}
"""

                extraction = (
                    structured_llm.invoke(
                        retry_prompt
                    )
                )

                if isinstance(
                    extraction,
                    dict,
                ):
                    extraction = (
                        MedicalExtraction
                        .model_validate(
                            extraction
                        )
                    )

                if not isinstance(
                    extraction,
                    MedicalExtraction,
                ):
                    raise TypeError(
                        "Retry returned invalid output"
                    )

            except Exception as retry_error:

                error_count += 1

                print(
                    f"[WARNING] Chunk {chunk_id} failed",
                    flush=True,
                )

                print(
                    f"First error: "
                    f"{type(first_error).__name__}: "
                    f"{first_error}",
                    flush=True,
                )

                print(
                    f"Retry error: "
                    f"{type(retry_error).__name__}: "
                    f"{retry_error}",
                    flush=True,
                )

                continue

        # ------------------------------------------------------
        # Entities
        # ------------------------------------------------------

        entity_lookup = {}

        for extracted_entity in (
            extraction.entities
        ):

            name = (
                extracted_entity.name.strip()
            )

            if not name:
                continue

            entity_type_name = (
                normalize_entity_type(
                    extracted_entity.entity_type
                )
            )

            try:
                entity_type = EntityType(
                    entity_type_name
                )
            except ValueError:
                entity_type = (
                    EntityType.CONDITION
                )

            entity = create_entity(
                name=name,
                entity_type=entity_type,
                chunk_id=chunk_id,
                citation=citation,
                aliases=(
                    extracted_entity.aliases
                ),
                confidence=0.85,
            )

            graph.add_entity(entity)

            entity_lookup[
                name.lower()
            ] = entity

            for alias in (
                extracted_entity.aliases
            ):
                entity_lookup[
                    alias.lower()
                ] = entity

        # ------------------------------------------------------
        # Relations
        # ------------------------------------------------------

        for extracted_relation in (
            extraction.relations
        ):

            source_name = (
                extracted_relation.source.strip()
            )

            target_name = (
                extracted_relation.target.strip()
            )

            if not source_name or not target_name:
                continue

            source_entity = (
                entity_lookup.get(
                    source_name.lower()
                )
            )

            target_entity = (
                entity_lookup.get(
                    target_name.lower()
                )
            )

            if (
                source_entity is None
                or target_entity is None
            ):
                continue

            if (
                source_entity.entity_id
                == target_entity.entity_id
            ):
                continue

            relation_name = (
                normalize_relation_type(
                    extracted_relation.relation
                )
            )

            if relation_name is None:
                continue

            try:

                relation_type = RelationType(
                    relation_name
                )

            except ValueError:

                continue

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
                evidence_text=(
                    extracted_relation.evidence
                ),
                confidence=(
                    extracted_relation.confidence
                ),
            )

            graph.add_relation(
                relation
            )

        # ------------------------------------------------------
        # Progress
        # ------------------------------------------------------

        if (
            index == 1
            or index % 10 == 0
            or index == len(chunk_records)
        ):

            print(
                f"Processed {index}/"
                f"{len(chunk_records)} "
                f"| entities={graph.entity_count()} "
                f"| relations={graph.relation_count()} "
                f"| errors={error_count} "
                f"| retries={retry_count}",
                flush=True,
            )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    print(
        "\nSaving graph...",
        flush=True,
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            graph.to_dict(),
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    print(
        "\n" + "=" * 70,
        flush=True,
    )

    print(
        "GRAPH CONSTRUCTION COMPLETED",
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )

    print(
        f"Chunks processed : "
        f"{len(chunk_records)}",
        flush=True,
    )

    print(
        f"Entities created : "
        f"{graph.entity_count()}",
        flush=True,
    )

    print(
        f"Relations created: "
        f"{graph.relation_count()}",
        flush=True,
    )

    print(
        f"Graph nodes      : "
        f"{graph.node_count()}",
        flush=True,
    )

    print(
        f"Graph edges      : "
        f"{graph.edge_count()}",
        flush=True,
    )

    print(
        f"Extraction errors: "
        f"{error_count}",
        flush=True,
    )

    print(
        f"Retries          : "
        f"{retry_count}",
        flush=True,
    )

    print(
        f"Output           : "
        f"{OUTPUT_FILE}",
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )


def main():

    try:
        build_graph()

    except BaseException as exc:

        print(
            "\n" + "=" * 70,
            flush=True,
        )

        print(
            "GRAPH CONSTRUCTION FAILED",
            flush=True,
        )

        print(
            "=" * 70,
            flush=True,
        )

        print(
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

        traceback.print_exc()

        raise


if __name__ == "__main__":
    main()