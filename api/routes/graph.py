import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException


router = APIRouter(
    prefix="/graph",
    tags=["Knowledge Graph"],
)


ROOT_DIR = Path(__file__).resolve().parents[2]

GRAPH_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "graph.json"
)


CORE_ENTITY_TYPES = {
    "disease",
    "drug",
    "treatment",
    "procedure",
    "condition",
    "symptom",
    "outcome",
    "lab test",
    "organization",
    "therapy",
    "anatomical location",
    "attribute",
}


def load_graph_data():
    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            f"Graph file not found: {GRAPH_FILE}"
        )

    with GRAPH_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def normalize_entity(item: dict) -> dict:
    return {
        "id": str(
            item.get(
                "entity_id",
                item.get("id", ""),
            )
        ),
        "name": str(
            item.get(
                "name",
                item.get("label", ""),
            )
        ),
        "type": str(
            item.get(
                "entity_type",
                item.get("type", "unknown"),
            )
        ).lower(),
        "confidence": item.get("confidence"),
        "aliases": item.get(
            "aliases",
            [],
        ),
        "description": item.get(
            "description",
            "",
        ),
        "source_chunks": item.get(
            "source_chunks",
            [],
        ),
        "citations": item.get(
            "citations",
            [],
        ),
    }


def normalize_relation(item: dict) -> dict:
    return {
        "id": str(
            item.get(
                "relation_id",
                "",
            )
        ),
        "source": str(
            item.get(
                "source_entity_id",
                item.get("source", ""),
            )
        ),
        "target": str(
            item.get(
                "target_entity_id",
                item.get("target", ""),
            )
        ),
        "relation": str(
            item.get(
                "relation_type",
                item.get(
                    "relation",
                    "related_to",
                ),
            )
        ).lower(),
        "confidence": item.get(
            "confidence"
        ),
        "evidence": item.get(
            "evidence_text",
            "",
        ),
        "citations": item.get(
            "citations",
            [],
        ),
    }


def build_adjacency(
    relations: list[dict],
) -> dict[str, set]:
    adjacency: dict[str, set] = {}

    for relation in relations:
        source = relation["source"]
        target = relation["target"]

        if not source or not target:
            continue

        adjacency.setdefault(
            source,
            set(),
        ).add(target)

        adjacency.setdefault(
            target,
            set(),
        ).add(source)

    return adjacency


def find_scd_seed_ids(
    entities: list[dict],
) -> list[str]:
    seeds = []

    for entity in entities:
        entity_id = entity["id"].lower()
        name = entity["name"].strip().lower()

        if (
            entity_id
            == "disease:sickle_cell_disease"
        ):
            seeds.append(entity["id"])
            continue

        if name in {
            "sickle cell disease",
            "sickle-cell disease",
            "sickle cell anemia",
            "sickle-cell anemia",
            "sickle cell anaemia",
        }:
            seeds.append(entity["id"])

    return list(dict.fromkeys(seeds))


def build_clinical_overview(
    entities: list[dict],
    relations: list[dict],
    max_nodes: int,
) -> tuple[list[dict], list[dict]]:
    """
    Create a clinically meaningful initial graph around SCD
    instead of displaying an arbitrary slice of graph.json.
    """

    entity_map = {
        entity["id"]: entity
        for entity in entities
    }

    adjacency = build_adjacency(
        relations
    )

    degree = {
        node_id: len(neighbors)
        for node_id, neighbors
        in adjacency.items()
    }

    seed_ids = find_scd_seed_ids(
        entities
    )

    if not seed_ids:
        seed_ids = [
            entity["id"]
            for entity in sorted(
                entities,
                key=lambda item: degree.get(
                    item["id"],
                    0,
                ),
                reverse=True,
            )[:3]
        ]

    selected = set(seed_ids)

    # First hop
    for seed_id in seed_ids:
        selected.update(
            adjacency.get(
                seed_id,
                set(),
            )
        )

    # Second hop
    first_hop = list(selected)

    for node_id in first_hop:
        selected.update(
            adjacency.get(
                node_id,
                set(),
            )
        )

    # Keep the most connected nodes while always
    # retaining the SCD seed.
    ranked = sorted(
        selected,
        key=lambda node_id: (
            node_id not in seed_ids,
            -degree.get(
                node_id,
                0,
            ),
        ),
    )

    selected = set(
        ranked[:max_nodes]
    )

    output_entities = [
        entity_map[node_id]
        for node_id in selected
        if node_id in entity_map
    ]

    output_relations = [
        relation
        for relation in relations
        if (
            relation["source"]
            in selected
            and relation["target"]
            in selected
        )
    ]

    return (
        output_entities,
        output_relations,
    )


def build_keyword_view(
    entities: list[dict],
    relations: list[dict],
    keywords: set[str],
    max_nodes: int,
) -> tuple[list[dict], list[dict]]:
    """
    Build a focused graph around domain-specific keywords.
    """

    entity_map = {
        entity["id"]: entity
        for entity in entities
    }

    matching_ids = set()

    for entity in entities:
        text = (
            f"{entity['name']} "
            f"{' '.join(entity.get('aliases', []))}"
        ).lower()

        if any(
            keyword in text
            for keyword in keywords
        ):
            matching_ids.add(
                entity["id"]
            )

    if "disease:sickle_cell_disease" in entity_map:
        matching_ids.add(
            "disease:sickle_cell_disease"
        )

    adjacency = build_adjacency(
        relations
    )

    selected = set(
        matching_ids
    )

    for node_id in list(
        matching_ids
    ):
        selected.update(
            adjacency.get(
                node_id,
                set(),
            )
        )

    if len(selected) > max_nodes:
        degree = {
            node_id: len(
                adjacency.get(
                    node_id,
                    set(),
                )
            )
            for node_id in selected
        }

        selected = set(
            sorted(
                selected,
                key=lambda node_id:
                degree.get(
                    node_id,
                    0,
                ),
                reverse=True,
            )[:max_nodes]
        )

    output_entities = [
        entity_map[node_id]
        for node_id in selected
        if node_id in entity_map
    ]

    output_relations = [
        relation
        for relation in relations
        if (
            relation["source"] in selected
            and relation["target"] in selected
        )
    ]

    return (
        output_entities,
        output_relations,
    )


@router.get("")
def get_graph(
    view: str = "overview",
    entity_type: Optional[str] = None,
    relation_type: Optional[str] = None,
    max_nodes: int = 120,
):
    try:
        max_nodes = max(
            20,
            min(
                max_nodes,
                300,
            ),
        )

        data = load_graph_data()

        raw_entities = data.get(
            "entities",
            data.get(
                "nodes",
                [],
            ),
        )

        raw_relations = data.get(
            "relations",
            data.get(
                "edges",
                [],
            ),
        )

        entities = [
            normalize_entity(item)
            for item in raw_entities
            if isinstance(
                item,
                dict,
            )
        ]

        relations = [
            normalize_relation(item)
            for item in raw_relations
            if isinstance(
                item,
                dict,
            )
        ]

        view = (
            view.strip().lower()
            if view
            else "overview"
        )

        if view == "overview":
            entities, relations = (
                build_clinical_overview(
                    entities,
                    relations,
                    max_nodes,
                )
            )

        elif view == "treatments":
            entities, relations = (
                build_keyword_view(
                    entities,
                    relations,
                    {
                        "hydroxyurea",
                        "transfusion",
                        "exchange transfusion",
                        "treatment",
                        "therapy",
                        "drug",
                        "analgesic",
                    },
                    max_nodes,
                )
            )

        elif view == "complications":
            entities, relations = (
                build_keyword_view(
                    entities,
                    relations,
                    {
                        "acute chest syndrome",
                        "stroke",
                        "vaso-occlusive",
                        "pain",
                        "renal",
                        "pulmonary",
                        "iron overload",
                        "complication",
                    },
                    max_nodes,
                )
            )

        elif view == "transfusion":
            entities, relations = (
                build_keyword_view(
                    entities,
                    relations,
                    {
                        "transfusion",
                        "exchange",
                        "red blood cell",
                        "iron chelation",
                        "alloimmunization",
                    },
                    max_nodes,
                )
            )

        elif view == "pregnancy":
            entities, relations = (
                build_keyword_view(
                    entities,
                    relations,
                    {
                        "pregnancy",
                        "pregnant",
                        "maternal",
                        "antenatal",
                        "birth",
                        "fetal",
                        "neonatal",
                    },
                    max_nodes,
                )
            )

        else:
            entities, relations = (
                build_clinical_overview(
                    entities,
                    relations,
                    max_nodes,
                )
            )

        # ----------------------------------------------------
        # Entity type filter
        # ----------------------------------------------------

        if (
            entity_type
            and entity_type.lower()
            != "all"
        ):
            selected_type = (
                entity_type.strip().lower()
            )

            entities = [
                entity
                for entity in entities
                if entity["type"]
                == selected_type
            ]

        entity_ids = {
            entity["id"]
            for entity in entities
        }

        # ----------------------------------------------------
        # Relation filter
        # ----------------------------------------------------

        if (
            relation_type
            and relation_type.lower()
            != "all"
        ):
            selected_relation = (
                relation_type.strip().lower()
            )

            relations = [
                relation
                for relation in relations
                if relation["relation"]
                == selected_relation
            ]

        relations = [
            relation
            for relation in relations
            if (
                relation["source"]
                in entity_ids
                and relation["target"]
                in entity_ids
            )
        ]

        available_entity_types = sorted(
            {
                entity["type"]
                for entity in (
                    normalize_entity(item)
                    for item in raw_entities
                    if isinstance(
                        item,
                        dict,
                    )
                )
            }
            | CORE_ENTITY_TYPES
        )

        available_relations = sorted(
            {
                relation["relation"]
                for relation in (
                    normalize_relation(item)
                    for item in raw_relations
                    if isinstance(
                        item,
                        dict,
                    )
                )
            }
        )

        return {
            "view": view,
            "nodes": entities,
            "edges": relations,
            "total_nodes": len(
                entities
            ),
            "total_edges": len(
                relations
            ),
            "available_entity_types":
                available_entity_types,
            "available_relations":
                available_relations,
            "available_views": [
                "overview",
                "treatments",
                "complications",
                "transfusion",
                "pregnancy",
            ],
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not load knowledge graph: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc