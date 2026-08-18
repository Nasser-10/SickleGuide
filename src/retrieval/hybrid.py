from typing import Dict, List, Tuple

from langchain_core.documents import Document

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.vector_store import VectorStore
from src.graph.graph_retriever import GraphRetriever


DEFAULT_DENSE_K = 15
DEFAULT_BM25_K = 15
DEFAULT_GRAPH_K = 15
DEFAULT_FINAL_K = 10
DEFAULT_RRF_K = 60


# ============================================================
# Query Expansion
# ============================================================

MEDICAL_EXPANSIONS = {
    "acute chest syndrome": [
        "acute chest syndrome",
        "ACS",
    ],
    "acs": [
        "acute chest syndrome",
        "ACS",
    ],
    "treatment": [
        "treatment",
        "treatments",
        "therapy",
        "interventions",
        "management",
    ],
    "treatments": [
        "treatment",
        "treatments",
        "therapy",
        "interventions",
        "management",
    ],
    "evaluated": [
        "evaluated",
        "interventions",
        "treatment",
        "therapy",
    ],
    "recommended": [
        "recommended",
        "suggested",
        "recommendation",
        "guideline",
    ],
    "blood transfusion": [
        "blood transfusion",
        "transfusion",
        "exchange transfusion",
        "red blood cell transfusion",
    ],
}


def expand_query(
    query: str,
) -> str:
    """
    Expand a clinical query for retrieval only.

    The expansion is never passed to generation as medical evidence.
    """

    query = query.strip()

    terms = [
        query
    ]

    normalized = query.lower()

    for trigger, expansions in (
        MEDICAL_EXPANSIONS.items()
    ):
        if trigger in normalized:
            for expansion in expansions:
                if expansion.lower() not in normalized:
                    terms.append(
                        expansion
                    )

    # SCD terminology normalization.
    if (
        "sickle cell disease"
        in normalized
        and "SCD" not in query
    ):
        terms.append(
            "SCD"
        )

    if (
        "acute chest syndrome"
        in normalized
        and "ACS" not in query
    ):
        terms.append(
            "ACS"
        )

    # Deduplicate while preserving order.
    seen = set()
    final_terms = []

    for term in terms:

        normalized_term = (
            term.strip().lower()
        )

        if not normalized_term:
            continue

        if normalized_term in seen:
            continue

        seen.add(
            normalized_term
        )

        final_terms.append(
            term.strip()
        )

    return " ".join(
        final_terms
    )


# ============================================================
# Dense + BM25 Hybrid
# ============================================================

class HybridRetriever:
    """
    Dense + BM25 retriever.

    Kept for backward compatibility.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_store: VectorStore,
        dense_k: int = DEFAULT_DENSE_K,
        bm25_k: int = DEFAULT_BM25_K,
        final_k: int = DEFAULT_FINAL_K,
        rrf_k: int = DEFAULT_RRF_K,
    ):
        self.bm25_retriever = (
            bm25_retriever
        )

        self.vector_store = (
            vector_store
        )

        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.final_k = final_k
        self.rrf_k = rrf_k

    @staticmethod
    def _document_key(
        document: Document,
    ) -> str:

        chunk_id = (
            document.metadata.get(
                "chunk_id"
            )
        )

        if chunk_id is not None:
            return f"chunk:{chunk_id}"

        return (
            document.metadata.get(
                "source",
                "unknown",
            )
            + "|"
            + str(
                document.metadata.get(
                    "page_number",
                    "unknown",
                )
            )
            + "|"
            + document.page_content[:100]
        )

    def retrieve(
        self,
        query: str,
        final_k: int | None = None,
    ) -> List[Document]:

        if not query or not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        final_k = (
            final_k
            if final_k is not None
            else self.final_k
        )

        expanded_query = expand_query(
            query
        )

        dense_results = (
            self.vector_store.similarity_search(
                expanded_query,
                k=self.dense_k,
            )
        )

        bm25_results = (
            self.bm25_retriever.search(
                expanded_query,
                k=self.bm25_k,
            )
        )

        scores = {}
        documents = {}

        for rank, document in enumerate(
            dense_results,
            start=1,
        ):

            key = self._document_key(
                document
            )

            documents[key] = document

            scores[key] = (
                scores.get(
                    key,
                    0.0,
                )
                + 1.0
                / (
                    self.rrf_k
                    + rank
                )
            )

        for rank, document in enumerate(
            bm25_results,
            start=1,
        ):

            key = self._document_key(
                document
            )

            documents[key] = document

            scores[key] = (
                scores.get(
                    key,
                    0.0,
                )
                + 1.0
                / (
                    self.rrf_k
                    + rank
                )
            )

        ranked = sorted(
            scores.keys(),
            key=lambda key: scores[key],
            reverse=True,
        )

        results = []

        for rank, key in enumerate(
            ranked[:final_k],
            start=1,
        ):

            document = documents[key]

            metadata = (
                document.metadata.copy()
            )

            metadata[
                "hybrid_score"
            ] = float(
                scores[key]
            )

            metadata[
                "retrieval_method"
            ] = "hybrid_rrf"

            metadata[
                "retrieval_rank"
            ] = rank

            metadata[
                "expanded_query"
            ] = expanded_query

            results.append(
                Document(
                    page_content=(
                        document.page_content
                    ),
                    metadata=metadata,
                )
            )

        return results


# ============================================================
# Unified Dense + BM25 + Graph
# ============================================================

class HybridGraphRetriever:
    """
    Unified retrieval layer:

        Original Query
             ↓
        Query Expansion
             ↓
      ┌──────┼──────┐
      ↓      ↓      ↓
    Dense   BM25   Graph
      │      │      │
      └──────┼──────┘
             ↓
            RRF
             ↓
        Candidate set

    Reranking stays in reranker.py.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_store: VectorStore,
        graph_retriever: GraphRetriever,
        dense_k: int = DEFAULT_DENSE_K,
        bm25_k: int = DEFAULT_BM25_K,
        graph_k: int = DEFAULT_GRAPH_K,
        final_k: int = DEFAULT_FINAL_K,
        rrf_k: int = DEFAULT_RRF_K,
    ):

        if dense_k <= 0:
            raise ValueError(
                "dense_k must be > 0"
            )

        if bm25_k <= 0:
            raise ValueError(
                "bm25_k must be > 0"
            )

        if graph_k <= 0:
            raise ValueError(
                "graph_k must be > 0"
            )

        if final_k <= 0:
            raise ValueError(
                "final_k must be > 0"
            )

        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be > 0"
            )

        self.bm25_retriever = (
            bm25_retriever
        )

        self.vector_store = (
            vector_store
        )

        self.graph_retriever = (
            graph_retriever
        )

        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.graph_k = graph_k
        self.final_k = final_k
        self.rrf_k = rrf_k

    # ========================================================
    # Identity
    # ========================================================

    @staticmethod
    def _document_key(
        document: Document,
    ) -> str:

        chunk_id = (
            document.metadata.get(
                "chunk_id"
            )
        )

        if chunk_id is not None:
            return f"chunk:{chunk_id}"

        source = document.metadata.get(
            "source",
            "unknown",
        )

        page = document.metadata.get(
            "page_number",
            "unknown",
        )

        return (
            f"{source}|"
            f"{page}|"
            f"{document.page_content[:120]}"
        )

    # ========================================================
    # Retrieval
    # ========================================================

    def retrieve(
        self,
        query: str,
        final_k: int | None = None,
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

        final_k = (
            final_k
            if final_k is not None
            else self.final_k
        )

        expanded_query = expand_query(
            query
        )

        print(
            "\n[Unified Retrieval]",
            flush=True,
        )

        print(
            f"  Original query: "
            f"{query}",
            flush=True,
        )

        print(
            f"  Expanded query: "
            f"{expanded_query}",
            flush=True,
        )

        # -----------------------------------------------------
        # Dense
        # -----------------------------------------------------

        dense_results = (
            self.vector_store.similarity_search(
                expanded_query,
                k=self.dense_k,
            )
        )

        print(
            f"  Dense: "
            f"{len(dense_results)}",
            flush=True,
        )

        # -----------------------------------------------------
        # BM25
        # -----------------------------------------------------

        bm25_results = (
            self.bm25_retriever.search(
                expanded_query,
                k=self.bm25_k,
            )
        )

        print(
            f"  BM25: "
            f"{len(bm25_results)}",
            flush=True,
        )

        # -----------------------------------------------------
        # Graph
        # -----------------------------------------------------

        graph_results = (
            self.graph_retriever.retrieve(
                query,
                max_hops=2,
                max_entities=20,
                max_documents=self.graph_k,
            )
        )

        print(
            f"  Graph: "
            f"{len(graph_results)}",
            flush=True,
        )

        # -----------------------------------------------------
        # RRF
        # -----------------------------------------------------

        scores: Dict[
            str,
            float,
        ] = {}

        documents: Dict[
            str,
            Document,
        ] = {}

        channels = [
            (
                "dense",
                dense_results,
            ),
            (
                "bm25",
                bm25_results,
            ),
            (
                "graph",
                graph_results,
            ),
        ]

        for channel_name, results in channels:

            for rank, document in enumerate(
                results,
                start=1,
            ):

                key = (
                    self._document_key(
                        document
                    )
                )

                documents[key] = document

                scores[key] = (
                    scores.get(
                        key,
                        0.0,
                    )
                    + 1.0
                    / (
                        self.rrf_k
                        + rank
                    )
                )

        ranked_keys = sorted(
            scores.keys(),
            key=lambda key: scores[key],
            reverse=True,
        )

        # -----------------------------------------------------
        # Candidate set
        # -----------------------------------------------------

        results = []

        for rank, key in enumerate(
            ranked_keys[
                :final_k
            ],
            start=1,
        ):

            document = documents[key]

            metadata = (
                document.metadata.copy()
            )

            metadata[
                "unified_rrf_score"
            ] = float(
                scores[key]
            )

            metadata[
                "unified_retrieval_rank"
            ] = rank

            metadata[
                "retrieval_method"
            ] = (
                "dense_bm25_graph_rrf"
            )

            metadata[
                "original_query"
            ] = query

            metadata[
                "expanded_query"
            ] = expanded_query

            results.append(
                Document(
                    page_content=(
                        document.page_content
                    ),
                    metadata=metadata,
                )
            )

        print(
            f"  Fused: {len(results)}",
            flush=True,
        )

        return results

    def retrieve_with_scores(
        self,
        query: str,
        final_k: int | None = None,
    ) -> List[
        Tuple[Document, float]
    ]:

        documents = self.retrieve(
            query=query,
            final_k=final_k,
        )

        return [
            (
                document,
                float(
                    document.metadata.get(
                        "unified_rrf_score",
                        0.0,
                    )
                ),
            )
            for document in documents
        ]


# ============================================================
# Factories
# ============================================================

def create_hybrid_retriever(
    bm25_retriever: BM25Retriever,
    vector_store: VectorStore,
    dense_k: int = DEFAULT_DENSE_K,
    bm25_k: int = DEFAULT_BM25_K,
    final_k: int = DEFAULT_FINAL_K,
    rrf_k: int = DEFAULT_RRF_K,
) -> HybridRetriever:

    return HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_store=vector_store,
        dense_k=dense_k,
        bm25_k=bm25_k,
        final_k=final_k,
        rrf_k=rrf_k,
    )


def create_hybrid_graph_retriever(
    bm25_retriever: BM25Retriever,
    vector_store: VectorStore,
    graph_retriever: GraphRetriever,
    dense_k: int = DEFAULT_DENSE_K,
    bm25_k: int = DEFAULT_BM25_K,
    graph_k: int = DEFAULT_GRAPH_K,
    final_k: int = DEFAULT_FINAL_K,
    rrf_k: int = DEFAULT_RRF_K,
) -> HybridGraphRetriever:

    return HybridGraphRetriever(
        bm25_retriever=bm25_retriever,
        vector_store=vector_store,
        graph_retriever=graph_retriever,
        dense_k=dense_k,
        bm25_k=bm25_k,
        graph_k=graph_k,
        final_k=final_k,
        rrf_k=rrf_k,
    )