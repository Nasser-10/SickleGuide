from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from langchain_core.documents import Document

from src.generation.llm import create_rag_engine


router = APIRouter(
    prefix="/search",
    tags=["Search"],
)


# ============================================================
# Shared engine
# ============================================================

_rag_engine = None


def get_rag_engine():
    """
    Reuse the same RAG engine instance.
    """

    global _rag_engine

    if _rag_engine is None:
        _rag_engine = create_rag_engine()

    return _rag_engine


# ============================================================
# Models
# ============================================================

class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )


class SearchResult(BaseModel):
    rank: int
    content: str
    source: str
    page_number: Optional[int] = None
    citation: str
    score: Optional[float] = None
    retrieval_method: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]


# ============================================================
# Search endpoint
# ============================================================

@router.post(
    "",
    response_model=SearchResponse,
)
def search(
    request: SearchRequest,
):
    """
    Run retrieval + reranking without generation.
    """

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    try:

        engine = get_rag_engine()

        engine.initialize()

        candidates = (
            engine.unified_retriever.retrieve(
                query,
                final_k=engine.candidate_k,
            )
        )

        reranked = (
            engine.reranker.rerank(
                query=query,
                documents=candidates,
                top_k=request.top_k,
            )
        )

        results = []

        for rank, document in enumerate(
            reranked,
            start=1,
        ):

            metadata = (
                document.metadata
                or {}
            )

            citation = metadata.get(
                "citation",
                (
                    f"{metadata.get('source', 'Unknown source')}"
                    f" — Page "
                    f"{metadata.get('page_number', 'Unknown')}"
                ),
            )

            score = metadata.get(
                "reranker_score"
            )

            results.append(
                SearchResult(
                    rank=rank,
                    content=(
                        document.page_content[:3000]
                    ),
                    source=str(
                        metadata.get(
                            "source",
                            "Unknown source",
                        )
                    ),
                    page_number=metadata.get(
                        "page_number"
                    ),
                    citation=str(
                        citation
                    ),
                    score=(
                        float(score)
                        if score is not None
                        else None
                    ),
                    retrieval_method=metadata.get(
                        "retrieval_method"
                    ),
                )
            )

        return SearchResponse(
            query=query,
            results=results,
        )

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Search failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc