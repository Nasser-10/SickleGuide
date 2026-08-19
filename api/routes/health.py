from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter

from src.generation.llm import create_rag_engine


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


_rag_engine = None


def get_rag_engine():
    global _rag_engine

    if _rag_engine is None:
        _rag_engine = create_rag_engine()

    return _rag_engine


@router.get("")
def health():
    """
    Lightweight service health check.

    Does not force the heavy RAG stack to load.
    """

    root_dir = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    chunks_file = (
        root_dir
        / "data"
        / "processed"
        / "chunks.json"
    )

    graph_file = (
        root_dir
        / "data"
        / "processed"
        / "graph.json"
    )

    chroma_dir = (
        root_dir
        / "data"
        / "processed"
        / "chroma"
    )

    return {
        "status": "ok",
        "service": "SickleGuide",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "pipeline": {
            "chunks_exists": chunks_file.exists(),
            "graph_exists": graph_file.exists(),
            "chroma_exists": chroma_dir.exists(),
        },
        "rag_engine": {
            "initialized": (
                _rag_engine is not None
                and getattr(
                    _rag_engine,
                    "_initialized",
                    False,
                )
            ),
            "model": (
                _rag_engine.model_name
                if _rag_engine is not None
                else "qwen2.5:7b"
            ),
        },
    }