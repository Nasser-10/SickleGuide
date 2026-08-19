import json
import shutil
import time

from pathlib import Path
from typing import List

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from langchain_core.documents import Document

from src.ingestion.pdf_loader import load_pdf
from src.ingestion.cleaner import clean_documents
from src.ingestion.markdown_converter import (
    convert_documents_to_markdown,
)
from src.chunking.chunker import create_chunks
from src.chunking.metadata import (
    enrich_chunk_metadata,
)
from src.retrieval.bm25 import (
    create_bm25_retriever,
)
from src.retrieval.hybrid import (
    create_hybrid_graph_retriever,
)


router = APIRouter(
    prefix="/data",
    tags=["Knowledge Base"],
)


ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

RAW_DIR = (
    ROOT_DIR
    / "data"
    / "raw"
)

CHUNKS_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "chunks.json"
)


def safe_filename(
    filename: str,
) -> str:

    name = Path(
        filename
    ).name

    timestamp = int(
        time.time()
    )

    return (
        f"{timestamp}_{name}"
    )


def load_saved_chunks():

    if not CHUNKS_FILE.exists():
        return []

    with CHUNKS_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_chunks(
    chunks,
):

    CHUNKS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with CHUNKS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )


def refresh_runtime_engine(
    new_documents: List[Document],
):

    from api.routes.chat import (
        get_rag_engine,
    )

    engine = get_rag_engine()

    engine.initialize()

    engine.documents.extend(
        new_documents
    )

    engine.bm25 = (
        create_bm25_retriever(
            engine.documents
        )
    )

    engine.unified_retriever = (
        create_hybrid_graph_retriever(
            bm25_retriever=engine.bm25,
            vector_store=engine.vector_store,
            graph_retriever=engine.graph_retriever,
            dense_k=engine.dense_k,
            bm25_k=engine.bm25_k,
            graph_k=engine.graph_k,
            final_k=engine.candidate_k,
        )
    )


@router.get("")
def list_data():

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks = load_saved_chunks()

    source_counts = {}

    for item in chunks:

        metadata = item.get(
            "metadata",
            {},
        )

        source = metadata.get(
            "source",
            "Unknown",
        )

        source_counts[source] = (
            source_counts.get(
                source,
                0,
            )
            + 1
        )

    files = []

    for path in sorted(
        RAW_DIR.glob("*.pdf")
    ):

        files.append(
            {
                "name": path.name,
                "size_mb": round(
                    path.stat().st_size
                    / 1024
                    / 1024,
                    2,
                ),
                "chunks": source_counts.get(
                    path.name,
                    0,
                ),
            }
        )

    return {
        "files": files,
        "total_files": len(files),
        "total_chunks": len(chunks),
    }


@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Missing filename.",
        )

    if not file.filename.lower().endswith(
        ".pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_name = safe_filename(
        file.filename
    )

    output_path = (
        RAW_DIR
        / saved_name
    )

    try:

        with output_path.open(
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer,
            )

        documents = load_pdf(
            str(output_path)
        )

        if not documents:
            raise RuntimeError(
                "PDF produced no documents."
            )

        documents = clean_documents(
            documents
        )

        documents = (
            convert_documents_to_markdown(
                documents
            )
        )

        chunks = create_chunks(
            documents
        )

        chunks = enrich_chunk_metadata(
            chunks
        )

        if not chunks:
            raise RuntimeError(
                "PDF produced no chunks."
            )

        timestamp = int(
            time.time()
        )

        serialized_new = []

        new_documents = []

        for local_index, chunk in enumerate(
            chunks
        ):

            metadata = (
                chunk.metadata.copy()
            )

            unique_chunk_id = (
                f"upload_{timestamp}_"
                f"{local_index}"
            )

            metadata["source"] = (
                saved_name
            )

            metadata["chunk_id"] = (
                unique_chunk_id
            )

            metadata[
                "document_key"
            ] = (
                f"{saved_name}::"
                f"{unique_chunk_id}"
            )

            metadata[
                "uploaded"
            ] = True

            new_chunk = Document(
                page_content=(
                    chunk.page_content
                ),
                metadata=metadata,
            )

            new_documents.append(
                new_chunk
            )

            serialized_new.append(
                {
                    "page_content":
                        new_chunk.page_content,
                    "metadata":
                        metadata,
                }
            )

        # --------------------------------------------
        # Persist chunks
        # --------------------------------------------

        existing = load_saved_chunks()

        existing.extend(
            serialized_new
        )

        save_chunks(
            existing
        )

        # --------------------------------------------
        # Persistent vector indexing
        # --------------------------------------------

        from src.retrieval.vector_store import (
            create_vector_store,
        )

        vector_store = (
            create_vector_store()
        )

        vector_store.add_documents(
            new_documents
        )

        # --------------------------------------------
        # Refresh current runtime
        # --------------------------------------------

        refresh_runtime_engine(
            new_documents
        )

        return {
            "success": True,
            "filename": saved_name,
            "pages": len(documents),
            "chunks_added": len(
                new_documents
            ),
            "total_chunks": len(
                existing
            ),
            "message": (
                "PDF processed and added "
                "to the SickleGuide knowledge base. "
                "It is immediately searchable through "
                "dense and BM25 retrieval."
            ),
        }

    except Exception as exc:

        if output_path.exists():
            output_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=(
                "PDF ingestion failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc