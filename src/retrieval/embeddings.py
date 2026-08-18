from typing import List, Optional
import os

from langchain_core.documents import Document


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL_NAME = "BAAI/bge-m3"

DEFAULT_DEVICE = "cuda"

DEFAULT_EMBEDDING_DIMENSION = 1024


# ============================================================
# Hugging Face offline configuration
# ============================================================

def _enable_local_huggingface_mode() -> None:
    """
    Force Hugging Face libraries to use the local cache.

    SickleGuide already downloaded BGE-M3 successfully, so
    retrieval should not depend on an active internet connection.
    """

    os.environ.setdefault(
        "HF_HUB_OFFLINE",
        "1",
    )

    os.environ.setdefault(
        "TRANSFORMERS_OFFLINE",
        "1",
    )


# ============================================================
# Device
# ============================================================

def _resolve_device(
    device: str,
) -> str:
    """
    Resolve requested device.
    """

    if (
        device == "cuda"
        and _cuda_available()
    ):
        return "cuda"

    return "cpu"


def _cuda_available() -> bool:
    """
    Check CUDA availability without importing
    any extra package.
    """

    try:
        import torch

        return bool(
            torch.cuda.is_available()
        )

    except Exception:
        return False


# ============================================================
# Embedding model
# ============================================================

def get_embedding_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = DEFAULT_DEVICE,
    local_files_only: bool = True,
):
    """
    Create the BGE-M3 LangChain embedding model.

    Important:
        local_files_only=True prevents Hugging Face network
        requests during retrieval/inference.

    Returns:
        HuggingFaceEmbeddings instance.
    """

    _enable_local_huggingface_mode()

    from langchain_huggingface import (
        HuggingFaceEmbeddings,
    )

    resolved_device = _resolve_device(
        device
    )

    print(
        f"Embedding model: {model_name}",
        flush=True,
    )

    print(
        f"Embedding device: {resolved_device}",
        flush=True,
    )

    if local_files_only:

        print(
            "Embedding mode: local cache only",
            flush=True,
        )

    model_kwargs = {
        "device": resolved_device,
        "local_files_only": local_files_only,
    }

    encode_kwargs = {
        "normalize_embeddings": True,
        "batch_size": 32,
    }

    try:

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    except Exception as exc:

        raise RuntimeError(
            "\n"
            "Failed to load the local embedding model.\n"
            f"Model: {model_name}\n"
            f"Device: {resolved_device}\n"
            f"Local files only: {local_files_only}\n\n"
            "The model must already exist in the "
            "Hugging Face local cache.\n"
            "BGE-M3 was successfully loaded earlier "
            "in this project, so this usually indicates "
            "a cache/configuration mismatch."
        ) from exc

    return embeddings


# ============================================================
# Document embeddings
# ============================================================

def embed_documents(
    documents: List[Document],
    embeddings=None,
) -> List[List[float]]:
    """
    Generate embeddings for LangChain Documents.
    """

    if not isinstance(
        documents,
        list,
    ):
        raise TypeError(
            "documents must be a list"
        )

    if embeddings is None:
        embeddings = get_embedding_model()

    texts = [
        document.page_content
        for document in documents
    ]

    if not texts:
        return []

    return embeddings.embed_documents(
        texts
    )


# ============================================================
# Query embedding
# ============================================================

def embed_query(
    query: str,
    embeddings=None,
) -> List[float]:
    """
    Generate a BGE-M3 embedding for a query.
    """

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

    if embeddings is None:
        embeddings = get_embedding_model()

    return embeddings.embed_query(
        query
    )


# ============================================================
# Dimension
# ============================================================

def get_embedding_dimension(
    embeddings=None,
) -> int:
    """
    Return BGE-M3 embedding dimension.
    """

    if embeddings is None:
        embeddings = get_embedding_model()

    vector = embeddings.embed_query(
        "Sickle cell disease"
    )

    dimension = len(
        vector
    )

    return dimension


# ============================================================
# Model validation
# ============================================================

def validate_embedding_model(
    embeddings=None,
) -> dict:
    """
    Run a small local validation.

    This is useful before initializing Chroma.
    """

    if embeddings is None:
        embeddings = get_embedding_model()

    test_text = (
        "Hydroxyurea is used in "
        "sickle cell disease management."
    )

    vector = embeddings.embed_query(
        test_text
    )

    dimension = len(
        vector
    )

    return {
        "model": DEFAULT_MODEL_NAME,
        "dimension": dimension,
        "expected_dimension": (
            DEFAULT_EMBEDDING_DIMENSION
        ),
        "dimension_ok": (
            dimension
            == DEFAULT_EMBEDDING_DIMENSION
        ),
        "device": getattr(
            embeddings,
            "_client",
            None,
        ).device.type
        if getattr(
            embeddings,
            "_client",
            None,
        ) is not None
        and hasattr(
            getattr(
                embeddings,
                "_client",
                None,
            ),
            "device",
        )
        else "unknown",
    }