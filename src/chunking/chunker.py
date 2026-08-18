from typing import List

from langchain_core.documents import Document


# ============================================================
# Chunking configuration
# ============================================================

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200


# ============================================================
# Text helpers
# ============================================================

def _split_by_separators(
    text: str,
    chunk_size: int,
) -> List[str]:
    """
    Recursively split text using semantic separators.

    Order:
        paragraph
        line
        sentence
        word
        character
    """

    if len(text) <= chunk_size:
        return [text.strip()]

    separators = [
        "\n\n",
        "\n",
        ". ",
        "? ",
        "! ",
        "; ",
        ", ",
        " ",
        "",
    ]

    for separator in separators:

        if separator == "":
            # Final character-level fallback.
            return [
                text[i:i + chunk_size].strip()
                for i in range(
                    0,
                    len(text),
                    chunk_size,
                )
                if text[i:i + chunk_size].strip()
            ]

        if separator not in text:
            continue

        parts = [
            part.strip()
            for part in text.split(separator)
            if part.strip()
        ]

        if len(parts) <= 1:
            continue

        chunks = []
        current = ""

        for part in parts:

            candidate = (
                f"{current}{separator}{part}"
                if current
                else part
            )

            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(
                    current.strip()
                )

            if len(part) <= chunk_size:
                current = part
            else:
                nested_chunks = (
                    _split_by_separators(
                        part,
                        chunk_size,
                    )
                )

                chunks.extend(
                    nested_chunks[:-1]
                )

                current = (
                    nested_chunks[-1]
                    if nested_chunks
                    else ""
                )

        if current:
            chunks.append(
                current.strip()
            )

        return chunks

    return [text.strip()]


def _apply_overlap(
    chunks: List[str],
    overlap: int,
) -> List[str]:
    """
    Add character overlap between adjacent chunks.

    Overlap is intentionally conservative to avoid
    huge duplication in medical documents.
    """

    if (
        overlap <= 0
        or len(chunks) <= 1
    ):
        return chunks

    result = [chunks[0]]

    for index in range(1, len(chunks)):

        previous = chunks[index - 1]
        current = chunks[index]

        overlap_text = previous[
            max(0, len(previous) - overlap):
        ].strip()

        if overlap_text:
            merged = (
                overlap_text
                + "\n\n"
                + current
            )
        else:
            merged = current

        result.append(
            merged.strip()
        )

    return result


def _split_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[str]:
    """
    Create recursive, overlap-aware chunks.
    """

    if not text.strip():
        return []

    chunks = _split_by_separators(
        text,
        chunk_size,
    )

    chunks = [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]

    return _apply_overlap(
        chunks,
        chunk_overlap,
    )


# ============================================================
# Public API
# ============================================================

def create_chunks(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Create RAG chunks from LangChain Documents.

    Important:
    - Uses LangChain Document objects.
    - Preserves source/page metadata.
    - Does not import langchain-text-splitters.
    - Never crosses PDF page boundaries.
    - Keeps citations attached to every chunk.
    """

    if not isinstance(
        documents,
        list,
    ):
        raise TypeError(
            "documents must be a list"
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative"
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    chunks: List[Document] = []

    total_documents = len(
        documents
    )

    print(
        f"      Starting chunking: "
        f"{total_documents} documents",
        flush=True,
    )

    for document_index, document in enumerate(
        documents,
        start=1,
    ):

        if not isinstance(
            document,
            Document,
        ):
            raise TypeError(
                "Every item must be a "
                "LangChain Document"
            )

        text = (
            document.page_content
            or ""
        ).strip()

        if not text:
            continue

        split_chunks = _split_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for local_index, chunk_text in enumerate(
            split_chunks
        ):

            metadata = (
                document.metadata.copy()
            )

            metadata.update(
                {
                    "document_index": (
                        document_index
                    ),
                    "chunk_local_index": (
                        local_index
                    ),
                    "chunk_size": (
                        len(chunk_text)
                    ),
                }
            )

            source = metadata.get(
                "source",
                "unknown",
            )

            page = metadata.get(
                "page_number",
                "unknown",
            )

            metadata["citation"] = (
                f"{source} — Page {page}"
            )

            chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=metadata,
                )
            )

        if (
            document_index == 1
            or document_index % 25 == 0
            or document_index
            == total_documents
        ):
            print(
                f"      Processed "
                f"{document_index}/"
                f"{total_documents} "
                f"| chunks={len(chunks)}",
                flush=True,
            )

    # Global stable IDs.
    for index, chunk in enumerate(
        chunks
    ):

        chunk.metadata[
            "chunk_id"
        ] = str(index)

        chunk.metadata[
            "chunk_index"
        ] = index

    print(
        f"      Chunking completed: "
        f"{len(chunks)} chunks",
        flush=True,
    )

    return chunks