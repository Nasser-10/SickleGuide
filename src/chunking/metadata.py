from typing import List

from langchain_core.documents import Document


def enrich_chunk_metadata(
    chunks: List[Document],
) -> List[Document]:
    """
    Final metadata enrichment for retrieval,
    citation, evaluation, and Graph RAG.
    """

    enriched = []

    for index, chunk in enumerate(chunks):

        metadata = chunk.metadata.copy()

        metadata["chunk_id"] = metadata.get(
            "chunk_id",
            str(index),
        )

        metadata["chunk_index"] = metadata.get(
            "chunk_index",
            index,
        )

        metadata["chunk_size"] = len(
            chunk.page_content
        )

        metadata["source"] = metadata.get(
            "source",
            "unknown",
        )

        metadata["page_number"] = metadata.get(
            "page_number",
            0,
        )

        metadata["document_type"] = metadata.get(
            "document_type",
            "medical_guideline",
        )

        metadata["citation"] = (
            f"{metadata['source']} "
            f"— Page {metadata['page_number']}"
        )

        enriched.append(
            Document(
                page_content=chunk.page_content,
                metadata=metadata,
            )
        )

    return enriched