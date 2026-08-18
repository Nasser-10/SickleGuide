from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.retrieval.embeddings import get_embedding_model


DEFAULT_COLLECTION_NAME = "sickleguide"
DEFAULT_PERSIST_DIRECTORY = "data/processed/chroma"


class VectorStore:
    """
    Persistent Chroma vector store for SickleGuide.

    Responsibilities:
    - Create/load a persistent Chroma collection.
    - Store LangChain Documents and their metadata.
    - Perform dense similarity search.
    - Keep vector storage separate from the source chunks.json.
    """

    def __init__(
        self,
        persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model=None,
    ):
        self.persist_directory = Path(
            persist_directory
        )

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.collection_name = collection_name

        self.embedding_model = (
            embedding_model
            if embedding_model is not None
            else get_embedding_model()
        )

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=str(
                self.persist_directory
            ),
        )

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 64,
    ) -> int:
        """
        Add documents to Chroma in batches.

        Returns:
            Number of documents added.
        """

        if not documents:
            return 0

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0"
            )

        total_added = 0

        for start in range(
            0,
            len(documents),
            batch_size,
        ):
            batch = documents[
                start:start + batch_size
            ]

            self.vector_store.add_documents(
                batch
            )

            total_added += len(batch)

            print(
                f"Indexed "
                f"{total_added}/{len(documents)} "
                f"documents",
                flush=True,
            )

        return total_added

    def similarity_search(
        self,
        query: str,
        k: int = 5,
    ) -> List[Document]:
        """
        Retrieve the top-k most similar documents.
        """

        if not query or not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than 0"
            )

        return self.vector_store.similarity_search(
            query,
            k=k,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents together with Chroma similarity scores.
        """

        if not query or not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than 0"
            )

        return (
            self.vector_store
            .similarity_search_with_score(
                query,
                k=k,
            )
        )

    def count(self) -> int:
        """
        Return number of vectors in the collection.
        """

        return self.vector_store._collection.count()

    def get(
        self,
        limit: int = 5,
    ) -> List[Document]:
        """
        Return a small sample of stored documents.
        """

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0"
            )

        result = self.vector_store._collection.get(
            limit=limit,
            include=[
                "documents",
                "metadatas",
            ],
        )

        documents = []

        texts = result.get(
            "documents",
            [],
        )

        metadatas = result.get(
            "metadatas",
            [],
        )

        for text, metadata in zip(
            texts,
            metadatas,
        ):
            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata or {},
                )
            )

        return documents


def create_vector_store(
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model=None,
) -> VectorStore:
    """
    Factory function for the SickleGuide vector store.
    """

    return VectorStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )