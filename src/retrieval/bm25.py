from typing import List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    BM25 lexical retriever for SickleGuide.

    Uses LangChain Document objects while keeping the
    original metadata and citation information.
    """

    def __init__(
        self,
        documents: List[Document],
    ):
        if not isinstance(documents, list):
            raise TypeError(
                "documents must be a list"
            )

        if not documents:
            raise ValueError(
                "documents cannot be empty"
            )

        self.documents = documents

        tokenized_documents = [
            self._tokenize(
                document.page_content
            )
            for document in documents
        ]

        self.bm25 = BM25Okapi(
            tokenized_documents
        )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Simple medical-safe tokenizer.

        Keeps alphanumeric medical terms and
        does not aggressively stem or remove terms.
        """

        text = text.lower()

        tokens = []

        current = []

        for char in text:

            if char.isalnum() or char in "-_/":

                current.append(char)

            else:

                if current:
                    tokens.append(
                        "".join(current)
                    )
                    current = []

        if current:
            tokens.append(
                "".join(current)
            )

        return tokens

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> List[Document]:
        """
        Return top-k BM25 documents.
        """

        results = self.search_with_scores(
            query=query,
            k=k,
        )

        return [
            document
            for document, _ in results
        ]

    def search_with_scores(
        self,
        query: str,
        k: int = 5,
    ) -> List[Tuple[Document, float]]:
        """
        Return top-k documents with raw BM25 scores.
        """

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string"
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty"
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than 0"
            )

        query_tokens = self._tokenize(
            query
        )

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:k]

        return [
            (
                self.documents[index],
                float(scores[index]),
            )
            for index in ranked_indices
        ]

    def count(self) -> int:
        """
        Return number of indexed documents.
        """

        return len(self.documents)


def create_bm25_retriever(
    documents: List[Document],
) -> BM25Retriever:
    """
    Factory function for SickleGuide BM25 retrieval.
    """

    return BM25Retriever(
        documents=documents
    )