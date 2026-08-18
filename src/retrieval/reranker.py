from typing import List, Tuple
import re

import torch

from langchain_core.documents import Document


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_FINAL_K = 5
DEFAULT_MAX_LENGTH = 512


# ============================================================
# Query intent
# ============================================================

TREATMENT_TERMS = {
    "treat",
    "treats",
    "treatment",
    "treatments",
    "therapy",
    "therapies",
    "management",
    "manage",
    "managed",
    "medication",
    "medications",
    "drug",
    "drugs",
    "recommend",
    "recommended",
    "recommendation",
    "recommendations",
}


DIAGNOSIS_TERMS = {
    "diagnose",
    "diagnosis",
    "diagnostic",
    "screening",
    "screen",
    "test",
    "testing",
    "laboratory",
    "lab",
    "imaging",
    "mri",
}


COMPLICATION_TERMS = {
    "complication",
    "complications",
    "adverse",
    "side",
    "risk",
    "risks",
    "toxicity",
}


MONITORING_TERMS = {
    "monitor",
    "monitoring",
    "follow",
    "follow-up",
    "surveillance",
    "check",
    "checks",
}


STOPWORDS = {
    "what",
    "which",
    "when",
    "where",
    "who",
    "how",
    "are",
    "the",
    "for",
    "with",
    "from",
    "does",
    "this",
    "that",
    "these",
    "those",
    "and",
    "or",
    "can",
    "may",
    "should",
    "used",
    "use",
    "about",
    "into",
    "over",
    "under",
}


# ============================================================
# Text utilities
# ============================================================

def _normalize(text: str) -> str:
    text = (
        str(text)
        .lower()
        .strip()
    )

    text = re.sub(
        r"[^\w\s-]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in _normalize(text).split()
        if (
            len(token) >= 3
            and token not in STOPWORDS
        )
    }


def _detect_query_intent(query: str) -> str:
    tokens = _tokens(query)

    if tokens & TREATMENT_TERMS:
        return "treatment"

    if tokens & DIAGNOSIS_TERMS:
        return "diagnosis"

    if tokens & COMPLICATION_TERMS:
        return "complication"

    if tokens & MONITORING_TERMS:
        return "monitoring"

    return "general"


# ============================================================
# Candidate quality filtering
# ============================================================

def _is_bibliography(text: str) -> bool:
    normalized = _normalize(text)

    doi_count = len(
        re.findall(
            r"\bdoi\b|10\.\d{4,9}/\S+",
            normalized,
        )
    )

    url_count = len(
        re.findall(
            r"https?://|www\.",
            normalized,
        )
    )

    year_count = len(
        re.findall(
            r"\b(?:19|20)\d{2}\b",
            normalized,
        )
    )

    numbered_refs = len(
        re.findall(
            r"\b\d{1,4}\.\s+[A-Z][A-Za-z-]+",
            text,
        )
    )

    first_lines = [
        _normalize(line)
        for line in text.splitlines()[:8]
        if line.strip()
    ]

    for line in first_lines:
        if (
            line == "references"
            or line.startswith("references ")
            or line == "bibliography"
        ):
            return True

    if (
        numbered_refs >= 5
        and year_count >= 5
    ):
        return True

    if (
        doi_count >= 2
        and year_count >= 5
    ):
        return True

    if (
        url_count >= 3
        and year_count >= 5
    ):
        return True

    if (
        len(normalized) < 1800
        and year_count >= 8
        and (
            doi_count
            + url_count
            + numbered_refs
        ) >= 6
    ):
        return True

    return False


def _is_boilerplate(text: str) -> bool:
    normalized = _normalize(text)

    terms = [
        "all rights reserved",
        "creative commons",
        "under the terms of this licence",
        "cover photo",
        "freepik",
        "liable for damages",
        "responsibility for the interpretation",
    ]

    matches = sum(
        term in normalized
        for term in terms
    )

    return matches >= 2


def _is_title_only_chunk(
    document: Document,
) -> bool:
    """
    Detect title / cover / heading-only chunks.

    These are poor evidence for RAG even if their
    embedding similarity is high.
    """

    text = (
        document.page_content or ""
    ).strip()

    normalized = _normalize(
        text
    )

    if not normalized:
        return True

    # Very short chunks are almost always headings/title metadata.
    if len(normalized) < 180:
        return True

    # A single Markdown heading with no substantial body.
    non_empty_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if (
        len(non_empty_lines) <= 3
        and all(
            line.startswith("#")
            or len(line) < 180
            for line in non_empty_lines
        )
    ):
        return True

    return False


def _candidate_quality(
    document: Document,
) -> float:
    """
    Score evidence quality before cross-encoder reranking.
    """

    text = (
        document.page_content or ""
    ).strip()

    if not text:
        return 0.0

    if _is_bibliography(text):
        return 0.0

    if _is_boilerplate(text):
        return 0.0

    if _is_title_only_chunk(document):
        return 0.0

    normalized = _normalize(text)

    score = 1.0

    if len(normalized) >= 300:
        score += 0.25

    if len(normalized) >= 700:
        score += 0.25

    if "recommendation" in normalized:
        score += 0.35

    if "recommended" in normalized:
        score += 0.25

    if "patients" in normalized:
        score += 0.15

    if "evidence" in normalized:
        score += 0.15

    return score


def _query_match_score(
    query: str,
    document: Document,
) -> float:
    query_tokens = _tokens(query)

    document_tokens = _tokens(
        document.page_content
    )

    if not query_tokens:
        return 0.0

    overlap = (
        query_tokens
        & document_tokens
    )

    return (
        len(overlap)
        / len(query_tokens)
    )


def _intent_score(
    query_intent: str,
    document: Document,
) -> float:
    text = _normalize(
        document.page_content
    )

    score = 0.0

    if query_intent == "treatment":

        if "recommended" in text:
            score += 0.20

        if "treatment" in text:
            score += 0.15

        if "therapy" in text:
            score += 0.10

        if "management" in text:
            score += 0.10

        if "transfusion" in text:
            score += 0.10

        if "hydroxyurea" in text:
            score += 0.10

    elif query_intent == "diagnosis":

        if "diagnosis" in text:
            score += 0.20

        if "screening" in text:
            score += 0.15

        if "laboratory" in text:
            score += 0.10

    elif query_intent == "monitoring":

        if "monitoring" in text:
            score += 0.20

        if "follow-up" in text:
            score += 0.15

        if "screening" in text:
            score += 0.10

    elif query_intent == "complication":

        if "complication" in text:
            score += 0.20

        if "adverse" in text:
            score += 0.15

        if "risk" in text:
            score += 0.10

    return score


# ============================================================
# BGE Reranker
# ============================================================

class BGEReranker:
    """
    BGE Reranker v2 M3.

    Unified Retrieval candidates are filtered first,
    then reranked with the BGE cross-encoder.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str = "cuda",
        max_length: int = DEFAULT_MAX_LENGTH,
    ):

        self.model_name = model_name
        self.max_length = max_length

        if (
            device == "cuda"
            and torch.cuda.is_available()
        ):
            self.device = torch.device(
                "cuda"
            )
        else:
            self.device = torch.device(
                "cpu"
            )

        print(
            f"Loading reranker: {model_name}",
            flush=True,
        )

        print(
            f"Reranker device: {self.device}",
            flush=True,
        )

        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                model_name
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        print(
            "Reranker loaded successfully.",
            flush=True,
        )

    def _score_pairs(
        self,
        pairs: List[Tuple[str, str]],
    ) -> List[float]:

        if not pairs:
            return []

        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(
                self.device
            )
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            logits = (
                self.model(
                    **inputs
                )
                .logits
                .view(-1)
            )

            scores = torch.sigmoid(
                logits
            )

        return (
            scores
            .detach()
            .cpu()
            .tolist()
        )

    def score(
        self,
        query: str,
        document: Document,
    ) -> float:

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

        if not isinstance(
            document,
            Document,
        ):
            raise TypeError(
                "document must be a LangChain Document"
            )

        text = (
            document.page_content or ""
        ).strip()

        if not text:
            return 0.0

        scores = self._score_pairs(
            [
                (
                    query,
                    text,
                )
            ]
        )

        return float(
            scores[0]
        )

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = DEFAULT_FINAL_K,
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

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        if not documents:
            return []

        query_intent = (
            _detect_query_intent(
                query
            )
        )

        # -----------------------------------------------------
        # Quality filter
        # -----------------------------------------------------

        valid_documents = []

        rejected = 0

        for document in documents:

            quality = (
                _candidate_quality(
                    document
                )
            )

            if quality <= 0:
                rejected += 1
                continue

            valid_documents.append(
                (
                    document,
                    quality,
                )
            )

        print(
            f"Reranker candidates: "
            f"{len(documents)}",
            flush=True,
        )

        print(
            f"Rejected low-quality: "
            f"{rejected}",
            flush=True,
        )

        if not valid_documents:
            return []

        print(
            f"Reranking "
            f"{len(valid_documents)} "
            f"quality candidates...",
            flush=True,
        )

        pairs = [
            (
                query,
                document.page_content,
            )
            for document, _
            in valid_documents
        ]

        bge_scores = (
            self._score_pairs(
                pairs
            )
        )

        ranked = []

        for index, (
            document,
            quality,
        ) in enumerate(
            valid_documents
        ):

            bge_score = float(
                bge_scores[index]
            )

            lexical_score = (
                _query_match_score(
                    query,
                    document,
                )
            )

            intent_score = (
                _intent_score(
                    query_intent,
                    document,
                )
            )

            # BGE remains the dominant signal.
            final_score = (
                0.82 * bge_score
                + 0.10 * lexical_score
                + 0.08 * intent_score
            )

            metadata = (
                document.metadata.copy()
            )

            metadata[
                "bge_score"
            ] = bge_score

            metadata[
                "reranker_lexical_score"
            ] = lexical_score

            metadata[
                "reranker_intent_score"
            ] = intent_score

            metadata[
                "reranker_quality_score"
            ] = quality

            metadata[
                "reranker_score"
            ] = final_score

            metadata[
                "query_intent"
            ] = query_intent

            ranked.append(
                (
                    Document(
                        page_content=(
                            document.page_content
                        ),
                        metadata=metadata,
                    ),
                    final_score,
                )
            )

        ranked.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        final_documents = []

        for rank, (
            document,
            score,
        ) in enumerate(
            ranked[:top_k],
            start=1,
        ):

            metadata = (
                document.metadata.copy()
            )

            metadata[
                "reranker_rank"
            ] = rank

            metadata[
                "retrieval_method"
            ] = (
                "dense_bm25_graph_rrf_bge"
            )

            final_documents.append(
                Document(
                    page_content=(
                        document.page_content
                    ),
                    metadata=metadata,
                )
            )

        return final_documents

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: int = DEFAULT_FINAL_K,
    ) -> List[
        Tuple[Document, float]
    ]:

        reranked = self.rerank(
            query=query,
            documents=documents,
            top_k=top_k,
        )

        return [
            (
                document,
                float(
                    document.metadata[
                        "reranker_score"
                    ]
                ),
            )
            for document in reranked
        ]


def create_reranker(
    model_name: str = DEFAULT_RERANKER_MODEL,
    device: str = "cuda",
    max_length: int = DEFAULT_MAX_LENGTH,
) -> BGEReranker:

    return BGEReranker(
        model_name=model_name,
        device=device,
        max_length=max_length,
    )