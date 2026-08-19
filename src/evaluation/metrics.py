from typing import Dict, Iterable, List, Sequence

from langchain_core.documents import Document


# ============================================================
# Text helpers
# ============================================================

def normalize_text(text: str) -> str:
    return str(text).lower().strip()


def contains_keyword(document: Document, keyword: str) -> bool:
    return normalize_text(keyword) in normalize_text(document.page_content)


def source_matches(document: Document, expected_sources: Sequence[str]) -> bool:
    source = normalize_text(document.metadata.get("source", ""))
    return any(normalize_text(expected) == source for expected in expected_sources)


def page_matches(document: Document, expected_pages: Sequence[int]) -> bool:
    page = document.metadata.get("page_number")
    if page is None:
        return False
    try:
        page = int(page)
    except (ValueError, TypeError):
        return False
    return page in expected_pages


# ============================================================
# Relevance
# ============================================================

def document_is_relevant(
    document: Document,
    expected_sources: Sequence[str],
    expected_pages: Sequence[int] | None = None,
    expected_keywords: Sequence[str] | None = None,
) -> bool:
    """A document is relevant when its source matches and its page or keywords match."""
    if not isinstance(document, Document):
        return False

    if not source_matches(document, expected_sources):
        return False

    if not expected_pages and not expected_keywords:
        return True

    page_ok = bool(expected_pages) and page_matches(document, expected_pages)
    keyword_ok = bool(expected_keywords) and any(
        contains_keyword(document, keyword) for keyword in expected_keywords
    )
    return page_ok or keyword_ok


# ============================================================
# Retrieval metrics
# ============================================================

def precision_at_k(
    documents: Sequence[Document],
    expected_sources: Sequence[str],
    expected_pages: Sequence[int] | None = None,
    expected_keywords: Sequence[str] | None = None,
    k: int = 5,
) -> float:
    """Fraction of top-k retrieved documents that are relevant."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    top_documents = list(documents)[:k]
    if not top_documents:
        return 0.0

    relevant = sum(
        document_is_relevant(
            document,
            expected_sources,
            expected_pages,
            expected_keywords,
        )
        for document in top_documents
    )
    return float(relevant / len(top_documents))


def recall_at_k(
    documents: Sequence[Document],
    expected_sources: Sequence[str],
    expected_pages: Sequence[int] | None = None,
    expected_keywords: Sequence[str] | None = None,
    k: int = 5,
) -> float:
    """Binary retrieval recall: whether at least one relevant result appears in top-k."""
    if k <= 0:
        raise ValueError("k must be greater than 0")
    top_documents = list(documents)[:k]
    return float(any(
        document_is_relevant(
            document,
            expected_sources,
            expected_pages,
            expected_keywords,
        )
        for document in top_documents
    ))


def source_recall_at_k(
    documents: Sequence[Document],
    expected_sources: Sequence[str],
    k: int = 5,
) -> float:
    if not expected_sources:
        return 1.0

    top_documents = list(documents)[:k]
    found = set()
    for document in top_documents:
        source = document.metadata.get("source")
        if not source:
            continue
        normalized_source = normalize_text(source)
        for expected in expected_sources:
            if normalized_source == normalize_text(expected):
                found.add(expected)

    return len(found) / len(expected_sources)


# ============================================================
# MRR
# ============================================================

def reciprocal_rank(
    documents: Sequence[Document],
    expected_sources: Sequence[str],
    expected_pages: Sequence[int] | None = None,
    expected_keywords: Sequence[str] | None = None,
) -> float:
    for rank, document in enumerate(documents, start=1):
        if document_is_relevant(
            document,
            expected_sources,
            expected_pages,
            expected_keywords,
        ):
            return 1.0 / rank
    return 0.0


# ============================================================
# Citation metrics
# ============================================================

def citation_numbers_valid(answer: str, citation_map: Dict[int, str]) -> bool:
    import re

    if not answer:
        return False

    numbers = set()
    for value in re.findall(r"\[\s*Evidence\s+(\d+)\s*\]", answer, flags=re.IGNORECASE):
        numbers.add(int(value))
    for value in re.findall(r"(?<!Evidence\s)\[\s*(\d+)\s*\]", answer, flags=re.IGNORECASE):
        numbers.add(int(value))

    return all(number in citation_map for number in numbers)


# ============================================================
# Answer term coverage
# ============================================================

def answer_term_coverage(answer: str, expected_terms: Sequence[str] | None) -> float:
    if not expected_terms:
        return 1.0

    normalized_answer = normalize_text(answer)
    found = sum(normalize_text(term) in normalized_answer for term in expected_terms)
    return found / len(expected_terms)


# ============================================================
# Safety metric
# ============================================================

def safety_expectation_score(actual_requires_notice: bool, expected_requires_notice: bool) -> float:
    return float(actual_requires_notice == expected_requires_notice)


# ============================================================
# Aggregate summary
# ============================================================

def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def summarize_retrieval_metrics(case_results: List[Dict]) -> Dict[str, float]:
    return {
        "precision@5": mean(item["precision@5"] for item in case_results),
        "recall@5": mean(item["recall@5"] for item in case_results),
        "recall@10": mean(item["recall@10"] for item in case_results),
        "recall@20": mean(item["recall@20"] for item in case_results),
        "reranked_precision@5": mean(item["reranked_precision@5"] for item in case_results),
        "mrr": mean(item["mrr"] for item in case_results),
        "source_recall@10": mean(item["source_recall@10"] for item in case_results),
    }
