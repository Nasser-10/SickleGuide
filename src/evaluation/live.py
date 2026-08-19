from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.evaluation.dataset import get_retrieval_cases
from src.evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank

_MAX_HISTORY = 100
_history = deque(maxlen=_MAX_HISTORY)


def _find_case(query: str):
    normalized = " ".join(query.lower().split())
    for case in get_retrieval_cases():
        if " ".join(case.query.lower().split()) == normalized:
            return case
    return None


def record_chat_evaluation(query: str, result: Dict[str, Any]) -> Dict[str, Any]:
    grounding = result.get("grounding_review") or {}
    citation = result.get("citation_validation") or {}
    final_documents = result.get("final_documents") or []
    case = _find_case(query)

    record: Dict[str, Any] = {
        "id": f"eval-{datetime.now(timezone.utc).timestamp():.6f}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "grounded": bool(grounding.get("grounded", False)),
        "citations_valid": bool(citation.get("valid", False)),
        "evidence_count": len(final_documents),
        "labeled": case is not None,
        "precision@5": None,
        "recall@5": None,
        "mrr": None,
    }

    if case is not None:
        record["precision@5"] = precision_at_k(final_documents, case.expected_sources, case.expected_pages, case.expected_keywords, k=5)
        record["recall@5"] = recall_at_k(final_documents, case.expected_sources, case.expected_pages, case.expected_keywords, k=5)
        record["mrr"] = reciprocal_rank(final_documents, case.expected_sources, case.expected_pages, case.expected_keywords)

    _history.appendleft(record)
    return record


def latest_evaluation() -> Optional[Dict[str, Any]]:
    return _history[0] if _history else None


def evaluation_history(limit: int = 50) -> list[Dict[str, Any]]:
    return list(_history)[: max(1, min(limit, _MAX_HISTORY))]
