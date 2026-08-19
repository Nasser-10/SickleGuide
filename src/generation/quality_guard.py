from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.generation.citation import (
    build_citation_map,
    format_answer_with_citations,
    validate_citations,
)
from src.generation.prompt import build_grounded_regeneration_prompt
from src.generation.safety import apply_safety_notice


MAX_GROUNDING_REPAIRS = 2
MAX_CITATION_REPAIRS = 1


def _evidence_text(documents) -> str:
    blocks = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        citation = metadata.get(
            "citation",
            f"{metadata.get('source', 'Unknown source')} — Page {metadata.get('page_number', 'Unknown')}",
        )
        blocks.append(
            f"[Evidence {index}]\nCitation: {citation}\nContent:\n{document.page_content}"
        )
    return "\n\n".join(blocks)


def _repair_grounding(engine, query: str, result: Dict[str, Any]) -> Dict[str, Any]:
    documents = result.get("final_documents", [])
    if not documents:
        return result

    previous_answer = result.get("final_answer") or result.get("grounded_answer") or result.get("raw_answer", "")
    review = result.get("grounding_review") or {}
    unsupported = review.get("unsupported_claims") or []
    safety_instruction = result.get("safety_instruction", "")

    prompt = build_grounded_regeneration_prompt(
        query=query,
        documents=documents,
        previous_answer=previous_answer,
        unsupported_claims=unsupported,
        safety_instruction=safety_instruction,
    )
    prompt += """

FINAL QUALITY REQUIREMENT:
Return ONLY claims directly supported by the retrieved evidence.
Every medical claim must include an inline [Evidence N] citation.
If the evidence is insufficient, say so explicitly.
Do not use facts from conversation history as evidence.
"""

    response = engine.llm.invoke(prompt)
    repaired_answer = str(getattr(response, "content", response)).strip()
    if not repaired_answer:
        return result

    evidence_text = _evidence_text(documents)
    review_prompt = f"""
You are a strict medical RAG verifier.

QUESTION:
{query}

EVIDENCE:
{evidence_text}

ANSWER:
{repaired_answer}

Return a structured grounding decision.
A claim is grounded only when directly supported or faithfully paraphrased by the evidence.
"""
    review = engine.reviewer.invoke(review_prompt)
    review_dict = review.model_dump() if hasattr(review, "model_dump") else dict(review)

    if not bool(review_dict.get("grounded", False)):
        return result

    citation_map = build_citation_map(documents)
    citation_validation = validate_citations(repaired_answer, citation_map)
    final_answer = format_answer_with_citations(repaired_answer, documents)
    final_answer = apply_safety_notice(final_answer, result.get("safety_result"))

    updated = dict(result)
    updated.update(
        {
            "raw_answer": repaired_answer,
            "grounded_answer": repaired_answer,
            "final_answer": final_answer,
            "grounding_review": review_dict,
            "citation_validation": citation_validation,
            "quality_guard_repaired": True,
        }
    )
    return updated


def _repair_citations(engine, query: str, result: Dict[str, Any]) -> Dict[str, Any]:
    documents = result.get("final_documents", [])
    if not documents:
        return result

    answer = result.get("grounded_answer") or result.get("raw_answer") or result.get("final_answer", "")
    evidence_text = _evidence_text(documents)
    prompt = f"""
You are a citation repairer for a clinical RAG system.

QUESTION:
{query}

EVIDENCE:
{evidence_text}

ANSWER:
{answer}

Rewrite the answer with the same meaning but add an inline [Evidence N]
citation immediately after every important medical claim.
Use ONLY evidence numbers that exist above.
Do not add facts, recommendations, doses, effectiveness claims, or conclusions.
If a statement cannot be supported, remove it.
Return ONLY the repaired answer.
"""

    response = engine.llm.invoke(prompt)
    repaired = str(getattr(response, "content", response)).strip()
    if not repaired:
        return result

    citation_map = build_citation_map(documents)
    validation = validate_citations(repaired, citation_map)
    if not validation.get("valid"):
        return result

    updated = dict(result)
    updated.update(
        {
            "grounded_answer": repaired,
            "raw_answer": repaired,
            "final_answer": format_answer_with_citations(repaired, documents),
            "citation_validation": validation,
            "quality_guard_citation_repaired": True,
        }
    )
    return updated


def robust_invoke(engine, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    result = engine.invoke(query, conversation_history=conversation_history)

    for _ in range(MAX_GROUNDING_REPAIRS):
        review = result.get("grounding_review") or {}
        if bool(review.get("grounded", False)):
            break
        repaired = _repair_grounding(engine, query, result)
        if repaired.get("quality_guard_repaired"):
            result = repaired
            break
        result = repaired

    citation_validation = result.get("citation_validation") or {}
    if not citation_validation.get("valid", False):
        repaired = _repair_citations(engine, query, result)
        if repaired.get("quality_guard_citation_repaired"):
            result = repaired

    return result
