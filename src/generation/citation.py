from typing import Dict, List
import re

from langchain_core.documents import Document


def get_document_citation(document: Document) -> str:
    if not isinstance(document, Document):
        raise TypeError("document must be a LangChain Document")
    metadata = document.metadata or {}
    citation = metadata.get("citation")
    if citation:
        return str(citation).strip()
    return f"{metadata.get('source', 'Unknown source')} — Page {metadata.get('page_number', 'Unknown page')}"


def build_citation_map(documents: List[Document]) -> Dict[int, str]:
    if not isinstance(documents, list):
        raise TypeError("documents must be a list")
    return {
        index: get_document_citation(document)
        for index, document in enumerate(documents, start=1)
        if isinstance(document, Document)
    }


def extract_evidence_numbers(text: str) -> List[int]:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    numbers: List[int] = []
    for pattern in (
        r"\[\s*Evidence\s+(\d+)\s*\]",
        r"(?<!Evidence\s)\[\s*(\d+)\s*\]",
    ):
        for value in re.findall(pattern, text, flags=re.IGNORECASE):
            number = int(value)
            if number not in numbers:
                numbers.append(number)
    return numbers


def validate_citations(answer: str, citation_map: Dict[int, str]) -> Dict:
    evidence_numbers = extract_evidence_numbers(answer)
    valid = [number for number in evidence_numbers if number in citation_map]
    invalid = [number for number in evidence_numbers if number not in citation_map]
    return {
        "valid": bool(answer and answer.strip() and evidence_numbers and not invalid),
        "valid_evidence_numbers": valid,
        "invalid_evidence_numbers": invalid,
        "citation_count": len(evidence_numbers),
    }


def build_citation_block(documents: List[Document]) -> str:
    citation_map = build_citation_map(documents)
    if not citation_map:
        return ""
    lines = ["Verified sources:"]
    lines.extend(f"[{number}] {citation}" for number, citation in citation_map.items())
    return "\n".join(lines)


def _remove_invalid_citations(answer: str, invalid_numbers: List[int]) -> str:
    cleaned = answer
    for number in invalid_numbers:
        cleaned = re.sub(rf"\[\s*Evidence\s+{number}\s*\]", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(rf"\[\s*{number}\s*\]", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def format_answer_with_citations(answer: str, documents: List[Document]) -> str:
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")
    answer = answer.strip()
    if not answer:
        return ""

    citation_map = build_citation_map(documents)
    initial = validate_citations(answer, citation_map)
    cleaned_answer = _remove_invalid_citations(answer, initial["invalid_evidence_numbers"])
    source_block = build_citation_block(documents)
    return f"{cleaned_answer}\n\n{source_block}" if source_block else cleaned_answer


def get_citations(documents: List[Document]) -> List[str]:
    citations = []
    seen = set()
    for document in documents:
        citation = get_document_citation(document)
        if citation and citation not in seen:
            seen.add(citation)
            citations.append(citation)
    return citations
