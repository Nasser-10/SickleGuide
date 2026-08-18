from dataclasses import dataclass
from typing import List, Tuple
import re

from langchain_core.documents import Document


# ============================================================
# Safety configuration
# ============================================================

MIN_EVIDENCE_DOCUMENTS = 1

HIGH_RISK_TERMS = {
    "emergency",
    "chest pain",
    "difficulty breathing",
    "shortness of breath",
    "severe pain",
    "unconscious",
    "seizure",
    "stroke",
    "heavy bleeding",
    "fainting",
    "suicidal",
    "overdose",
}

PERSONAL_MEDICAL_ADVICE_TERMS = {
    "should i take",
    "should i stop",
    "what dose should i take",
    "my dose",
    "my medication",
    "can i take",
    "can i stop",
    "is it safe for me",
    "what should i do",
}

DIAGNOSIS_TERMS = {
    "do i have",
    "am i sick",
    "diagnose me",
    "what disease do i have",
    "is this sickle cell",
}


# ============================================================
# Safety result
# ============================================================

@dataclass
class SafetyResult:
    """
    Result of safety assessment.
    """

    safe_to_generate: bool

    requires_urgent_care_notice: bool

    requires_clinician_notice: bool

    insufficient_evidence: bool

    reason: str

    notices: List[str]


# ============================================================
# Text utilities
# ============================================================

def _normalize(
    text: str,
) -> str:
    """
    Normalize text for rule-based safety checks.
    """

    text = (
        str(text)
        .lower()
        .strip()
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


# ============================================================
# Detection helpers
# ============================================================

def contains_high_risk_signal(
    query: str,
) -> bool:
    """
    Detect possible emergency/high-risk language.
    """

    normalized = _normalize(
        query
    )

    return any(
        term in normalized
        for term in HIGH_RISK_TERMS
    )


def requests_personal_medical_advice(
    query: str,
) -> bool:
    """
    Detect requests for individualized treatment decisions.
    """

    normalized = _normalize(
        query
    )

    return any(
        term in normalized
        for term in PERSONAL_MEDICAL_ADVICE_TERMS
    )


def requests_diagnosis(
    query: str,
) -> bool:
    """
    Detect direct diagnosis requests.
    """

    normalized = _normalize(
        query
    )

    return any(
        term in normalized
        for term in DIAGNOSIS_TERMS
    )


# ============================================================
# Evidence validation
# ============================================================

def validate_evidence(
    documents: List[Document],
) -> bool:
    """
    Verify that retrieval actually returned usable evidence.
    """

    if not documents:
        return False

    usable = 0

    for document in documents:

        if not isinstance(
            document,
            Document,
        ):
            continue

        text = (
            document.page_content
            or ""
        ).strip()

        if len(text) >= 50:
            usable += 1

    return (
        usable
        >= MIN_EVIDENCE_DOCUMENTS
    )


# ============================================================
# Safety assessment
# ============================================================

def assess_query(
    query: str,
    documents: List[Document],
) -> SafetyResult:
    """
    Assess whether the query can safely proceed to generation.

    This layer does NOT provide medical advice.
    It only determines safeguards needed by the generation layer.
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

    notices = []

    high_risk = contains_high_risk_signal(
        query
    )

    personal_advice = (
        requests_personal_medical_advice(
            query
        )
    )

    diagnosis_request = (
        requests_diagnosis(
            query
        )
    )

    insufficient_evidence = (
        not validate_evidence(
            documents
        )
    )

    # --------------------------------------------------------
    # Emergency notice
    # --------------------------------------------------------

    if high_risk:

        notices.append(
            "The query may describe an urgent or "
            "potentially serious medical situation. "
            "The system should recommend prompt "
            "professional medical evaluation rather "
            "than relying on this assistant alone."
        )

    # --------------------------------------------------------
    # Personal medical advice notice
    # --------------------------------------------------------

    if personal_advice:

        notices.append(
            "The response must not provide individualized "
            "medication changes, dosing decisions, or "
            "treatment instructions. It should remain "
            "general and evidence-grounded."
        )

    # --------------------------------------------------------
    # Diagnosis notice
    # --------------------------------------------------------

    if diagnosis_request:

        notices.append(
            "The assistant must not diagnose the user. "
            "It may explain information from the supplied "
            "clinical sources and recommend professional "
            "evaluation where appropriate."
        )

    # --------------------------------------------------------
    # Evidence failure
    # --------------------------------------------------------

    if insufficient_evidence:

        notices.append(
            "There is insufficient retrieved evidence "
            "to provide a reliable evidence-grounded answer."
        )

    return SafetyResult(
        safe_to_generate=True,
        requires_urgent_care_notice=high_risk,
        requires_clinician_notice=(
            personal_advice
            or diagnosis_request
        ),
        insufficient_evidence=(
            insufficient_evidence
        ),
        reason=(
            "Safety assessment completed."
        ),
        notices=notices,
    )


# ============================================================
# Safety instructions for generation
# ============================================================

def build_safety_instruction(
    safety_result: SafetyResult,
) -> str:
    """
    Convert safety assessment into deterministic instructions
    for the generation prompt.
    """

    if not isinstance(
        safety_result,
        SafetyResult,
    ):
        raise TypeError(
            "safety_result must be a SafetyResult"
        )

    instructions = []

    instructions.append(
        "Do not provide personalized diagnosis "
        "or treatment decisions."
    )

    instructions.append(
        "Do not invent medication doses, "
        "contraindications, or recommendations."
    )

    instructions.append(
        "Use only information supported by "
        "the retrieved evidence."
    )

    if (
        safety_result
        .requires_urgent_care_notice
    ):

        instructions.append(
            "Include a clear notice that potentially "
            "urgent symptoms require prompt professional "
            "medical evaluation."
        )

    if (
        safety_result
        .requires_clinician_notice
    ):

        instructions.append(
            "Include a clinician-consultation notice "
            "for individualized decisions."
        )

    if (
        safety_result
        .insufficient_evidence
    ):

        instructions.append(
            "Explicitly state that the supplied sources "
            "do not contain enough evidence to answer "
            "the question reliably."
        )

    return "\n".join(
        f"- {instruction}"
        for instruction in instructions
    )


# ============================================================
# Final safety notice
# ============================================================

def build_safety_notice(
    safety_result: SafetyResult,
) -> str:
    """
    Build user-facing safety notices.
    """

    if not isinstance(
        safety_result,
        SafetyResult,
    ):
        raise TypeError(
            "safety_result must be a SafetyResult"
        )

    if not safety_result.notices:
        return ""

    lines = [
        "Safety note:"
    ]

    for notice in (
        safety_result.notices
    ):
        lines.append(
            f"- {notice}"
        )

    return "\n".join(
        lines
    )


# ============================================================
# Utility
# ============================================================

def apply_safety_notice(
    answer: str,
    safety_result: SafetyResult,
) -> str:
    """
    Append deterministic safety notices to an answer.
    """

    if not isinstance(
        answer,
        str,
    ):
        raise TypeError(
            "answer must be a string"
        )

    answer = answer.strip()

    notice = build_safety_notice(
        safety_result
    )

    if not notice:
        return answer

    if not answer:
        return notice

    return (
        f"{answer}\n\n{notice}"
    )