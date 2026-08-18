from typing import List

from langchain_core.documents import Document


# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = """
You are SickleGuide, an evidence-grounded clinical assistant
for sickle cell disease information.

You MUST answer using ONLY the retrieved evidence.

ABSOLUTE GROUNDING RULES:

1. A medical claim is allowed ONLY when the retrieved evidence
   explicitly supports that claim.
2. Do NOT use your pretrained knowledge to fill gaps.
3. If a treatment, drug, dose, contraindication, monitoring plan,
   outcome, or recommendation is not explicitly supported by the
   retrieved evidence, DO NOT mention it.
4. Never turn a study finding into a recommendation unless the
   source explicitly gives that recommendation.
5. Never infer "most effective", "best", "first-line", "preferred",
   "superior", or "recommended" unless the source explicitly uses
   that meaning.
6. IMPORTANT:
   "higher quality evidence", "lower quality evidence",
   "very low certainty", "limited evidence", or
   "more evidence is needed" do NOT mean that a treatment is
   more effective, preferred, or recommended.
7. Do not infer comparative effectiveness from the order in which
   treatments appear in the source.
8. Do not infer a recommendation simply because an intervention is
   mentioned in a clinical question, research question, or evidence
   summary.
9. If the source lists several interventions being evaluated,
   report them as evaluated interventions, not as recommended treatments,
   unless the source explicitly recommends one.
10. General medical knowledge does NOT count as retrieved evidence.
11. Do not diagnose the user.
12. Do not provide individualized treatment decisions.
13. Preserve the exact level of certainty reported by the source.
14. Do not fabricate citations.
15. Every important medical claim MUST have a supporting
    [Evidence N] citation.
16. The citation must support the exact claim being made.
17. If the evidence answers only part of the question, answer only
    that part and explicitly state the limitation.
18. If the evidence does not support a direct answer, say so.
19. Do not mention treatments merely because they are familiar
    or commonly used outside the supplied evidence.

SPECIAL RULE FOR BROAD TREATMENT QUESTIONS:

When the question asks for "treatments for SCD" but the retrieved
evidence covers specific scenarios such as acute chest syndrome,
pregnancy, transfusion support, pain, or pediatric care:

- Do NOT combine those scenario-specific interventions into one
  universal treatment list.
- Clearly label the clinical scenario when mentioning an intervention.
- Say when the available evidence is scenario-specific.
- Do NOT generalize a treatment from one scenario to all patients
  with SCD.

IMPORTANT:
It is better to give a short incomplete answer that is fully
supported than a complete answer containing unsupported facts.

GROUNDING > COMPLETENESS.
"""


# ============================================================
# Evidence formatting
# ============================================================

def format_evidence(
    documents: List[Document],
) -> str:
    """
    Format final retrieved/reranked evidence.
    """

    if not isinstance(
        documents,
        list,
    ):
        raise TypeError(
            "documents must be a list"
        )

    if not documents:
        return (
            "NO RETRIEVED EVIDENCE WAS PROVIDED."
        )

    blocks = []

    evidence_index = 1

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

        if not text:
            continue

        metadata = (
            document.metadata
            or {}
        )

        source = metadata.get(
            "source",
            "Unknown source",
        )

        page = metadata.get(
            "page_number",
            "Unknown page",
        )

        citation = metadata.get(
            "citation",
            f"{source} — Page {page}",
        )

        retrieval_method = metadata.get(
            "retrieval_method",
            "unknown",
        )

        reranker_score = metadata.get(
            "reranker_score"
        )

        graph_relations = metadata.get(
            "graph_supporting_relations",
            [],
        )

        lines = [
            f"[Evidence {evidence_index}]",
            f"Source: {source}",
            f"Page: {page}",
            f"Citation: {citation}",
            f"Retrieval method: {retrieval_method}",
        ]

        if reranker_score is not None:
            lines.append(
                f"Reranker score: "
                f"{float(reranker_score):.4f}"
            )

        if graph_relations:
            lines.append(
                "Graph relations: "
                + ", ".join(
                    map(
                        str,
                        graph_relations,
                    )
                )
            )

        lines.append(
            "Content:"
        )

        lines.append(
            text
        )

        blocks.append(
            "\n".join(lines)
        )

        evidence_index += 1

    if not blocks:
        return (
            "NO VALID RETRIEVED EVIDENCE WAS PROVIDED."
        )

    return (
        "\n\n"
        + "\n\n".join(
            blocks
        )
    )


# ============================================================
# Main Prompt
# ============================================================

def build_prompt(
    query: str,
    documents: List[Document],
    safety_instruction: str = "",
) -> str:
    """
    Build the grounded generation prompt.
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

    evidence = format_evidence(
        documents
    )

    safety_block = ""

    if safety_instruction:
        safety_block = (
            "\n\nSAFETY INSTRUCTIONS:\n"
            f"{safety_instruction}"
        )

    return f"""
{SYSTEM_PROMPT}

USER QUESTION:
{query}

RETRIEVED EVIDENCE:
{evidence}

{safety_block}

Before writing each medical claim, ask:

"Does the evidence explicitly support this exact statement?"

If NO:
- Do not write it.

If the evidence says:
- "evaluated"
  → say "was evaluated"
- "considered"
  → say "was considered"
- "limited evidence"
  → say "evidence was limited"
- "very low certainty"
  → say "certainty was very low"
- "recommended"
  → say "recommended"
- "suggested"
  → say "suggested"

DO NOT convert one into another.

Especially:
"very low certainty except for opioids"
MUST NOT become
"opioids are the most effective treatment."

If a clinical question asks what treatment is most effective,
that does NOT mean the source answered that question.

Citation format:
[Evidence 1]
[Evidence 2]

Every medical claim needs an exact supporting evidence citation.

Never invent evidence numbers.
Never use outside medical knowledge.
"""


# ============================================================
# Grounded Regeneration Prompt
# ============================================================

def build_grounded_regeneration_prompt(
    query: str,
    documents: List[Document],
    previous_answer: str,
    unsupported_claims: List[str],
    safety_instruction: str = "",
) -> str:
    """
    Regenerate an answer after grounding failure.
    """

    evidence = format_evidence(
        documents
    )

    claims = "\n".join(
        f"- {claim}"
        for claim in unsupported_claims
    )

    return f"""
You are correcting a medical RAG answer.

USER QUESTION:
{query}

ORIGINAL ANSWER:
{previous_answer}

UNSUPPORTED CLAIMS:
{claims}

RETRIEVED EVIDENCE:
{evidence}

SAFETY INSTRUCTIONS:
{safety_instruction}

Rewrite the answer from scratch.

STRICT CORRECTION RULES:

1. Remove every unsupported claim.
2. Do NOT paraphrase an unsupported claim into a weaker-looking
   version that keeps the same unsupported meaning.
3. Use ONLY retrieved evidence.
4. Do not use general medical knowledge.
5. Do not infer treatment recommendations.
6. Do not infer comparative effectiveness.
7. Do not infer "most effective", "best", "preferred",
   "first-line", or "superior".
8. Do not convert evidence certainty into treatment effectiveness.
9. If an intervention was merely evaluated, say that it was evaluated.
10. If the source reports uncertainty, preserve that uncertainty.
11. Do not combine scenario-specific findings into universal SCD advice.
12. Every medical claim must have a supporting [Evidence N] citation.
13. If the evidence cannot answer the question reliably, say that
    the retrieved evidence is insufficient.

For example:

BAD:
"Opioids are the most effective treatment."

GOOD:
"The evidence review examined opioids among the interventions
considered for acute chest syndrome, and reported that the overall
certainty of evidence was very low." [Evidence N]

Return ONLY the corrected answer.
"""


# ============================================================
# No Evidence
# ============================================================

def build_no_evidence_prompt(
    query: str,
    safety_instruction: str = "",
) -> str:

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

    return f"""
{SYSTEM_PROMPT}

USER QUESTION:
{query}

NO VALID RETRIEVED EVIDENCE IS AVAILABLE.

Do not answer from your own medical knowledge.

State clearly that the currently retrieved SickleGuide evidence
is insufficient to answer the question reliably.

{safety_instruction}
"""