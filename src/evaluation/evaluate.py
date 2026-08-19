from typing import Dict, List

from src.evaluation.dataset import (
    EvaluationCase,
    get_evaluation_cases,
    get_retrieval_cases,
    get_safety_cases,
)

from src.evaluation.metrics import (
    answer_term_coverage,
    mean,
    recall_at_k,
    reciprocal_rank,
    source_recall_at_k,
)

from src.generation.llm import (
    create_rag_engine,
)


# ============================================================
# Retrieval Evaluation
# ============================================================

def evaluate_retrieval(
    engine,
    cases: List[EvaluationCase] | None = None,
) -> Dict:
    """
    Evaluate unified retrieval + reranking.
    """

    if cases is None:
        cases = get_retrieval_cases()

    results = []

    for index, case in enumerate(
        cases,
        start=1,
    ):

        print(
            "\n" + "-" * 70,
            flush=True,
        )

        print(
            f"Retrieval case "
            f"{index}/{len(cases)}",
            flush=True,
        )

        print(
            f"Query: {case.query}",
            flush=True,
        )

        # ----------------------------------------------------
        # Unified candidates
        # ----------------------------------------------------

        candidates = (
            engine.unified_retriever.retrieve(
                case.query,
                final_k=engine.candidate_k,
            )
        )

        # ----------------------------------------------------
        # Reranked candidates
        # ----------------------------------------------------

        reranked = (
            engine.reranker.rerank(
                query=case.query,
                documents=candidates,
                top_k=engine.final_k,
            )
        )

        # ----------------------------------------------------
        # Metrics over candidates
        # ----------------------------------------------------

        candidate_recall_5 = recall_at_k(
            candidates,
            case.expected_sources,
            case.expected_pages,
            case.expected_keywords,
            k=5,
        )

        candidate_recall_10 = recall_at_k(
            candidates,
            case.expected_sources,
            case.expected_pages,
            case.expected_keywords,
            k=10,
        )

        candidate_recall_20 = recall_at_k(
            candidates,
            case.expected_sources,
            case.expected_pages,
            case.expected_keywords,
            k=20,
        )

        # ----------------------------------------------------
        # Metrics over reranked results
        # ----------------------------------------------------

        reranked_recall_5 = recall_at_k(
            reranked,
            case.expected_sources,
            case.expected_pages,
            case.expected_keywords,
            k=5,
        )

        reranked_mrr = reciprocal_rank(
            reranked,
            case.expected_sources,
            case.expected_pages,
            case.expected_keywords,
        )

        source_recall_10 = source_recall_at_k(
            candidates,
            case.expected_sources,
            k=10,
        )

        result = {
            "query": case.query,
            "recall@5": candidate_recall_5,
            "recall@10": candidate_recall_10,
            "recall@20": candidate_recall_20,
            "reranked_recall@5": reranked_recall_5,
            "mrr": reranked_mrr,
            "source_recall@10": source_recall_10,
            "candidate_count": len(
                candidates
            ),
            "reranked_count": len(
                reranked
            ),
            "top_sources": [
                document.metadata.get(
                    "source",
                    "",
                )
                for document in reranked
            ],
            "top_pages": [
                document.metadata.get(
                    "page_number"
                )
                for document in reranked
            ],
        }

        results.append(
            result
        )

        print(
            f"Recall@5  : {candidate_recall_5:.3f}",
            flush=True,
        )

        print(
            f"Recall@10 : {candidate_recall_10:.3f}",
            flush=True,
        )

        print(
            f"Recall@20 : {candidate_recall_20:.3f}",
            flush=True,
        )

        print(
            f"Rerank R@5: {reranked_recall_5:.3f}",
            flush=True,
        )

        print(
            f"MRR       : {reranked_mrr:.3f}",
            flush=True,
        )

    summary = {
        "cases": len(results),
        "candidate_recall@5": mean(
            item["recall@5"]
            for item in results
        ),
        "candidate_recall@10": mean(
            item["recall@10"]
            for item in results
        ),
        "candidate_recall@20": mean(
            item["recall@20"]
            for item in results
        ),
        "reranked_recall@5": mean(
            item["reranked_recall@5"]
            for item in results
        ),
        "mrr": mean(
            item["mrr"]
            for item in results
        ),
        "source_recall@10": mean(
            item["source_recall@10"]
            for item in results
        ),
    }

    return {
        "summary": summary,
        "cases": results,
    }


# ============================================================
# End-to-End Evaluation
# ============================================================

def evaluate_end_to_end(
    engine,
    cases: List[EvaluationCase] | None = None,
) -> Dict:
    """
    Run the full LangGraph pipeline.

    This is intentionally separate because it invokes the
    local Qwen model and is more expensive.
    """

    if cases is None:
        cases = get_retrieval_cases()

    results = []

    for index, case in enumerate(
        cases,
        start=1,
    ):

        print(
            "\n" + "-" * 70,
            flush=True,
        )

        print(
            f"End-to-end case "
            f"{index}/{len(cases)}",
            flush=True,
        )

        print(
            f"Query: {case.query}",
            flush=True,
        )

        result = engine.invoke(
            case.query
        )

        answer = result.get(
            "final_answer",
            "",
        )

        grounding_review = result.get(
            "grounding_review",
            {},
        )

        citation_validation = result.get(
            "citation_validation",
            {},
        )

        grounding_ok = bool(
            grounding_review.get(
                "grounded",
                False,
            )
        )

        citation_ok = bool(
            citation_validation.get(
                "valid",
                False,
            )
        )

        answer_coverage = (
            answer_term_coverage(
                answer,
                case.expected_answer_terms,
            )
        )

        results.append(
            {
                "query": case.query,
                "grounded": grounding_ok,
                "citations_valid": citation_ok,
                "answer_term_coverage": answer_coverage,
                "answer": answer,
                "grounding_review": grounding_review,
                "citation_validation": citation_validation,
            }
        )

        print(
            f"Grounded              : {grounding_ok}",
            flush=True,
        )

        print(
            f"Citations valid       : {citation_ok}",
            flush=True,
        )

        print(
            f"Answer term coverage  : "
            f"{answer_coverage:.3f}",
            flush=True,
        )

    summary = {
        "cases": len(results),
        "grounded_rate": mean(
            float(
                item["grounded"]
            )
            for item in results
        ),
        "citation_valid_rate": mean(
            float(
                item["citations_valid"]
            )
            for item in results
        ),
        "answer_term_coverage": mean(
            item[
                "answer_term_coverage"
            ]
            for item in results
        ),
    }

    return {
        "summary": summary,
        "cases": results,
    }


# ============================================================
# Main evaluation entry
# ============================================================

def run_evaluation(
    run_end_to_end: bool = False,
) -> Dict:

    print(
        "=" * 70,
        flush=True,
    )

    print(
        "SickleGuide Evaluation",
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )

    engine = create_rag_engine()

    print(
        "\nInitializing engine...",
        flush=True,
    )

    engine.initialize()

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70,
        flush=True,
    )

    print(
        "RETRIEVAL EVALUATION",
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )

    retrieval = evaluate_retrieval(
        engine
    )

    print(
        "\nRetrieval summary:",
        flush=True,
    )

    for key, value in (
        retrieval["summary"].items()
    ):
        print(
            f"{key:25s}: "
            f"{value:.3f}"
            if isinstance(
                value,
                float,
            )
            else f"{key:25s}: {value}",
            flush=True,
        )

    # --------------------------------------------------------
    # End-to-end
    # --------------------------------------------------------

    end_to_end = None

    if run_end_to_end:

        print(
            "\n" + "=" * 70,
            flush=True,
        )

        print(
            "END-TO-END EVALUATION",
            flush=True,
        )

        print(
            "=" * 70,
            flush=True,
        )

        end_to_end = (
            evaluate_end_to_end(
                engine
            )
        )

        print(
            "\nEnd-to-end summary:",
            flush=True,
        )

        for key, value in (
            end_to_end["summary"].items()
        ):
            print(
                f"{key:25s}: "
                f"{value:.3f}"
                if isinstance(
                    value,
                    float,
                )
                else f"{key:25s}: {value}",
                flush=True,
            )

    return {
        "retrieval": retrieval,
        "end_to_end": end_to_end,
    }