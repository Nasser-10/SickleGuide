from typing import Dict, List

from src.evaluation.dataset import EvaluationCase, get_retrieval_cases
from src.evaluation.metrics import (
    answer_term_coverage,
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    source_recall_at_k,
)
from src.generation.llm import create_rag_engine


def evaluate_retrieval(engine, cases: List[EvaluationCase] | None = None) -> Dict:
    if cases is None:
        cases = get_retrieval_cases()

    results = []
    for index, case in enumerate(cases, start=1):
        print("\n" + "-" * 70, flush=True)
        print(f"Retrieval case {index}/{len(cases)}", flush=True)
        print(f"Query: {case.query}", flush=True)

        candidates = engine.unified_retriever.retrieve(
            case.query,
            final_k=engine.candidate_k,
        )
        reranked = engine.reranker.rerank(
            query=case.query,
            documents=candidates,
            top_k=engine.final_k,
        )

        candidate_precision_5 = precision_at_k(
            candidates, case.expected_sources, case.expected_pages, case.expected_keywords, k=5
        )
        candidate_recall_5 = recall_at_k(
            candidates, case.expected_sources, case.expected_pages, case.expected_keywords, k=5
        )
        candidate_recall_10 = recall_at_k(
            candidates, case.expected_sources, case.expected_pages, case.expected_keywords, k=10
        )
        candidate_recall_20 = recall_at_k(
            candidates, case.expected_sources, case.expected_pages, case.expected_keywords, k=20
        )
        reranked_precision_5 = precision_at_k(
            reranked, case.expected_sources, case.expected_pages, case.expected_keywords, k=5
        )
        reranked_recall_5 = recall_at_k(
            reranked, case.expected_sources, case.expected_pages, case.expected_keywords, k=5
        )
        reranked_mrr = reciprocal_rank(
            reranked, case.expected_sources, case.expected_pages, case.expected_keywords
        )
        source_recall_10 = source_recall_at_k(candidates, case.expected_sources, k=10)

        result = {
            "query": case.query,
            "precision@5": candidate_precision_5,
            "recall@5": candidate_recall_5,
            "recall@10": candidate_recall_10,
            "recall@20": candidate_recall_20,
            "reranked_precision@5": reranked_precision_5,
            "reranked_recall@5": reranked_recall_5,
            "mrr": reranked_mrr,
            "source_recall@10": source_recall_10,
            "candidate_count": len(candidates),
            "reranked_count": len(reranked),
            "top_sources": [d.metadata.get("source", "") for d in reranked],
            "top_pages": [d.metadata.get("page_number") for d in reranked],
        }
        results.append(result)

        print(f"Precision@5          : {candidate_precision_5:.3f}", flush=True)
        print(f"Recall@5             : {candidate_recall_5:.3f}", flush=True)
        print(f"Reranked Precision@5 : {reranked_precision_5:.3f}", flush=True)
        print(f"Reranked Recall@5    : {reranked_recall_5:.3f}", flush=True)
        print(f"MRR                  : {reranked_mrr:.3f}", flush=True)

    summary = {
        "cases": len(results),
        "candidate_precision@5": mean(item["precision@5"] for item in results),
        "candidate_recall@5": mean(item["recall@5"] for item in results),
        "candidate_recall@10": mean(item["recall@10"] for item in results),
        "candidate_recall@20": mean(item["recall@20"] for item in results),
        "reranked_precision@5": mean(item["reranked_precision@5"] for item in results),
        "reranked_recall@5": mean(item["reranked_recall@5"] for item in results),
        "mrr": mean(item["mrr"] for item in results),
        "source_recall@10": mean(item["source_recall@10"] for item in results),
    }
    return {"summary": summary, "cases": results}


def evaluate_end_to_end(engine, cases: List[EvaluationCase] | None = None) -> Dict:
    if cases is None:
        cases = get_retrieval_cases()

    results = []
    for index, case in enumerate(cases, start=1):
        print("\n" + "-" * 70, flush=True)
        print(f"End-to-end case {index}/{len(cases)}", flush=True)
        print(f"Query: {case.query}", flush=True)

        result = engine.invoke(case.query)
        answer = result.get("final_answer", "")
        grounding_review = result.get("grounding_review", {})
        citation_validation = result.get("citation_validation", {})
        grounding_ok = bool(grounding_review.get("grounded", False))
        citation_ok = bool(citation_validation.get("valid", False))
        answer_coverage = answer_term_coverage(answer, case.expected_answer_terms)

        results.append({
            "query": case.query,
            "grounded": grounding_ok,
            "citations_valid": citation_ok,
            "answer_term_coverage": answer_coverage,
            "answer": answer,
            "grounding_review": grounding_review,
            "citation_validation": citation_validation,
        })

        print(f"Grounded             : {grounding_ok}", flush=True)
        print(f"Citations valid      : {citation_ok}", flush=True)
        print(f"Answer term coverage : {answer_coverage:.3f}", flush=True)

    summary = {
        "cases": len(results),
        "grounded_rate": mean(float(item["grounded"]) for item in results),
        "citation_valid_rate": mean(float(item["citations_valid"]) for item in results),
        "answer_term_coverage": mean(item["answer_term_coverage"] for item in results),
    }
    return {"summary": summary, "cases": results}


def run_evaluation(run_end_to_end: bool = False) -> Dict:
    print("=" * 70, flush=True)
    print("SickleGuide Evaluation", flush=True)
    print("=" * 70, flush=True)

    engine = create_rag_engine()
    print("\nInitializing engine...", flush=True)
    engine.initialize()

    print("\n" + "=" * 70, flush=True)
    print("RETRIEVAL EVALUATION", flush=True)
    print("=" * 70, flush=True)
    retrieval = evaluate_retrieval(engine)

    end_to_end = None
    if run_end_to_end:
        print("\n" + "=" * 70, flush=True)
        print("END-TO-END EVALUATION", flush=True)
        print("=" * 70, flush=True)
        end_to_end = evaluate_end_to_end(engine)

    return {"retrieval": retrieval, "end_to_end": end_to_end}
