from typing import Dict, List, Sequence

from src.evaluation.dataset import EvaluationCase, get_retrieval_cases
from src.evaluation.metrics import (
    answer_term_coverage,
    mean,
    retrieval_metrics_at_k,
    source_recall_at_k,
)
from src.generation.llm import create_rag_engine

K_VALUES: Sequence[int] = (3, 5, 10)


def evaluate_retrieval(engine, cases: List[EvaluationCase] | None = None) -> Dict:
    if cases is None:
        cases = get_retrieval_cases()

    results = []
    for index, case in enumerate(cases, start=1):
        print("\n" + "-" * 70, flush=True)
        print(f"Retrieval case {index}/{len(cases)}", flush=True)
        print(f"Query: {case.query}", flush=True)

        # One retrieval + one rerank pass per case. All K metrics are calculated
        # from these same ranked lists, so K=3/5/10 comparisons are consistent.
        candidates = engine.unified_retriever.retrieve(case.query, final_k=engine.candidate_k)
        reranked = engine.reranker.rerank(query=case.query, documents=candidates, top_k=max(10, engine.final_k))

        candidate_k_metrics = retrieval_metrics_at_k(
            candidates,
            case.expected_sources,
            case.expected_pages,
            case.expected_keywords,
            K_VALUES,
        )
        reranked_k_metrics = retrieval_metrics_at_k(
            reranked,
            case.expected_sources,
            case.expected_pages,
            case.expected_keywords,
            K_VALUES,
        )
        source_recall_10 = source_recall_at_k(candidates, case.expected_sources, k=10)

        result = {
            "query": case.query,
            "k_metrics": candidate_k_metrics,
            "reranked_k_metrics": reranked_k_metrics,
            "source_recall@10": source_recall_10,
            "candidate_count": len(candidates),
            "reranked_count": len(reranked),
            "top_sources": [d.metadata.get("source", "") for d in reranked[:10]],
            "top_pages": [d.metadata.get("page_number") for d in reranked[:10]],
        }
        results.append(result)

        for k in K_VALUES:
            print(
                f"K={k:<2} | Precision {candidate_k_metrics[str(k)]['precision']:.3f} | "
                f"Recall {candidate_k_metrics[str(k)]['recall']:.3f} | "
                f"MRR {candidate_k_metrics[str(k)]['mrr']:.3f} | "
                f"Reranked P {reranked_k_metrics[str(k)]['precision']:.3f}",
                flush=True,
            )

    summary = {
        "cases": len(results),
        "k_values": list(K_VALUES),
        "precision@3": mean(item["k_metrics"]["3"]["precision"] for item in results),
        "precision@5": mean(item["k_metrics"]["5"]["precision"] for item in results),
        "precision@10": mean(item["k_metrics"]["10"]["precision"] for item in results),
        "reranked_precision@3": mean(item["reranked_k_metrics"]["3"]["precision"] for item in results),
        "reranked_precision@5": mean(item["reranked_k_metrics"]["5"]["precision"] for item in results),
        "reranked_precision@10": mean(item["reranked_k_metrics"]["10"]["precision"] for item in results),
        "recall@3": mean(item["k_metrics"]["3"]["recall"] for item in results),
        "recall@5": mean(item["k_metrics"]["5"]["recall"] for item in results),
        "recall@10": mean(item["k_metrics"]["10"]["recall"] for item in results),
        "reranked_recall@3": mean(item["reranked_k_metrics"]["3"]["recall"] for item in results),
        "reranked_recall@5": mean(item["reranked_k_metrics"]["5"]["recall"] for item in results),
        "reranked_recall@10": mean(item["reranked_k_metrics"]["10"]["recall"] for item in results),
        "mrr@3": mean(item["k_metrics"]["3"]["mrr"] for item in results),
        "mrr@5": mean(item["k_metrics"]["5"]["mrr"] for item in results),
        "mrr@10": mean(item["k_metrics"]["10"]["mrr"] for item in results),
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
        print(f"Grounded: {grounding_ok} | Citations: {citation_ok} | Coverage: {answer_coverage:.3f}", flush=True)

    summary = {
        "cases": len(results),
        "grounded_rate": mean(float(item["grounded"]) for item in results),
        "citation_valid_rate": mean(float(item["citations_valid"]) for item in results),
        "answer_term_coverage": mean(item["answer_term_coverage"] for item in results),
    }
    return {"summary": summary, "cases": results}


def run_evaluation(run_end_to_end: bool = False) -> Dict:
    """Run the fast benchmark by default; E2E LLM evaluation is opt-in."""
    print("=" * 70, flush=True)
    print("SickleGuide Evaluation", flush=True)
    print("=" * 70, flush=True)

    engine = create_rag_engine()
    print("\nInitializing engine...", flush=True)
    engine.initialize()

    print("\n" + "=" * 70, flush=True)
    print("RETRIEVAL BENCHMARK (K=3 / 5 / 10)", flush=True)
    print("=" * 70, flush=True)
    retrieval = evaluate_retrieval(engine)

    end_to_end = None
    if run_end_to_end:
        print("\n" + "=" * 70, flush=True)
        print("END-TO-END EVALUATION", flush=True)
        print("=" * 70, flush=True)
        end_to_end = evaluate_end_to_end(engine)

    return {"retrieval": retrieval, "end_to_end": end_to_end, "benchmark": {"k_values": list(K_VALUES), "e2e_enabled": run_end_to_end}}
