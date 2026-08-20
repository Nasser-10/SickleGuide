from typing import Callable, Dict, List, Sequence, Optional

from src.evaluation.dataset import EvaluationCase, get_retrieval_cases
from src.evaluation.metrics import answer_term_coverage, mean, retrieval_metrics_at_k, source_recall_at_k
from src.generation.llm import create_rag_engine

K_VALUES: Sequence[int] = (3, 5, 10)
ProgressCallback = Optional[Callable[[Dict], None]]


def _progress(callback: ProgressCallback, payload: Dict) -> None:
    if callback:
        callback(payload)


def _retrieval_summary(results: List[Dict]) -> Dict[str, float]:
    return {
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


def evaluate_retrieval(engine, cases: List[EvaluationCase] | None = None, progress_callback: ProgressCallback = None) -> Dict:
    cases = cases or get_retrieval_cases()
    results = []
    total = len(cases)
    for index, case in enumerate(cases, start=1):
        _progress(progress_callback, {"stage": "retrieval", "progress": int((index - 1) / max(total, 1) * 70), "completed": index - 1, "total": total, "message": f"Evaluating retrieval case {index}/{total}"})
        candidates = engine.unified_retriever.retrieve(case.query, final_k=engine.candidate_k)
        reranked = engine.reranker.rerank(query=case.query, documents=candidates, top_k=max(10, engine.final_k))
        candidate_k_metrics = retrieval_metrics_at_k(candidates, case.expected_sources, case.expected_pages, case.expected_keywords, K_VALUES)
        reranked_k_metrics = retrieval_metrics_at_k(reranked, case.expected_sources, case.expected_pages, case.expected_keywords, K_VALUES)
        source_recall_10 = source_recall_at_k(candidates, case.expected_sources, k=10)
        result = {"query": case.query, "k_metrics": candidate_k_metrics, "reranked_k_metrics": reranked_k_metrics, "source_recall@10": source_recall_10, "candidate_count": len(candidates), "reranked_count": len(reranked), "top_sources": [d.metadata.get("source", "") for d in reranked[:10]], "top_pages": [d.metadata.get("page_number") for d in reranked[:10]]}
        results.append(result)
        summary = _retrieval_summary(results)
        _progress(progress_callback, {"stage": "retrieval", "progress": int(index / max(total, 1) * 70), "completed": index, "total": total, "message": f"Case {index}/{total} complete", "partial": {"retrieval": {"summary": summary, "cases": results}}})
    return {"summary": _retrieval_summary(results), "cases": results, "k_values": list(K_VALUES)}


def evaluate_end_to_end(engine, cases: List[EvaluationCase] | None = None, progress_callback: ProgressCallback = None, start_progress: int = 70) -> Dict:
    cases = cases or get_retrieval_cases()
    results = []
    total = len(cases)
    for index, case in enumerate(cases, start=1):
        _progress(progress_callback, {"stage": "grounding", "progress": start_progress + int((index - 1) / max(total, 1) * (100 - start_progress)), "completed": index - 1, "total": total, "message": f"Checking grounding and citations {index}/{total}"})
        result = engine.invoke(case.query)
        answer = result.get("final_answer", "")
        grounding_review = result.get("grounding_review", {})
        citation_validation = result.get("citation_validation", {})
        results.append({"query": case.query, "grounded": bool(grounding_review.get("grounded", False)), "citations_valid": bool(citation_validation.get("valid", False)), "answer_term_coverage": answer_term_coverage(answer, case.expected_answer_terms), "answer": answer, "grounding_review": grounding_review, "citation_validation": citation_validation})
        summary = {"cases": len(results), "grounded_rate": mean(float(item["grounded"]) for item in results), "citation_valid_rate": mean(float(item["citations_valid"]) for item in results), "answer_term_coverage": mean(item["answer_term_coverage"] for item in results)}
        _progress(progress_callback, {"stage": "grounding", "progress": start_progress + int(index / max(total, 1) * (100 - start_progress)), "completed": index, "total": total, "message": f"E2E case {index}/{total} complete", "partial": {"end_to_end": {"summary": summary, "cases": results}}})
    summary = {"cases": len(results), "grounded_rate": mean(float(item["grounded"]) for item in results), "citation_valid_rate": mean(float(item["citations_valid"]) for item in results), "answer_term_coverage": mean(item["answer_term_coverage"] for item in results)}
    return {"summary": summary, "cases": results}


def run_evaluation(run_end_to_end: bool = False, progress_callback: ProgressCallback = None) -> Dict:
    _progress(progress_callback, {"stage": "initializing", "progress": 2, "message": "Initializing retrieval and reranking engine..."})
    engine = create_rag_engine()
    engine.initialize()
    _progress(progress_callback, {"stage": "ready", "progress": 5, "message": "Engine ready. Starting benchmark..."})
    retrieval = evaluate_retrieval(engine, progress_callback=progress_callback)
    end_to_end = None
    if run_end_to_end:
        end_to_end = evaluate_end_to_end(engine, progress_callback=progress_callback)
    _progress(progress_callback, {"stage": "complete", "progress": 100, "message": "Evaluation complete"})
    return {"retrieval": retrieval, "end_to_end": end_to_end, "benchmark": {"k_values": list(K_VALUES), "e2e_enabled": run_end_to_end}}
