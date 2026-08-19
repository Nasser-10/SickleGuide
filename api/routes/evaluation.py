import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


class EvaluationRequest(BaseModel):
    full: bool = False


RUBRIC = [
    {
        "id": "retrieval_quality",
        "title": "Retrieval Quality",
        "description": "Measures whether the retrieval and reranking stages surface the most relevant clinical evidence.",
        "metrics": ["candidate_recall@5", "reranked_recall@5", "mrr"],
        "mode": "automatic",
    },
    {
        "id": "grounding_faithfulness",
        "title": "Answer Grounding & Faithfulness",
        "description": "Checks whether generated answers stay supported by retrieved evidence and whether citations are valid.",
        "metrics": ["grounded_rate", "citation_valid_rate", "answer_term_coverage"],
        "mode": "automatic",
    },
    {
        "id": "architecture_fullstack",
        "title": "System Architecture & Full-Stack Implementation",
        "description": "Assesses the end-to-end ingestion, retrieval, graph, API and frontend integration.",
        "metrics": [],
        "mode": "demo_review",
    },
    {
        "id": "evaluation_metrics",
        "title": "Evaluation & Metrics Implementation",
        "description": "Assesses whether the system has reproducible evaluation datasets, retrieval metrics and end-to-end quality reporting.",
        "metrics": ["recall@k", "mrr", "grounding", "citation_validity"],
        "mode": "automatic_plus_review",
    },
    {
        "id": "clinical_safety",
        "title": "Clinical Safety & Responsible AI",
        "description": "Reviews evidence-first behavior, citation support, uncertainty handling and safe failure when evidence is insufficient.",
        "metrics": ["grounded_rate", "citation_valid_rate"],
        "mode": "automatic_plus_review",
    },
    {
        "id": "presentation_demo",
        "title": "Presentation, Communication & Live Demo",
        "description": "Demo-facing criterion covering clarity, usability, workflow visibility and reliability during a live walkthrough.",
        "metrics": [],
        "mode": "demo_review",
    },
    {
        "id": "innovation",
        "title": "Innovation & Out-of-the-Box Thinking",
        "description": "Highlights differentiating capabilities such as graph retrieval, evidence fusion, claim checking and transparent evidence exploration.",
        "metrics": [],
        "mode": "demo_review",
    },
]


def run_eval_sync(full: bool):
    from src.evaluation.evaluate import run_evaluation
    return run_evaluation(run_end_to_end=full)


def _rubric_snapshot(report: dict[str, Any], full: bool) -> list[dict[str, Any]]:
    retrieval = (report.get("retrieval") or {}).get("summary") or {}
    end_to_end = (report.get("end_to_end") or {}).get("summary") or {}

    measured = {
        "retrieval_quality": bool(retrieval),
        "grounding_faithfulness": bool(end_to_end),
        "architecture_fullstack": True,
        "evaluation_metrics": bool(retrieval),
        "clinical_safety": bool(end_to_end),
        "presentation_demo": True,
        "innovation": True,
    }

    result = []
    for item in RUBRIC:
        result.append({
            **item,
            "status": "measured" if measured[item["id"]] and item["mode"] != "demo_review" else "demo review",
            "score": None,
        })
    return result


@router.post("/run")
async def run_evaluation(request: EvaluationRequest):
    try:
        result = await asyncio.to_thread(run_eval_sync, request.full)
        return {
            "success": True,
            "mode": "full" if request.full else "retrieval",
            "report": result,
            "rubric": _rubric_snapshot(result, request.full),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {type(exc).__name__}: {exc}",
        ) from exc


@router.get("/rubric")
async def get_evaluation_rubric():
    return {"rubric": RUBRIC}
