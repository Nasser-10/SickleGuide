import copy
import threading
import uuid
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


class EvaluationRequest(BaseModel):
    full: bool = False


RUBRIC = [
    {"id": "retrieval_quality", "title": "Retrieval Quality", "description": "Measures whether retrieval and reranking surface relevant clinical evidence.", "metrics": ["precision@3", "precision@5", "precision@10", "recall@3", "recall@5", "recall@10", "mrr@3", "mrr@5", "mrr@10"], "mode": "automatic"},
    {"id": "grounding_faithfulness", "title": "Answer Grounding & Faithfulness", "description": "Checks whether generated answers stay supported by retrieved evidence and citations.", "metrics": ["grounded_rate", "citation_valid_rate", "answer_term_coverage"], "mode": "full_e2e"},
    {"id": "architecture_fullstack", "title": "System Architecture & Full-Stack Implementation", "description": "Assesses ingestion, retrieval, graph, API and frontend integration.", "metrics": [], "mode": "demo_review"},
    {"id": "evaluation_metrics", "title": "Evaluation & Metrics Implementation", "description": "Assesses reproducible evaluation datasets, retrieval metrics and end-to-end reporting.", "metrics": ["precision@3/5/10", "recall@3/5/10", "mrr@3/5/10"], "mode": "automatic"},
    {"id": "clinical_safety", "title": "Clinical Safety & Responsible AI", "description": "Reviews evidence-first behavior, uncertainty handling and safe failure.", "metrics": ["grounded_rate", "citation_valid_rate"], "mode": "full_e2e"},
    {"id": "presentation_demo", "title": "Presentation, Communication & Live Demo", "description": "Demo-facing criterion covering clarity, usability and reliability.", "metrics": [], "mode": "demo_review"},
    {"id": "innovation", "title": "Innovation & Out-of-the-Box Thinking", "description": "Highlights graph retrieval, evidence fusion, claim checking and transparent evidence exploration.", "metrics": [], "mode": "demo_review"},
]

_jobs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


def _set(job_id: str, **updates: Any) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(updates)


def _get(job_id: str):
    with _lock:
        return copy.deepcopy(_jobs.get(job_id))


def _run(job_id: str, full: bool) -> None:
    try:
        from src.evaluation.evaluate import run_evaluation

        def progress(event: Dict[str, Any]) -> None:
            _set(job_id, stage=event.get("stage", "working"), progress=int(event.get("progress", 0)), message=event.get("message", "Working..."), partial=event.get("partial"))

        report = run_evaluation(run_end_to_end=full, progress_callback=progress)
        _set(job_id, status="completed", stage="complete", progress=100, message="Evaluation complete", partial=report, report=report)
    except Exception as exc:
        _set(job_id, status="failed", stage="error", message=f"{type(exc).__name__}: {exc}")


@router.post("/run")
async def start_evaluation(request: EvaluationRequest):
    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {"job_id": job_id, "status": "running", "stage": "queued", "progress": 0, "message": "Evaluation queued...", "mode": "full" if request.full else "fast", "partial": None, "report": None}
    threading.Thread(target=_run, args=(job_id, request.full), daemon=True).start()
    return {"success": True, "job_id": job_id, "mode": "full" if request.full else "fast", "status": "running"}


@router.get("/run/{job_id}")
async def evaluation_status(job_id: str):
    result = _get(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Evaluation job not found.")
    return result


@router.get("/rubric")
async def get_evaluation_rubric():
    return {"rubric": RUBRIC}
