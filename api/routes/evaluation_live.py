from fastapi import APIRouter, Query

from src.evaluation.live import evaluation_history, latest_evaluation

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


@router.get("/live")
async def get_live_evaluation(limit: int = Query(default=50, ge=1, le=100)):
    return {
        "latest": latest_evaluation(),
        "history": evaluation_history(limit),
    }
