import asyncio
from typing import Any

from fastapi import (
    APIRouter,
    HTTPException,
)

from pydantic import BaseModel


router = APIRouter(
    prefix="/evaluation",
    tags=["Evaluation"],
)


class EvaluationRequest(
    BaseModel
):
    full: bool = False


def run_eval_sync(
    full: bool,
):

    from src.evaluation.evaluate import (
        run_evaluation,
    )

    return run_evaluation(
        run_end_to_end=full
    )


@router.post("/run")
async def run_evaluation(
    request: EvaluationRequest,
):

    try:

        result = await asyncio.to_thread(
            run_eval_sync,
            request.full,
        )

        return {
            "success": True,
            "mode": (
                "full"
                if request.full
                else "retrieval"
            ),
            "report": result,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Evaluation failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc