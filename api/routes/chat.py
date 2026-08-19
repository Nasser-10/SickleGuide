import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.evaluation.live import record_chat_evaluation
from src.generation.llm import create_rag_engine
from src.memory.store import get_memory_store

router = APIRouter(prefix="/chat", tags=["Chat"])
_rag_engine = None
_memory = get_memory_store()


def get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = create_rag_engine()
    return _rag_engine


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    chat_id: Optional[str] = Field(default=None, max_length=100)
    history: List[ChatMessage] = Field(default_factory=list, max_length=12)


class Source(BaseModel):
    evidence_id: Optional[int] = None
    source: str
    page_number: Optional[int] = None
    citation: str
    reranker_score: Optional[float] = None


class ChatResponse(BaseModel):
    query: str
    chat_id: Optional[str]
    answer: str
    grounded: bool
    citations_valid: bool
    sources: List[Source]
    grounding_review: Dict[str, Any]
    citation_validation: Optional[Dict[str, Any]] = None


def _safe_chat_id(chat_id: Optional[str]) -> str:
    value = str(chat_id or "").strip()
    if not value or len(value) > 100:
        raise HTTPException(status_code=400, detail="A valid chat_id is required for persistent memory.")
    return value


def _get_history(request: ChatRequest, chat_id: str) -> List[Dict[str, str]]:
    stored = _memory.load(chat_id)
    if stored:
        return stored
    return [{"role": m.role, "content": m.content} for m in request.history]


def build_chat_response(result: Dict[str, Any], query: str, chat_id: Optional[str]) -> ChatResponse:
    sources = []
    for index, document in enumerate(result.get("final_documents", []), start=1):
        metadata = document.metadata or {}
        citation = metadata.get("citation", f"{metadata.get('source', 'Unknown source')} — Page {metadata.get('page_number', 'Unknown')}")
        score = metadata.get("reranker_score")
        sources.append(Source(evidence_id=index, source=str(metadata.get("source", "Unknown source")), page_number=metadata.get("page_number"), citation=str(citation), reranker_score=float(score) if score is not None else None))
    grounding_review = result.get("grounding_review", {})
    citation_validation = result.get("citation_validation")
    return ChatResponse(query=query, chat_id=chat_id, answer=result.get("final_answer", ""), grounded=bool(grounding_review.get("grounded", False)), citations_valid=bool(citation_validation.get("valid", False)) if citation_validation else False, sources=sources, grounding_review=grounding_review, citation_validation=citation_validation)


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    chat_id = _safe_chat_id(request.chat_id)
    try:
        history = _get_history(request, chat_id)
        result = get_rag_engine().invoke(query, conversation_history=history)
        answer = result.get("final_answer", "")
        _memory.append_turn(chat_id, query, answer)
        record_chat_evaluation(query, result)
        return build_chat_response(result, query, chat_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SickleGuide failed to process the request: {type(exc).__name__}: {exc}") from exc


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    chat_id = _safe_chat_id(request.chat_id)
    history = _get_history(request, chat_id)

    async def event_stream():
        try:
            for stage, message in [("retrieval", "Searching clinical evidence..."), ("reranking", "Ranking the most relevant evidence..."), ("generation", "Generating an evidence-grounded answer...")]:
                yield "data: " + json.dumps({"type": "status", "stage": stage, "message": message}) + "\n\n"
                await asyncio.sleep(0)

            result = await asyncio.to_thread(get_rag_engine().invoke, query, history)
            response = build_chat_response(result, query, chat_id)

            words = response.answer.split(" ")
            for index, word in enumerate(words):
                token = word if index == len(words) - 1 else word + " "
                yield "data: " + json.dumps({"type": "token", "content": token}, ensure_ascii=False) + "\n\n"
                await asyncio.sleep(0.018)

            _memory.append_turn(chat_id, query, response.answer)
            yield "data: " + json.dumps({"type": "sources", "sources": [source.model_dump() for source in response.sources]}, ensure_ascii=False) + "\n\n"
            yield "data: " + json.dumps({"type": "evaluation_status", "stage": "starting", "progress": 5, "message": "Evaluating retrieval quality..."}) + "\n\n"
            yield "data: " + json.dumps({"type": "evaluation_status", "stage": "grounding", "progress": 45, "message": "Checking grounding and evidence support..."}) + "\n\n"
            live_evaluation = await asyncio.to_thread(record_chat_evaluation, query, result)
            yield "data: " + json.dumps({"type": "evaluation_status", "stage": "complete", "progress": 100, "message": "Evaluation complete"}) + "\n\n"
            yield "data: " + json.dumps({"type": "evaluation_result", "evaluation": live_evaluation}, ensure_ascii=False) + "\n\n"
            yield "data: " + json.dumps({"type": "done", "chat_id": chat_id, "grounded": response.grounded, "citations_valid": response.citations_valid}) + "\n\n"
        except Exception as exc:
            yield "data: " + json.dumps({"type": "error", "message": f"{type(exc).__name__}: {exc}"}) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
