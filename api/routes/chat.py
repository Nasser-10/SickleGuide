import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.generation.llm import create_rag_engine
from src.evaluation.dataset import get_retrieval_cases
from src.evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank

router = APIRouter(prefix="/chat", tags=["Chat"])

_rag_engine = None
_chat_memory: Dict[str, List[Dict[str, str]]] = {}
_MAX_MEMORY_MESSAGES = 12


def get_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = create_rag_engine()
    return _rag_engine


def get_memory(chat_id: Optional[str]) -> List[Dict[str, str]]:
    if not chat_id:
        return []
    return list(_chat_memory.get(chat_id, []))[-_MAX_MEMORY_MESSAGES:]


def save_memory(chat_id: Optional[str], messages: List[Dict[str, str]]) -> None:
    if not chat_id:
        return
    cleaned = [
        {"role": str(m["role"]), "content": str(m["content"])[:4000]}
        for m in messages[-_MAX_MEMORY_MESSAGES:]
        if m.get("role") in {"user", "assistant"} and m.get("content")
    ]
    _chat_memory[chat_id] = cleaned


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
    live_evaluation: Dict[str, Any] = {}


def build_chat_response(result: Dict[str, Any], query: str, chat_id: Optional[str]) -> ChatResponse:
    sources = []
    for index, document in enumerate(result.get("final_documents", []), start=1):
        metadata = document.metadata or {}
        citation = metadata.get(
            "citation",
            f"{metadata.get('source', 'Unknown source')} — Page {metadata.get('page_number', 'Unknown')}",
        )
        score = metadata.get("reranker_score")
        sources.append(Source(
            evidence_id=index,
            source=str(metadata.get("source", "Unknown source")),
            page_number=metadata.get("page_number"),
            citation=str(citation),
            reranker_score=float(score) if score is not None else None,
        ))

    grounding_review = result.get("grounding_review", {})
    citation_validation = result.get("citation_validation")
    return ChatResponse(
        query=query,
        chat_id=chat_id,
        answer=result.get("final_answer", ""),
        grounded=bool(grounding_review.get("grounded", False)),
        citations_valid=bool(citation_validation.get("valid", False)) if citation_validation else False,
        sources=sources,
        grounding_review=grounding_review,
        citation_validation=citation_validation,
        live_evaluation=result.get("live_evaluation", {}),
    )


def _live_evaluation(result: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Run cheap post-answer checks. Precision is benchmark-grounded when the query matches a labeled case."""
    final_documents = result.get("final_documents", [])
    grounding = result.get("grounding_review", {})
    citation = result.get("citation_validation", {})

    live = {
        "grounded": bool(grounding.get("grounded", False)),
        "citations_valid": bool(citation.get("valid", False)),
        "evidence_count": len(final_documents),
        "precision@5": None,
        "recall@5": None,
        "mrr": None,
        "precision_status": "available only for labeled evaluation questions",
    }

    normalized_query = query.strip().lower()
    try:
        case = next((c for c in get_retrieval_cases() if c.query.strip().lower() == normalized_query), None)
        if case:
            documents = result.get("retrieved_documents", final_documents)
            live["precision@5"] = precision_at_k(
                documents, case.expected_sources, case.expected_pages, case.expected_keywords, k=5
            )
            live["recall@5"] = recall_at_k(
                documents, case.expected_sources, case.expected_pages, case.expected_keywords, k=5
            )
            live["mrr"] = reciprocal_rank(
                documents, case.expected_sources, case.expected_pages, case.expected_keywords
            )
            live["precision_status"] = "benchmark case"
    except Exception:
        pass

    return live


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        stored = get_memory(request.chat_id)
        supplied = [{"role": m.role, "content": m.content} for m in request.history]
        history = stored or supplied
        result = get_rag_engine().invoke(query, conversation_history=history)
        result["live_evaluation"] = _live_evaluation(result, query)
        save_memory(request.chat_id, history + [{"role": "user", "content": query}, {"role": "assistant", "content": result.get("final_answer", "")}])
        return build_chat_response(result, query, request.chat_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SickleGuide failed to process the request: {type(exc).__name__}: {exc}") from exc


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    stored = get_memory(request.chat_id)
    supplied = [{"role": m.role, "content": m.content} for m in request.history]
    history = stored or supplied

    async def event_stream():
        try:
            for stage, message in [
                ("retrieval", "Searching clinical evidence..."),
                ("reranking", "Ranking the most relevant evidence..."),
                ("generation", "Generating an evidence-grounded answer..."),
            ]:
                yield "data: " + json.dumps({"type": "status", "stage": stage, "message": message}) + "\n\n"
                await asyncio.sleep(0)

            result = await asyncio.to_thread(get_rag_engine().invoke, query, history)
            result["live_evaluation"] = _live_evaluation(result, query)
            response = build_chat_response(result, query, request.chat_id)

            save_memory(request.chat_id, history + [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response.answer},
            ])

            yield "data: " + json.dumps({
                "type": "grounding",
                "grounded": response.grounded,
                "citations_valid": response.citations_valid,
            }) + "\n\n"

            yield "data: " + json.dumps({
                "type": "live_evaluation",
                "evaluation": response.live_evaluation,
            }, ensure_ascii=False) + "\n\n"

            words = response.answer.split(" ")
            for index, word in enumerate(words):
                token = word if index == len(words) - 1 else word + " "
                yield "data: " + json.dumps({"type": "token", "content": token}, ensure_ascii=False) + "\n\n"
                await asyncio.sleep(0.018)

            yield "data: " + json.dumps({
                "type": "sources",
                "sources": [source.model_dump() for source in response.sources],
            }, ensure_ascii=False) + "\n\n"

            yield "data: " + json.dumps({
                "type": "done",
                "chat_id": response.chat_id,
                "grounded": response.grounded,
                "citations_valid": response.citations_valid,
            }) + "\n\n"
        except Exception as exc:
            yield "data: " + json.dumps({"type": "error", "message": f"{type(exc).__name__}: {exc}"}) + "\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.get("/memory/{chat_id}")
def get_chat_memory(chat_id: str):
    return {"chat_id": chat_id, "messages": get_memory(chat_id)}


@router.delete("/memory/{chat_id}")
def clear_chat_memory(chat_id: str):
    _chat_memory.pop(chat_id, None)
    return {"chat_id": chat_id, "cleared": True}
