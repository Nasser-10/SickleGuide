from typing import Any, Dict, List, Optional, TypedDict
import json

from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from src.generation.citation import format_answer_with_citations, validate_citations
from src.generation.prompt import build_grounded_regeneration_prompt, build_no_evidence_prompt, build_prompt
from src.generation.safety import apply_safety_notice, assess_query, build_safety_instruction
from src.retrieval.bm25 import create_bm25_retriever
from src.retrieval.vector_store import create_vector_store
from src.graph.graph_retriever import create_graph_retriever, load_graph
from src.retrieval.hybrid import create_hybrid_graph_retriever
from src.retrieval.reranker import create_reranker

DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.0
# Keep answers concise for an interactive local demo; grounding/citation checks still run.
DEFAULT_NUM_PREDICT = 800
DEFAULT_DENSE_K = 15
DEFAULT_BM25_K = 15
DEFAULT_GRAPH_K = 15
DEFAULT_CANDIDATE_K = 20
DEFAULT_FINAL_K = 5
MAX_GROUNDING_RETRIES = 1
MAX_HISTORY_MESSAGES = 12

class GroundingReview(BaseModel):
    grounded: bool = Field(description="True when all important medical claims are supported directly or faithfully paraphrased from retrieved evidence.")
    unsupported_claims: List[str] = Field(default_factory=list)
    reasoning: str = Field(default="")

class RAGState(TypedDict, total=False):
    query: str
    conversation_history: List[Dict[str, str]]
    retrieved_documents: List[Document]
    final_documents: List[Document]
    safety_result: Any
    safety_instruction: str
    raw_answer: str
    grounded_answer: str
    final_answer: str
    citation_validation: Dict[str, Any]
    grounding_review: Dict[str, Any]
    grounding_retry_count: int
    grounding_failed: bool
    error: Optional[str]

class SickleGuideRAG:
    def __init__(self, model_name: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL, temperature: float = DEFAULT_TEMPERATURE, num_predict: int = DEFAULT_NUM_PREDICT, dense_k: int = DEFAULT_DENSE_K, bm25_k: int = DEFAULT_BM25_K, graph_k: int = DEFAULT_GRAPH_K, candidate_k: int = DEFAULT_CANDIDATE_K, final_k: int = DEFAULT_FINAL_K, graph_path: str = "data/processed/graph.json", chunks_path: str = "data/processed/chunks.json"):
        self.model_name, self.base_url, self.temperature, self.num_predict = model_name, base_url, temperature, num_predict
        self.dense_k, self.bm25_k, self.graph_k, self.candidate_k, self.final_k = dense_k, bm25_k, graph_k, candidate_k, final_k
        self.graph_path, self.chunks_path = graph_path, chunks_path
        self._initialized = False
        self.llm = self.documents = self.bm25 = self.vector_store = self.graph = self.graph_retriever = self.unified_retriever = self.reranker = self.reviewer = self.rag_graph = None

    def initialize(self) -> None:
        if self._initialized: return
        with open(self.chunks_path, "r", encoding="utf-8") as file: data = json.load(file)
        self.documents = [Document(page_content=item["page_content"], metadata=item.get("metadata", {})) for item in data]
        if not self.documents: raise RuntimeError("No processed documents found.")
        self.bm25 = create_bm25_retriever(self.documents)
        self.vector_store = create_vector_store()
        self.graph = load_graph(self.graph_path)
        self.graph_retriever = create_graph_retriever(self.graph, self.documents)
        self.unified_retriever = create_hybrid_graph_retriever(bm25_retriever=self.bm25, vector_store=self.vector_store, graph_retriever=self.graph_retriever, dense_k=self.dense_k, bm25_k=self.bm25_k, graph_k=self.graph_k, final_k=self.candidate_k)
        self.reranker = create_reranker()
        self.llm = ChatOllama(model=self.model_name, base_url=self.base_url, temperature=self.temperature, num_predict=self.num_predict)
        self.reviewer = self.llm.with_structured_output(GroundingReview, method="json_schema")
        self.rag_graph = self._build_graph(); self._initialized = True

    @staticmethod
    def _clean_history(history: Optional[List[Dict[str, str]]]) -> List[Dict[str, str]]:
        if not history: return []
        cleaned = []
        for item in history[-MAX_HISTORY_MESSAGES:]:
            if not isinstance(item, dict): continue
            role = str(item.get("role", "")).strip().lower(); content = str(item.get("content", "")).strip()
            if role in {"user", "assistant"} and content: cleaned.append({"role": role, "content": content[:4000]})
        return cleaned

    @staticmethod
    def _format_history(history: List[Dict[str, str]]) -> str:
        return "\n".join(f"{item['role'].upper()}: {item['content']}" for item in history) if history else "No previous conversation."

    def _safety_node(self, state: RAGState) -> RAGState:
        result = assess_query(state["query"], [])
        return {"safety_result": result, "safety_instruction": build_safety_instruction(result)}

    def _retrieve_node(self, state: RAGState) -> RAGState:
        candidates = self.unified_retriever.retrieve(state["query"], final_k=self.candidate_k)
        return {"retrieved_documents": candidates}

    def _rerank_node(self, state: RAGState) -> RAGState:
        docs = self.reranker.rerank(query=state["query"], documents=state.get("retrieved_documents", []), top_k=self.final_k)
        return {"final_documents": docs}

    def _generation_node(self, state: RAGState) -> RAGState:
        query, documents = state["query"], state.get("final_documents", [])
        history = self._clean_history(state.get("conversation_history", []))
        safety_result = assess_query(query, documents); safety_instruction = build_safety_instruction(safety_result)
        prompt = build_prompt(query=query, documents=documents, safety_instruction=safety_instruction) if documents else build_no_evidence_prompt(query=query, safety_instruction=safety_instruction)
        prompt += f"\n\nCONVERSATION CONTEXT:\n\n{self._format_history(history)}\n\nIMPORTANT:\nThe conversation context is ONLY for continuity, never evidence. Only retrieved evidence may support medical facts. If evidence is insufficient, say so."
        response = self.llm.invoke(prompt); content = str(getattr(response, "content", response)).strip()
        return {"raw_answer": content, "grounded_answer": content, "safety_result": safety_result, "safety_instruction": safety_instruction, "grounding_retry_count": 0, "grounding_failed": False}

    def _grounding_review_node(self, state: RAGState) -> RAGState:
        query = state["query"]; answer = state.get("grounded_answer", state.get("raw_answer", "")); documents = state.get("final_documents", []); evidence_blocks = []
        for i, document in enumerate(documents, 1):
            citation = document.metadata.get("citation") or f"{document.metadata.get('source', 'Unknown source')} — Page {document.metadata.get('page_number', 'Unknown')}"
            evidence_blocks.append(f"[Evidence {i}]\nCitation: {citation}\nContent:\n{document.page_content}")
        evidence_text = "\n\n".join(evidence_blocks)
        prompt = f"""You are a strict but fair medical RAG grounding evaluator.\nQUESTION:\n{query}\n\nRETRIEVED EVIDENCE:\n{evidence_text}\n\nGENERATED ANSWER:\n{answer}\n\nA claim is grounded only when directly supported or faithfully paraphrased by the evidence. Do not use general medical knowledge, infer recommendations, comparative effectiveness, or unsupported conclusions. Preserve uncertainty. Do not treat conversation as evidence. Faithful concise summaries count as grounded. Return a structured grounding decision."""
        review = self.reviewer.invoke(prompt)
        return {"grounding_review": review.model_dump() if isinstance(review, GroundingReview) else GroundingReview.model_validate(review).model_dump()}

    def _route_after_grounding(self, state: RAGState) -> str:
        review = state.get("grounding_review", {}); grounded = bool(review.get("grounded", False)); retry = state.get("grounding_retry_count", 0)
        if grounded: return "citations"
        if retry < MAX_GROUNDING_RETRIES: return "regenerate"
        return "grounding_failure"

    def _regenerate_node(self, state: RAGState) -> RAGState:
        prompt = build_grounded_regeneration_prompt(query=state["query"], documents=state.get("final_documents", []), previous_answer=state.get("grounded_answer", ""), unsupported_claims=state.get("grounding_review", {}).get("unsupported_claims", []), safety_instruction=state.get("safety_instruction", "")); prompt += f"\n\nCONVERSATION CONTEXT:\n{self._format_history(self._clean_history(state.get('conversation_history', [])))}\n\nConversation context is NOT evidence. Only retrieved evidence can support medical claims."
        response = self.llm.invoke(prompt); content = str(getattr(response, "content", response)).strip(); retry = state.get("grounding_retry_count", 0)
        return {"grounded_answer": content, "raw_answer": content, "grounding_retry_count": retry + 1}

    def _grounding_failure_node(self, state: RAGState) -> RAGState:
        documents = state.get("final_documents", []); safe_answer = "I could not generate a sufficiently evidence-grounded answer to this question from the retrieved SickleGuide sources. The available evidence did not support all of the claims required for a reliable answer."
        if documents:
            seen = set(); lines = []
            for i, d in enumerate(documents, 1):
                citation = d.metadata.get("citation") or f"{d.metadata.get('source', 'Unknown source')} — Page {d.metadata.get('page_number', 'Unknown')}"
                if citation not in seen: seen.add(citation); lines.append(f"[{i}] {citation}")
            if lines: safe_answer += "\n\nThe retrieved sources were:\n" + "\n".join(lines)
        return {"grounding_failed": True, "grounded_answer": safe_answer, "final_answer": safe_answer}

    def _citation_node(self, state: RAGState) -> RAGState:
        answer = state.get("grounded_answer", state.get("raw_answer", "")); documents = state.get("final_documents", []); final_answer = format_answer_with_citations(answer, documents); citation_map = {}
        for i, document in enumerate(documents, 1): citation_map[i] = document.metadata.get("citation") or f"{document.metadata.get('source', 'Unknown source')} — Page {document.metadata.get('page_number', 'Unknown')}"
        return {"final_answer": final_answer, "citation_validation": validate_citations(final_answer, citation_map)}

    def _safety_output_node(self, state: RAGState) -> RAGState:
        return {"final_answer": apply_safety_notice(state.get("final_answer", ""), state["safety_result"])}

    def _build_graph(self):
        workflow = StateGraph(RAGState)
        for name, node in [("safety", self._safety_node),("retrieve", self._retrieve_node),("rerank", self._rerank_node),("generate", self._generation_node),("grounding_review", self._grounding_review_node),("regenerate", self._regenerate_node),("grounding_failure", self._grounding_failure_node),("citations", self._citation_node),("safety_output", self._safety_output_node)]: workflow.add_node(name,node)
        workflow.set_entry_point("safety");workflow.add_edge("safety","retrieve");workflow.add_edge("retrieve","rerank");workflow.add_edge("rerank","generate");workflow.add_edge("generate","grounding_review");workflow.add_conditional_edges("grounding_review",self._route_after_grounding,{"regenerate":"regenerate","citations":"citations","grounding_failure":"grounding_failure"});workflow.add_edge("regenerate","grounding_review");workflow.add_edge("grounding_failure","safety_output");workflow.add_edge("citations","safety_output");workflow.add_edge("safety_output",END);return workflow.compile()

    def invoke(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        if not isinstance(query,str): raise TypeError("query must be a string")
        query=query.strip()
        if not query: raise ValueError("query cannot be empty")
        self.initialize();return self.rag_graph.invoke({"query":query,"conversation_history":self._clean_history(conversation_history)})

    def answer(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str: return self.invoke(query,conversation_history).get("final_answer","")


def create_rag_engine(model_name: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL, temperature: float = DEFAULT_TEMPERATURE, num_predict: int = DEFAULT_NUM_PREDICT) -> SickleGuideRAG: return SickleGuideRAG(model_name=model_name,base_url=base_url,temperature=temperature,num_predict=num_predict)

def get_llm(model_name: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL, temperature: float = DEFAULT_TEMPERATURE, num_predict: int = DEFAULT_NUM_PREDICT): return ChatOllama(model=model_name,base_url=base_url,temperature=temperature,num_predict=num_predict)

def generate(prompt: str, llm=None) -> str:
    if not isinstance(prompt,str): raise TypeError("prompt must be a string")
    if not prompt.strip(): raise ValueError("prompt cannot be empty")
    model=llm or get_llm();response=model.invoke(prompt);return str(getattr(response,"content",response)).strip()
