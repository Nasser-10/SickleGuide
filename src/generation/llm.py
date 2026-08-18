from typing import Any, Dict, List, Optional, TypedDict
import json

from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from src.generation.citation import (
    format_answer_with_citations,
    validate_citations,
)

from src.generation.prompt import (
    build_grounded_regeneration_prompt,
    build_no_evidence_prompt,
    build_prompt,
)

from src.generation.safety import (
    apply_safety_notice,
    assess_query,
    build_safety_instruction,
)

from src.retrieval.bm25 import (
    create_bm25_retriever,
)

from src.retrieval.vector_store import (
    create_vector_store,
)

from src.graph.graph_retriever import (
    create_graph_retriever,
    load_graph,
)

from src.retrieval.hybrid import (
    create_hybrid_graph_retriever,
)

from src.retrieval.reranker import (
    create_reranker,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_MODEL = "qwen2.5:7b"

DEFAULT_BASE_URL = "http://localhost:11434"

DEFAULT_TEMPERATURE = 0.0

DEFAULT_NUM_PREDICT = 1200

DEFAULT_DENSE_K = 15

DEFAULT_BM25_K = 15

DEFAULT_GRAPH_K = 15

DEFAULT_CANDIDATE_K = 20

DEFAULT_FINAL_K = 5

MAX_GROUNDING_RETRIES = 1


# ============================================================
# Grounding Review Schema
# ============================================================

class GroundingReview(BaseModel):
    """
    Strict medical RAG grounding review.
    """

    grounded: bool = Field(
        description=(
            "True when all important medical claims in the answer "
            "are supported directly or faithfully paraphrased from "
            "the retrieved evidence."
        )
    )

    unsupported_claims: List[str] = Field(
        default_factory=list,
        description=(
            "Important medical claims that are not supported "
            "by the retrieved evidence."
        ),
    )

    reasoning: str = Field(
        default="",
        description=(
            "Brief explanation of the grounding decision."
        ),
    )


# ============================================================
# LangGraph State
# ============================================================

class RAGState(TypedDict, total=False):

    query: str

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


# ============================================================
# SickleGuide RAG Engine
# ============================================================

class SickleGuideRAG:

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = DEFAULT_TEMPERATURE,
        num_predict: int = DEFAULT_NUM_PREDICT,
        dense_k: int = DEFAULT_DENSE_K,
        bm25_k: int = DEFAULT_BM25_K,
        graph_k: int = DEFAULT_GRAPH_K,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        final_k: int = DEFAULT_FINAL_K,
        graph_path: str = "data/processed/graph.json",
        chunks_path: str = "data/processed/chunks.json",
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.num_predict = num_predict

        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.graph_k = graph_k
        self.candidate_k = candidate_k
        self.final_k = final_k

        self.graph_path = graph_path
        self.chunks_path = chunks_path

        self._initialized = False

        self.llm = None
        self.documents = None
        self.bm25 = None
        self.vector_store = None
        self.graph = None
        self.graph_retriever = None
        self.unified_retriever = None
        self.reranker = None
        self.reviewer = None
        self.rag_graph = None

    # ========================================================
    # Initialization
    # ========================================================

    def initialize(self) -> None:

        if self._initialized:
            return

        print(
            "\n" + "=" * 70,
            flush=True,
        )

        print(
            "Initializing SickleGuide RAG Engine",
            flush=True,
        )

        print(
            "=" * 70,
            flush=True,
        )

        # ----------------------------------------------------
        # Documents
        # ----------------------------------------------------

        print(
            "\n[1/8] Loading processed chunks...",
            flush=True,
        )

        with open(
            self.chunks_path,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        self.documents = [
            Document(
                page_content=item["page_content"],
                metadata=item.get(
                    "metadata",
                    {},
                ),
            )
            for item in data
        ]

        print(
            f"      Documents: {len(self.documents)}",
            flush=True,
        )

        if not self.documents:
            raise RuntimeError(
                "No processed documents found."
            )

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        print(
            "\n[2/8] Initializing BM25...",
            flush=True,
        )

        self.bm25 = create_bm25_retriever(
            self.documents
        )

        print(
            f"      Indexed: {self.bm25.count()}",
            flush=True,
        )

        # ----------------------------------------------------
        # Chroma
        # ----------------------------------------------------

        print(
            "\n[3/8] Loading Chroma...",
            flush=True,
        )

        self.vector_store = create_vector_store()

        print(
            f"      Vectors: {self.vector_store.count()}",
            flush=True,
        )

        # ----------------------------------------------------
        # Graph
        # ----------------------------------------------------

        print(
            "\n[4/8] Loading medical graph...",
            flush=True,
        )

        self.graph = load_graph(
            self.graph_path
        )

        print(
            f"      Nodes: {self.graph.node_count()}",
            flush=True,
        )

        print(
            f"      Edges: {self.graph.edge_count()}",
            flush=True,
        )

        self.graph_retriever = (
            create_graph_retriever(
                self.graph,
                self.documents,
            )
        )

        # ----------------------------------------------------
        # Unified retrieval
        # ----------------------------------------------------

        print(
            "\n[5/8] Creating unified retriever...",
            flush=True,
        )

        self.unified_retriever = (
            create_hybrid_graph_retriever(
                bm25_retriever=self.bm25,
                vector_store=self.vector_store,
                graph_retriever=self.graph_retriever,
                dense_k=self.dense_k,
                bm25_k=self.bm25_k,
                graph_k=self.graph_k,
                final_k=self.candidate_k,
            )
        )

        print(
            "      Dense + BM25 + Graph + RRF ready.",
            flush=True,
        )

        # ----------------------------------------------------
        # Reranker
        # ----------------------------------------------------

        print(
            "\n[6/8] Loading BGE reranker...",
            flush=True,
        )

        self.reranker = create_reranker()

        # ----------------------------------------------------
        # Generation LLM
        # ----------------------------------------------------

        print(
            "\n[7/8] Loading generation LLM...",
            flush=True,
        )

        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            num_predict=self.num_predict,
        )

        # ----------------------------------------------------
        # Grounding reviewer
        # ----------------------------------------------------

        print(
            "\n[8/8] Creating grounding reviewer...",
            flush=True,
        )

        self.reviewer = (
            self.llm.with_structured_output(
                GroundingReview,
                method="json_schema",
            )
        )

        # ----------------------------------------------------
        # LangGraph
        # ----------------------------------------------------

        print(
            "\nBuilding LangGraph workflow...",
            flush=True,
        )

        self.rag_graph = self._build_graph()

        self._initialized = True

        print(
            "\n" + "=" * 70,
            flush=True,
        )

        print(
            "SickleGuide RAG Engine READY",
            flush=True,
        )

        print(
            "=" * 70,
            flush=True,
        )

    # ========================================================
    # Safety
    # ========================================================

    def _safety_node(
        self,
        state: RAGState,
    ) -> RAGState:

        query = state["query"]

        print(
            "\n[LangGraph] Safety check...",
            flush=True,
        )

        safety_result = assess_query(
            query,
            [],
        )

        return {
            "safety_result": safety_result,
            "safety_instruction": (
                build_safety_instruction(
                    safety_result
                )
            ),
        }

    # ========================================================
    # Retrieval
    # ========================================================

    def _retrieve_node(
        self,
        state: RAGState,
    ) -> RAGState:

        query = state["query"]

        print(
            "\n[LangGraph] Unified retrieval...",
            flush=True,
        )

        candidates = (
            self.unified_retriever.retrieve(
                query,
                final_k=self.candidate_k,
            )
        )

        print(
            f"[LangGraph] Candidates: {len(candidates)}",
            flush=True,
        )

        return {
            "retrieved_documents": candidates,
        }

    # ========================================================
    # Reranking
    # ========================================================

    def _rerank_node(
        self,
        state: RAGState,
    ) -> RAGState:

        query = state["query"]

        candidates = state.get(
            "retrieved_documents",
            [],
        )

        print(
            "\n[LangGraph] Reranking...",
            flush=True,
        )

        final_documents = self.reranker.rerank(
            query=query,
            documents=candidates,
            top_k=self.final_k,
        )

        print(
            f"[LangGraph] Final evidence: "
            f"{len(final_documents)}",
            flush=True,
        )

        return {
            "final_documents": final_documents,
        }

    # ========================================================
    # Generation
    # ========================================================

    def _generation_node(
        self,
        state: RAGState,
    ) -> RAGState:

        query = state["query"]

        documents = state.get(
            "final_documents",
            [],
        )

        safety_result = assess_query(
            query,
            documents,
        )

        safety_instruction = (
            build_safety_instruction(
                safety_result
            )
        )

        if documents:

            prompt = build_prompt(
                query=query,
                documents=documents,
                safety_instruction=safety_instruction,
            )

        else:

            prompt = build_no_evidence_prompt(
                query=query,
                safety_instruction=safety_instruction,
            )

        print(
            "\n[LangGraph] Generating answer...",
            flush=True,
        )

        response = self.llm.invoke(
            prompt
        )

        content = getattr(
            response,
            "content",
            str(response),
        )

        content = str(
            content
        ).strip()

        return {
            "raw_answer": content,
            "grounded_answer": content,
            "safety_result": safety_result,
            "safety_instruction": safety_instruction,
            "grounding_retry_count": 0,
            "grounding_failed": False,
        }

    # ========================================================
    # Grounding Review
    # ========================================================

    def _grounding_review_node(
        self,
        state: RAGState,
    ) -> RAGState:

        query = state["query"]

        answer = state.get(
            "grounded_answer",
            state.get(
                "raw_answer",
                "",
            ),
        )

        documents = state.get(
            "final_documents",
            [],
        )

        print(
            "\n[LangGraph] Grounding review...",
            flush=True,
        )

        evidence_blocks = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            citation = document.metadata.get(
                "citation",
                (
                    f"{document.metadata.get('source', 'Unknown source')}"
                    f" — Page "
                    f"{document.metadata.get('page_number', 'Unknown')}"
                ),
            )

            evidence_blocks.append(
                "\n".join(
                    [
                        f"[Evidence {index}]",
                        f"Citation: {citation}",
                        "Content:",
                        document.page_content,
                    ]
                )
            )

        evidence_text = "\n\n".join(
            evidence_blocks
        )

        review_prompt = f"""
You are a strict but fair medical RAG grounding evaluator.

QUESTION:
{query}

RETRIEVED EVIDENCE:
{evidence_text}

GENERATED ANSWER:
{answer}

YOUR JOB:
Determine whether the important medical claims in the generated
answer are supported by the retrieved evidence.

GROUNDING RULES:

1. Direct statements in the evidence count as support.
2. Faithful paraphrases of direct statements count as support.
3. A concise summary of a sentence in the evidence counts as support
   when it preserves the original meaning.
4. If a clinical question in the evidence explicitly lists
   interventions being evaluated, an answer that accurately says
   those interventions "were evaluated" is SUPPORTED.
5. Do not require the answer to copy the evidence word-for-word.
6. Do not confuse "evaluated" with "recommended".
7. Do not confuse "suggested" with "strongly recommended".
8. Do not confuse "very low certainty" with "ineffective".
9. Do not confuse "very low certainty" with "effective".
10. Do not infer comparative effectiveness unless explicitly supported.
11. Do not infer a universal recommendation from a scenario-specific
    recommendation.
12. General medical knowledge is NOT evidence.
13. If the generated answer contains a claim that cannot be traced
    to the evidence, mark it unsupported.
14. A claim is NOT unsupported merely because the evidence is
    presented as a clinical question; if the answer accurately
    describes what that question explicitly asks or evaluates,
    that statement is grounded.
15. Be conservative about medical conclusions, but do not reject
    faithful summaries of explicit source text.

IMPORTANT EXAMPLE:

Evidence:
"KQ15. What is the most effective treatment among transfusion,
exchange transfusion, supportive therapy, steroids, and/or
antibiotics?"

Supported answer:
"The evidence review evaluated transfusion, exchange transfusion,
supportive therapy, steroids, and antibiotics for ACS."

NOT supported:
"Transfusion is recommended as the best treatment."

The first is a faithful description of the source.
The second adds an unsupported clinical conclusion.

GROUNDING DECISION:

Set grounded=true when every important medical claim is either:
- directly stated in the evidence, or
- a faithful paraphrase that preserves the original meaning.

Set grounded=false only for claims that add unsupported medical
information, recommendations, effectiveness claims, or conclusions.

If grounded=false, list the unsupported medical claims explicitly.
"""

        review = self.reviewer.invoke(
            review_prompt
        )

        if isinstance(
            review,
            GroundingReview,
        ):
            review_dict = review.model_dump()

        else:
            review_dict = (
                GroundingReview
                .model_validate(
                    review
                )
                .model_dump()
            )

        print(
            f"[LangGraph] Grounded: "
            f"{review_dict['grounded']}",
            flush=True,
        )

        unsupported = review_dict.get(
            "unsupported_claims",
            [],
        )

        for claim in unsupported:
            print(
                f"  - {claim}",
                flush=True,
            )

        return {
            "grounding_review": review_dict,
        }

    # ========================================================
    # Grounding routing
    # ========================================================

    def _route_after_grounding(
        self,
        state: RAGState,
    ) -> str:

        review = state.get(
            "grounding_review",
            {},
        )

        grounded = bool(
            review.get(
                "grounded",
                False,
            )
        )

        retry_count = state.get(
            "grounding_retry_count",
            0,
        )

        if grounded:
            return "citations"

        if retry_count < MAX_GROUNDING_RETRIES:
            return "regenerate"

        return "grounding_failure"

    # ========================================================
    # Regeneration
    # ========================================================

    def _regenerate_node(
        self,
        state: RAGState,
    ) -> RAGState:

        query = state["query"]

        previous_answer = state.get(
            "grounded_answer",
            "",
        )

        documents = state.get(
            "final_documents",
            [],
        )

        review = state.get(
            "grounding_review",
            {},
        )

        unsupported_claims = review.get(
            "unsupported_claims",
            [],
        )

        safety_instruction = state.get(
            "safety_instruction",
            "",
        )

        print(
            "\n[LangGraph] Regenerating due to grounding failure...",
            flush=True,
        )

        prompt = (
            build_grounded_regeneration_prompt(
                query=query,
                documents=documents,
                previous_answer=previous_answer,
                unsupported_claims=unsupported_claims,
                safety_instruction=safety_instruction,
            )
        )

        response = self.llm.invoke(
            prompt
        )

        content = getattr(
            response,
            "content",
            str(response),
        )

        content = str(
            content
        ).strip()

        retry_count = state.get(
            "grounding_retry_count",
            0,
        )

        return {
            "grounded_answer": content,
            "raw_answer": content,
            "grounding_retry_count": (
                retry_count + 1
            ),
        }

    # ========================================================
    # Fail Closed
    # ========================================================

    def _grounding_failure_node(
        self,
        state: RAGState,
    ) -> RAGState:

        documents = state.get(
            "final_documents",
            [],
        )

        print(
            "\n[LangGraph] Grounding failed after retry.",
            flush=True,
        )

        print(
            "[LangGraph] Failing closed instead of returning "
            "an unsupported medical answer.",
            flush=True,
        )

        safe_answer = (
            "I could not generate a sufficiently "
            "evidence-grounded answer to this question "
            "from the retrieved SickleGuide sources. "
            "The available evidence did not support all "
            "of the claims required for a reliable answer."
        )

        if documents:

            source_lines = []

            seen = set()

            for index, document in enumerate(
                documents,
                start=1,
            ):

                citation = document.metadata.get(
                    "citation"
                )

                if not citation:
                    citation = (
                        f"{document.metadata.get('source', 'Unknown source')}"
                        f" — Page "
                        f"{document.metadata.get('page_number', 'Unknown')}"
                    )

                if citation in seen:
                    continue

                seen.add(
                    citation
                )

                source_lines.append(
                    f"[{index}] {citation}"
                )

            if source_lines:

                safe_answer += (
                    "\n\nThe retrieved sources were:\n"
                    + "\n".join(
                        source_lines
                    )
                )

        return {
            "grounding_failed": True,
            "grounded_answer": safe_answer,
            "final_answer": safe_answer,
        }

    # ========================================================
    # Citation Validation
    # ========================================================

    def _citation_node(
        self,
        state: RAGState,
    ) -> RAGState:

        answer = state.get(
            "grounded_answer",
            state.get(
                "raw_answer",
                "",
            ),
        )

        documents = state.get(
            "final_documents",
            [],
        )

        print(
            "\n[LangGraph] Validating citations...",
            flush=True,
        )

        citation_map = {
            index: document.metadata.get(
                "citation",
                (
                    f"{document.metadata.get('source', 'Unknown source')}"
                    f" — Page "
                    f"{document.metadata.get('page_number', 'Unknown')}"
                ),
            )
            for index, document in enumerate(
                documents,
                start=1,
            )
        }

        citation_validation = (
            validate_citations(
                answer,
                citation_map,
            )
        )

        final_answer = (
            format_answer_with_citations(
                answer,
                documents,
            )
        )

        return {
            "final_answer": final_answer,
            "citation_validation": (
                citation_validation
            ),
        }

    # ========================================================
    # Safety Output
    # ========================================================

    def _safety_output_node(
        self,
        state: RAGState,
    ) -> RAGState:

        answer = state.get(
            "final_answer",
            "",
        )

        safety_result = state[
            "safety_result"
        ]

        final_answer = (
            apply_safety_notice(
                answer,
                safety_result,
            )
        )

        return {
            "final_answer": final_answer,
        }

    # ========================================================
    # LangGraph
    # ========================================================

    def _build_graph(self):

        workflow = StateGraph(
            RAGState
        )

        workflow.add_node(
            "safety",
            self._safety_node,
        )

        workflow.add_node(
            "retrieve",
            self._retrieve_node,
        )

        workflow.add_node(
            "rerank",
            self._rerank_node,
        )

        workflow.add_node(
            "generate",
            self._generation_node,
        )

        workflow.add_node(
            "grounding_review",
            self._grounding_review_node,
        )

        workflow.add_node(
            "regenerate",
            self._regenerate_node,
        )

        workflow.add_node(
            "grounding_failure",
            self._grounding_failure_node,
        )

        workflow.add_node(
            "citations",
            self._citation_node,
        )

        workflow.add_node(
            "safety_output",
            self._safety_output_node,
        )

        workflow.set_entry_point(
            "safety"
        )

        workflow.add_edge(
            "safety",
            "retrieve",
        )

        workflow.add_edge(
            "retrieve",
            "rerank",
        )

        workflow.add_edge(
            "rerank",
            "generate",
        )

        workflow.add_edge(
            "generate",
            "grounding_review",
        )

        workflow.add_conditional_edges(
            "grounding_review",
            self._route_after_grounding,
            {
                "regenerate": "regenerate",
                "citations": "citations",
                "grounding_failure": "grounding_failure",
            },
        )

        workflow.add_edge(
            "regenerate",
            "grounding_review",
        )

        workflow.add_edge(
            "grounding_failure",
            "safety_output",
        )

        workflow.add_edge(
            "citations",
            "safety_output",
        )

        workflow.add_edge(
            "safety_output",
            END,
        )

        return workflow.compile()

    # ========================================================
    # Public API
    # ========================================================

    def invoke(
        self,
        query: str,
    ) -> Dict[str, Any]:

        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "query must be a string"
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty"
            )

        self.initialize()

        print(
            "\n" + "=" * 70,
            flush=True,
        )

        print(
            f"SICKLEGUIDE QUERY:\n{query}",
            flush=True,
        )

        print(
            "=" * 70,
            flush=True,
        )

        return self.rag_graph.invoke(
            {
                "query": query,
            }
        )

    def answer(
        self,
        query: str,
    ) -> str:

        result = self.invoke(
            query
        )

        return result.get(
            "final_answer",
            "",
        )


# ============================================================
# Factory
# ============================================================

def create_rag_engine(
    model_name: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    num_predict: int = DEFAULT_NUM_PREDICT,
) -> SickleGuideRAG:

    return SickleGuideRAG(
        model_name=model_name,
        base_url=base_url,
        temperature=temperature,
        num_predict=num_predict,
    )


# ============================================================
# Backward-compatible helpers
# ============================================================

def get_llm(
    model_name: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    num_predict: int = DEFAULT_NUM_PREDICT,
):

    return ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
        num_predict=num_predict,
    )


def generate(
    prompt: str,
    llm=None,
) -> str:

    if not isinstance(
        prompt,
        str,
    ):
        raise TypeError(
            "prompt must be a string"
        )

    prompt = prompt.strip()

    if not prompt:
        raise ValueError(
            "prompt cannot be empty"
        )

    if llm is None:
        llm = get_llm()

    response = llm.invoke(
        prompt
    )

    content = getattr(
        response,
        "content",
        str(response),
    )

    return str(
        content
    ).strip()


def test_llm(
    llm=None,
) -> dict:

    if llm is None:
        llm = get_llm()

    response = llm.invoke(
        "Reply with exactly: SickleGuide LLM OK"
    )

    content = getattr(
        response,
        "content",
        str(response),
    )

    return {
        "success": True,
        "response": str(
            content
        ).strip(),
    }