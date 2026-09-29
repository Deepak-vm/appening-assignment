"""LangGraph pipeline: retrieve → check_relevance → generate | fallback."""

import statistics
from typing import TypedDict

from langgraph.graph import END, StateGraph
from openai import OpenAI

from .config import settings
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .vectorstore import get_index

# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    question: str
    chunks: list[dict]   # list of {text, page, score}
    scores: list[float]
    answer: str
    confidence: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_openai_client = OpenAI(api_key=settings.openai_api_key)
_index = None  # lazily initialised to avoid cold-start in tests


def _get_index():
    global _index
    if _index is None:
        _index = get_index()
    return _index


def _embed_query(text: str) -> list[float]:
    response = _openai_client.embeddings.create(
        model=settings.embedding_model,
        input=[text],
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def retrieve(state: RAGState) -> dict:
    """Embed the question and fetch top-k chunks from Pinecone."""
    vector = _embed_query(state["question"])
    result = _get_index().query(
        vector=vector,
        top_k=settings.top_k,
        include_metadata=True,
    )
    chunks = []
    scores = []
    for match in result.matches:
        chunks.append(
            {
                "text": match.metadata["text"],
                "page": int(match.metadata["page"]),
                "score": float(match.score),
            }
        )
        scores.append(float(match.score))
    return {"chunks": chunks, "scores": scores}


def check_relevance(state: RAGState) -> str:
    """Routing function: returns 'generate' or 'fallback'."""
    if not state["scores"]:
        return "fallback"
    top_score = max(state["scores"])
    return "generate" if top_score >= settings.relevance_threshold else "fallback"


def generate(state: RAGState) -> dict:
    """Call the LLM with the retrieved context and return answer + confidence."""
    context_parts = []
    for i, chunk in enumerate(state["chunks"], start=1):
        context_parts.append(f"[{i}] (page {chunk['page']})\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    user_message = USER_PROMPT_TEMPLATE.format(
        context=context,
        question=state["question"],
    )

    response = _openai_client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )
    answer = response.choices[0].message.content.strip()

    # Confidence = mean of top-k cosine scores (a retrieval-side heuristic).
    # Formula: mean(scores[:top_k]).  Scores are already in [0, 1] for cosine.
    confidence = round(statistics.mean(state["scores"]), 4) if state["scores"] else 0.0

    return {"answer": answer, "confidence": confidence}


def fallback(state: RAGState) -> dict:
    """Return a canned refusal when retrieval scores are below the threshold."""
    return {
        "answer": "The document does not contain enough information to answer this question.",
        "confidence": round(max(state["scores"]) if state["scores"] else 0.0, 4),
    }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(RAGState)

    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("fallback", fallback)

    g.set_entry_point("retrieve")
    g.add_conditional_edges(
        "retrieve",
        check_relevance,
        {"generate": "generate", "fallback": "fallback"},
    )
    g.add_edge("generate", END)
    g.add_edge("fallback", END)

    return g.compile()


# Module-level compiled graph used by the API.
rag_graph = build_graph()
