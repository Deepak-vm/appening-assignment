import statistics
from typing import Literal, TypedDict

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from .config import settings
from .prompts import NOT_FOUND_MESSAGE, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .vectorstore import get_index

_embedder = GoogleGenerativeAIEmbeddings(
    model=settings.embedding_model,
    google_api_key=settings.gemini_api_key,
)
_llm = ChatGroq(api_key=settings.groq_api_key, model=settings.llm_model)


class RAGState(TypedDict):
    question: str
    chunks: list[dict]  # each: {text, page, score}
    answer: str
    confidence: float


def _embed_query(text: str) -> list[float]:
    return _embedder.embed_query(text)


def _confidence(chunks: list[dict]) -> float:
    """Mean retrieval similarity — a retrieval-side heuristic, not a probability."""
    return round(statistics.mean(c["score"] for c in chunks), 4) if chunks else 0.0


# nodes

def retrieve(state: RAGState) -> dict:
    """Embed the question and fetch the top-k chunks from Pinecone."""
    result = get_index().query(
        vector=_embed_query(state["question"]),
        top_k=settings.top_k,
        include_metadata=True,
    )
    chunks = [
        {"text": m.metadata["text"], "page": int(m.metadata["page"]), "score": float(m.score)}
        for m in result.matches
    ]
    return {"chunks": chunks}


def route_by_relevance(state: RAGState) -> Literal["generate", "fallback"]:
    top = max((c["score"] for c in state["chunks"]), default=0.0)
    return "generate" if top >= settings.relevance_threshold else "fallback"


def generate(state: RAGState) -> dict:
    context = "\n\n".join(
        f"[{i}] (page {c['page']})\n{c['text']}"
        for i, c in enumerate(state["chunks"], start=1)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            context=context, question=state["question"])},
    ]
    response = _llm.invoke(messages)
    return {
        "answer": response.content.strip(),
        "confidence": _confidence(state["chunks"]),
    }


def fallback(state: RAGState) -> dict:
    return {"answer": NOT_FOUND_MESSAGE, "confidence": _confidence(state["chunks"])}


# graph

def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("fallback", fallback)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", route_by_relevance)
    g.add_edge("generate", END)
    g.add_edge("fallback", END)
    return g.compile()


rag_graph = build_graph()
