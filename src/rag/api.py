import logging
from fastapi import FastAPI, HTTPException
from .graph import rag_graph
from .schemas import ChatRequest, ChatResponse, Source

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="Answers questions strictly from the Agentic AI eBook.",
)


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["rag"])
def chat(body: ChatRequest):
    state = rag_graph.invoke({"question": body.question, "chunks": []})

    sources = [
        Source(text=c["text"], page=c["page"], score=c["score"])
        for c in state.get("chunks", [])
    ]
    return ChatResponse(
        answer=state["answer"],
        confidence=state["confidence"],
        sources=sources,
    )
