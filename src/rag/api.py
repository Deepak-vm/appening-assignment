"""FastAPI application exposing POST /chat and GET /health."""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .graph import rag_graph
from .schemas import ChatRequest, ChatResponse, Source

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="Answers questions strictly from the Agentic AI eBook.",
    version="0.1.0",
)


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["rag"])
def chat(body: ChatRequest):
    try:
        state = rag_graph.invoke({"question": body.question, "chunks": [], "scores": []})
    except Exception as exc:
        logger.exception("Graph execution failed")
        raise HTTPException(status_code=500, detail="Internal pipeline error") from exc

    sources = [
        Source(text=c["text"], page=c["page"], score=c["score"])
        for c in state.get("chunks", [])
    ]
    return ChatResponse(
        answer=state["answer"],
        confidence=state["confidence"],
        sources=sources,
    )
