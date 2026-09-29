from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class Source(BaseModel):
    text: str
    page: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: list[Source]
