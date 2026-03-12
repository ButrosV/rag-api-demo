from pydantic import BaseModel, Field
from datetime import datetime, timezone


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="User query for RAG pipeline")


class ContextChunk(BaseModel):
    id: str = Field(..., description="Chunk identifier (p2_c1)")
    text: str = Field(
        ..., max_length=1200, description="Chunk context text (800 char pipeline chunking + 400 char buffer)"
    )
    page: int | None = Field(default=None, description="Source page (if available)")
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Cosine similarity score (1.0->perfect) for chunk vs question"
    )


class AskResponse(BaseModel):
    answer: str = Field(..., min_length=1, max_length=2000, description="LLM-generated answer string")
    contexts: list[ContextChunk] = Field(
        ..., min_length=1, max_length=10, description="List of retrieved context chunks (max 10)"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Response generation timestamp (UTC)"
    )
