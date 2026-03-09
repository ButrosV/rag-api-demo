from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str


class ContextChunk(BaseModel):
    id: str
    text: str
    page: int | None = None
    score: float


class AskResponse(BaseModel):
    answer: str
    contexts: list[ContextChunk]

