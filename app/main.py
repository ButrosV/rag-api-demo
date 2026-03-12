from fastapi import FastAPI, HTTPException

from app.schemas import AskResponse, AskRequest
from app.rag_pipeline import api_retrieve_and_answer

app = FastAPI(title="NVIDIA RAG API")


@app.get("/health")
async def health() -> dict[str, str]:
    """Basic health check - returns 200 if API + index are ready"""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """
    Handle RAG question about input document.
    Strategy:
    1. Validate non-empty question
    2. Retrieve relevant chunks via vector search
    3. Generate grounded LLM answer with context

    :param request: Question about NVIDIA document
    :return: Answer with supporting context chunks and scores
    :raises HTTPException: 400 for empty questions
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    return api_retrieve_and_answer(
        request.question, top_k=5
    )  #  leave top_k=5 fdespite being optional - FastAPI doesn't handle optional query params in POST body well - it treats them as required and throws 422 if missing, even with defaults in the function signature
