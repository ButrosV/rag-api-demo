import logging
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException

from app.schemas import AskResponse, AskRequest
from app.rag_pipeline import api_retrieve_and_answer
from app.config import get_settings, setup_logging


settings = get_settings()
setup_logging(settings.log_level)

logger = logging.getLogger(__name__)

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

    request_id = str(uuid4())[:8]
    start = time.monotonic()

    response = api_retrieve_and_answer(
        request.question, top_k=5
    )  # leave top_k=5 despite being optional - FastAPI doesn't handle optional query params in POST body well - it treats them as required and throws 422 if missing, even with defaults in the function signature

    elapsed_ms = int((time.monotonic() - start) * 1000)
    answer_preview = response.answer[:80].replace("\n", " ")
    if len(response.answer) > 80:
        answer_preview += "…"

    logger.info(
        "ask_complete request_id=%s question_len=%d latency_ms=%d num_chunks=%d answer_len=%d answer_preview=%r",
        request_id,
        len(request.question),
        elapsed_ms,
        len(response.contexts),
        len(response.answer),
        answer_preview,
    )

    return response
