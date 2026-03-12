from dataclasses import dataclass

from chromadb import PersistentClient
from openai import OpenAI, APIError, RateLimitError
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from ingestion.build_index import embedd_chunks
from ingestion.loaders import Chunk
from app.schemas import AskResponse, ContextChunk

settings = get_settings()
client_emb = PersistentClient(path=settings.index_dir)
collection = client_emb.get_collection(settings.collection_name)
client_llm = OpenAI(api_key=settings.openai_api_key)


@dataclass
class RetrievedChunk:
    id: str
    text: str
    page: int | None
    score: float


def retrieve(question: str, top_k: int = 5) -> list[RetrievedChunk]:
    """
    Retrieve top_k most relevant text chunks via vector similarity search.
    Embed the question using the same model as document chunks, queries the ChromaDB vector store,
    and convert distance scores to similarity scores (1.0 = perfect match, 0.0 = no similarity).

    :param question: Natural language query about the document.
    :param top_k: Maximum number of most relevant chunks to return. Defaults to 5.
    :return: List of RetrievedChunk objects ranked by relevance, each containing chunk ID,
             text content, source page number, and normalized similarity score [0.0, 1.0].
    """
    query_chunk = Chunk(id="query", text=question, page=None, start_char=0, length=len(question))
    q_embedding = embedd_chunks([query_chunk])[0]

    results = collection.query(query_embeddings=[q_embedding], n_results=top_k)

    mapped_chunks = []
    for id, document, metadata, distance in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0], strict=True
    ):
        mapped_chunks.append(RetrievedChunk(id=id, text=document, page=metadata["page"], score=1 - distance))

    return mapped_chunks


def build_prompt(question: str, contexts: list[RetrievedChunk]) -> tuple[str, list[dict]]:
    """
    Construct OpenAI chat completion messages for RAG pipeline.

    Create standard 2-message format: system instruction + user query with embedded contexts.
    Formats retrieved chunks with page numbers and clear separators for optimal LLM comprehension.

    :param question: User question about NVIDIA document.
    :param contexts: Retrieved chunks ranked by relevance (highest scores first).
    :return: Tuple of (raw system prompt string, OpenAI messages list) for chat completion.
    """
    SYSTEM_PROMPT = """You are an assistant that answers questions based ONLY 
    on the provided NVIDIA document context.
    If the answer isn't in the context, say "I don't know".
    Keep answers to 2-3 sentences maximum and cite context implicitly."""

    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    context_lines = "\n---\n".join([f"(Page {c.page}) {c.text}" for c in contexts])

    context_message = {
        "role": "user",
        "content": f"""Context:
                       ---
                       {context_lines}
                       ---
                       Question: {question}""",
    }

    return (SYSTEM_PROMPT, [system_message, context_message])


# tenacity retry decorator more fro production than MVP, but added here to demonstrate how to handle transient OpenAI API errors gracefully.
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=16),
    retry=(lambda e: isinstance(e, (APIError, RateLimitError))),
    reraise=True,
)
def prompt_llm(messages: list[dict], temperature: float = 0.1, max_tokens: int = 250) -> str:
    """
    Generate LLM response for RAG with retry logic.
    Retries 3x on API/RateLimit errors (4s→8s→16s backoff).

    :param messages: OpenAI chat messages list
    :param temperature: Randomness (0.1=default for conservative/factual RAG)
    :param max_tokens: Response length limit (250=default)

    :return: Cleaned answer string
    :raises APIError: After 3 failed retries
    """
    response = client_llm.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    answer = response.choices[0].message.content.strip()

    return answer


def retrieve_and_answer(question: str, top_k: int = 5) -> tuple[str, list[RetrievedChunk]]:
    """
    Complete RAG pipeline: retrieve → prompt → LLM answer.

    Strategy:
    1. Vector search top_k chunks from NVIDIA index
    2. Build OpenAI messages with context + question
    3. Generate grounded answer with retries

    :param question: User query about the search document
    :param top_k: Number of context chunks (default: 5)
    :return: (answer string, retrieved chunks list)
    """
    chunks_search = retrieve(question, top_k=top_k)
    _, messages = build_prompt(question, chunks_search)
    answer = prompt_llm(messages)

    return answer, chunks_search


def api_retrieve_and_answer(question: str, top_k: int = 5) -> AskResponse:
    """
    Format RAG output for API response.

    Converts RetrievedChunk to ContextChunk and packages with LLM answer.

    :param question: User query for RAG pipeline
    :param top_k: Number of context chunks to retrieve (default: 5)
    :return: AskResponse object containing answer and context details
    """
    answer, retrieved_chunks = retrieve_and_answer(question, top_k=top_k)

    context_chunks = [
        ContextChunk(id=chunk.id, text=chunk.text, page=chunk.page, score=chunk.score) for chunk in retrieved_chunks
    ]

    return AskResponse(answer=answer, contexts=context_chunks)
