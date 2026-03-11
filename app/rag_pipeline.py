from dataclasses import dataclass

from chromadb import PersistentClient

from app.config import get_settings
from ingestion.build_index import embedd_chunks
from ingestion.loaders import Chunk

settings = get_settings()
client = PersistentClient(path=settings.index_dir)
collection = client.get_collection(settings.collection_name)


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
    """Build system + user messages with structured contexts."""
    raise NotImplementedError


def retrieve_and_answer(question: str) -> tuple[str, list[RetrievedChunk]]:
    # Placeholder implementation to be filled in later.
    raise NotImplementedError
